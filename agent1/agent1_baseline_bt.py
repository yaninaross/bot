# %% Agent 1 — 5-Way Filter Comparison Backtest
#
# Five strategies compared:
#   A) BASELINE:         no CS filter, no pattern filter
#   B) CS ONLY:          CS filter,    no pattern filter
#   C) SUST.DROP ONLY:   no CS filter, sustained_drop pattern filter
#   D) CS + SUST.DROP:   CS filter  +  sustained_drop pattern filter
#   E) CS+SD+GUARDS:     D + gap_max 75% + Polygon float >= 10M (hard exclude)
#
# Colab-cell friendly. ~8-10 min runtime (API-bound).

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

BACKTEST_START = '2026-01-02'
BACKTEST_END   = '2026-09-12'

GAP_MIN_PCT    = 5.0
GAP_MAX_PCT    = 75.0
PRICE_MIN      = 10.0
FLOAT_MIN      = 10_000_000
TOP_N          = 5
PRE_FILTER_N   = 15

PEAK_WINDOW    = 3
MAX_PEAK_BAR   = 12
FALLBACK_BAR   = 15

CAPITAL        = 100_000

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "YOUR_POLYGON_API_KEY")
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
# ASYNC TICKER REFERENCE FETCH (asset type for CS filter)
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
            return tk, {
                'type': res.get('type'),
                'name': res.get('name', ''),
                'shares': res.get('weighted_shares_outstanding'),
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


# ═══════════════════════════════════════════════════════════════════════════════
# OPEN PATTERN CLASSIFICATION (E7 logic)
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL HIGH DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def find_local_high(bars, window=PEAK_WINDOW, max_peak_bar=MAX_PEAK_BAR,
                    fallback_bar=FALLBACK_BAR):
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


# ═══════════════════════════════════════════════════════════════════════════════
# SCANNER — returns ALL qualifying candidates sorted by gap
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
            'volume_today': today[tk]['v'],
        })

    candidates.sort(key=lambda x: -x['gap_pct'])

    nxt = _next_n_td(dt_str, 2)
    t1 = nxt[0].strftime('%Y-%m-%d') if len(nxt) >= 1 else None
    t2 = nxt[1].strftime('%Y-%m-%d') if len(nxt) >= 2 else None
    for s in candidates:
        s['t1_date'] = t1
        s['t2_date'] = t2
    return candidates


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

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
# ═══════════════════════════════════════════════════════════════════════════════
#
#  BACKTEST ENGINE
#
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest():
    t_total = time.time()
    timings = {}

    print(f"\n{'='*80}")
    print(f"  AGENT 1 — 5-WAY FILTER COMPARISON")
    print(f"  {BACKTEST_START} → {BACKTEST_END}")
    print(f"  Universe: ALL US equities")
    print(f"  A) Baseline:    top {TOP_N} by gap, no additional filters")
    print(f"  B) CS Only:     common stock filter, then top {TOP_N}")
    print(f"  C) Sust.Drop:   sustained_drop pattern filter, then top {TOP_N}")
    print(f"  D) CS+Sust.Drop: CS + sustained_drop, then top {TOP_N}")
    print(f"  E) CS+SD+Guards: D + gap<={GAP_MAX_PCT:.0f}% + Polygon float>={FLOAT_MIN/1e6:.0f}M (hard)")
    print(f"  Entry: local_high + fallback  |  Hold: T+2 close")
    print(f"  Capital: ${CAPITAL:,.0f}/day, equal split, non-compounding")
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

    # ── Phase 1: Scan ──
    t1 = time.time()
    print(f"\n  Phase 1: Scanning for gap-ups (all US equities)...")
    all_candidates = {}
    total_cands = 0
    for i, dt in enumerate(dates):
        cands = scan_day(dt)
        all_candidates[dt] = cands
        total_cands += len(cands)
        if (i+1) % 30 == 0 or i == n_dates - 1:
            print(f"    [{i+1:>3}/{n_dates}] {dt}  |  {total_cands} total  |  {time.time()-t_total:.0f}s")
    timings['scan'] = time.time() - t1

    # ── Phase 2: Fetch minute bars for top PRE_FILTER_N per day ──
    t2 = time.time()
    all_pairs = set()
    for dt, cands in all_candidates.items():
        for c in cands[:PRE_FILTER_N]:
            all_pairs.add((c['ticker'], dt))

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
    timings['bars'] = time.time() - t2
    print(f"  → Got bars for {len(all_bars)}/{len(all_pairs)} pairs")

    # ── Phase 3: Fetch ticker reference data (asset type for CS filter) ──
    t3 = time.time()
    all_tickers = set()
    for cands in all_candidates.values():
        for c in cands[:PRE_FILTER_N]:
            all_tickers.add(c['ticker'])
    print(f"\n  Phase 3: Fetching reference data for {len(all_tickers)} tickers...")
    tk_list = sorted(all_tickers)
    for c_start in range(0, len(tk_list), 200):
        chunk = tk_list[c_start:c_start + 200]
        fetch_ticker_refs(chunk)
        done = min(c_start + 200, len(tk_list))
        if done % 400 == 0 or done == len(tk_list):
            print(f"    {done}/{len(tk_list)}  |  {time.time()-t_total:.0f}s")
    timings['refs'] = time.time() - t3
    n_cs = sum(1 for tk in all_tickers if _ticker_ref.get(tk, {}).get('type') == 'CS')
    n_etf = sum(1 for tk in all_tickers if _ticker_ref.get(tk, {}).get('type') in _ETF_TYPES)
    n_has_float = sum(1 for tk in all_tickers if _ticker_ref.get(tk, {}).get('shares') is not None)
    n_float_ok = sum(1 for tk in all_tickers
                     if (_ticker_ref.get(tk, {}).get('shares') or 0) >= FLOAT_MIN)
    print(f"  → {n_cs} CS, {n_etf} ETF, {len(all_tickers) - n_cs - n_etf} other")
    print(f"  → Float data: {n_has_float}/{len(all_tickers)} have shares, "
          f"{n_float_ok} >= {FLOAT_MIN/1e6:.0f}M")

    # ── Phase 4: Classify patterns + find entries ──
    t4 = time.time()
    pattern_cache = {}
    entry_cache = {}
    for key, rth_bars in all_bars.items():
        pattern_cache[key] = classify_open_pattern(rth_bars)
        if rth_bars and len(rth_bars) >= PEAK_WINDOW * 2 + 1:
            entry_cache[key] = find_local_high(rth_bars)
        else:
            entry_cache[key] = None
    timings['classify'] = time.time() - t4

    pat_counts = defaultdict(int)
    for p in pattern_cache.values():
        pat_counts[p] += 1
    print(f"\n  Phase 4: Pattern classification")
    for pat in ['sustained_drop', 'drop_then_recover', 'mild_drop', 'flat_or_bounce', 'no_data']:
        print(f"    {pat:25s}  {pat_counts.get(pat, 0):>5d}  "
              f"({pat_counts.get(pat, 0)/len(pattern_cache)*100:.1f}%)" if pattern_cache else '')

    # ── Phase 5: Fetch outcome dates ──
    print(f"\n  Phase 5: Fetching outcome dates...")
    outcome_dates = set()
    for cands in all_candidates.values():
        for c in cands[:PRE_FILTER_N]:
            outcome_dates.add(c['scan_date'])
            if c.get('t1_date'): outcome_dates.add(c['t1_date'])
            if c.get('t2_date'): outcome_dates.add(c['t2_date'])
    to_fetch = sorted(od for od in outcome_dates if od not in _gd_cache)
    for od in to_fetch:
        try: fetch_grouped_daily(od)
        except: pass
    timings['outcomes'] = time.time() - t4

    # ── Phase 6: Build trades for all 5 scenarios ──
    t5 = time.time()
    print(f"\n  Phase 6: Building trade lists...")

    def _build_trade(c):
        tk, dt = c['ticker'], c['scan_date']
        key = (tk, dt)
        entry = entry_cache.get(key)
        if entry is None:
            return None
        tc = copy.deepcopy(c)
        tc['entry_price'] = entry['entry_price']
        tc['entry_time'] = entry['entry_time']
        tc['entry_type'] = entry['entry_type']
        tc['open_pattern'] = pattern_cache.get(key, 'no_data')
        tc['asset_type'] = _ticker_ref.get(tk, {}).get('type', 'UNK')
        ep = tc['entry_price']
        t0d = _gd_cache.get(dt, {}).get(tk, {})
        tc['pnl_eod'] = round((ep - t0d['c']) / ep * 100, 3) if t0d.get('c') else None
        t1d = _gd_cache.get(tc.get('t1_date', ''), {}).get(tk, {})
        tc['pnl_t1'] = round((ep - t1d['c']) / ep * 100, 3) if t1d.get('c') else None
        t2d = _gd_cache.get(tc.get('t2_date', ''), {}).get(tk, {})
        tc['pnl_t2'] = round((ep - t2d['c']) / ep * 100, 3) if t2d.get('c') else None
        return tc

    baseline_trades = []
    cso_trades = []
    sdo_trades = []
    cssd_trades = []
    guard_trades = []
    bl_cnt = {'lh': 0, 'fb': 0, 'ne': 0, 'nb': 0}
    co_cnt = {'lh': 0, 'fb': 0, 'ne': 0, 'nb': 0}
    so_cnt = {'lh': 0, 'fb': 0, 'ne': 0, 'nb': 0}
    cd_cnt = {'lh': 0, 'fb': 0, 'ne': 0, 'nb': 0}
    gu_cnt = {'lh': 0, 'fb': 0, 'ne': 0, 'nb': 0}

    def _append_trade(c, trade_list, cnt):
        key = (c['ticker'], c['scan_date'])
        trade = _build_trade(c)
        if trade:
            cnt['lh' if trade['entry_type'] == 'local_high' else 'fb'] += 1
            trade_list.append(trade)
        elif key not in all_bars:
            cnt['nb'] += 1
        else:
            cnt['ne'] += 1

    for dt in dates:
        cands = all_candidates.get(dt, [])

        # A) Baseline: top N by gap (no filters)
        for c in cands[:TOP_N]:
            key = (c['ticker'], dt)
            c['open_pattern'] = pattern_cache.get(key, 'no_data')
            c['asset_type'] = _ticker_ref.get(c['ticker'], {}).get('type', 'UNK')
            _append_trade(c, baseline_trades, bl_cnt)

        # B) CS Only: filter to CS, then top N
        cs_qualified = []
        for c in cands[:PRE_FILTER_N]:
            tk = c['ticker']
            at = _ticker_ref.get(tk, {}).get('type', 'UNK')
            if at != 'CS':
                continue
            cc = copy.deepcopy(c)
            cc['asset_type'] = at
            cc['open_pattern'] = pattern_cache.get((tk, dt), 'no_data')
            cs_qualified.append(cc)
        for c in cs_qualified[:TOP_N]:
            _append_trade(c, cso_trades, co_cnt)

        # C) Sust.Drop Only: filter to sustained_drop (any asset type), then top N
        sd_qualified = []
        for c in cands[:PRE_FILTER_N]:
            tk = c['ticker']
            key = (tk, dt)
            pat = pattern_cache.get(key, 'no_data')
            if pat != 'sustained_drop':
                continue
            cc = copy.deepcopy(c)
            cc['open_pattern'] = pat
            cc['asset_type'] = _ticker_ref.get(tk, {}).get('type', 'UNK')
            sd_qualified.append(cc)
        for c in sd_qualified[:TOP_N]:
            _append_trade(c, sdo_trades, so_cnt)

        # D) CS + sustained_drop: filter CS + pattern, then top N
        cssd_qualified = []
        for c in cands[:PRE_FILTER_N]:
            tk = c['ticker']
            key = (tk, dt)
            at = _ticker_ref.get(tk, {}).get('type', 'UNK')
            if at != 'CS':
                continue
            pat = pattern_cache.get(key, 'no_data')
            if pat != 'sustained_drop':
                continue
            cc = copy.deepcopy(c)
            cc['open_pattern'] = pat
            cc['asset_type'] = at
            cssd_qualified.append(cc)
        for c in cssd_qualified[:TOP_N]:
            _append_trade(c, cssd_trades, cd_cnt)

        # E) CS+SD+GUARDS: CS + sustained_drop + gap<=75% + float>=10M (Polygon, hard exclude)
        gap_max = GAP_MAX_PCT / 100.0
        guards_qualified = []
        for c in cands[:PRE_FILTER_N]:
            tk = c['ticker']
            if c['gap_pct'] > gap_max:
                continue
            ref = _ticker_ref.get(tk, {})
            poly_shares = ref.get('shares')
            if poly_shares is None or poly_shares < FLOAT_MIN:
                continue
            at = ref.get('type', 'UNK')
            if at != 'CS':
                continue
            key = (tk, dt)
            pat = pattern_cache.get(key, 'no_data')
            if pat != 'sustained_drop':
                continue
            cc = copy.deepcopy(c)
            cc['open_pattern'] = pat
            cc['asset_type'] = at
            guards_qualified.append(cc)
        for c in guards_qualified[:TOP_N]:
            _append_trade(c, guard_trades, gu_cnt)

    bl_t2 = [t for t in baseline_trades if t.get('pnl_t2') is not None]
    co_t2 = [t for t in cso_trades if t.get('pnl_t2') is not None]
    so_t2 = [t for t in sdo_trades if t.get('pnl_t2') is not None]
    cd_t2 = [t for t in cssd_trades if t.get('pnl_t2') is not None]
    gu_t2 = [t for t in guard_trades if t.get('pnl_t2') is not None]
    timings['build'] = time.time() - t5
    elapsed = time.time() - t_total

    for lbl, trades, cnt in [('Baseline', bl_t2, bl_cnt), ('CS Only', co_t2, co_cnt),
                              ('Sust.Drop', so_t2, so_cnt), ('CS+SD', cd_t2, cd_cnt),
                              ('CS+SD+G', gu_t2, gu_cnt)]:
        print(f"\n  → {lbl:14s}  {len(trades)} trades with T+2 outcome")
        print(f"    local_high: {cnt['lh']}  |  fallback: {cnt['fb']}  |  "
              f"no_entry: {cnt['ne']}  |  no_bars: {cnt['nb']}")
    print(f"\n  {'─'*50}")
    for step, dur in [('Scan', timings['scan']), ('Bars', timings['bars']),
                      ('References', timings['refs']),
                      ('Classify+Outcomes', timings['outcomes']),
                      ('Build', timings['build'])]:
        print(f"    {step:<20s}  {dur:>6.0f}s  ({dur/elapsed*100:>4.0f}%)")
    print(f"  TOTAL: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    return bl_t2, co_t2, so_t2, cd_t2, gu_t2, dates


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD RESULT
# ═══════════════════════════════════════════════════════════════════════════════

def build_result(trades, dates, label):
    pnls = [t['pnl_t2'] for t in trades]
    m = compute_metrics(pnls, label)

    daily_pnl = defaultdict(list)
    for t in trades:
        daily_pnl[t['scan_date']].append(t['pnl_t2'])

    equity = [CAPITAL]
    eq_dates = []
    running_capital = CAPITAL
    for dt in sorted(daily_pnl):
        n = len(daily_pnl[dt])
        pos_size = running_capital / n
        day_dollar = sum(pos_size * (r / 100) for r in daily_pnl[dt])
        running_capital += day_dollar
        equity.append(running_capital)
        eq_dates.append(dt)

    cum_nc = 0.0
    daily_stats = []
    for dt in dates:
        dp = daily_pnl.get(dt, [])
        if dp:
            n = len(dp)
            pos_size = CAPITAL / n
            day_dollar = sum(pos_size * (r / 100) for r in dp)
        else:
            day_dollar = 0.0
            n = 0
        cum_nc += day_dollar
        daily_stats.append({'date': dt, 'n_trades': n,
                            'day_pnl_dollar': day_dollar, 'cum_pnl': cum_nc})

    eq_arr = np.array(equity)
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / peak * 100
    max_dd = float(dd.min())
    active_days = sum(1 for ds in daily_stats if ds['n_trades'] > 0)
    tpd = len(trades) / active_days if active_days > 0 else 0

    return {
        'label': label, 'trades': trades, 'metrics': m,
        'equity': equity, 'eq_dates': eq_dates,
        'daily_stats': daily_stats, 'max_dd': max_dd,
        'final_capital': running_capital, 'cum_nc': cum_nc,
        'active_days': active_days, 'tpd': tpd,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

def print_results(all_res, dates):
    print(f"\n\n{'='*110}")
    print(f"  5-WAY COMPARISON — {dates[0]} → {dates[-1]}")
    print(f"  Capital: ${CAPITAL:,.0f}/day  |  Non-compounding")
    print(f"{'='*110}")

    # ── Main comparison table via tabulate ──
    bl = all_res[0]
    headers = ['Metric'] + [r['label'] for r in all_res]
    def _row(name, fn):
        return [name] + [fn(r) for r in all_res]

    tbl = [
        _row('Trades',       lambda r: f"{r['metrics']['n']}"),
        _row('Active days',  lambda r: f"{r['active_days']}"),
        _row('Trades/day',   lambda r: f"{r['tpd']:.1f}"),
        _row('Avg PnL',      lambda r: f"{r['metrics']['avg']:+.2f}%"),
        _row('Median PnL',   lambda r: f"{r['metrics']['med']:+.2f}%"),
        _row('Trimmed Mean',  lambda r: f"{r['metrics']['trimmed_mean']:+.2f}%"),
        _row('Win Rate',     lambda r: f"{r['metrics']['wr']*100:.0f}%"),
        _row('Sharpe',       lambda r: f"{r['metrics']['sharpe']:.3f}"),
        _row('Profit Factor', lambda r: f"{r['metrics']['pf']:.2f}"),
        _row('95% CI',       lambda r: f"[{r['metrics']['ci_lo']:+.2f}, {r['metrics']['ci_hi']:+.2f}]"),
        _row('Total $ (NC)', lambda r: f"${r['cum_nc']:+,.0f}"),
        _row('Total $ (CMP)', lambda r: f"${r['final_capital']-CAPITAL:+,.0f}"),
        _row('Max DD',       lambda r: f"{r['max_dd']:.1f}%"),
    ]
    print()
    print(tabulate(tbl, headers=headers, tablefmt='simple', stralign='right'))

    # ── Delta table vs baseline ──
    print(f"\n  ── DELTA vs BASELINE ──")
    bl_m = bl['metrics']
    delta_headers = ['Metric'] + [r['label'] for r in all_res[1:]]
    delta_tbl = []
    for metric, key, fmt, pct in [
        ('Avg PnL', 'avg', '+.2f', True),
        ('Median', 'med', '+.2f', True),
        ('Win Rate', 'wr', '+.0f', False),
        ('Sharpe', 'sharpe', '+.3f', False),
        ('PF', 'pf', '+.2f', False),
    ]:
        row = [metric]
        for r in all_res[1:]:
            v = r['metrics'][key] - bl_m[key]
            if key == 'wr':
                v *= 100
            s = '%' if pct else ''
            row.append(f"{v:{fmt}}{s}")
        delta_tbl.append(row)
    delta_tbl.append(['$ (NC)'] + [f"${r['cum_nc']-bl['cum_nc']:+,.0f}" for r in all_res[1:]])
    print(tabulate(delta_tbl, headers=delta_headers, tablefmt='simple', stralign='right'))

    # ── Pattern breakdown for baseline ──
    print(f"\n  ── OPEN PATTERN BREAKDOWN (Baseline trades) ──")
    pat_groups = defaultdict(list)
    for t in bl['trades']:
        pat_groups[t.get('open_pattern', 'no_data')].append(t['pnl_t2'])
    rows = []
    for pat in ['sustained_drop', 'drop_then_recover', 'mild_drop', 'flat_or_bounce', 'no_data']:
        pnls = pat_groups.get(pat, [])
        if not pnls: continue
        a = np.array(pnls)
        sh = np.mean(a) / np.std(a, ddof=1) if len(a) > 1 and np.std(a, ddof=1) > 0 else 0
        rows.append([pat, len(a), f"{np.mean(a):+.2f}%", f"{np.median(a):+.2f}%",
                     f"{(a > 0).mean() * 100:.0f}%", f"{sh:.3f}"])
    print(tabulate(rows, headers=['Pattern', 'n', 'Avg', 'Median', 'WR', 'Sharpe'],
                   tablefmt='simple'))

    # ── Asset type breakdown for baseline ──
    print(f"\n  ── ASSET TYPE BREAKDOWN (Baseline trades) ──")
    at_groups = defaultdict(list)
    for t in bl['trades']:
        at_groups[t.get('asset_type', 'UNK')].append(t['pnl_t2'])
    rows = []
    for at in sorted(at_groups, key=lambda x: -len(at_groups[x])):
        pnls = at_groups[at]
        a = np.array(pnls)
        sh = np.mean(a) / np.std(a, ddof=1) if len(a) > 1 and np.std(a, ddof=1) > 0 else 0
        rows.append([at, len(a), f"{np.mean(a):+.2f}%", f"{(a > 0).mean() * 100:.0f}%",
                     f"{sh:.3f}"])
    print(tabulate(rows, headers=['Type', 'n', 'Avg', 'WR', 'Sharpe'], tablefmt='simple'))

    # ── Entry type per scenario ──
    for res in all_res:
        trades = res['trades']
        lh = [t['pnl_t2'] for t in trades if t['entry_type'] == 'local_high']
        fb = [t['pnl_t2'] for t in trades if t['entry_type'] == 'fallback']
        print(f"\n  ── ENTRY TYPE — {res['label']} ──")
        for lbl, arr in [('local_high', lh), ('fallback', fb)]:
            if arr:
                a = np.array(arr)
                sh = np.mean(a) / np.std(a, ddof=1) if len(a) > 1 and np.std(a, ddof=1) > 0 else 0
                print(f"    {lbl:12s}  n={len(a):<4d}  avg={np.mean(a):+.2f}%  "
                      f"WR={float((a>0).mean())*100:.0f}%  Sharpe={sh:.3f}")

    # ── Monthly comparison ──
    print(f"\n  ── MONTHLY COMPARISON ──")
    monthly = {r['label']: defaultdict(list) for r in all_res}
    for res in all_res:
        for t in res['trades']:
            monthly[res['label']][t['scan_date'][:7]].append(t['pnl_t2'])
    all_months = sorted(set(mo for m in monthly.values() for mo in m.keys()))
    mo_headers = ['Month']
    for r in all_res:
        mo_headers += [f"{r['label'][:6]} n", f"{r['label'][:6]} avg"]
    rows = []
    for mo in all_months:
        row = [mo]
        for r in all_res:
            pnls = monthly[r['label']].get(mo, [])
            row.append(f"{len(pnls)}")
            row.append(f"{np.mean(pnls):+.2f}%" if pnls else '—')
        rows.append(row)
    print(tabulate(rows, headers=mo_headers, tablefmt='simple'))

    # ── Worst / Best for each ──
    for res in all_res:
        trades_s = sorted(res['trades'], key=lambda x: x['pnl_t2'])
        for lbl, sl in [('WORST 5', trades_s[:5]), ('BEST 5', trades_s[-5:])]:
            print(f"\n  ── {lbl} — {res['label']} ──")
            for t in sl:
                in_float = '✓' if t['ticker'] in _FLOAT else ' '
                print(f"    {t['scan_date']} {t['ticker']:>6s}  gap={t['gap_pct']*100:>6.1f}%  "
                      f"entry={t['entry_type']:>10s}  pnl={t['pnl_t2']:>+8.2f}%  "
                      f"float={in_float}  pat={t.get('open_pattern','?')}")

    print()


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS — 5-way overlay on every panel
# ═══════════════════════════════════════════════════════════════════════════════

COLORS = ['#2c3e50', '#e67e22', '#8e44ad', '#27ae60', '#e74c3c']

def plot_results(all_res, dates):
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        BASE_DIR = os.getcwd()

    _ALL = [(r, COLORS[i], r['label']) for i, r in enumerate(all_res)]

    # ── Figure 1: Equity, Drawdown, Monthly, Distribution ──
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    title_parts = [f"{r['label']}: {r['metrics']['n']}t Sh={r['metrics']['sharpe']:.3f}"
                   for r in all_res]
    fig.suptitle(
        f'Agent 1 — 5-Way Comparison: {dates[0]} → {dates[-1]}\n' +
        '  |  '.join(title_parts),
        fontsize=9, fontweight='bold')

    # 1. Equity curves
    ax = axes[0, 0]
    for res, color, lbl in _ALL:
        eq_d = [datetime.strptime(dates[0], '%Y-%m-%d')] + \
               [datetime.strptime(d, '%Y-%m-%d') for d in res['eq_dates']]
        ax.plot(eq_d, res['equity'], color=color, linewidth=2,
                label=f'{lbl} (${res["final_capital"]:,.0f})')
    ax.axhline(CAPITAL, color='#999', ls='--', lw=0.8)
    ax.set_title('Equity Curves (compounding)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    # 2. Drawdown
    ax = axes[0, 1]
    for res, color, lbl in _ALL:
        eq_d = [datetime.strptime(dates[0], '%Y-%m-%d')] + \
               [datetime.strptime(d, '%Y-%m-%d') for d in res['eq_dates']]
        eq_arr = np.array(res['equity'])
        peak = np.maximum.accumulate(eq_arr)
        dd = (eq_arr - peak) / peak * 100
        ax.fill_between(eq_d, dd, 0, alpha=0.12, color=color)
        ax.plot(eq_d, dd, color=color, linewidth=1.5,
                label=f'{lbl} ({dd.min():.1f}%)')
    ax.set_title('Drawdown')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    # 3. Monthly avg PnL — grouped bars (4 bars per month)
    ax = axes[1, 0]
    monthly = {}
    for res in all_res:
        m = defaultdict(list)
        for t in res['trades']:
            m[t['scan_date'][:7]].append(t['pnl_t2'])
        monthly[res['label']] = m
    months = sorted(set(mo for m in monthly.values() for mo in m.keys()))
    x = np.arange(len(months))
    n_bars = len(all_res)
    w = 0.8 / n_bars
    for i, (res, color, lbl) in enumerate(_ALL):
        avgs = [np.mean(monthly[lbl].get(mo, [0])) for mo in months]
        ax.bar(x + (i - n_bars/2 + 0.5) * w, avgs, w, color=color, alpha=0.8, label=lbl)
    ax.set_xticks(x)
    ax.set_xticklabels([mo[5:] for mo in months], fontsize=9)
    ax.axhline(0, color='#333', lw=0.8)
    ax.set_title('Monthly Avg PnL (%)')
    ax.set_ylabel('Avg T+2 PnL (%)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2, axis='y')

    # 4. PnL distribution — overlapping histograms
    ax = axes[1, 1]
    all_pnls = []
    for res in all_res:
        all_pnls.extend(t['pnl_t2'] for t in res['trades'])
    clip_lo, clip_hi = np.percentile(all_pnls, 1), np.percentile(all_pnls, 99)
    bins = np.linspace(clip_lo, clip_hi, 50)
    for res, color, lbl in _ALL:
        pnls = [t['pnl_t2'] for t in res['trades']]
        ax.hist(pnls, bins=bins, color=color, alpha=0.35, label=f'{lbl} ({np.mean(pnls):+.2f}%)')
        ax.axvline(np.mean(pnls), color=color, ls='--', lw=1.5)
    ax.axvline(0, color='#333', lw=1.2)
    ax.set_xlabel('T+2 PnL (%)')
    ax.set_title('PnL Distribution')
    ax.legend(fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    p1 = os.path.join(BASE_DIR, 'agent1_baseline_bt.png')
    plt.savefig(p1, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n  Saved: {p1}")

    # ── Figure 2: Rolling metrics ──
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    roll_window = 20

    ax = axes[0]
    for res, color, lbl in _ALL:
        ts = sorted(res['trades'], key=lambda t: t['scan_date'])
        pnls_s = [t['pnl_t2'] for t in ts]
        td = [datetime.strptime(t['scan_date'], '%Y-%m-%d') for t in ts]
        if len(pnls_s) >= roll_window:
            rolling = pd.Series(pnls_s).rolling(roll_window).mean()
            ax.plot(td, rolling, color=color, linewidth=2,
                    label=f'{lbl} {roll_window}-trade MA')
        ax.axhline(np.mean(pnls_s), color=color, lw=1, ls='--', alpha=0.4)
    ax.axhline(0, color='#333', lw=0.8, ls=':')
    ax.set_ylabel('Avg PnL (%)')
    ax.set_title(f'Rolling {roll_window}-Trade Average PnL', fontsize=12, fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for res, color, lbl in _ALL:
        ts = sorted(res['trades'], key=lambda t: t['scan_date'])
        pnls_s = [t['pnl_t2'] for t in ts]
        td = [datetime.strptime(t['scan_date'], '%Y-%m-%d') for t in ts]
        if len(pnls_s) >= roll_window:
            wins = [1 if p > 0 else 0 for p in pnls_s]
            rolling_wr = pd.Series(wins).rolling(roll_window).mean() * 100
            ax.plot(td, rolling_wr, color=color, linewidth=2, label=f'{lbl}')
        overall_wr = np.mean([1 if p > 0 else 0 for p in pnls_s]) * 100
        ax.axhline(overall_wr, color=color, lw=1, ls='--', alpha=0.4)
    ax.axhline(50, color='#999', lw=1, ls=':', label='50%')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title(f'Rolling {roll_window}-Trade Win Rate', fontsize=12, fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    p2 = os.path.join(BASE_DIR, 'agent1_baseline_rolling.png')
    plt.savefig(p2, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {p2}")

    # ── Figure 3: Cumulative non-compounding P&L ──
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    for res, color, lbl in _ALL:
        ds = res['daily_stats']
        d_dates = [datetime.strptime(s['date'], '%Y-%m-%d') for s in ds]
        cum = [s['cum_pnl'] for s in ds]
        ax.plot(d_dates, cum, color=color, linewidth=2,
                label=f'{lbl} (${res["cum_nc"]:+,.0f})')
    ax.axhline(0, color='#333', lw=0.8, ls=':')
    ax.set_title('Cumulative P&L (non-compounding, $100K/day)', fontsize=12, fontweight='bold')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p3 = os.path.join(BASE_DIR, 'agent1_baseline_cumulative.png')
    plt.savefig(p3, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {p3}")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

bl_trades, co_trades, so_trades, cd_trades, gu_trades, dates = run_backtest()
bl_result = build_result(bl_trades, dates, 'Baseline')
co_result = build_result(co_trades, dates, 'CS Only')
so_result = build_result(so_trades, dates, 'Sust.Drop')
cd_result = build_result(cd_trades, dates, 'CS+SD')
gu_result = build_result(gu_trades, dates, 'CS+SD+G')
all_results = [bl_result, co_result, so_result, cd_result, gu_result]
print_results(all_results, dates)
plot_results(all_results, dates)
