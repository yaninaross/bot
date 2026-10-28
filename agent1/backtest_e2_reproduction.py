# %% Agent 1 — E2 Locked Config Backtest
#
# Runs the LOCKED strategy (v1.1) for a specified period.
# Config: CS + sustained_drop, SOFT float, N=1, exit 17:00 ET,
#   SL=-50% disaster cap (intraday), no skip rule, no gap max.
#
# Change BACKTEST_START / BACKTEST_END below to run any period.
#
# Colab-cell friendly. Single data pass.

import os, sys, time, json, re, asyncio, copy
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
# CONFIG — E2 BENCHMARK (DO NOT CHANGE)
# ═══════════════════════════════════════════════════════════════════════════════

BACKTEST_START = '2026-09-11'
BACKTEST_END   = '2026-09-17'

GAP_MIN_PCT    = 5.0
PRICE_MIN      = 10.0
FLOAT_MIN      = 10_000_000
TOP_N          = 1
PRE_FILTER_N   = 30

PEAK_WINDOW    = 3
MAX_PEAK_BAR   = 12
FALLBACK_BAR   = 15

EXIT_TIME      = '17:00'
SCORING_TIME_ET = '09:20'
PM_MIN_BARS    = 5
STOP_LOSS_PCT  = 0.50

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


# ═══════════════════════════════════════════════════════════════════════════════
# ASYNC MINUTE BAR FETCH (premarket + RTH + after-hours)
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN CLASSIFICATION + LOCAL HIGH DETECTION
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
# SCANNER
# ═══════════════════════════════════════════════════════════════════════════════

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
    max_loss = float(a.min())
    max_gain = float(a.max())
    return {'label':label,'n':n,'avg':avg,'med':med,'std':std,'wr':wr,
            'sharpe':sharpe,'pf':pf,'ci_lo':avg-1.96*se,'ci_hi':avg+1.96*se,
            'trimmed_mean':tm,'max_loss':max_loss,'max_gain':max_gain}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA PIPELINE (single pass, shared across all variants)
# ═══════════════════════════════════════════════════════════════════════════════

def run_data_pipeline():
    t_total = time.time()

    print(f"\n{'='*80}")
    print(f"  E2 REPRODUCTION + ABLATION — Data Pipeline")
    print(f"  {BACKTEST_START} → {BACKTEST_END}")
    print(f"{'='*80}")

    d = datetime.strptime(BACKTEST_START, '%Y-%m-%d').date()
    end = datetime.strptime(BACKTEST_END, '%Y-%m-%d').date()
    dates = []
    while d <= end:
        if _is_td(d): dates.append(d.strftime('%Y-%m-%d'))
        d += timedelta(days=1)
    print(f"\n  {len(dates)} trading days")

    # Phase 1: Scan
    t1 = time.time()
    print(f"\n  Phase 1: Scanning for gap-ups...")
    all_candidates = {}
    total_cands = 0
    for i, dt in enumerate(dates):
        cands = scan_day(dt)
        all_candidates[dt] = cands
        total_cands += len(cands)
        if (i+1) % 30 == 0 or i == len(dates) - 1:
            print(f"    [{i+1:>3}/{len(dates)}] {dt}  |  {total_cands} total  |  "
                  f"{time.time()-t_total:.0f}s")

    # Phase 2: Fetch minute bars
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
        if done % 500 == 0 or done == len(pairs_list):
            print(f"    {done}/{len(pairs_list)}  |  {time.time()-t_total:.0f}s")
    print(f"  → Got bars for {len(all_bars)}/{len(all_pairs)} pairs")

    # Phase 3: Fetch ticker reference data
    t3 = time.time()
    all_tickers = set()
    for cands in all_candidates.values():
        for c in cands[:PRE_FILTER_N]:
            all_tickers.add(c['ticker'])
    print(f"\n  Phase 3: Fetching reference for {len(all_tickers)} tickers...")
    tk_list = sorted(all_tickers)
    for c_start in range(0, len(tk_list), 200):
        chunk = tk_list[c_start:c_start + 200]
        fetch_ticker_refs(chunk)
        done = min(c_start + 200, len(tk_list))
        if done % 400 == 0 or done == len(tk_list):
            print(f"    {done}/{len(tk_list)}  |  {time.time()-t_total:.0f}s")

    # Phase 4: Classify patterns + find entries
    pattern_cache = {}
    entry_cache = {}
    for key, bar_data in all_bars.items():
        rth_bars = bar_data.get('rth', []) if isinstance(bar_data, dict) else bar_data
        pattern_cache[key] = classify_open_pattern(rth_bars)
        if rth_bars and len(rth_bars) >= PEAK_WINDOW * 2 + 1:
            entry_cache[key] = find_local_high(rth_bars)
        else:
            entry_cache[key] = None

    pat_counts = defaultdict(int)
    for p in pattern_cache.values():
        pat_counts[p] += 1
    print(f"\n  Phase 4: Pattern classification")
    for pat in ['sustained_drop', 'drop_then_recover', 'mild_drop', 'flat_or_bounce', 'no_data']:
        if pattern_cache:
            print(f"    {pat:25s}  {pat_counts.get(pat, 0):>5d}  "
                  f"({pat_counts.get(pat, 0)/len(pattern_cache)*100:.1f}%)")

    # Phase 5: Fetch outcome dates
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

    elapsed = time.time() - t_total
    print(f"\n  Data pipeline: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    return all_candidates, all_bars, pattern_cache, entry_cache, dates, t_total


# ═══════════════════════════════════════════════════════════════════════════════
# VARIANT RUNNER — parameterized trade builder
# ═══════════════════════════════════════════════════════════════════════════════

def run_variant(all_candidates, all_bars, pattern_cache, entry_cache, dates,
                label, exit_time='17:00', top_n=1, hard_float=False,
                pm_rerank=False, pnl_cap=None, skip_15pct=False,
                use_t2_exit=False, stop_loss_pct=None):

    def _get_exit_price(tk, dt, entry_time, exit_t, entry_price=None, sl_pct=None):
        bar_data = all_bars.get((tk, dt))
        if bar_data is None:
            return None, False, None
        session_bars = bar_data.get('rth', []) + bar_data.get('ah', [])
        exit_price = None
        sl_hit = False
        sl_exit_time = None
        sl_price = entry_price * (1 + sl_pct) if entry_price and sl_pct else None
        for t_str, bar in session_bars:
            if t_str > exit_t:
                break
            if t_str >= entry_time:
                if sl_price and bar['h'] >= sl_price:
                    exit_price = sl_price
                    sl_hit = True
                    sl_exit_time = t_str
                    break
                exit_price = bar['c']
        return exit_price, sl_hit, sl_exit_time

    trades = []
    cnt = {'lh': 0, 'fb': 0, 'ne': 0, 'nb': 0, 'skip': 0, 'float_ex': 0}

    # Optionally apply PM reranking
    if pm_rerank:
        working_candidates = {}
        gap_min_final = GAP_MIN_PCT / 100.0
        for dt in dates:
            cands = [copy.deepcopy(c) for c in all_candidates.get(dt, [])]
            for c in cands:
                tk = c['ticker']
                bar_data = all_bars.get((tk, dt))
                if bar_data is None:
                    continue
                pm_bars = bar_data.get('pm', [])
                if not pm_bars:
                    continue
                pm_before = [(t, d) for t, d in pm_bars if t <= SCORING_TIME_ET]
                if len(pm_before) < PM_MIN_BARS:
                    continue
                pm_price = pm_before[-1][1]['c']
                if pm_price and pm_price > 0:
                    c['gap_pct'] = (pm_price - c['prior_close']) / c['prior_close']
                    c['pm_price'] = pm_price
            reranked = [c for c in cands if c['gap_pct'] >= gap_min_final]
            reranked.sort(key=lambda x: -x['gap_pct'])
            working_candidates[dt] = reranked
    else:
        working_candidates = all_candidates

    for dt in dates:
        cands = working_candidates.get(dt, [])

        cs_qualified = []
        for c in cands[:PRE_FILTER_N]:
            tk = c['ticker']
            ref = _ticker_ref.get(tk, {})
            at = ref.get('type', 'UNK')
            if at != 'CS':
                continue
            pattern = pattern_cache.get((tk, dt), 'no_data')
            if pattern != 'sustained_drop':
                continue
            if hard_float and tk not in _FLOAT:
                api_shares = ref.get('shares')
                if api_shares is not None and api_shares < FLOAT_MIN:
                    cnt['float_ex'] += 1
                    continue
                if api_shares is None:
                    cnt['float_ex'] += 1
                    continue
            cc = copy.deepcopy(c)
            cc['asset_type'] = at
            cc['open_pattern'] = pattern
            cs_qualified.append(cc)

        for c in cs_qualified[:top_n]:
            tk, dt_s = c['ticker'], c['scan_date']
            entry = entry_cache.get((tk, dt_s))
            if entry is None:
                if (tk, dt_s) not in all_bars:
                    cnt['nb'] += 1
                else:
                    cnt['ne'] += 1
                continue

            if skip_15pct:
                rth_bars = all_bars.get((tk, dt_s), {}).get('rth', [])
                if rth_bars and len(rth_bars) > 15:
                    open_p = rth_bars[0][1]['o']
                    bar15_c = rth_bars[15][1]['c']
                    if open_p > 0 and (open_p - bar15_c) / open_p > 0.15:
                        cnt['skip'] += 1
                        continue

            tc = copy.deepcopy(c)
            tc['entry_price'] = entry['entry_price']
            tc['entry_time'] = entry['entry_time']
            tc['entry_type'] = entry['entry_type']
            ep = tc['entry_price']

            if use_t2_exit:
                t2d = _gd_cache.get(tc.get('t2_date', ''), {}).get(tk, {})
                if t2d.get('c'):
                    tc['pnl_eod'] = round((ep - t2d['c']) / ep * 100, 3)
                else:
                    tc['pnl_eod'] = None
                tc['sl_hit'] = False
                tc['sl_exit_time'] = None
            else:
                exit_price, sl_hit, sl_exit_time = _get_exit_price(
                    tk, dt_s, tc['entry_time'], exit_time,
                    entry_price=ep, sl_pct=stop_loss_pct)
                tc['sl_hit'] = sl_hit
                tc['sl_exit_time'] = sl_exit_time
                if exit_price is not None:
                    tc['pnl_eod'] = round((ep - exit_price) / ep * 100, 3)
                else:
                    t0d = _gd_cache.get(dt_s, {}).get(tk, {})
                    if t0d.get('c'):
                        tc['pnl_eod'] = round((ep - t0d['c']) / ep * 100, 3)
                    else:
                        tc['pnl_eod'] = None

            if pnl_cap is not None and tc.get('pnl_eod') is not None:
                tc['pnl_eod'] = max(tc['pnl_eod'], pnl_cap)

            if tc.get('pnl_eod') is not None:
                cnt['lh' if tc['entry_type'] == 'local_high' else 'fb'] += 1
                trades.append(tc)

    pnls = [t['pnl_eod'] for t in trades]
    m = compute_metrics(pnls, label)

    cum_nc = 0.0
    daily_pnl = defaultdict(list)
    for t in trades:
        daily_pnl[t['scan_date']].append(t['pnl_eod'])
    for dt in dates:
        dp = daily_pnl.get(dt, [])
        if dp:
            n = len(dp)
            pos_size = CAPITAL / n
            cum_nc += sum(pos_size * (r / 100) for r in dp)

    active_days = sum(1 for dt in dates if dt in daily_pnl)

    return {
        'label': label,
        'n': m['n'], 'avg': m['avg'], 'med': m['med'],
        'trimmed_mean': m['trimmed_mean'], 'std': m['std'],
        'wr': m['wr'], 'sharpe': m['sharpe'], 'pf': m['pf'],
        'ci_lo': m['ci_lo'], 'ci_hi': m['ci_hi'],
        'max_loss': m.get('max_loss', 0), 'max_gain': m.get('max_gain', 0),
        'cum_nc': cum_nc, 'active_days': active_days,
        'cnt': cnt, 'trades': trades,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RUN — Locked config (v1.1) with per-trade detail
# ═══════════════════════════════════════════════════════════════════════════════

all_candidates, all_bars, pattern_cache, entry_cache, dates, t_total = run_data_pipeline()

# Run with SL (locked config v1.1)
locked = run_variant(all_candidates, all_bars, pattern_cache, entry_cache, dates,
                     label='LOCKED_v1.1', exit_time='17:00', top_n=1,
                     hard_float=False, pm_rerank=False, pnl_cap=None,
                     skip_15pct=False, use_t2_exit=False,
                     stop_loss_pct=STOP_LOSS_PCT)

# Also run without SL for comparison
raw = run_variant(all_candidates, all_bars, pattern_cache, entry_cache, dates,
                  label='NO_SL', exit_time='17:00', top_n=1,
                  hard_float=False, pm_rerank=False, pnl_cap=None,
                  skip_15pct=False, use_t2_exit=False,
                  stop_loss_pct=None)


print(f"\n\n{'='*80}")
print(f"  AGENT 1 — LAST WEEK RESULTS")
print(f"  {BACKTEST_START} → {BACKTEST_END}  ({len(dates)} trading days)")
print(f"  Config: Locked v1.2 (N=1, CS+SD, exit 17:00, SL=-50%)")
print(f"{'='*80}")


# ─── PER-TRADE DETAIL ───────────────────────────────────────────────────────

print(f"\n  ─── TRADE LOG ───\n")

trades = locked['trades']
raw_trades = raw['trades']
raw_by_date = {t['scan_date']: t for t in raw_trades}

if not trades:
    print(f"  No trades in this period.\n")
    no_trade_days = [dt for dt in dates if dt not in {t['scan_date'] for t in trades}]
    if no_trade_days:
        print(f"  Days with no qualifying gap-up:")
        for dt in no_trade_days:
            n_cands = len(all_candidates.get(dt, []))
            print(f"    {dt}  ({n_cands} gap candidates, none passed CS+SD filter)")
else:
    print(f"  {'Date':>12s}  {'Ticker':>6s}  {'Gap%':>6s}  {'Entry$':>8s}  {'Type':>10s}  "
          f"{'SL$':>8s}  {'SL Hit':>6s}  {'PnL%':>8s}  {'Raw PnL%':>9s}  {'$ PnL':>10s}")
    print(f"  {'─'*12}  {'─'*6}  {'─'*6}  {'─'*8}  {'─'*10}  "
          f"{'─'*8}  {'─'*6}  {'─'*8}  {'─'*9}  {'─'*10}")

    cum_dol = 0.0
    for t in sorted(trades, key=lambda x: x['scan_date']):
        sl_price = t['entry_price'] * (1 + STOP_LOSS_PCT)
        dol = CAPITAL * t['pnl_eod'] / 100
        cum_dol += dol
        sl_str = 'YES' if t.get('sl_hit') else 'no'

        raw_t = raw_by_date.get(t['scan_date'])
        raw_pnl_str = f"{raw_t['pnl_eod']:>+8.1f}%" if raw_t else '      N/A'

        print(f"  {t['scan_date']:>12s}  {t['ticker']:>6s}  "
              f"+{t['gap_pct']*100:>4.1f}%  ${t['entry_price']:>7.2f}  "
              f"{t['entry_type']:>10s}  ${sl_price:>7.2f}  {sl_str:>6s}  "
              f"{t['pnl_eod']:>+7.1f}%  {raw_pnl_str}  ${dol:>+9,.0f}")

    print(f"\n  {'':>12s}  {'':>6s}  {'':>6s}  {'':>8s}  {'':>10s}  "
          f"{'':>8s}  {'':>6s}  {'':>8s}  {'':>9s}  {'─'*10}")
    print(f"  {'TOTAL':>12s}  {'':>6s}  {'':>6s}  {'':>8s}  {'':>10s}  "
          f"{'':>8s}  {'':>6s}  {'':>8s}  {'':>9s}  ${cum_dol:>+9,.0f}")

    # Days with no trade
    trade_dates = {t['scan_date'] for t in trades}
    no_trade_days = [dt for dt in dates if dt not in trade_dates]
    if no_trade_days:
        print(f"\n  Days with no trade:")
        for dt in no_trade_days:
            n_cands = len(all_candidates.get(dt, []))
            n_sd = sum(1 for c in all_candidates.get(dt, [])[:PRE_FILTER_N]
                       if pattern_cache.get((c['ticker'], dt)) == 'sustained_drop')
            print(f"    {dt}  ({n_cands} gap candidates, {n_sd} sustained_drop)")


# ─── SUMMARY ─────────────────────────────────────────────────────────────────

print(f"\n\n  ─── SUMMARY ───\n")

for r, lbl in [(locked, 'With SL -50%'), (raw, 'Without SL')]:
    n_sl = sum(1 for t in r['trades'] if t.get('sl_hit'))
    print(f"  {lbl}:")
    print(f"    Trades:     {r['n']}")
    print(f"    Avg PnL:    {r['avg']:+.2f}%")
    print(f"    Win Rate:   {r['wr']*100:.0f}%")
    print(f"    Total $ NC: ${r['cum_nc']:+,.0f}")
    if lbl == 'With SL -50%':
        print(f"    SL hits:    {n_sl}")
    print(f"    Max Loss:   {r['max_loss']:+.1f}%")
    print(f"    Max Gain:   {r['max_gain']:+.1f}%")
    print(f"    Entry: lh={r['cnt']['lh']} fb={r['cnt']['fb']}")
    print()

sl_delta = locked['cum_nc'] - raw['cum_nc']
print(f"  SL impact: ${sl_delta:+,.0f} "
      f"({'SL helped' if sl_delta > 0 else 'SL cost money (no bad trades this week)'})")

results = [locked]


# ═══════════════════════════════════════════════════════════════════════════════
# CHART — Per-trade PnL waterfall
# ═══════════════════════════════════════════════════════════════════════════════

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

if locked['trades']:
    sorted_trades = sorted(locked['trades'], key=lambda x: x['scan_date'])
    fig, ax = plt.subplots(figsize=(max(10, len(sorted_trades) * 2), 6))

    labels_t = [f"{t['scan_date'][-5:]}\n{t['ticker']}" for t in sorted_trades]
    pnls_t = [t['pnl_eod'] for t in sorted_trades]
    dollars_t = [CAPITAL * t['pnl_eod'] / 100 for t in sorted_trades]
    colors_t = ['#e74c3c' if t.get('sl_hit') else ('#27ae60' if p > 0 else '#c0392b')
                for t, p in zip(sorted_trades, pnls_t)]

    x = np.arange(len(labels_t))
    bars = ax.bar(x, dollars_t, color=colors_t, alpha=0.85, edgecolor='black', linewidth=0.3)
    ax.axhline(0, color='black', lw=0.8)

    for b, d, p in zip(bars, dollars_t, pnls_t):
        va = 'bottom' if d >= 0 else 'top'
        offset = 200 if d >= 0 else -200
        ax.text(b.get_x() + b.get_width()/2, d + offset,
                f"${d/1000:+.1f}K\n({p:+.1f}%)", ha='center', va=va, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_t, fontsize=9)
    ax.set_ylabel('$ PnL (non-compounding, $100K/day)')
    ax.set_title(f'Agent 1 — Last Week: {BACKTEST_START} → {BACKTEST_END}\n'
                 f'{locked["n"]} trades, ${locked["cum_nc"]:+,.0f} total, '
                 f'SL=-50% active (red outline = SL hit)',
                 fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    p1 = os.path.join(BASE_DIR, 'last_week_pnl.png')
    plt.savefig(p1, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n  Saved: {p1}")
else:
    print(f"\n  No trades to chart.")


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

results_dir = os.path.join(BASE_DIR, 'results')
os.makedirs(results_dir, exist_ok=True)

output = {
    'run_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
    'period': f'{BACKTEST_START} to {BACKTEST_END}',
    'n_trading_days': len(dates),
    'config': 'Locked v1.2 (N=1, CS+SD, exit 17:00, SL=-50%)',
    'summary': {
        'n': locked['n'], 'avg_pnl': round(locked['avg'], 2),
        'win_rate': round(locked['wr'], 3),
        'sharpe': round(locked['sharpe'], 3),
        'total_dollars_nc': round(locked['cum_nc']),
        'max_loss': round(locked['max_loss'], 2),
        'max_gain': round(locked['max_gain'], 2),
        'sl_hits': sum(1 for t in locked['trades'] if t.get('sl_hit')),
    },
    'trades': [{
        'date': t['scan_date'], 'ticker': t['ticker'],
        'gap_pct': round(t['gap_pct'] * 100, 1),
        'entry_price': round(t['entry_price'], 2),
        'entry_type': t['entry_type'],
        'pnl_pct': round(t['pnl_eod'], 1),
        'sl_hit': t.get('sl_hit', False),
        'dollar_pnl': round(CAPITAL * t['pnl_eod'] / 100),
    } for t in sorted(locked['trades'], key=lambda x: x['scan_date'])],
}

json_path = os.path.join(results_dir, 'last_week_results.json')
with open(json_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"  Saved: {json_path}")

print(f"\n  Total runtime: {time.time() - t_total:.0f}s ({(time.time() - t_total)/60:.1f} min)")
