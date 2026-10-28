# %% Agent 1 — Time Decay Analysis: Exit Time Sweep by Period
#
# Run AFTER backtest_e2_reproduction.py or agent1_eval_sl_skip.py in same
# Colab session to reuse cached data. If run standalone, pipeline executes
# automatically (~10 min).
#
# QUESTION: Has the optimal exit time shifted in Jul-Sep 2026?
# If so, the fade is exhausting faster in the new regime.
#
# Sweeps 9 intraday exit times + T+1 / T+2, splits Jan-Jun vs Jul-Sep,
# compares avg PnL, Sharpe, WR, total $.

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
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter

try:
    from google.colab import auth
    try: auth.authenticate_user()
    except: pass
except: pass


# =====================================================================================
# CONFIG — E2 BENCHMARK (DO NOT CHANGE)
# =====================================================================================

BACKTEST_START = '2026-01-02'
BACKTEST_END   = '2026-09-11'

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

CAPITAL        = 100_000

# Period split
PERIOD_JAN_JUN = ('2026-01', '2026-06')   # inclusive month range
PERIOD_JUL_SEP = ('2026-07', '2026-09')

# Exit times to sweep (intraday ET)
EXIT_TIMES_INTRADAY = ['10:00', '10:30', '11:00', '12:00', '13:00',
                       '14:00', '15:00', '16:00', '17:00']

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


# =====================================================================================
# HELPERS
# =====================================================================================

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

try:
    _gd_cache
except NameError:
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

try:
    _ticker_ref
except NameError:
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


# =====================================================================================
# DATA PIPELINE (skips if already loaded from prior cell)
# =====================================================================================

PIPELINE_CACHED = False
try:
    _ = all_bars; _ = pattern_cache; _ = entry_cache; _ = all_candidates; _ = dates
    PIPELINE_CACHED = True
    print("  Using cached pipeline data from prior cell")
except NameError:
    pass

if not PIPELINE_CACHED:
    t_total = time.time()
    print(f"\n{'='*80}")
    print(f"  TIME DECAY ANALYSIS -- Data Pipeline")
    print(f"  {BACKTEST_START} -> {BACKTEST_END}")
    print(f"{'='*80}")

    d = datetime.strptime(BACKTEST_START, '%Y-%m-%d').date()
    end = datetime.strptime(BACKTEST_END, '%Y-%m-%d').date()
    dates = []
    while d <= end:
        if _is_td(d): dates.append(d.strftime('%Y-%m-%d'))
        d += timedelta(days=1)
    print(f"\n  {len(dates)} trading days")

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
    print(f"  -> Got bars for {len(all_bars)}/{len(all_pairs)} pairs")

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

    pattern_cache = {}
    entry_cache = {}
    for key, bar_data in all_bars.items():
        rth_bars = bar_data.get('rth', []) if isinstance(bar_data, dict) else bar_data
        pattern_cache[key] = classify_open_pattern(rth_bars)
        if rth_bars and len(rth_bars) >= PEAK_WINDOW * 2 + 1:
            entry_cache[key] = find_local_high(rth_bars)
        else:
            entry_cache[key] = None

    outcome_dates = set()
    for cands in all_candidates.values():
        for c in cands[:PRE_FILTER_N]:
            outcome_dates.add(c['scan_date'])
            if c.get('t1_date'): outcome_dates.add(c['t1_date'])
            if c.get('t2_date'): outcome_dates.add(c['t2_date'])
    for od in sorted(od for od in outcome_dates if od not in _gd_cache):
        try: fetch_grouped_daily(od)
        except: pass

    print(f"\n  Pipeline: {time.time()-t_total:.0f}s")


# Ensure ticker refs are populated
_ref_tickers_needed = set()
for _cands in all_candidates.values():
    for _c in _cands[:PRE_FILTER_N]:
        _ref_tickers_needed.add(_c['ticker'])
_ref_missing = [tk for tk in _ref_tickers_needed if tk not in _ticker_ref]
if _ref_missing:
    print(f"  Fetching ticker refs for {len(_ref_missing)} tickers...")
    for _cs in range(0, len(_ref_missing), 200):
        _chunk = _ref_missing[_cs:_cs + 200]
        fetch_ticker_refs(_chunk)
    print(f"  -> _ticker_ref has {len(_ticker_ref)} entries")
del _ref_tickers_needed, _ref_missing


# =====================================================================================
# BUILD E2 BASELINE TRADES (enriched with full bar data reference)
# =====================================================================================

def build_e2_trades_enriched():
    """Build E2 baseline trade list. Each trade includes entry info +
    references to bar data and candidate metadata for exit-time sweeping."""
    trades = []
    for dt in dates:
        cands = all_candidates.get(dt, [])
        cs_qualified = []
        for c in cands[:PRE_FILTER_N]:
            tk = c['ticker']
            ref = _ticker_ref.get(tk, {})
            if ref.get('type', 'UNK') != 'CS':
                continue
            if pattern_cache.get((tk, dt), 'no_data') != 'sustained_drop':
                continue
            cc = copy.deepcopy(c)
            cc['asset_type'] = 'CS'
            cc['open_pattern'] = 'sustained_drop'
            cs_qualified.append(cc)

        for c in cs_qualified[:TOP_N]:
            tk, dt_s = c['ticker'], c['scan_date']
            entry = entry_cache.get((tk, dt_s))
            if entry is None:
                continue

            bar_data = all_bars.get((tk, dt_s))
            if bar_data is None:
                continue

            tc = copy.deepcopy(c)
            tc['entry_price'] = entry['entry_price']
            tc['entry_time'] = entry['entry_time']
            tc['entry_type'] = entry['entry_type']
            trades.append(tc)

    return trades

print(f"\n{'='*80}")
print(f"  Building E2 baseline trades with intraday enrichment...")
print(f"{'='*80}")

e2_trades = build_e2_trades_enriched()
print(f"  {len(e2_trades)} trades built")


# =====================================================================================
# COMPUTE PnL AT EACH EXIT TIME FOR EACH TRADE
# =====================================================================================

def get_intraday_exit_price(tk, dt_s, entry_time, exit_time_target):
    """Get the close of the last bar at or before exit_time_target,
    but only bars at or after entry_time. Returns None if no bar found."""
    bar_data = all_bars.get((tk, dt_s))
    if bar_data is None:
        return None
    rth_bars = bar_data.get('rth', [])
    ah_bars = bar_data.get('ah', [])
    session_bars = rth_bars + ah_bars

    exit_price = None
    for t_str, bar in session_bars:
        if t_str > exit_time_target:
            break
        if t_str >= entry_time:
            exit_price = bar['c']
    return exit_price


def get_tn_exit_price(tk, dt_s, n_days):
    """Get the daily close on the Nth next trading day via _gd_cache."""
    nxt = _next_n_td(dt_s, n_days)
    if len(nxt) < n_days:
        return None
    target_date = nxt[n_days - 1].strftime('%Y-%m-%d')
    daily = _gd_cache.get(target_date, {}).get(tk, {})
    if daily.get('c'):
        return daily['c']
    # Fetch if missing
    try:
        gd = fetch_grouped_daily(target_date)
        return gd.get(tk, {}).get('c')
    except Exception:
        return None


# All exit labels in order (for table columns)
EXIT_LABELS = EXIT_TIMES_INTRADAY + ['T+1', 'T+2']

print(f"\n  Computing PnL at {len(EXIT_LABELS)} exit points for {len(e2_trades)} trades...")

for t in e2_trades:
    tk = t['ticker']
    dt_s = t['scan_date']
    ep = t['entry_price']
    entry_time = t['entry_time']

    t['pnl_by_exit'] = {}

    # Intraday exits
    for exit_t in EXIT_TIMES_INTRADAY:
        if exit_t < entry_time:
            # Exit time is before entry -- cannot exit, skip
            t['pnl_by_exit'][exit_t] = None
            continue
        exit_price = get_intraday_exit_price(tk, dt_s, entry_time, exit_t)
        if exit_price is not None and ep > 0:
            t['pnl_by_exit'][exit_t] = round((ep - exit_price) / ep * 100, 3)
        else:
            # Fallback: use grouped daily close if intraday bars missing
            daily = _gd_cache.get(dt_s, {}).get(tk, {})
            if daily.get('c') and ep > 0:
                t['pnl_by_exit'][exit_t] = round((ep - daily['c']) / ep * 100, 3)
            else:
                t['pnl_by_exit'][exit_t] = None

    # T+1 exit
    t1_price = get_tn_exit_price(tk, dt_s, 1)
    if t1_price is not None and ep > 0:
        t['pnl_by_exit']['T+1'] = round((ep - t1_price) / ep * 100, 3)
    else:
        t['pnl_by_exit']['T+1'] = None

    # T+2 exit
    t2_price = get_tn_exit_price(tk, dt_s, 2)
    if t2_price is not None and ep > 0:
        t['pnl_by_exit']['T+2'] = round((ep - t2_price) / ep * 100, 3)
    else:
        t['pnl_by_exit']['T+2'] = None

    # Also store month for splitting
    t['month'] = dt_s[:7]  # e.g. '2026-01'

print(f"  Done.")


# =====================================================================================
# SPLIT INTO PERIODS
# =====================================================================================

def in_period(month_str, period_range):
    """Check if month_str (e.g. '2026-03') is within period_range ('2026-01','2026-06')."""
    return period_range[0] <= month_str <= period_range[1]

trades_jan_jun = [t for t in e2_trades if in_period(t['month'], PERIOD_JAN_JUN)]
trades_jul_sep = [t for t in e2_trades if in_period(t['month'], PERIOD_JUL_SEP)]

print(f"\n  Jan-Jun: {len(trades_jan_jun)} trades")
print(f"  Jul-Sep: {len(trades_jul_sep)} trades")


# =====================================================================================
# COMPUTE METRICS BY EXIT TIME x PERIOD
# =====================================================================================

def metrics_for_exit(trades_list, exit_label):
    """Compute metrics for a list of trades at a specific exit time."""
    pnls = [t['pnl_by_exit'].get(exit_label) for t in trades_list]
    pnls = [p for p in pnls if p is not None]
    m = compute_metrics(pnls, exit_label)
    total_dol = sum(CAPITAL * p / 100 for p in pnls)
    m['total_dol'] = round(total_dol)
    return m


# Build results table: {exit_label: {period: metrics}}
results_table = {}
for ex in EXIT_LABELS:
    results_table[ex] = {
        'jan_jun': metrics_for_exit(trades_jan_jun, ex),
        'jul_sep': metrics_for_exit(trades_jul_sep, ex),
        'all': metrics_for_exit(e2_trades, ex),
    }


# =====================================================================================
# OUTPUT: 2D COMPARISON TABLE
# =====================================================================================

print(f"\n\n{'='*80}")
print(f"  TIME DECAY ANALYSIS: Exit Time Sweep")
print(f"  {BACKTEST_START} -> {BACKTEST_END}")
print(f"  {len(e2_trades)} total trades | Jan-Jun: {len(trades_jan_jun)} | Jul-Sep: {len(trades_jul_sep)}")
print(f"{'='*80}")

# Header
print(f"\n  {'':>8s}  {'--- Jan-Jun ---':^44s}  {'--- Jul-Sep ---':^44s}")
print(f"  {'Exit':>8s}  {'n':>4s}  {'Avg%':>7s}  {'Med%':>7s}  {'WR%':>5s}  "
      f"{'Sharpe':>7s}  {'Total$':>10s}  "
      f"{'n':>4s}  {'Avg%':>7s}  {'Med%':>7s}  {'WR%':>5s}  "
      f"{'Sharpe':>7s}  {'Total$':>10s}")
print(f"  {'---':>8s}  {'---':>4s}  {'------':>7s}  {'------':>7s}  {'----':>5s}  "
      f"{'------':>7s}  {'---------':>10s}  "
      f"{'---':>4s}  {'------':>7s}  {'------':>7s}  {'----':>5s}  "
      f"{'------':>7s}  {'---------':>10s}")

best_jj_avg = -999
best_jj_exit = None
best_js_avg = -999
best_js_exit = None

for ex in EXIT_LABELS:
    jj = results_table[ex]['jan_jun']
    js = results_table[ex]['jul_sep']

    if jj['n'] > 0 and jj['avg'] > best_jj_avg:
        best_jj_avg = jj['avg']
        best_jj_exit = ex
    if js['n'] > 0 and js['avg'] > best_js_avg:
        best_js_avg = js['avg']
        best_js_exit = ex

    jj_str = (f"  {jj['n']:>4d}  {jj['avg']:>+6.2f}%  {jj['med']:>+6.2f}%  "
              f"{jj['wr']*100:>4.0f}%  {jj['sharpe']:>7.3f}  ${jj['total_dol']:>+9,d}"
              if jj['n'] > 0 else
              f"  {'--':>4s}  {'--':>7s}  {'--':>7s}  {'--':>5s}  {'--':>7s}  {'--':>10s}")

    js_str = (f"  {js['n']:>4d}  {js['avg']:>+6.2f}%  {js['med']:>+6.2f}%  "
              f"{js['wr']*100:>4.0f}%  {js['sharpe']:>7.3f}  ${js['total_dol']:>+9,d}"
              if js['n'] > 0 else
              f"  {'--':>4s}  {'--':>7s}  {'--':>7s}  {'--':>5s}  {'--':>7s}  {'--':>10s}")

    print(f"  {ex:>8s}{jj_str}{js_str}")


# =====================================================================================
# KEY ANALYSIS
# =====================================================================================

print(f"\n\n{'='*80}")
print(f"  KEY FINDINGS")
print(f"{'='*80}")

print(f"\n  1. OPTIMAL EXIT TIME")
print(f"     Jan-Jun: {best_jj_exit} (avg {best_jj_avg:+.2f}%)")
print(f"     Jul-Sep: {best_js_exit} (avg {best_js_avg:+.2f}%)")

if best_jj_exit != best_js_exit:
    print(f"\n     ** EXIT TIME HAS SHIFTED **")
    print(f"     Jan-Jun peaked at {best_jj_exit}, Jul-Sep peaks at {best_js_exit}.")
    print(f"     The fade is exhausting {'faster' if best_js_exit < best_jj_exit else 'slower'} "
          f"in the new regime.")
else:
    print(f"\n     Optimal exit time is STABLE at {best_jj_exit} across both periods.")


# Fade exhaustion point: where marginal PnL gain from holding longer turns negative
print(f"\n  2. FADE EXHAUSTION POINT")
print(f"     (Exit time where holding LONGER starts to hurt -- marginal gain < 0)")

for period_label, period_key in [('Jan-Jun', 'jan_jun'), ('Jul-Sep', 'jul_sep')]:
    prev_avg = None
    exhaustion_exit = None
    for ex in EXIT_TIMES_INTRADAY:
        m = results_table[ex][period_key]
        if m['n'] == 0:
            continue
        cur_avg = m['avg']
        if prev_avg is not None and cur_avg < prev_avg:
            exhaustion_exit = ex
            break
        prev_avg = cur_avg
    if exhaustion_exit:
        print(f"     {period_label}: fade exhausts at {exhaustion_exit} "
              f"(PnL starts declining after the prior exit time)")
    else:
        print(f"     {period_label}: NO exhaustion -- PnL keeps rising through {EXIT_TIMES_INTRADAY[-1]}")

# Marginal PnL table
print(f"\n  3. MARGINAL PnL GAIN (change from previous exit time)")
print(f"\n  {'Exit':>8s}  {'Jan-Jun':>10s}  {'Jul-Sep':>10s}")
print(f"  {'---':>8s}  {'----------':>10s}  {'----------':>10s}")

prev_jj = None
prev_js = None
for ex in EXIT_LABELS:
    jj = results_table[ex]['jan_jun']
    js = results_table[ex]['jul_sep']

    jj_delta = f"{jj['avg'] - prev_jj:>+.2f}%" if prev_jj is not None and jj['n'] > 0 else '--'
    js_delta = f"{js['avg'] - prev_js:>+.2f}%" if prev_js is not None and js['n'] > 0 else '--'

    print(f"  {ex:>8s}  {jj_delta:>10s}  {js_delta:>10s}")

    if jj['n'] > 0:
        prev_jj = jj['avg']
    if js['n'] > 0:
        prev_js = js['avg']


# =====================================================================================
# MONTHLY HEATMAP DATA
# =====================================================================================

# Collect all months present in the data
all_months = sorted(set(t['month'] for t in e2_trades))
monthly_data = {}  # {exit_label: {month: metrics}}
for ex in EXIT_LABELS:
    monthly_data[ex] = {}
    for month in all_months:
        month_trades = [t for t in e2_trades if t['month'] == month]
        monthly_data[ex][month] = metrics_for_exit(month_trades, ex)

# Print monthly breakdown
print(f"\n\n  4. MONTHLY AVG PnL BY EXIT TIME")
header = f"  {'Exit':>8s}"
for m in all_months:
    header += f"  {m[-2:]:>6s}"
print(header)
print(f"  {'---':>8s}" + f"  {'------':>6s}" * len(all_months))

for ex in EXIT_LABELS:
    row = f"  {ex:>8s}"
    for m in all_months:
        md = monthly_data[ex][m]
        if md['n'] > 0:
            row += f"  {md['avg']:>+5.1f}%"
        else:
            row += f"  {'--':>6s}"
    print(row)


# =====================================================================================
# CHARTS
# =====================================================================================

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()


# --- Chart 1: Avg PnL by exit time -- two lines (Jan-Jun vs Jul-Sep) ---

fig, ax = plt.subplots(figsize=(14, 7))

x_labels = EXIT_LABELS
x = np.arange(len(x_labels))

jj_avgs = [results_table[ex]['jan_jun']['avg'] if results_table[ex]['jan_jun']['n'] > 0
           else float('nan') for ex in EXIT_LABELS]
js_avgs = [results_table[ex]['jul_sep']['avg'] if results_table[ex]['jul_sep']['n'] > 0
           else float('nan') for ex in EXIT_LABELS]

ax.plot(x, jj_avgs, 'o-', color='#2980b9', linewidth=2.5, markersize=8,
        label=f'Jan-Jun (n={len(trades_jan_jun)})', zorder=3)
ax.plot(x, js_avgs, 's--', color='#e74c3c', linewidth=2.5, markersize=8,
        label=f'Jul-Sep (n={len(trades_jul_sep)})', zorder=3)

# Annotate values
for i, (jj, js) in enumerate(zip(jj_avgs, js_avgs)):
    if not np.isnan(jj):
        ax.annotate(f'{jj:+.1f}%', (x[i], jj), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=8, color='#2980b9')
    if not np.isnan(js):
        ax.annotate(f'{js:+.1f}%', (x[i], js), textcoords="offset points",
                    xytext=(0, -16), ha='center', fontsize=8, color='#e74c3c')

ax.axhline(0, color='black', lw=0.8, alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
ax.set_ylabel('Avg PnL %', fontsize=11)
ax.set_title(f'Agent 1 SHORT Gap-Fade: Avg PnL by Exit Time\n'
             f'Jan-Jun vs Jul-Sep 2026 ({len(e2_trades)} total trades)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
p1 = os.path.join(BASE_DIR, 'time_decay_avg_pnl.png')
plt.savefig(p1, dpi=150, bbox_inches='tight')
plt.show()
print(f"\n  Saved: {p1}")


# --- Chart 2: Sharpe by exit time -- two lines ---

fig, ax = plt.subplots(figsize=(14, 7))

jj_sharpe = [results_table[ex]['jan_jun']['sharpe'] if results_table[ex]['jan_jun']['n'] > 0
             else float('nan') for ex in EXIT_LABELS]
js_sharpe = [results_table[ex]['jul_sep']['sharpe'] if results_table[ex]['jul_sep']['n'] > 0
             else float('nan') for ex in EXIT_LABELS]

ax.plot(x, jj_sharpe, 'o-', color='#2980b9', linewidth=2.5, markersize=8,
        label=f'Jan-Jun (n={len(trades_jan_jun)})', zorder=3)
ax.plot(x, js_sharpe, 's--', color='#e74c3c', linewidth=2.5, markersize=8,
        label=f'Jul-Sep (n={len(trades_jul_sep)})', zorder=3)

for i, (jj, js) in enumerate(zip(jj_sharpe, js_sharpe)):
    if not np.isnan(jj):
        ax.annotate(f'{jj:.2f}', (x[i], jj), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=8, color='#2980b9')
    if not np.isnan(js):
        ax.annotate(f'{js:.2f}', (x[i], js), textcoords="offset points",
                    xytext=(0, -16), ha='center', fontsize=8, color='#e74c3c')

ax.axhline(0, color='black', lw=0.8, alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
ax.set_ylabel('Sharpe Ratio (avg/std)', fontsize=11)
ax.set_title(f'Agent 1 SHORT Gap-Fade: Sharpe by Exit Time\n'
             f'Jan-Jun vs Jul-Sep 2026',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
p2 = os.path.join(BASE_DIR, 'time_decay_sharpe.png')
plt.savefig(p2, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved: {p2}")


# --- Chart 3: Heatmap of avg PnL (rows=exit times, cols=months) ---

heatmap_data = []
for ex in EXIT_LABELS:
    row = []
    for m in all_months:
        md = monthly_data[ex][m]
        row.append(md['avg'] if md['n'] > 0 else float('nan'))
    heatmap_data.append(row)

hm_array = np.array(heatmap_data)

fig, ax = plt.subplots(figsize=(max(10, len(all_months) * 1.2), max(8, len(EXIT_LABELS) * 0.6)))

# Use diverging colormap centered at 0
vmax = max(abs(np.nanmin(hm_array)), abs(np.nanmax(hm_array)))
vmax = max(vmax, 1.0)  # minimum range
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

im = ax.imshow(hm_array, cmap='RdYlGn', norm=norm, aspect='auto')
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Avg PnL %', fontsize=10)

# Axis labels
ax.set_xticks(np.arange(len(all_months)))
month_labels = [m[-2:] for m in all_months]
ax.set_xticklabels(month_labels, fontsize=10)
ax.set_xlabel('Month (2026)', fontsize=11)
ax.set_yticks(np.arange(len(EXIT_LABELS)))
ax.set_yticklabels(EXIT_LABELS, fontsize=10)
ax.set_ylabel('Exit Time', fontsize=11)

# Annotate cells
for i in range(len(EXIT_LABELS)):
    for j in range(len(all_months)):
        val = hm_array[i, j]
        if not np.isnan(val):
            n_trades_cell = monthly_data[EXIT_LABELS[i]][all_months[j]]['n']
            color = 'black' if abs(val) < vmax * 0.6 else 'white'
            ax.text(j, i, f'{val:+.1f}\nn={n_trades_cell}',
                    ha='center', va='center', fontsize=7, color=color)

# Draw vertical line between Jun and Jul
jun_idx = None
for j, m in enumerate(all_months):
    if m == '2026-06':
        jun_idx = j
        break
if jun_idx is not None:
    ax.axvline(jun_idx + 0.5, color='black', linewidth=2, linestyle='--', alpha=0.7)
    ax.text(jun_idx + 0.5, -0.8, 'Jan-Jun | Jul-Sep',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_title(f'Agent 1 SHORT Gap-Fade: Avg PnL Heatmap by Exit Time x Month\n'
             f'{len(e2_trades)} trades, {BACKTEST_START} -> {BACKTEST_END}',
             fontsize=12, fontweight='bold')

plt.tight_layout()
p3 = os.path.join(BASE_DIR, 'time_decay_heatmap.png')
plt.savefig(p3, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved: {p3}")


# =====================================================================================
# BOTTOM LINE
# =====================================================================================

print(f"\n\n{'='*80}")
print(f"  BOTTOM LINE")
print(f"{'='*80}")

print(f"\n  Jan-Jun optimal: {best_jj_exit} at {best_jj_avg:+.2f}% avg")
print(f"  Jul-Sep optimal: {best_js_exit} at {best_js_avg:+.2f}% avg")

if best_jj_exit != best_js_exit:
    # Find the PnL at 17:00 for Jul-Sep to quantify the cost of late exit
    js_17 = results_table['17:00']['jul_sep']
    js_best = results_table[best_js_exit]['jul_sep']
    cost_of_late = js_17['avg'] - js_best['avg'] if js_17['n'] > 0 and js_best['n'] > 0 else 0
    dol_cost = js_17['total_dol'] - js_best['total_dol'] if js_17['n'] > 0 else 0

    print(f"\n  REGIME SHIFT DETECTED:")
    print(f"  In Jul-Sep, holding to 17:00 vs exiting at {best_js_exit}:")
    print(f"    Avg PnL difference: {cost_of_late:+.2f}% per trade")
    print(f"    Dollar cost: ${dol_cost:+,.0f}")
    if cost_of_late < 0:
        print(f"\n  The fade is exhausting FASTER in Jul-Sep.")
        print(f"  Stocks that gap up are recovering sooner, eating into SHORT profits")
        print(f"  when held too long. Consider moving exit to {best_js_exit}.")
    else:
        print(f"\n  The fade is exhausting SLOWER in Jul-Sep (holding longer still helps).")
else:
    print(f"\n  NO SHIFT. Optimal exit is stable at {best_jj_exit}.")
    # But still check if the MAGNITUDE changed
    mag_diff = best_js_avg - best_jj_avg
    print(f"  Magnitude change: {mag_diff:+.2f}% (Jul-Sep vs Jan-Jun)")
    if abs(mag_diff) > 2:
        if mag_diff < 0:
            print(f"  The fade is WEAKER in Jul-Sep -- same optimal time but lower PnL.")
        else:
            print(f"  The fade is STRONGER in Jul-Sep -- same optimal time but higher PnL.")


# =====================================================================================
# SAVE RESULTS JSON
# =====================================================================================

results_dir = os.path.join(BASE_DIR, 'results')
os.makedirs(results_dir, exist_ok=True)

output = {
    'run_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
    'period': f'{BACKTEST_START} to {BACKTEST_END}',
    'n_trades': len(e2_trades),
    'n_jan_jun': len(trades_jan_jun),
    'n_jul_sep': len(trades_jul_sep),
    'optimal_exit': {
        'jan_jun': best_jj_exit,
        'jul_sep': best_js_exit,
        'shift_detected': best_jj_exit != best_js_exit,
    },
    'by_exit_time': {},
    'monthly_heatmap': {},
}

for ex in EXIT_LABELS:
    jj = results_table[ex]['jan_jun']
    js = results_table[ex]['jul_sep']
    a = results_table[ex]['all']
    output['by_exit_time'][ex] = {
        'jan_jun': {
            'n': jj['n'], 'avg': round(jj['avg'], 2), 'med': round(jj['med'], 2),
            'wr': round(jj['wr'], 3), 'sharpe': round(jj['sharpe'], 3),
            'total_dol': jj['total_dol'],
            'max_loss': round(jj['max_loss'], 2), 'max_gain': round(jj['max_gain'], 2),
        },
        'jul_sep': {
            'n': js['n'], 'avg': round(js['avg'], 2), 'med': round(js['med'], 2),
            'wr': round(js['wr'], 3), 'sharpe': round(js['sharpe'], 3),
            'total_dol': js['total_dol'],
            'max_loss': round(js['max_loss'], 2), 'max_gain': round(js['max_gain'], 2),
        },
        'all': {
            'n': a['n'], 'avg': round(a['avg'], 2), 'med': round(a['med'], 2),
            'wr': round(a['wr'], 3), 'sharpe': round(a['sharpe'], 3),
            'total_dol': a['total_dol'],
            'max_loss': round(a['max_loss'], 2), 'max_gain': round(a['max_gain'], 2),
        },
    }

for ex in EXIT_LABELS:
    output['monthly_heatmap'][ex] = {}
    for m in all_months:
        md = monthly_data[ex][m]
        output['monthly_heatmap'][ex][m] = {
            'n': md['n'],
            'avg': round(md['avg'], 2) if md['n'] > 0 else None,
            'wr': round(md['wr'], 3) if md['n'] > 0 else None,
            'sharpe': round(md['sharpe'], 3) if md['n'] > 0 else None,
        }

json_path = os.path.join(results_dir, 'time_decay_analysis.json')
with open(json_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\n  Saved: {json_path}")

print(f"\n{'='*80}")
print(f"  DONE -- Time decay analysis complete")
print(f"{'='*80}")
