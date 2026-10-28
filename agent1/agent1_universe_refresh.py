# %% Agent 1 — Universe Refresh: Build NEW_FLOAT from Recent Gap Data
#
# Run AFTER backtest_e2_reproduction.py or agent1_eval_sl_skip.py in same
# Colab session to reuse cached data. If run standalone, pipeline executes
# automatically (~10 min).
#
# Purpose: The hardcoded _FLOAT dict (217 tickers) was built 1.5 years ago.
# This script scans ALL gap-up candidates from the backtest period, ranks
# tickers by gap magnitude, frequency, and mean-reversion rate, selects
# the top 217 as NEW_FLOAT, and compares performance vs the original.

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
from matplotlib.ticker import FuncFormatter

try:
    from google.colab import auth
    try: auth.authenticate_user()
    except: pass
except: pass


# =============================================================================
# CONFIG
# =============================================================================

BACKTEST_START = '2026-01-02'
BACKTEST_END   = '2026-09-11'

GAP_MIN_PCT    = 5.0          # for E2 trade selection
GAP_SCAN_PCT   = 3.0          # for universe building (lower bar to catch more)
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

NEW_FLOAT_SIZE = 217          # match original dict size

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "")
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

ORIG_FLOAT_SET = set(_FLOAT.keys())


# =============================================================================
# HELPERS (identical to agent1_eval_sl_skip.py)
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


# =============================================================================
# DATA PIPELINE (skips if already loaded from prior cell)
# =============================================================================

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
    print(f"  UNIVERSE REFRESH — Data Pipeline")
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


# =============================================================================
# BUILD E2 BASELINE TRADES (for performance comparison)
# =============================================================================

try:
    _ = e2_trades
    if len(e2_trades) > 0 and 'pnl_eod' in e2_trades[0]:
        print(f"  Using cached e2_trades ({len(e2_trades)} trades) from prior cell")
    else:
        raise NameError
except NameError:
    print(f"\n  Building E2 baseline trades...")
    e2_trades = []
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
            ep = tc['entry_price']

            rth_bars = bar_data.get('rth', [])
            ah_bars = bar_data.get('ah', [])
            session_bars = rth_bars + ah_bars

            exit_price = None
            for t_str, bar in session_bars:
                if t_str > EXIT_TIME:
                    break
                if t_str >= tc['entry_time']:
                    exit_price = bar['c']
            if exit_price is None:
                daily = _gd_cache.get(dt_s, {}).get(tk, {})
                if daily.get('c'):
                    exit_price = daily['c']
            if exit_price is None:
                continue

            tc['pnl_eod'] = round((ep - exit_price) / ep * 100, 3)
            e2_trades.append(tc)

    print(f"  -> {len(e2_trades)} E2 baseline trades built")


# =============================================================================
# STEP 1: BUILD NEW_FLOAT UNIVERSE FROM all_candidates
# =============================================================================

print(f"\n\n{'='*80}")
print(f"  STEP 1: BUILD NEW_FLOAT UNIVERSE FROM GAP DATA")
print(f"  Scanning all_candidates for tickers that gapped >= {GAP_SCAN_PCT}%")
print(f"  Period: {BACKTEST_START} -> {BACKTEST_END} ({len(dates)} trading days)")
print(f"{'='*80}")

# Collect per-ticker gap statistics across all dates
# all_candidates already has every gap-up stock (>= 3%), not just top N
ticker_gap_days = defaultdict(list)  # ticker -> [(date, gap_pct), ...]

for dt, cands in all_candidates.items():
    for c in cands:
        tk = c['ticker']
        gap = c['gap_pct']
        if gap >= GAP_SCAN_PCT / 100.0:
            ticker_gap_days[tk].append((dt, gap))

n_total_tickers = len(ticker_gap_days)
print(f"\n  {n_total_tickers} unique tickers gapped >= {GAP_SCAN_PCT}% at least once")

# Filter to CS only (exclude ETF/ETN/etc)
# Need to fetch refs for any tickers not yet in _ticker_ref
all_gap_tickers = set(ticker_gap_days.keys())
missing_refs = [tk for tk in all_gap_tickers if tk not in _ticker_ref]
if missing_refs:
    print(f"  Fetching reference for {len(missing_refs)} additional tickers...")
    for c_start in range(0, len(missing_refs), 200):
        chunk = missing_refs[c_start:c_start + 200]
        fetch_ticker_refs(chunk)
        done = min(c_start + 200, len(missing_refs))
        if done % 400 == 0 or done == len(missing_refs):
            print(f"    {done}/{len(missing_refs)}")

cs_gap_tickers = {}
n_etf_dropped = 0
n_unknown_type = 0
for tk, gap_list in ticker_gap_days.items():
    ref = _ticker_ref.get(tk, {})
    at = ref.get('type', 'UNK')
    if at == 'CS':
        cs_gap_tickers[tk] = gap_list
    elif at in {'ETF', 'ETV', 'ETN', 'ETP'}:
        n_etf_dropped += 1
    else:
        n_unknown_type += 1

print(f"  After CS filter: {len(cs_gap_tickers)} CS tickers "
      f"(dropped {n_etf_dropped} ETF/ETN, {n_unknown_type} unknown/other)")

# Compute per-ticker stats
n_trading_days = len(dates)

ticker_stats = {}
for tk, gap_list in cs_gap_tickers.items():
    gaps = [g for _, g in gap_list]
    gap_dates = [d for d, _ in gap_list]
    n_gap = len(gaps)

    avg_gap = float(np.mean(gaps))
    gap_freq = n_gap / n_trading_days

    # Mean reversion rate: fraction of gap days where close < open
    n_reverted = 0
    n_checked = 0
    for d in gap_dates:
        daily = _gd_cache.get(d, {}).get(tk, {})
        if daily and daily.get('o', 0) > 0 and daily.get('c', 0) > 0:
            n_checked += 1
            if daily['c'] < daily['o']:
                n_reverted += 1
    mr_rate = n_reverted / n_checked if n_checked > 0 else 0

    # Float data: use _FLOAT if available, else _ticker_ref
    float_shares = _FLOAT.get(tk)
    if float_shares is None:
        ref = _ticker_ref.get(tk, {})
        float_shares = ref.get('shares')

    ticker_stats[tk] = {
        'ticker': tk,
        'avg_gap_magnitude': avg_gap,
        'gap_frequency': gap_freq,
        'mean_reversion_rate': mr_rate,
        'n_gap_days': n_gap,
        'n_checked': n_checked,
        'float_shares': float_shares,
        'in_orig_float': tk in ORIG_FLOAT_SET,
    }

print(f"\n  Computed stats for {len(ticker_stats)} CS tickers")

# Normalize and compute composite score
all_avg_gaps = [s['avg_gap_magnitude'] for s in ticker_stats.values()]
all_freqs = [s['gap_frequency'] for s in ticker_stats.values()]
all_mr = [s['mean_reversion_rate'] for s in ticker_stats.values()]

def min_max_norm(val, vals):
    mn, mx = min(vals), max(vals)
    if mx == mn: return 0.5
    return (val - mn) / (mx - mn)

for tk, s in ticker_stats.items():
    norm_gap = min_max_norm(s['avg_gap_magnitude'], all_avg_gaps)
    norm_freq = min_max_norm(s['gap_frequency'], all_freqs)
    norm_mr = min_max_norm(s['mean_reversion_rate'], all_mr)
    s['norm_gap'] = norm_gap
    s['norm_freq'] = norm_freq
    s['norm_mr'] = norm_mr
    s['composite_score'] = 0.4 * norm_gap + 0.3 * norm_freq + 0.3 * norm_mr

# Rank and select top NEW_FLOAT_SIZE
ranked = sorted(ticker_stats.values(), key=lambda x: -x['composite_score'])
new_float_list = ranked[:NEW_FLOAT_SIZE]
NEW_FLOAT_SET = set(s['ticker'] for s in new_float_list)

# Build NEW_FLOAT dict (use existing float value or _ticker_ref shares)
NEW_FLOAT = {}
for s in new_float_list:
    tk = s['ticker']
    if s['float_shares'] is not None:
        NEW_FLOAT[tk] = s['float_shares']
    else:
        NEW_FLOAT[tk] = 0  # unknown float, will be included but flagged

print(f"  Selected top {NEW_FLOAT_SIZE} -> NEW_FLOAT")

# Print composite score distribution
scores = [s['composite_score'] for s in new_float_list]
print(f"\n  Composite score range: {min(scores):.3f} - {max(scores):.3f}")
print(f"  Median composite: {np.median(scores):.3f}")
cutoff_score = new_float_list[-1]['composite_score']
print(f"  Cutoff score (#{NEW_FLOAT_SIZE}): {cutoff_score:.3f}")

# Show top 20
print(f"\n  ---- TOP 20 NEW_FLOAT BY COMPOSITE SCORE ----\n")
print(f"  {'Rank':>4s}  {'Ticker':>6s}  {'Score':>6s}  {'AvgGap%':>8s}  "
      f"{'GapFreq':>7s}  {'MR Rate':>7s}  {'N gaps':>6s}  "
      f"{'In Orig':>7s}  {'Float':>12s}")
print(f"  {'----':>4s}  {'------':>6s}  {'------':>6s}  {'--------':>8s}  "
      f"{'-------':>7s}  {'-------':>7s}  {'------':>6s}  "
      f"{'-------':>7s}  {'------------':>12s}")

for i, s in enumerate(new_float_list[:20]):
    in_orig = 'YES' if s['in_orig_float'] else 'new'
    flt = f"{s['float_shares']:>12,}" if s['float_shares'] else '     unknown'
    print(f"  {i+1:>4d}  {s['ticker']:>6s}  {s['composite_score']:>6.3f}  "
          f"{s['avg_gap_magnitude']*100:>7.1f}%  {s['gap_frequency']:>7.4f}  "
          f"{s['mean_reversion_rate']:>7.2f}  {s['n_gap_days']:>6d}  "
          f"{in_orig:>7s}  {flt}")


# =============================================================================
# STEP 2: COMPARE UNIVERSES
# =============================================================================

print(f"\n\n{'='*80}")
print(f"  STEP 2: UNIVERSE COMPARISON")
print(f"{'='*80}")

overlap = ORIG_FLOAT_SET & NEW_FLOAT_SET
orig_only = ORIG_FLOAT_SET - NEW_FLOAT_SET
new_only = NEW_FLOAT_SET - ORIG_FLOAT_SET

print(f"\n  Original _FLOAT:     {len(ORIG_FLOAT_SET):>4d} tickers")
print(f"  NEW_FLOAT:           {len(NEW_FLOAT_SET):>4d} tickers")
print(f"  Overlap:             {len(overlap):>4d} tickers ({len(overlap)/len(ORIG_FLOAT_SET)*100:.1f}%)")
print(f"  Original only:       {len(orig_only):>4d} (dropped from new)")
print(f"  New entrants:        {len(new_only):>4d} (not in original)")

# Jaccard similarity
jaccard = len(overlap) / len(ORIG_FLOAT_SET | NEW_FLOAT_SET)
print(f"  Jaccard similarity:  {jaccard:.3f}")

# Show dropped tickers with their stats (if they were in the gap data)
print(f"\n  ---- DROPPED FROM ORIGINAL (not in NEW_FLOAT) ----")
print(f"  These {len(orig_only)} tickers were in the original _FLOAT but did NOT")
print(f"  score high enough to make the new top-{NEW_FLOAT_SIZE}.\n")

dropped_with_stats = []
for tk in sorted(orig_only):
    s = ticker_stats.get(tk)
    if s:
        dropped_with_stats.append(s)
    else:
        dropped_with_stats.append({
            'ticker': tk, 'composite_score': 0, 'avg_gap_magnitude': 0,
            'gap_frequency': 0, 'mean_reversion_rate': 0, 'n_gap_days': 0,
        })

# Sort by composite score descending to show near-misses first
dropped_with_stats.sort(key=lambda x: -x['composite_score'])

print(f"  {'Ticker':>6s}  {'Score':>6s}  {'AvgGap%':>8s}  {'Freq':>6s}  "
      f"{'MR Rate':>7s}  {'N gaps':>6s}  {'Reason':>20s}")
print(f"  {'------':>6s}  {'------':>6s}  {'--------':>8s}  {'------':>6s}  "
      f"{'-------':>7s}  {'------':>6s}  {'--------------------':>20s}")

for s in dropped_with_stats[:30]:
    reason = 'below cutoff' if s['composite_score'] > 0 else 'never gapped >=3%'
    print(f"  {s['ticker']:>6s}  {s['composite_score']:>6.3f}  "
          f"{s['avg_gap_magnitude']*100:>7.1f}%  {s['gap_frequency']:>6.4f}  "
          f"{s['mean_reversion_rate']:>7.2f}  {s['n_gap_days']:>6d}  {reason:>20s}")

if len(dropped_with_stats) > 30:
    print(f"  ... and {len(dropped_with_stats) - 30} more")

# Show new entrants
print(f"\n\n  ---- NEW ENTRANTS (in NEW_FLOAT, not in original) ----")
print(f"  These {len(new_only)} tickers were NOT in the original _FLOAT but")
print(f"  scored high enough to enter the new top-{NEW_FLOAT_SIZE}.\n")

new_entrant_stats = [ticker_stats[tk] for tk in new_only if tk in ticker_stats]
new_entrant_stats.sort(key=lambda x: -x['composite_score'])

print(f"  {'Rank':>4s}  {'Ticker':>6s}  {'Score':>6s}  {'AvgGap%':>8s}  "
      f"{'Freq':>6s}  {'MR Rate':>7s}  {'N gaps':>6s}  {'Float':>12s}")
print(f"  {'----':>4s}  {'------':>6s}  {'------':>6s}  {'--------':>8s}  "
      f"{'------':>6s}  {'-------':>7s}  {'------':>6s}  {'------------':>12s}")

for i, s in enumerate(new_entrant_stats[:30]):
    flt = f"{s['float_shares']:>12,}" if s['float_shares'] else '     unknown'
    print(f"  {i+1:>4d}  {s['ticker']:>6s}  {s['composite_score']:>6.3f}  "
          f"{s['avg_gap_magnitude']*100:>7.1f}%  {s['gap_frequency']:>6.4f}  "
          f"{s['mean_reversion_rate']:>7.2f}  {s['n_gap_days']:>6d}  {flt}")

if len(new_entrant_stats) > 30:
    print(f"  ... and {len(new_entrant_stats) - 30} more")


# =============================================================================
# STEP 3: PERFORMANCE COMPARISON — tag E2 trades by universe membership
# =============================================================================

print(f"\n\n{'='*80}")
print(f"  STEP 3: PERFORMANCE BY UNIVERSE MEMBERSHIP")
print(f"  Tagging {len(e2_trades)} E2 trades by _FLOAT / NEW_FLOAT membership")
print(f"{'='*80}")

# Tag each trade
group_trades = defaultdict(list)  # group -> list of pnl
for t in e2_trades:
    tk = t['ticker']
    in_orig = tk in ORIG_FLOAT_SET
    in_new = tk in NEW_FLOAT_SET
    if in_orig and in_new:
        group = 'BOTH'
    elif in_orig and not in_new:
        group = 'ORIG_ONLY'
    elif not in_orig and in_new:
        group = 'NEW_ONLY'
    else:
        group = 'NEITHER'
    group_trades[group].append(t['pnl_eod'])

print(f"\n  {'Group':>12s}  {'N':>5s}  {'Avg PnL':>8s}  {'Median':>7s}  "
      f"{'WR':>5s}  {'Sharpe':>7s}  {'CI Low':>7s}  {'CI High':>7s}")
print(f"  {'------------':>12s}  {'-----':>5s}  {'--------':>8s}  {'-------':>7s}  "
      f"{'-----':>5s}  {'-------':>7s}  {'-------':>7s}  {'-------':>7s}")

all_group_metrics = {}
for grp in ['BOTH', 'ORIG_ONLY', 'NEW_ONLY', 'NEITHER']:
    pnls = group_trades.get(grp, [])
    m = compute_metrics(pnls, grp)
    all_group_metrics[grp] = m
    if m['n'] > 0:
        print(f"  {grp:>12s}  {m['n']:>5d}  {m['avg']:>+7.2f}%  {m['med']:>+6.2f}%  "
              f"{m['wr']*100:>4.0f}%  {m['sharpe']:>7.3f}  {m['ci_lo']:>+6.2f}%  "
              f"{m['ci_hi']:>+6.2f}%")
    else:
        print(f"  {grp:>12s}  {m['n']:>5d}  {'---':>8s}  {'---':>7s}  "
              f"{'---':>5s}  {'---':>7s}  {'---':>7s}  {'---':>7s}")

# Overall metrics
all_pnls = [t['pnl_eod'] for t in e2_trades]
all_m = compute_metrics(all_pnls, 'ALL')
print(f"  {'ALL':>12s}  {all_m['n']:>5d}  {all_m['avg']:>+7.2f}%  {all_m['med']:>+6.2f}%  "
      f"{all_m['wr']*100:>4.0f}%  {all_m['sharpe']:>7.3f}  {all_m['ci_lo']:>+6.2f}%  "
      f"{all_m['ci_hi']:>+6.2f}%")


# =============================================================================
# STEP 4: MONTHLY PERFORMANCE COMPARISON (Jul-Sep focus)
# =============================================================================

print(f"\n\n{'='*80}")
print(f"  STEP 4: MONTHLY BREAKDOWN — ORIG vs NEW UNIVERSE TRADES")
print(f"{'='*80}")

# Monthly breakdown of e2_trades tagged by group
monthly = defaultdict(lambda: defaultdict(list))
for t in e2_trades:
    tk = t['ticker']
    month = t['scan_date'][:7]  # YYYY-MM
    in_orig = tk in ORIG_FLOAT_SET
    in_new = tk in NEW_FLOAT_SET
    if in_orig and in_new:
        grp = 'BOTH'
    elif in_orig and not in_new:
        grp = 'ORIG_ONLY'
    elif not in_orig and in_new:
        grp = 'NEW_ONLY'
    else:
        grp = 'NEITHER'
    monthly[month][grp].append(t['pnl_eod'])
    monthly[month]['ALL'].append(t['pnl_eod'])

print(f"\n  {'Month':>7s}  {'Group':>12s}  {'N':>4s}  {'Avg':>7s}  {'WR':>5s}  "
      f"{'Sharpe':>7s}  {'Total $':>10s}")
print(f"  {'-------':>7s}  {'------------':>12s}  {'----':>4s}  {'-------':>7s}  "
      f"{'-----':>5s}  {'-------':>7s}  {'----------':>10s}")

for month in sorted(monthly.keys()):
    for grp in ['ALL', 'BOTH', 'ORIG_ONLY', 'NEW_ONLY', 'NEITHER']:
        pnls = monthly[month].get(grp, [])
        if not pnls:
            continue
        m = compute_metrics(pnls, grp)
        total_dol = sum(CAPITAL * p / 100 for p in pnls)
        print(f"  {month:>7s}  {grp:>12s}  {m['n']:>4d}  {m['avg']:>+6.2f}%  "
              f"{m['wr']*100:>4.0f}%  {m['sharpe']:>7.3f}  ${total_dol:>+9,.0f}")
    print()


# =============================================================================
# STEP 5: TOP 20 NEW CANDIDATES — detailed analysis
# =============================================================================

print(f"\n{'='*80}")
print(f"  STEP 5: TOP 20 NEW CANDIDATES (in NEW_FLOAT, not in original _FLOAT)")
print(f"{'='*80}")

# Find e2_trades for these new tickers
e2_by_ticker = defaultdict(list)
for t in e2_trades:
    e2_by_ticker[t['ticker']].append(t)

top_new = new_entrant_stats[:20]

print(f"\n  {'#':>3s}  {'Ticker':>6s}  {'Score':>6s}  {'AvgGap':>7s}  {'Freq':>6s}  "
      f"{'MR':>5s}  {'N gap':>5s}  {'E2 n':>4s}  {'E2 avg':>7s}  {'E2 WR':>5s}  "
      f"{'Float':>12s}  {'Name':>20s}")
print(f"  {'---':>3s}  {'------':>6s}  {'------':>6s}  {'-------':>7s}  {'------':>6s}  "
      f"{'-----':>5s}  {'-----':>5s}  {'----':>4s}  {'-------':>7s}  {'-----':>5s}  "
      f"{'------------':>12s}  {'--------------------':>20s}")

for i, s in enumerate(top_new):
    tk = s['ticker']
    e2t = e2_by_ticker.get(tk, [])
    ref = _ticker_ref.get(tk, {})
    name = ref.get('name', '')[:20]
    flt = f"{s['float_shares']:>12,}" if s['float_shares'] else '     unknown'

    if e2t:
        e2_pnls = [t['pnl_eod'] for t in e2t]
        e2_avg = np.mean(e2_pnls)
        e2_wr = sum(1 for p in e2_pnls if p > 0) / len(e2_pnls)
        e2_avg_str = f"{e2_avg:>+6.1f}%"
        e2_wr_str = f"{e2_wr*100:>4.0f}%"
        e2_n_str = f"{len(e2t):>4d}"
    else:
        e2_avg_str = '    N/A'
        e2_wr_str = '  N/A'
        e2_n_str = '   0'

    print(f"  {i+1:>3d}  {tk:>6s}  {s['composite_score']:>6.3f}  "
          f"{s['avg_gap_magnitude']*100:>6.1f}%  {s['gap_frequency']:>6.4f}  "
          f"{s['mean_reversion_rate']:>5.2f}  {s['n_gap_days']:>5d}  "
          f"{e2_n_str}  {e2_avg_str}  {e2_wr_str}  {flt}  {name}")


# =============================================================================
# STEP 6: CHARTS
# =============================================================================

print(f"\n\n{'='*80}")
print(f"  STEP 6: GENERATING CHARTS")
print(f"{'='*80}")

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---- Chart 1: Universe Overlap Bar Chart ----

fig, ax = plt.subplots(figsize=(10, 6))
categories = ['Original _FLOAT\nonly', 'Overlap\n(both)', 'NEW_FLOAT\nonly']
counts = [len(orig_only), len(overlap), len(new_only)]
colors = ['#c44e52', '#4c72b0', '#55a868']
bars = ax.bar(categories, counts, color=colors, edgecolor='#333', linewidth=0.8, width=0.6)

for bar, cnt in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            str(cnt), ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylabel('Number of Tickers', fontsize=12)
ax.set_title(f'Universe Comparison: Original _FLOAT vs NEW_FLOAT\n'
             f'Overlap: {len(overlap)}/{len(ORIG_FLOAT_SET)} '
             f'({len(overlap)/len(ORIG_FLOAT_SET)*100:.1f}%)  |  '
             f'Jaccard: {jaccard:.3f}', fontsize=13)
ax.set_ylim(0, max(counts) * 1.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
chart1_path = os.path.join(RESULTS_DIR, 'universe_overlap.png')
fig.savefig(chart1_path, dpi=150, bbox_inches='tight')
print(f"  Chart 1 saved: {chart1_path}")
plt.close(fig)


# ---- Chart 2: Composite Score Distribution ----

fig, ax = plt.subplots(figsize=(12, 6))

# Scores for tickers in original _FLOAT (that also have stats)
orig_scores = [ticker_stats[tk]['composite_score']
               for tk in ORIG_FLOAT_SET if tk in ticker_stats]
new_scores = [s['composite_score'] for s in new_float_list]

bins = np.linspace(0, 1, 30)
ax.hist(orig_scores, bins=bins, alpha=0.6, label=f'Original _FLOAT (n={len(orig_scores)})',
        color='#c44e52', edgecolor='#333', linewidth=0.5)
ax.hist(new_scores, bins=bins, alpha=0.6, label=f'NEW_FLOAT (n={len(new_scores)})',
        color='#4c72b0', edgecolor='#333', linewidth=0.5)

# Cutoff line
ax.axvline(cutoff_score, color='#333', linestyle='--', linewidth=1.5,
           label=f'Cutoff = {cutoff_score:.3f}')

ax.set_xlabel('Composite Score', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Composite Score Distribution: Original vs NEW_FLOAT\n'
             '(0.4 * gap_magnitude + 0.3 * gap_frequency + 0.3 * mean_reversion_rate)',
             fontsize=12)
ax.legend(fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
chart2_path = os.path.join(RESULTS_DIR, 'composite_score_dist.png')
fig.savefig(chart2_path, dpi=150, bbox_inches='tight')
print(f"  Chart 2 saved: {chart2_path}")
plt.close(fig)


# ---- Chart 3: Performance by Group ----

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 3a: Avg PnL by group
grps = ['BOTH', 'ORIG_ONLY', 'NEW_ONLY', 'NEITHER']
grp_labels = ['Both', 'Orig Only', 'New Only', 'Neither']
grp_colors = ['#4c72b0', '#c44e52', '#55a868', '#8c8c8c']

avgs = [all_group_metrics[g]['avg'] for g in grps]
ns = [all_group_metrics[g]['n'] for g in grps]

ax = axes[0]
bar_positions = range(len(grps))
bars = ax.bar(bar_positions, avgs, color=grp_colors, edgecolor='#333', linewidth=0.8)
for bar, avg_val, n_val in zip(bars, avgs, ns):
    label_text = f"{avg_val:+.1f}%\nn={n_val}"
    y_pos = bar.get_height() if bar.get_height() >= 0 else bar.get_height()
    va = 'bottom' if bar.get_height() >= 0 else 'top'
    ax.text(bar.get_x() + bar.get_width()/2, y_pos, label_text,
            ha='center', va=va, fontsize=9, fontweight='bold')
ax.set_xticks(bar_positions)
ax.set_xticklabels(grp_labels, fontsize=10)
ax.axhline(0, color='#333', linewidth=0.5)
ax.set_ylabel('Avg PnL %', fontsize=11)
ax.set_title('Avg PnL by Universe Group', fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 3b: Win Rate by group
wrs = [all_group_metrics[g]['wr'] * 100 for g in grps]
ax = axes[1]
bars = ax.bar(bar_positions, wrs, color=grp_colors, edgecolor='#333', linewidth=0.8)
for bar, wr_val in zip(bars, wrs):
    if wr_val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{wr_val:.0f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_xticks(bar_positions)
ax.set_xticklabels(grp_labels, fontsize=10)
ax.set_ylabel('Win Rate %', fontsize=11)
ax.set_title('Win Rate by Universe Group', fontsize=12)
ax.set_ylim(0, 100)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 3c: Monthly Avg PnL for Q3 (Jul-Sep)
ax = axes[2]
q3_months = [m for m in sorted(monthly.keys()) if m >= '2026-07']
if q3_months:
    x = np.arange(len(q3_months))
    width = 0.2
    for i, (grp, color, label) in enumerate(zip(
            ['BOTH', 'ORIG_ONLY', 'NEW_ONLY'],
            ['#4c72b0', '#c44e52', '#55a868'],
            ['Both', 'Orig Only', 'New Only'])):
        vals = []
        for month in q3_months:
            pnls = monthly[month].get(grp, [])
            vals.append(np.mean(pnls) if pnls else 0)
        offset = (i - 1) * width
        ax.bar(x + offset, vals, width, color=color, edgecolor='#333',
               linewidth=0.5, label=label, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(q3_months, fontsize=10)
    ax.axhline(0, color='#333', linewidth=0.5)
    ax.set_ylabel('Avg PnL %', fontsize=11)
    ax.set_title('Q3 Monthly Avg PnL by Group', fontsize=12)
    ax.legend(fontsize=9, loc='best')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
else:
    ax.text(0.5, 0.5, 'No Q3 data', ha='center', va='center', fontsize=14,
            transform=ax.transAxes)
    ax.set_title('Q3 Monthly Avg PnL by Group', fontsize=12)

fig.suptitle('E2 Trade Performance by Universe Membership', fontsize=14, y=1.02)
fig.tight_layout()
chart3_path = os.path.join(RESULTS_DIR, 'performance_by_group.png')
fig.savefig(chart3_path, dpi=150, bbox_inches='tight')
print(f"  Chart 3 saved: {chart3_path}")
plt.close(fig)


# =============================================================================
# STEP 7: SAVE RESULTS
# =============================================================================

print(f"\n\n{'='*80}")
print(f"  STEP 7: SAVING RESULTS")
print(f"{'='*80}")

# new_universe_candidates.json — the new float dict + stats
new_candidates_out = []
for s in ranked[:NEW_FLOAT_SIZE]:
    new_candidates_out.append({
        'ticker': s['ticker'],
        'composite_score': round(s['composite_score'], 4),
        'avg_gap_magnitude': round(s['avg_gap_magnitude'] * 100, 2),
        'gap_frequency': round(s['gap_frequency'], 4),
        'mean_reversion_rate': round(s['mean_reversion_rate'], 3),
        'n_gap_days': s['n_gap_days'],
        'float_shares': s['float_shares'],
        'in_original_float': s['in_orig_float'],
    })

candidates_path = os.path.join(RESULTS_DIR, 'new_universe_candidates.json')
with open(candidates_path, 'w') as f:
    json.dump(new_candidates_out, f, indent=2)
print(f"  Saved: {candidates_path}")

# universe_comparison.json — overlap analysis + performance metrics
comparison_out = {
    'config': {
        'backtest_start': BACKTEST_START,
        'backtest_end': BACKTEST_END,
        'n_trading_days': n_trading_days,
        'gap_scan_pct': GAP_SCAN_PCT,
        'new_float_size': NEW_FLOAT_SIZE,
        'composite_weights': {'gap_magnitude': 0.4, 'gap_frequency': 0.3, 'mean_reversion_rate': 0.3},
        'cutoff_score': round(cutoff_score, 4),
    },
    'universe_stats': {
        'n_orig': len(ORIG_FLOAT_SET),
        'n_new': len(NEW_FLOAT_SET),
        'n_overlap': len(overlap),
        'overlap_pct': round(len(overlap) / len(ORIG_FLOAT_SET) * 100, 1),
        'jaccard_similarity': round(jaccard, 3),
        'n_orig_only': len(orig_only),
        'n_new_only': len(new_only),
        'orig_only_tickers': sorted(orig_only),
        'new_only_tickers': sorted(new_only),
        'overlap_tickers': sorted(overlap),
    },
    'performance_by_group': {},
    'monthly_performance': {},
    'new_float_dict': {tk: v for tk, v in sorted(NEW_FLOAT.items())},
}

for grp in ['BOTH', 'ORIG_ONLY', 'NEW_ONLY', 'NEITHER', 'ALL']:
    m = all_group_metrics.get(grp, compute_metrics([], grp))
    if grp == 'ALL':
        m = all_m
    comparison_out['performance_by_group'][grp] = {
        'n': m['n'], 'avg': round(m['avg'], 3),
        'median': round(m['med'], 3), 'wr': round(m['wr'], 3),
        'sharpe': round(m['sharpe'], 3),
        'ci_lo': round(m['ci_lo'], 3), 'ci_hi': round(m['ci_hi'], 3),
    }

for month in sorted(monthly.keys()):
    comparison_out['monthly_performance'][month] = {}
    for grp in ['ALL', 'BOTH', 'ORIG_ONLY', 'NEW_ONLY', 'NEITHER']:
        pnls = monthly[month].get(grp, [])
        if pnls:
            m = compute_metrics(pnls, grp)
            comparison_out['monthly_performance'][month][grp] = {
                'n': m['n'], 'avg': round(m['avg'], 3),
                'wr': round(m['wr'], 3), 'sharpe': round(m['sharpe'], 3),
            }

comparison_path = os.path.join(RESULTS_DIR, 'universe_comparison.json')
with open(comparison_path, 'w') as f:
    json.dump(comparison_out, f, indent=2)
print(f"  Saved: {comparison_path}")


# =============================================================================
# STEP 8: SUMMARY
# =============================================================================

print(f"\n\n{'='*80}")
print(f"  UNIVERSE REFRESH — SUMMARY")
print(f"{'='*80}")

print(f"\n  Period:              {BACKTEST_START} -> {BACKTEST_END} ({n_trading_days} trading days)")
print(f"  Total CS tickers:    {len(cs_gap_tickers)} gapped >= {GAP_SCAN_PCT}%")
print(f"  Original _FLOAT:     {len(ORIG_FLOAT_SET)} tickers")
print(f"  NEW_FLOAT:           {len(NEW_FLOAT_SET)} tickers")
print(f"  Overlap:             {len(overlap)} ({len(overlap)/len(ORIG_FLOAT_SET)*100:.1f}%)")
print(f"  Jaccard similarity:  {jaccard:.3f}")
print(f"  New entrants:        {len(new_only)}")
print(f"  Dropped:             {len(orig_only)}")

print(f"\n  Performance by universe group:")
for grp in ['BOTH', 'ORIG_ONLY', 'NEW_ONLY', 'NEITHER']:
    m = all_group_metrics[grp]
    if m['n'] > 0:
        print(f"    {grp:>12s}:  n={m['n']:>3d}  avg={m['avg']:>+6.2f}%  "
              f"WR={m['wr']*100:>4.0f}%  Sharpe={m['sharpe']:>6.3f}")
    else:
        print(f"    {grp:>12s}:  n=  0")

n_new_with_trades = sum(1 for s in new_entrant_stats if s['ticker'] in e2_by_ticker)
print(f"\n  New entrants with E2 trades: {n_new_with_trades}/{len(new_only)}")

# Turnover rate
turnover = len(new_only) / NEW_FLOAT_SIZE
print(f"  Turnover rate: {turnover*100:.1f}% of universe changed")

if turnover > 0.3:
    print(f"\n  ** HIGH TURNOVER ({turnover*100:.0f}%) -- the gap universe has shifted")
    print(f"     significantly since the original _FLOAT was built.")
    print(f"     Consider updating _FLOAT with NEW_FLOAT for live trading.")
elif turnover > 0.15:
    print(f"\n  ** MODERATE TURNOVER ({turnover*100:.0f}%) -- some universe drift.")
    print(f"     Review new entrants for inclusion in _FLOAT.")
else:
    print(f"\n  ** LOW TURNOVER ({turnover*100:.0f}%) -- universe is stable.")
    print(f"     No urgent need to update _FLOAT.")

print(f"\n  Charts saved to: {RESULTS_DIR}/")
print(f"    - universe_overlap.png")
print(f"    - composite_score_dist.png")
print(f"    - performance_by_group.png")
print(f"  Data saved to: {RESULTS_DIR}/")
print(f"    - new_universe_candidates.json")
print(f"    - universe_comparison.json")

print(f"\n  NEXT STEPS:")
print(f"  1. Review the NEW_FLOAT dict in universe_comparison.json")
print(f"  2. If turnover > 30%, test NEW_FLOAT in a full E2 backtest")
print(f"  3. Consider hybrid: keep overlap tickers, add top new entrants")
print(f"  4. Update float_cache.json with any new float values")

print(f"\n{'='*80}")
print(f"  DONE")
print(f"{'='*80}\n")
