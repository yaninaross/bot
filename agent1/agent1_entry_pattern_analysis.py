# %% Agent 1 — Entry Pattern Analysis
#
# Run AFTER backtest_e2_reproduction.py in same Colab session to reuse cached data.
# If run standalone, pipeline executes automatically (~10 min).
#
# Analyzes how entry pattern characteristics (local_high vs fallback,
# entry bar, pre-entry drop, local high magnitude) changed across
# Q1/Q2/Q3 2026, and sweeps local high window sizes on Q3 data.

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


# ===============================================================================
# CONFIG — E2 BENCHMARK (DO NOT CHANGE)
# ===============================================================================

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


# ===============================================================================
# HELPERS
# ===============================================================================

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


# ===============================================================================
# DATA PIPELINE (skips if already loaded from prior cell)
# ===============================================================================

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
    print(f"  ENTRY PATTERN ANALYSIS -- Data Pipeline")
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


# ===============================================================================
# BUILD E2 BASELINE TRADES (enriched with entry pattern data)
# ===============================================================================

def build_e2_trades_enriched():
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
            tc['peak_bar'] = entry.get('peak_bar')
            ep = tc['entry_price']

            rth_bars = bar_data.get('rth', [])
            ah_bars = bar_data.get('ah', [])
            session_bars = rth_bars + ah_bars

            # Final PnL at 17:00
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

            # Entry bar index (bar 0 = 09:30)
            entry_time_str = tc['entry_time']
            entry_bar_idx = None
            for bi, (t_str, _) in enumerate(rth_bars):
                if t_str == entry_time_str:
                    entry_bar_idx = bi
                    break
            tc['entry_bar_index'] = entry_bar_idx

            # Entry delay in minutes from 09:30
            if entry_bar_idx is not None:
                tc['entry_delay_min'] = entry_bar_idx
            else:
                tc['entry_delay_min'] = None

            # Local high magnitude: how high above open the bounce went
            open_price = rth_bars[0][1]['o'] if rth_bars else None
            if tc['entry_type'] == 'local_high' and tc['peak_bar'] is not None and open_price and open_price > 0:
                peak_idx = tc['peak_bar']
                if peak_idx < len(rth_bars):
                    lh_price = rth_bars[peak_idx][1]['h']
                    tc['lh_magnitude'] = round((lh_price - open_price) / open_price * 100, 3)
                else:
                    tc['lh_magnitude'] = None
            else:
                tc['lh_magnitude'] = None

            # Pre-entry drop: (open - entry_price) / open * 100
            if open_price and open_price > 0:
                tc['pre_entry_drop'] = round((open_price - ep) / open_price * 100, 3)
            else:
                tc['pre_entry_drop'] = None

            trades.append(tc)

    return trades

print(f"\n{'='*80}")
print(f"  Building E2 baseline trades with entry pattern enrichment...")
print(f"{'='*80}")

e2_trades = build_e2_trades_enriched()
raw_pnls = [t['pnl_eod'] for t in e2_trades]
base_m = compute_metrics(raw_pnls, 'E2_BASE')

print(f"  {len(e2_trades)} trades, avg {base_m['avg']:+.2f}%, "
      f"Sharpe {base_m['sharpe']:.3f}, WR {base_m['wr']*100:.0f}%")


# ===============================================================================
# PERIOD DEFINITIONS
# ===============================================================================

PERIODS = {
    'Q1': ('2026-01-01', '2026-03-31'),
    'Q2': ('2026-04-01', '2026-06-30'),
    'Q3': ('2026-07-01', '2026-09-30'),
}

MONTHS = [
    ('Jan', '2026-01'), ('Feb', '2026-02'), ('Mar', '2026-03'),
    ('Apr', '2026-04'), ('May', '2026-05'), ('Jun', '2026-06'),
    ('Jul', '2026-07'), ('Aug', '2026-08'), ('Sep', '2026-09'),
]

def get_period(dt_str):
    for name, (start, end) in PERIODS.items():
        if start <= dt_str <= end:
            return name
    return None

def get_month(dt_str):
    return dt_str[:7]  # '2026-01'


# ===============================================================================
# SECTION 1: PERIOD COMPARISON (Q1 vs Q2 vs Q3)
# ===============================================================================

print(f"\n\n{'='*80}")
print(f"  SECTION 1: PERIOD COMPARISON (Q1 vs Q2 vs Q3)")
print(f"{'='*80}")

period_trades = defaultdict(list)
for t in e2_trades:
    p = get_period(t['scan_date'])
    if p:
        period_trades[p].append(t)

# Period summary table
print(f"\n  --- PERIOD SUMMARY ---\n")
print(f"  {'Period':>6s}  {'n':>3s}  {'LH':>3s}  {'FB':>3s}  {'FB%':>5s}  "
      f"{'Avg Bar':>7s}  {'Med Bar':>7s}  {'LH Mag':>7s}  {'Drop%':>6s}  "
      f"{'LH PnL':>7s}  {'FB PnL':>7s}  {'LH Shrp':>7s}  {'FB Shrp':>7s}  "
      f"{'Avg PnL':>7s}  {'Sharpe':>7s}")
print(f"  {'-'*6}  {'-'*3}  {'-'*3}  {'-'*3}  {'-'*5}  "
      f"{'-'*7}  {'-'*7}  {'-'*7}  {'-'*6}  "
      f"{'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  "
      f"{'-'*7}  {'-'*7}")

period_summary = {}
for period_name in ['Q1', 'Q2', 'Q3']:
    trades = period_trades.get(period_name, [])
    n = len(trades)
    if n == 0:
        period_summary[period_name] = {'n': 0}
        continue

    lh_trades = [t for t in trades if t['entry_type'] == 'local_high']
    fb_trades = [t for t in trades if t['entry_type'] == 'fallback']
    n_lh = len(lh_trades)
    n_fb = len(fb_trades)
    fb_rate = n_fb / n * 100 if n > 0 else 0

    bars = [t['entry_bar_index'] for t in trades if t['entry_bar_index'] is not None]
    avg_bar = np.mean(bars) if bars else 0
    med_bar = np.median(bars) if bars else 0

    lh_mags = [t['lh_magnitude'] for t in lh_trades if t['lh_magnitude'] is not None]
    avg_lh_mag = np.mean(lh_mags) if lh_mags else 0

    drops = [t['pre_entry_drop'] for t in trades if t['pre_entry_drop'] is not None]
    avg_drop = np.mean(drops) if drops else 0

    lh_pnls = [t['pnl_eod'] for t in lh_trades]
    fb_pnls = [t['pnl_eod'] for t in fb_trades]
    all_pnls = [t['pnl_eod'] for t in trades]

    lh_m = compute_metrics(lh_pnls, f'{period_name}_LH')
    fb_m = compute_metrics(fb_pnls, f'{period_name}_FB')
    all_m = compute_metrics(all_pnls, f'{period_name}_ALL')

    period_summary[period_name] = {
        'n': n, 'n_lh': n_lh, 'n_fb': n_fb, 'fb_rate': fb_rate,
        'avg_bar': avg_bar, 'med_bar': med_bar,
        'avg_lh_mag': avg_lh_mag, 'avg_drop': avg_drop,
        'lh_avg': lh_m['avg'], 'fb_avg': fb_m['avg'],
        'lh_sharpe': lh_m['sharpe'], 'fb_sharpe': fb_m['sharpe'],
        'avg_pnl': all_m['avg'], 'sharpe': all_m['sharpe'],
    }

    lh_avg_s = f"{lh_m['avg']:+.2f}" if n_lh > 0 else '--'
    fb_avg_s = f"{fb_m['avg']:+.2f}" if n_fb > 0 else '--'
    lh_sh_s = f"{lh_m['sharpe']:.3f}" if n_lh > 0 else '--'
    fb_sh_s = f"{fb_m['sharpe']:.3f}" if n_fb > 0 else '--'

    print(f"  {period_name:>6s}  {n:>3d}  {n_lh:>3d}  {n_fb:>3d}  {fb_rate:>4.0f}%  "
          f"{avg_bar:>7.1f}  {med_bar:>7.1f}  {avg_lh_mag:>+6.2f}%  {avg_drop:>+5.2f}%  "
          f"{lh_avg_s:>7s}  {fb_avg_s:>7s}  {lh_sh_s:>7s}  {fb_sh_s:>7s}  "
          f"{all_m['avg']:>+6.2f}%  {all_m['sharpe']:>7.3f}")

print(f"\n  INTERPRETATION:")
print(f"  LH       = local_high entry (peak detected in bars 3-11)")
print(f"  FB       = fallback entry at bar 15 (09:45)")
print(f"  LH Mag   = how far above open the local high bounce went (% of open)")
print(f"  Drop%    = how much stock dropped from open to entry (% of open)")
print(f"  LH/FB PnL = avg PnL for each entry type separately")


# ===============================================================================
# SECTION 2: ENTRY BAR ANALYSIS
# ===============================================================================

print(f"\n\n{'='*80}")
print(f"  SECTION 2: ENTRY BAR ANALYSIS")
print(f"  PnL by entry bar index (bar 0 = 09:30)")
print(f"{'='*80}")

# Group trades by entry bar
bar_trades = defaultdict(list)
for t in e2_trades:
    bi = t.get('entry_bar_index')
    if bi is not None:
        bar_trades[bi].append(t)

# Overall PnL by entry bar
print(f"\n  --- OVERALL: PnL BY ENTRY BAR ---\n")
print(f"  {'Bar':>4s}  {'Time':>5s}  {'n':>3s}  {'Avg PnL':>8s}  {'Med PnL':>8s}  "
      f"{'WR':>5s}  {'Sharpe':>7s}  {'Type':>12s}")
print(f"  {'-'*4}  {'-'*5}  {'-'*3}  {'-'*8}  {'-'*8}  "
      f"{'-'*5}  {'-'*7}  {'-'*12}")

bar_results = {}
for bi in sorted(bar_trades.keys()):
    trades = bar_trades[bi]
    pnls = [t['pnl_eod'] for t in trades]
    m = compute_metrics(pnls, f'bar_{bi}')
    # Time string: bar 0=09:30, bar 1=09:31, etc.
    bar_time = f"09:{30+bi:02d}" if bi < 30 else f"10:{bi-30:02d}"
    # Predominant entry type
    n_lh = sum(1 for t in trades if t['entry_type'] == 'local_high')
    n_fb = sum(1 for t in trades if t['entry_type'] == 'fallback')
    etype = 'local_high' if n_lh >= n_fb else 'fallback'

    bar_results[bi] = m
    print(f"  {bi:>4d}  {bar_time:>5s}  {m['n']:>3d}  {m['avg']:>+7.2f}%  "
          f"{m['med']:>+7.2f}%  {m['wr']*100:>4.0f}%  {m['sharpe']:>7.3f}  {etype:>12s}")

# By-period PnL per entry bar
print(f"\n\n  --- PnL BY ENTRY BAR, PER PERIOD ---\n")
print(f"  {'Bar':>4s}  {'Time':>5s}  "
      f"{'Q1 n':>4s}  {'Q1 Avg':>7s}  "
      f"{'Q2 n':>4s}  {'Q2 Avg':>7s}  "
      f"{'Q3 n':>4s}  {'Q3 Avg':>7s}  "
      f"{'Shift?':>6s}")
print(f"  {'-'*4}  {'-'*5}  "
      f"{'-'*4}  {'-'*7}  "
      f"{'-'*4}  {'-'*7}  "
      f"{'-'*4}  {'-'*7}  "
      f"{'-'*6}")

all_bar_indices = sorted(bar_trades.keys())
period_bar_data = {}  # {period: {bar: metrics}}
for period_name in ['Q1', 'Q2', 'Q3']:
    period_bar_data[period_name] = {}
    trades = period_trades.get(period_name, [])
    p_bar_trades = defaultdict(list)
    for t in trades:
        bi = t.get('entry_bar_index')
        if bi is not None:
            p_bar_trades[bi].append(t)
    for bi in all_bar_indices:
        pnls = [t['pnl_eod'] for t in p_bar_trades.get(bi, [])]
        period_bar_data[period_name][bi] = compute_metrics(pnls)

for bi in all_bar_indices:
    bar_time = f"09:{30+bi:02d}" if bi < 30 else f"10:{bi-30:02d}"

    q1_m = period_bar_data['Q1'].get(bi, {'n': 0, 'avg': 0})
    q2_m = period_bar_data['Q2'].get(bi, {'n': 0, 'avg': 0})
    q3_m = period_bar_data['Q3'].get(bi, {'n': 0, 'avg': 0})

    q1_s = f"{q1_m['avg']:+.2f}" if q1_m['n'] > 0 else '  --'
    q2_s = f"{q2_m['avg']:+.2f}" if q2_m['n'] > 0 else '  --'
    q3_s = f"{q3_m['avg']:+.2f}" if q3_m['n'] > 0 else '  --'

    # Flag bars where Q3 avg is meaningfully worse than Q1+Q2
    shift = ''
    q12_pnls = []
    for pn in ['Q1', 'Q2']:
        for t in period_trades.get(pn, []):
            if t.get('entry_bar_index') == bi:
                q12_pnls.append(t['pnl_eod'])
    q3_pnls = [t['pnl_eod'] for t in period_trades.get('Q3', []) if t.get('entry_bar_index') == bi]
    if q12_pnls and q3_pnls:
        q12_avg = np.mean(q12_pnls)
        q3_avg = np.mean(q3_pnls)
        if q3_avg < q12_avg - 5:
            shift = '  <<'  # meaningful decline
        elif q3_avg > q12_avg + 5:
            shift = '  >>'  # meaningful improvement

    print(f"  {bi:>4d}  {bar_time:>5s}  "
          f"{q1_m['n']:>4d}  {q1_s:>7s}  "
          f"{q2_m['n']:>4d}  {q2_s:>7s}  "
          f"{q3_m['n']:>4d}  {q3_s:>7s}  "
          f"{shift}")

# Find optimal bar per period
print(f"\n  --- OPTIMAL ENTRY BAR BY PERIOD ---\n")
for period_name in ['Q1', 'Q2', 'Q3', 'Q1+Q2']:
    if period_name == 'Q1+Q2':
        ptrades = period_trades.get('Q1', []) + period_trades.get('Q2', [])
    else:
        ptrades = period_trades.get(period_name, [])

    p_bar = defaultdict(list)
    for t in ptrades:
        bi = t.get('entry_bar_index')
        if bi is not None:
            p_bar[bi].append(t['pnl_eod'])

    if not p_bar:
        print(f"  {period_name}: no trades")
        continue

    # Best bar = highest avg PnL with n >= 3
    viable = {bi: np.mean(pnls) for bi, pnls in p_bar.items() if len(pnls) >= 3}
    if viable:
        best_bi = max(viable, key=viable.get)
        bar_time = f"09:{30+best_bi:02d}" if best_bi < 30 else f"10:{best_bi-30:02d}"
        print(f"  {period_name:>5s}: best bar = {best_bi} ({bar_time}), "
              f"avg {viable[best_bi]:+.2f}%, n={len(p_bar[best_bi])}")
    else:
        print(f"  {period_name:>5s}: no bars with n >= 3")


# ===============================================================================
# SECTION 3: WINDOW SWEEP (Q3 data only)
# ===============================================================================

print(f"\n\n{'='*80}")
print(f"  SECTION 3: LOCAL HIGH WINDOW SWEEP (Q3 only)")
print(f"  Re-run entry detection with different peak window sizes")
print(f"  on Jul-Sep 2026 data to find if a different window helps Q3.")
print(f"{'='*80}")

WINDOW_SIZES = [2, 3, 4, 5]

# Also test fallback-only (skip local high entirely, always bar 15)
# Collect Q3 dates
q3_dates = [dt for dt in dates if '2026-07-01' <= dt <= '2026-09-30']

def run_window_sweep_q3(window_val, use_fallback_only=False):
    """Re-run entry detection + PnL for Q3 trades with a given window size."""
    trades_out = []
    for dt in q3_dates:
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
            cs_qualified.append(cc)

        for c in cs_qualified[:TOP_N]:
            tk, dt_s = c['ticker'], c['scan_date']
            bar_data = all_bars.get((tk, dt_s))
            if bar_data is None:
                continue
            rth_bars = bar_data.get('rth', [])
            if not rth_bars:
                continue

            # Re-run entry detection with this window
            if use_fallback_only:
                # Always use fallback bar 15
                if FALLBACK_BAR < len(rth_bars):
                    entry = {
                        'entry_price': rth_bars[FALLBACK_BAR][1]['c'],
                        'entry_time': rth_bars[FALLBACK_BAR][0],
                        'entry_type': 'fallback',
                        'peak_bar': None,
                    }
                else:
                    continue
            else:
                entry = find_local_high(rth_bars, window=window_val,
                                        max_peak_bar=MAX_PEAK_BAR,
                                        fallback_bar=FALLBACK_BAR)
            if entry is None:
                continue

            ep = entry['entry_price']
            ah_bars = bar_data.get('ah', [])
            session_bars = rth_bars + ah_bars

            exit_price = None
            for t_str, bar in session_bars:
                if t_str > EXIT_TIME:
                    break
                if t_str >= entry['entry_time']:
                    exit_price = bar['c']
            if exit_price is None:
                daily = _gd_cache.get(dt_s, {}).get(tk, {})
                if daily.get('c'):
                    exit_price = daily['c']
            if exit_price is None:
                continue

            pnl = round((ep - exit_price) / ep * 100, 3)
            trades_out.append({
                'ticker': tk, 'scan_date': dt_s,
                'entry_type': entry['entry_type'],
                'entry_time': entry['entry_time'],
                'entry_price': ep,
                'peak_bar': entry.get('peak_bar'),
                'pnl_eod': pnl,
            })

    return trades_out


print(f"\n  {'Window':>8s}  {'n':>3s}  {'LH':>3s}  {'FB':>3s}  {'FB%':>5s}  "
      f"{'Avg PnL':>8s}  {'Med PnL':>8s}  {'WR':>5s}  {'Sharpe':>7s}  "
      f"{'LH Avg':>7s}  {'FB Avg':>7s}")
print(f"  {'-'*8}  {'-'*3}  {'-'*3}  {'-'*3}  {'-'*5}  "
      f"{'-'*8}  {'-'*8}  {'-'*5}  {'-'*7}  "
      f"{'-'*7}  {'-'*7}")

window_results = []

for w in WINDOW_SIZES:
    trades = run_window_sweep_q3(w)
    pnls = [t['pnl_eod'] for t in trades]
    m = compute_metrics(pnls, f'w={w}')

    n_lh = sum(1 for t in trades if t['entry_type'] == 'local_high')
    n_fb = sum(1 for t in trades if t['entry_type'] == 'fallback')
    fb_rate = n_fb / len(trades) * 100 if trades else 0

    lh_pnls = [t['pnl_eod'] for t in trades if t['entry_type'] == 'local_high']
    fb_pnls = [t['pnl_eod'] for t in trades if t['entry_type'] == 'fallback']
    lh_m = compute_metrics(lh_pnls)
    fb_m = compute_metrics(fb_pnls)

    lh_s = f"{lh_m['avg']:+.2f}" if lh_pnls else '  --'
    fb_s = f"{fb_m['avg']:+.2f}" if fb_pnls else '  --'

    mark = ' <-- current' if w == PEAK_WINDOW else ''

    window_results.append({
        'window': w, 'n': m['n'], 'n_lh': n_lh, 'n_fb': n_fb,
        'fb_rate': round(fb_rate, 1),
        'avg_pnl': round(m['avg'], 3), 'med_pnl': round(m['med'], 3),
        'wr': round(m['wr'], 3), 'sharpe': round(m['sharpe'], 3),
        'lh_avg': round(lh_m['avg'], 3) if lh_pnls else None,
        'fb_avg': round(fb_m['avg'], 3) if fb_pnls else None,
    })

    print(f"  {'w='+str(w):>8s}  {m['n']:>3d}  {n_lh:>3d}  {n_fb:>3d}  {fb_rate:>4.0f}%  "
          f"{m['avg']:>+7.2f}%  {m['med']:>+7.2f}%  {m['wr']*100:>4.0f}%  {m['sharpe']:>7.3f}  "
          f"{lh_s:>7s}  {fb_s:>7s}{mark}")

# Fallback-only
fb_only_trades = run_window_sweep_q3(0, use_fallback_only=True)
fb_only_pnls = [t['pnl_eod'] for t in fb_only_trades]
fb_only_m = compute_metrics(fb_only_pnls, 'FB_ONLY')
n_fb_only = len(fb_only_trades)

window_results.append({
    'window': 'FB_ONLY', 'n': fb_only_m['n'], 'n_lh': 0, 'n_fb': n_fb_only,
    'fb_rate': 100.0,
    'avg_pnl': round(fb_only_m['avg'], 3), 'med_pnl': round(fb_only_m['med'], 3),
    'wr': round(fb_only_m['wr'], 3), 'sharpe': round(fb_only_m['sharpe'], 3),
    'lh_avg': None,
    'fb_avg': round(fb_only_m['avg'], 3) if fb_only_pnls else None,
})

print(f"  {'FB_ONLY':>8s}  {fb_only_m['n']:>3d}  {0:>3d}  {n_fb_only:>3d}  {100:>4.0f}%  "
      f"{fb_only_m['avg']:>+7.2f}%  {fb_only_m['med']:>+7.2f}%  {fb_only_m['wr']*100:>4.0f}%  "
      f"{fb_only_m['sharpe']:>7.3f}  "
      f"{'--':>7s}  {fb_only_m['avg']:>+6.2f}%")

# Find best window
numeric_windows = [r for r in window_results if isinstance(r['window'], int)]
if numeric_windows:
    best_w = max(numeric_windows, key=lambda r: r['avg_pnl'])
    print(f"\n  BEST WINDOW: w={best_w['window']} -> avg {best_w['avg_pnl']:+.2f}%, "
          f"Sharpe {best_w['sharpe']:.3f}")
    curr = next((r for r in numeric_windows if r['window'] == PEAK_WINDOW), None)
    if curr and best_w['window'] != PEAK_WINDOW:
        delta = best_w['avg_pnl'] - curr['avg_pnl']
        print(f"  vs current w={PEAK_WINDOW}: delta = {delta:+.2f}% avg PnL")


# ===============================================================================
# SECTION 4: MONTHLY FALLBACK RATE
# ===============================================================================

print(f"\n\n{'='*80}")
print(f"  SECTION 4: MONTHLY BREAKDOWN")
print(f"{'='*80}")

print(f"\n  {'Month':>5s}  {'n':>3s}  {'LH':>3s}  {'FB':>3s}  {'FB%':>5s}  "
      f"{'Avg PnL':>8s}  {'Sharpe':>7s}  {'LH Avg':>7s}  {'FB Avg':>7s}")
print(f"  {'-'*5}  {'-'*3}  {'-'*3}  {'-'*3}  {'-'*5}  "
      f"{'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}")

monthly_data = {}
for label, ym in MONTHS:
    mtrades = [t for t in e2_trades if t['scan_date'].startswith(ym)]
    n = len(mtrades)
    if n == 0:
        monthly_data[label] = {'n': 0, 'fb_rate': 0}
        print(f"  {label:>5s}  {0:>3d}  {0:>3d}  {0:>3d}  {'--':>5s}  "
              f"{'--':>8s}  {'--':>7s}  {'--':>7s}  {'--':>7s}")
        continue

    n_lh = sum(1 for t in mtrades if t['entry_type'] == 'local_high')
    n_fb = sum(1 for t in mtrades if t['entry_type'] == 'fallback')
    fb_rate = n_fb / n * 100

    pnls = [t['pnl_eod'] for t in mtrades]
    m = compute_metrics(pnls)

    lh_pnls = [t['pnl_eod'] for t in mtrades if t['entry_type'] == 'local_high']
    fb_pnls = [t['pnl_eod'] for t in mtrades if t['entry_type'] == 'fallback']
    lh_m = compute_metrics(lh_pnls)
    fb_m = compute_metrics(fb_pnls)

    lh_s = f"{lh_m['avg']:+.2f}" if lh_pnls else '  --'
    fb_s = f"{fb_m['avg']:+.2f}" if fb_pnls else '  --'

    monthly_data[label] = {
        'n': n, 'n_lh': n_lh, 'n_fb': n_fb, 'fb_rate': round(fb_rate, 1),
        'avg_pnl': round(m['avg'], 2), 'sharpe': round(m['sharpe'], 3),
    }

    print(f"  {label:>5s}  {n:>3d}  {n_lh:>3d}  {n_fb:>3d}  {fb_rate:>4.0f}%  "
          f"{m['avg']:>+7.2f}%  {m['sharpe']:>7.3f}  {lh_s:>7s}  {fb_s:>7s}")


# ===============================================================================
# CHARTS
# ===============================================================================

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

chart_colors = {
    'Q1': '#2980b9',  # blue
    'Q2': '#27ae60',  # green
    'Q3': '#e74c3c',  # red
}


# --- Chart 1: Entry bar distribution by period (stacked histogram) ----------

fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle('Entry Bar Distribution by Period (Q1 / Q2 / Q3)',
             fontsize=12, fontweight='bold')

# Collect entry bar indices per period
period_bars_data = {}
for pname in ['Q1', 'Q2', 'Q3']:
    period_bars_data[pname] = [t['entry_bar_index'] for t in period_trades.get(pname, [])
                                if t['entry_bar_index'] is not None]

# Find range of bar indices
all_bi = []
for v in period_bars_data.values():
    all_bi.extend(v)
if all_bi:
    bi_min, bi_max = min(all_bi), max(all_bi)
    bins = np.arange(bi_min - 0.5, bi_max + 1.5, 1)

    ax.hist([period_bars_data['Q1'], period_bars_data['Q2'], period_bars_data['Q3']],
            bins=bins, stacked=True,
            color=[chart_colors['Q1'], chart_colors['Q2'], chart_colors['Q3']],
            alpha=0.85, edgecolor='black', linewidth=0.3,
            label=[f"Q1 (n={len(period_bars_data['Q1'])})",
                   f"Q2 (n={len(period_bars_data['Q2'])})",
                   f"Q3 (n={len(period_bars_data['Q3'])})"])

ax.set_xlabel('Entry Bar Index (bar 0 = 09:30 ET)')
ax.set_ylabel('Number of Trades')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Add time labels on top for key bars
for bi_label in [3, 6, 9, 12, 15]:
    t_str = f"09:{30+bi_label:02d}" if bi_label < 30 else f"10:{bi_label-30:02d}"
    ax.annotate(t_str, xy=(bi_label, 0), xytext=(bi_label, ax.get_ylim()[1] * 0.95),
                ha='center', fontsize=7, color='#555555', style='italic')

plt.tight_layout(rect=[0, 0, 1, 0.94])
p1 = os.path.join(BASE_DIR, 'entry_bar_distribution.png')
plt.savefig(p1, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved: {p1}")


# --- Chart 2: Avg PnL by entry bar (line chart, overall + per period) -------

fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle('Average PnL by Entry Bar',
             fontsize=12, fontweight='bold')

# Overall line
overall_bars_sorted = sorted(bar_results.keys())
# Only show bars with at least 2 trades
viable_bars = [bi for bi in overall_bars_sorted if bar_results[bi]['n'] >= 2]
if viable_bars:
    ax.plot(viable_bars,
            [bar_results[bi]['avg'] for bi in viable_bars],
            'o-', color='#2c3e50', linewidth=2.5, markersize=8,
            label=f"Overall (n={sum(bar_results[bi]['n'] for bi in viable_bars)})",
            zorder=5)

# Per-period lines
for pname in ['Q1', 'Q2', 'Q3']:
    pbd = period_bar_data[pname]
    viable_p = sorted([bi for bi in pbd if pbd[bi]['n'] >= 2])
    if viable_p:
        ax.plot(viable_p,
                [pbd[bi]['avg'] for bi in viable_p],
                's--', color=chart_colors[pname], linewidth=1.5, markersize=5,
                alpha=0.7,
                label=f"{pname} (n={sum(pbd[bi]['n'] for bi in viable_p)})")

ax.axhline(0, color='black', lw=1, alpha=0.4)
ax.set_xlabel('Entry Bar Index (bar 0 = 09:30 ET)')
ax.set_ylabel('Average PnL %')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Add time labels
ax2 = ax.twiny()
tick_positions = [3, 6, 9, 12, 15]
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(tick_positions)
ax2.set_xticklabels([f"09:{30+b:02d}" for b in tick_positions], fontsize=8, color='#666666')
ax2.tick_params(axis='x', length=0)

plt.tight_layout(rect=[0, 0, 1, 0.92])
p2 = os.path.join(BASE_DIR, 'entry_bar_pnl.png')
plt.savefig(p2, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved: {p2}")


# --- Chart 3: Fallback rate by month (bar chart) ----------------------------

fig, ax1 = plt.subplots(figsize=(12, 5))
fig.suptitle('Fallback Rate and Average PnL by Month',
             fontsize=12, fontweight='bold')

month_labels = [label for label, _ in MONTHS]
fb_rates = []
avg_pnls_monthly = []
for label, ym in MONTHS:
    mtrades = [t for t in e2_trades if t['scan_date'].startswith(ym)]
    if mtrades:
        n_fb = sum(1 for t in mtrades if t['entry_type'] == 'fallback')
        fb_rates.append(n_fb / len(mtrades) * 100)
        avg_pnls_monthly.append(np.mean([t['pnl_eod'] for t in mtrades]))
    else:
        fb_rates.append(0)
        avg_pnls_monthly.append(0)

x = np.arange(len(month_labels))
width = 0.4

bars1 = ax1.bar(x, fb_rates, width, color='#e67e22', alpha=0.8, edgecolor='black',
                linewidth=0.3, label='Fallback Rate %')
ax1.set_xlabel('Month')
ax1.set_ylabel('Fallback Rate (%)', color='#e67e22')
ax1.set_xticks(x)
ax1.set_xticklabels(month_labels)
ax1.tick_params(axis='y', labelcolor='#e67e22')

# Overlay avg PnL on secondary axis
ax2 = ax1.twinx()
ax2.plot(x, avg_pnls_monthly, 'D-', color='#2c3e50', linewidth=2, markersize=6,
         label='Avg PnL %')
ax2.axhline(0, color='black', lw=0.8, alpha=0.3)
ax2.set_ylabel('Average PnL %', color='#2c3e50')
ax2.tick_params(axis='y', labelcolor='#2c3e50')

# Add count labels on bars
for i, (fb, n_label) in enumerate(zip(fb_rates, month_labels)):
    mtrades = [t for t in e2_trades if t['scan_date'].startswith(MONTHS[i][1])]
    if mtrades:
        ax1.text(i, fb + 1.5, f"n={len(mtrades)}", ha='center', fontsize=7, color='#555')

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

ax1.grid(True, alpha=0.2, axis='y')
plt.tight_layout(rect=[0, 0, 1, 0.94])
p3 = os.path.join(BASE_DIR, 'fallback_rate_monthly.png')
plt.savefig(p3, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved: {p3}")


# --- Chart 4: Window sweep results on Q3 (bar chart) ------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Local High Window Sweep -- Q3 (Jul-Sep 2026)',
             fontsize=12, fontweight='bold')

# Left: Avg PnL by window
ax = axes[0]
w_labels = [str(r['window']) for r in window_results]
w_pnls = [r['avg_pnl'] for r in window_results]
colors = []
for r in window_results:
    if r['window'] == PEAK_WINDOW:
        colors.append('#2980b9')  # current = blue
    elif isinstance(r['window'], str):
        colors.append('#95a5a6')  # fallback only = grey
    else:
        colors.append('#27ae60')  # other = green

bars = ax.bar(w_labels, w_pnls, color=colors, alpha=0.85,
              edgecolor='black', linewidth=0.3)
ax.axhline(0, color='black', lw=1, alpha=0.4)
ax.set_xlabel('Window Size')
ax.set_ylabel('Average PnL %')
ax.set_title('Avg PnL by Window Size')
ax.grid(True, alpha=0.3, axis='y')

# Label bars
for bar, pnl in zip(bars, w_pnls):
    y_pos = bar.get_height() + 0.3 if bar.get_height() >= 0 else bar.get_height() - 0.8
    ax.text(bar.get_x() + bar.get_width()/2, y_pos, f"{pnl:+.1f}%",
            ha='center', fontsize=9, fontweight='bold')

# Add annotation for current
for i, r in enumerate(window_results):
    if r['window'] == PEAK_WINDOW:
        ax.annotate('CURRENT', xy=(i, w_pnls[i]),
                    xytext=(i, w_pnls[i] + max(abs(p) for p in w_pnls) * 0.15),
                    ha='center', fontsize=8, color='#2980b9', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#2980b9', lw=1.5))

# Right: Sharpe by window
ax = axes[1]
w_sharpes = [r['sharpe'] for r in window_results]
bars = ax.bar(w_labels, w_sharpes, color=colors, alpha=0.85,
              edgecolor='black', linewidth=0.3)
ax.axhline(0, color='black', lw=1, alpha=0.4)
ax.set_xlabel('Window Size')
ax.set_ylabel('Sharpe Ratio')
ax.set_title('Sharpe by Window Size')
ax.grid(True, alpha=0.3, axis='y')

for bar, sh in zip(bars, w_sharpes):
    y_pos = bar.get_height() + 0.02 if bar.get_height() >= 0 else bar.get_height() - 0.04
    ax.text(bar.get_x() + bar.get_width()/2, y_pos, f"{sh:.3f}",
            ha='center', fontsize=9, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.92])
p4 = os.path.join(BASE_DIR, 'window_sweep_q3.png')
plt.savefig(p4, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved: {p4}")


# ===============================================================================
# SAVE RESULTS
# ===============================================================================

results_dir = os.path.join(BASE_DIR, 'results')
os.makedirs(results_dir, exist_ok=True)

output = {
    'run_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
    'period': f'{BACKTEST_START} to {BACKTEST_END}',
    'n_trades': len(e2_trades),
    'baseline': {
        'avg': round(base_m['avg'], 3),
        'sharpe': round(base_m['sharpe'], 3),
        'wr': round(base_m['wr'], 3),
    },
    'period_comparison': {
        pname: {
            k: round(v, 3) if isinstance(v, float) else v
            for k, v in pdata.items()
        }
        for pname, pdata in period_summary.items()
    },
    'entry_bar_analysis': {
        str(bi): {
            'n': m['n'], 'avg': round(m['avg'], 3),
            'med': round(m['med'], 3), 'wr': round(m['wr'], 3),
            'sharpe': round(m['sharpe'], 3),
        }
        for bi, m in bar_results.items()
    },
    'entry_bar_by_period': {
        pname: {
            str(bi): {'n': pbd[bi]['n'], 'avg': round(pbd[bi]['avg'], 3)}
            for bi in sorted(pbd.keys()) if pbd[bi]['n'] > 0
        }
        for pname, pbd in period_bar_data.items()
    },
    'window_sweep_q3': window_results,
    'monthly': monthly_data,
}

out_path = os.path.join(results_dir, 'entry_pattern_analysis.json')
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\n  Saved: {out_path}")


# ===============================================================================
# BOTTOM LINE
# ===============================================================================

print(f"\n\n{'='*80}")
print(f"  ENTRY PATTERN ANALYSIS -- BOTTOM LINE")
print(f"{'='*80}")

print(f"\n  TRADES: {len(e2_trades)} total, avg {base_m['avg']:+.2f}%, "
      f"Sharpe {base_m['sharpe']:.3f}")

for pname in ['Q1', 'Q2', 'Q3']:
    ps = period_summary.get(pname, {})
    if ps.get('n', 0) > 0:
        print(f"  {pname}: n={ps['n']}, avg {ps['avg_pnl']:+.2f}%, "
              f"Sharpe {ps['sharpe']:.3f}, FB rate {ps['fb_rate']:.0f}%")

n_lh_all = sum(1 for t in e2_trades if t['entry_type'] == 'local_high')
n_fb_all = sum(1 for t in e2_trades if t['entry_type'] == 'fallback')
lh_all_m = compute_metrics([t['pnl_eod'] for t in e2_trades if t['entry_type'] == 'local_high'])
fb_all_m = compute_metrics([t['pnl_eod'] for t in e2_trades if t['entry_type'] == 'fallback'])

print(f"\n  ENTRY TYPE SPLIT:")
print(f"  local_high: n={n_lh_all}, avg {lh_all_m['avg']:+.2f}%, Sharpe {lh_all_m['sharpe']:.3f}")
print(f"  fallback:   n={n_fb_all}, avg {fb_all_m['avg']:+.2f}%, Sharpe {fb_all_m['sharpe']:.3f}")

if numeric_windows:
    best_w = max(numeric_windows, key=lambda r: r['avg_pnl'])
    curr_w = next((r for r in numeric_windows if r['window'] == PEAK_WINDOW), None)
    print(f"\n  WINDOW SWEEP (Q3 only):")
    if curr_w:
        print(f"  Current w={PEAK_WINDOW}: avg {curr_w['avg_pnl']:+.2f}%, "
              f"Sharpe {curr_w['sharpe']:.3f}")
    print(f"  Best    w={best_w['window']}: avg {best_w['avg_pnl']:+.2f}%, "
          f"Sharpe {best_w['sharpe']:.3f}")
    print(f"  FB_ONLY:       avg {fb_only_m['avg']:+.2f}%, "
          f"Sharpe {fb_only_m['sharpe']:.3f}")

print(f"\n  Charts saved:")
print(f"    {p1}")
print(f"    {p2}")
print(f"    {p3}")
print(f"    {p4}")
print(f"  Results: {out_path}")

print(f"\n{'='*80}")
print(f"  DONE")
print(f"{'='*80}")
