# %% Agent 1 — Monthly Diagnostic: When Did the Edge Shift?
#
# Run AFTER agent1_eval_sl_skip.py in same Colab session to reuse e2_trades.
# If run standalone, pipeline executes automatically (~10 min).
#
# Computes monthly metrics (Jan-Sep 2026) to identify:
#   - When avg PnL first dropped below +10%
#   - When WR first dropped below 70%
#   - Whether decay is gradual or sudden
#   - Whether losses concentrate in specific tickers or spread

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


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — E2 BENCHMARK (DO NOT CHANGE)
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS (same as eval script)
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


# ═══════════════════════════════════════════════════════════════════════════════
# DATA PIPELINE (skips if already loaded from prior cell)
# ═══════════════════════════════════════════════════════════════════════════════

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
    print(f"  MONTHLY DIAGNOSTIC — Data Pipeline")
    print(f"  {BACKTEST_START} → {BACKTEST_END}")
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
    print(f"  → Got bars for {len(all_bars)}/{len(all_pairs)} pairs")

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
    print(f"  → _ticker_ref has {len(_ticker_ref)} entries")
del _ref_tickers_needed, _ref_missing


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD E2 TRADES (reuse from eval cell if available)
# ═══════════════════════════════════════════════════════════════════════════════

TRADES_CACHED = False
try:
    _ = e2_trades
    if len(e2_trades) > 0 and 'pnl_eod' in e2_trades[0]:
        TRADES_CACHED = True
        print(f"  Reusing {len(e2_trades)} e2_trades from prior cell")
except NameError:
    pass

if not TRADES_CACHED:
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

                max_adverse_pnl = tc['pnl_eod']
                for t_str, bar in session_bars:
                    if t_str > EXIT_TIME:
                        break
                    if t_str >= tc['entry_time']:
                        bar_pnl = (ep - bar['h']) / ep * 100
                        if bar_pnl < max_adverse_pnl:
                            max_adverse_pnl = bar_pnl
                tc['max_adverse_pnl'] = round(max_adverse_pnl, 3)

                trades.append(tc)

        return trades

    print(f"\n  Building E2 baseline trades...")
    e2_trades = build_e2_trades_enriched()
    print(f"  → {len(e2_trades)} trades")


# ═══════════════════════════════════════════════════════════════════════════════
# MONTHLY BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════════

MONTHS = ['2026-01','2026-02','2026-03','2026-04','2026-05',
          '2026-06','2026-07','2026-08','2026-09']
MONTH_LABELS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep']

monthly = {}
for m_key, m_lbl in zip(MONTHS, MONTH_LABELS):
    m_trades = [t for t in e2_trades if t['scan_date'].startswith(m_key)]
    m_pnls = [t['pnl_eod'] for t in m_trades]
    m_metrics = compute_metrics(m_pnls, m_lbl)

    best_t = max(m_trades, key=lambda t: t['pnl_eod']) if m_trades else None
    worst_t = min(m_trades, key=lambda t: t['pnl_eod']) if m_trades else None
    winners = [t for t in m_trades if t['pnl_eod'] > 0]
    losers = [t for t in m_trades if t['pnl_eod'] <= 0]
    total_dol = sum(CAPITAL * t['pnl_eod'] / 100 for t in m_trades)

    monthly[m_lbl] = {
        'key': m_key, 'trades': m_trades, 'pnls': m_pnls,
        'metrics': m_metrics, 'best': best_t, 'worst': worst_t,
        'n_win': len(winners), 'n_lose': len(losers),
        'total_dol': total_dol,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

base_pnls = [t['pnl_eod'] for t in e2_trades]
base_m = compute_metrics(base_pnls, 'FULL')
base_dol = sum(CAPITAL * t['pnl_eod'] / 100 for t in e2_trades)

print(f"\n\n{'='*80}")
print(f"  AGENT 1 — MONTHLY DIAGNOSTIC")
print(f"  When did the edge shift?")
print(f"  {BACKTEST_START} → {BACKTEST_END}")
print(f"  {len(e2_trades)} trades, avg {base_m['avg']:+.2f}%, "
      f"Sharpe {base_m['sharpe']:.3f}, ${base_dol:+,.0f}")
print(f"{'='*80}")


# ─── SECTION 1: MONTHLY METRICS TABLE ───────────────────────────────────────

print(f"\n  ─── SECTION 1: MONTHLY METRICS ───\n")

print(f"  {'Month':>5s}  {'n':>3s}  {'Win':>3s}  {'Lose':>4s}  {'WR%':>5s}  "
      f"{'Avg PnL':>8s}  {'Med PnL':>8s}  {'Sharpe':>7s}  "
      f"{'Total $':>12s}  {'Best':>32s}  {'Worst':>32s}")
print(f"  {'─'*5}  {'─'*3}  {'─'*3}  {'─'*4}  {'─'*5}  "
      f"{'─'*8}  {'─'*8}  {'─'*7}  "
      f"{'─'*12}  {'─'*32}  {'─'*32}")

first_avg_below_10 = None
first_wr_below_70 = None
cum_dol = 0.0

for m_lbl in MONTH_LABELS:
    md = monthly[m_lbl]
    m = md['metrics']
    n = m['n']
    cum_dol += md['total_dol']

    if n == 0:
        print(f"  {m_lbl:>5s}  {0:>3d}  {'':>3s}  {'':>4s}  {'':>5s}  "
              f"{'':>8s}  {'':>8s}  {'':>7s}  {'':>12s}  {'':>32s}  {'':>32s}")
        continue

    avg = m['avg']
    wr = m['wr'] * 100

    flags = ''
    if avg < 10 and first_avg_below_10 is None:
        first_avg_below_10 = m_lbl
        flags += ' ◄ avg<10%'
    if wr < 70 and first_wr_below_70 is None:
        first_wr_below_70 = m_lbl
        flags += ' ◄ WR<70%'

    best_str = f"{md['best']['ticker']:>5s} {md['best']['pnl_eod']:>+7.1f}% ({md['best']['scan_date'][-5:]})" if md['best'] else ''
    worst_str = f"{md['worst']['ticker']:>5s} {md['worst']['pnl_eod']:>+7.1f}% ({md['worst']['scan_date'][-5:]})" if md['worst'] else ''

    print(f"  {m_lbl:>5s}  {n:>3d}  {md['n_win']:>3d}  {md['n_lose']:>4d}  "
          f"{wr:>4.0f}%  {avg:>+7.2f}%  {m['med']:>+7.2f}%  {m['sharpe']:>7.3f}  "
          f"${md['total_dol']:>+11,.0f}  {best_str:>32s}  {worst_str:>32s}{flags}")

print(f"\n  {'TOTAL':>5s}  {len(e2_trades):>3d}  "
      f"{sum(md['n_win'] for md in monthly.values()):>3d}  "
      f"{sum(md['n_lose'] for md in monthly.values()):>4d}  "
      f"{base_m['wr']*100:>4.0f}%  {base_m['avg']:>+7.2f}%  "
      f"{base_m['med']:>+7.2f}%  {base_m['sharpe']:>7.3f}  "
      f"${base_dol:>+11,.0f}")


# ─── SECTION 2: CUMULATIVE EQUITY CURVE ─────────────────────────────────────

print(f"\n\n  ─── SECTION 2: CUMULATIVE $ BY MONTH ───\n")

cum = 0.0
print(f"  {'Month':>5s}  {'Month $':>12s}  {'Cumul $':>12s}  {'Bar':>40s}")
print(f"  {'─'*5}  {'─'*12}  {'─'*12}  {'─'*40}")

max_cum = max(abs(md['total_dol']) for md in monthly.values()) if monthly else 1
for m_lbl in MONTH_LABELS:
    md = monthly[m_lbl]
    cum += md['total_dol']
    bar_len = int(abs(md['total_dol']) / max_cum * 30) if max_cum > 0 else 0
    bar_char = '█' if md['total_dol'] >= 0 else '░'
    bar = bar_char * bar_len
    sign = '+' if md['total_dol'] >= 0 else ''
    print(f"  {m_lbl:>5s}  ${md['total_dol']:>+11,.0f}  ${cum:>+11,.0f}  "
          f"{'  ' if md['total_dol'] >= 0 else ''}{bar}")


# ─── SECTION 3: DECAY DIAGNOSIS ─────────────────────────────────────────────

print(f"\n\n  ─── SECTION 3: DECAY DIAGNOSIS ───\n")

if first_avg_below_10:
    print(f"  Avg PnL first dropped below +10%: {first_avg_below_10}")
else:
    print(f"  Avg PnL never dropped below +10%")

if first_wr_below_70:
    print(f"  Win rate first dropped below 70%: {first_wr_below_70}")
else:
    print(f"  Win rate never dropped below 70%")

avgs = [monthly[m]['metrics']['avg'] for m in MONTH_LABELS if monthly[m]['metrics']['n'] > 0]
if len(avgs) >= 4:
    first_half = avgs[:len(avgs)//2]
    second_half = avgs[len(avgs)//2:]
    h1_avg = np.mean(first_half)
    h2_avg = np.mean(second_half)
    print(f"\n  H1 avg of monthly avgs: {h1_avg:+.2f}%")
    print(f"  H2 avg of monthly avgs: {h2_avg:+.2f}%")
    delta = h2_avg - h1_avg
    if abs(delta) < 3:
        print(f"  Δ = {delta:+.2f}pp — STABLE (no significant decay)")
    elif delta < -10:
        print(f"  Δ = {delta:+.2f}pp — SUDDEN SHIFT (not gradual)")
    else:
        print(f"  Δ = {delta:+.2f}pp — GRADUAL DECAY")

month_avgs_arr = np.array([monthly[m]['metrics']['avg'] for m in MONTH_LABELS
                           if monthly[m]['metrics']['n'] > 0])
month_idx = np.arange(len(month_avgs_arr))
if len(month_avgs_arr) >= 3:
    slope, intercept = np.polyfit(month_idx, month_avgs_arr, 1)
    r = np.corrcoef(month_idx, month_avgs_arr)[0, 1]
    print(f"\n  Linear trend: slope = {slope:+.2f}pp/month, R = {r:+.3f}")
    if abs(r) < 0.3:
        print(f"  Weak correlation — no clear trend, more like regime jumps")
    elif r < -0.5:
        print(f"  Strong negative trend — consistent decay over time")
    else:
        print(f"  Trend direction: {'declining' if slope < 0 else 'improving'}")


# ─── SECTION 4: LOSER CONCENTRATION ─────────────────────────────────────────

print(f"\n\n  ─── SECTION 4: ARE LOSERS CONCENTRATED OR SPREAD? ───\n")

all_losers = [t for t in e2_trades if t['pnl_eod'] <= 0]
if all_losers:
    loser_tickers = defaultdict(list)
    for t in all_losers:
        loser_tickers[t['ticker']].append(t)

    print(f"  {len(all_losers)} losers across {len(loser_tickers)} unique tickers")
    print(f"  {len(e2_trades) - len(all_losers)} winners across "
          f"{len(set(t['ticker'] for t in e2_trades if t['pnl_eod'] > 0))} unique tickers")

    repeat_losers = {tk: trades for tk, trades in loser_tickers.items() if len(trades) > 1}
    if repeat_losers:
        print(f"\n  REPEAT LOSERS (same ticker lost >1 time):")
        for tk in sorted(repeat_losers, key=lambda x: -len(repeat_losers[x])):
            trades = repeat_losers[tk]
            total_loss = sum(CAPITAL * t['pnl_eod'] / 100 for t in trades)
            dates_str = ', '.join(t['scan_date'][-5:] for t in sorted(trades, key=lambda x: x['scan_date']))
            print(f"    {tk:>6s}: {len(trades)} losses, ${total_loss:+,.0f} — dates: {dates_str}")
    else:
        print(f"\n  No repeat losers — every losing ticker only lost once.")

    loser_months = defaultdict(int)
    for t in all_losers:
        m = t['scan_date'][:7]
        loser_months[m] += 1

    print(f"\n  LOSER DISTRIBUTION BY MONTH:")
    for m_key, m_lbl in zip(MONTHS, MONTH_LABELS):
        n_total = monthly[m_lbl]['metrics']['n']
        n_lose = loser_months.get(m_key, 0)
        if n_total > 0:
            pct = n_lose / n_total * 100
            bar = '█' * n_lose
            print(f"    {m_lbl}: {n_lose:>2d}/{n_total:>2d} losers ({pct:>4.0f}%)  {bar}")
        else:
            print(f"    {m_lbl}:  0/ 0 losers")

    loser_dol_by_month = defaultdict(float)
    for t in all_losers:
        m = t['scan_date'][:7]
        loser_dol_by_month[m] += CAPITAL * t['pnl_eod'] / 100

    worst_month_key = min(loser_dol_by_month, key=lambda x: loser_dol_by_month[x])
    worst_month_lbl = MONTH_LABELS[MONTHS.index(worst_month_key)]
    print(f"\n  Worst month for losses: {worst_month_lbl} "
          f"(${loser_dol_by_month[worst_month_key]:+,.0f} in losing trades)")

    big_losers = [t for t in all_losers if t['pnl_eod'] < -10]
    if big_losers:
        print(f"\n  BIG LOSERS (>-10% PnL): {len(big_losers)} trades")
        for t in sorted(big_losers, key=lambda x: x['pnl_eod']):
            dol = CAPITAL * t['pnl_eod'] / 100
            print(f"    {t['scan_date']}  {t['ticker']:>6s}  gap +{t['gap_pct']*100:.0f}%  "
                  f"PnL {t['pnl_eod']:>+7.1f}%  ${dol:>+9,.0f}  "
                  f"MAE {t.get('max_adverse_pnl', 0):>+6.1f}%")
        big_loser_dol = sum(CAPITAL * t['pnl_eod'] / 100 for t in big_losers)
        all_loser_dol = sum(CAPITAL * t['pnl_eod'] / 100 for t in all_losers)
        print(f"    → Big losers account for ${big_loser_dol:+,.0f} "
              f"of ${all_loser_dol:+,.0f} total losses "
              f"({big_loser_dol/all_loser_dol*100:.0f}%)")
    else:
        print(f"\n  No big losers (>-10% PnL). All losses are small.")


# ─── SECTION 5: TRADE FREQUENCY ─────────────────────────────────────────────

print(f"\n\n  ─── SECTION 5: TRADE FREQUENCY ───\n")

print(f"  {'Month':>5s}  {'TDs':>4s}  {'Trades':>6s}  {'Trades/Wk':>9s}  {'No-trade days':>14s}")
print(f"  {'─'*5}  {'─'*4}  {'─'*6}  {'─'*9}  {'─'*14}")

for m_key, m_lbl in zip(MONTHS, MONTH_LABELS):
    m_dates = [d for d in dates if d.startswith(m_key)]
    n_td = len(m_dates)
    n_trades = monthly[m_lbl]['metrics']['n']
    tpw = n_trades / (n_td / 5) if n_td > 0 else 0
    trade_dates = set(t['scan_date'] for t in monthly[m_lbl]['trades'])
    no_trade = n_td - len(trade_dates)
    print(f"  {m_lbl:>5s}  {n_td:>4d}  {n_trades:>6d}  {tpw:>8.1f}  {no_trade:>14d}")


# ─── SECTION 6: GAP SIZE EVOLUTION ──────────────────────────────────────────

print(f"\n\n  ─── SECTION 6: GAP SIZE EVOLUTION ───")
print(f"  Are we getting smaller gaps over time? (smaller gaps = less edge)\n")

print(f"  {'Month':>5s}  {'Avg Gap%':>8s}  {'Med Gap%':>8s}  {'Min Gap%':>8s}  {'Max Gap%':>8s}")
print(f"  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

for m_lbl in MONTH_LABELS:
    md = monthly[m_lbl]
    if md['metrics']['n'] == 0:
        print(f"  {m_lbl:>5s}  {'':>8s}  {'':>8s}  {'':>8s}  {'':>8s}")
        continue
    gaps = [t['gap_pct'] * 100 for t in md['trades']]
    print(f"  {m_lbl:>5s}  {np.mean(gaps):>+7.1f}%  {np.median(gaps):>+7.1f}%  "
          f"{min(gaps):>+7.1f}%  {max(gaps):>+7.1f}%")


# ─── BOTTOM LINE ─────────────────────────────────────────────────────────────

print(f"\n\n{'='*80}")
print(f"  DIAGNOSTIC BOTTOM LINE")
print(f"{'='*80}")

strong_months = [m for m in MONTH_LABELS if monthly[m]['metrics']['n'] > 0
                 and monthly[m]['metrics']['avg'] >= 10]
weak_months = [m for m in MONTH_LABELS if monthly[m]['metrics']['n'] > 0
               and monthly[m]['metrics']['avg'] < 10]
negative_months = [m for m in MONTH_LABELS if monthly[m]['metrics']['n'] > 0
                   and monthly[m]['metrics']['avg'] < 0]

print(f"\n  Strong months (avg >= +10%): {', '.join(strong_months) if strong_months else 'NONE'}")
print(f"  Weak months   (avg < +10%):  {', '.join(weak_months) if weak_months else 'NONE'}")
print(f"  Negative months (avg < 0%):  {', '.join(negative_months) if negative_months else 'NONE'}")

if len(weak_months) == 0:
    print(f"\n  VERDICT: No deterioration detected. Every month averages >= +10%.")
elif len(weak_months) <= 2:
    print(f"\n  VERDICT: Isolated weak months ({', '.join(weak_months)}). "
          f"Check if driven by 1-2 outlier trades or a regime event.")
elif len(strong_months) <= 2:
    print(f"\n  VERDICT: Broad deterioration. Only {', '.join(strong_months)} were strong. "
          f"Strategy may have lost structural edge.")
else:
    n_recent_weak = sum(1 for m in MONTH_LABELS[-3:] if m in weak_months)
    if n_recent_weak >= 2:
        print(f"\n  VERDICT: Recent decay — {n_recent_weak}/3 recent months are weak. "
              f"Edge may be fading.")
    else:
        print(f"\n  VERDICT: Mixed — some strong, some weak. No clear trend direction.")


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()


# ─── Chart 1: Monthly Avg PnL bar chart ─────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f'Agent 1 Monthly Diagnostic — {len(e2_trades)} trades, '
             f'{BACKTEST_START} → {BACKTEST_END}',
             fontsize=12, fontweight='bold')

x = np.arange(len(MONTH_LABELS))
avgs_plot = [monthly[m]['metrics']['avg'] for m in MONTH_LABELS]
n_trades_plot = [monthly[m]['metrics']['n'] for m in MONTH_LABELS]
wr_plot = [monthly[m]['metrics']['wr'] * 100 for m in MONTH_LABELS]
dol_plot = [monthly[m]['total_dol'] / 1000 for m in MONTH_LABELS]

# Panel 1: Avg PnL by month
ax = axes[0, 0]
colors = ['#27ae60' if a >= 10 else ('#f39c12' if a >= 0 else '#e74c3c') for a in avgs_plot]
bars = ax.bar(x, avgs_plot, color=colors, alpha=0.85, edgecolor='black', linewidth=0.3)
ax.axhline(10, color='blue', ls='--', alpha=0.5, label='10% threshold')
ax.axhline(0, color='black', lw=1)
for b, a, n in zip(bars, avgs_plot, n_trades_plot):
    va = 'bottom' if a >= 0 else 'top'
    ax.text(b.get_x() + b.get_width()/2, a + (1 if a >= 0 else -1),
            f"{a:+.1f}%\nn={n}", ha='center', va=va, fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(MONTH_LABELS)
ax.set_ylabel('Avg PnL %')
ax.set_title('Avg PnL by Month')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# Panel 2: Win rate by month
ax = axes[0, 1]
wr_colors = ['#27ae60' if w >= 70 else ('#f39c12' if w >= 50 else '#e74c3c') for w in wr_plot]
bars = ax.bar(x, wr_plot, color=wr_colors, alpha=0.85, edgecolor='black', linewidth=0.3)
ax.axhline(70, color='blue', ls='--', alpha=0.5, label='70% threshold')
ax.axhline(50, color='red', ls='--', alpha=0.3, label='50% (coin flip)')
for b, w in zip(bars, wr_plot):
    ax.text(b.get_x() + b.get_width()/2, w + 1,
            f"{w:.0f}%", ha='center', va='bottom', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(MONTH_LABELS)
ax.set_ylabel('Win Rate %')
ax.set_title('Win Rate by Month')
ax.set_ylim(0, 110)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# Panel 3: Monthly $ PnL
ax = axes[1, 0]
dol_colors = ['#27ae60' if d >= 0 else '#e74c3c' for d in dol_plot]
bars = ax.bar(x, dol_plot, color=dol_colors, alpha=0.85, edgecolor='black', linewidth=0.3)
ax.axhline(0, color='black', lw=1)
for b, d in zip(bars, dol_plot):
    va = 'bottom' if d >= 0 else 'top'
    ax.text(b.get_x() + b.get_width()/2, d + (5 if d >= 0 else -5),
            f"${d:+.0f}K", ha='center', va=va, fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(MONTH_LABELS)
ax.set_ylabel('$ PnL ($K, non-compounding)')
ax.set_title('Monthly Dollar PnL')
ax.grid(True, alpha=0.3, axis='y')

# Panel 4: Cumulative equity curve
ax = axes[1, 1]
cum_equity = np.cumsum([monthly[m]['total_dol'] for m in MONTH_LABELS]) / 1000
ax.plot(x, cum_equity, 'b-o', linewidth=2, markersize=6)
ax.fill_between(x, 0, cum_equity, alpha=0.15, color='blue')
ax.axhline(0, color='black', lw=0.5)
for xi, ce in zip(x, cum_equity):
    ax.text(xi, ce + 15, f"${ce:.0f}K", ha='center', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(MONTH_LABELS)
ax.set_ylabel('Cumulative $ ($K)')
ax.set_title('Cumulative Equity Curve')
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.94])
p1 = os.path.join(BASE_DIR, 'diagnostic_monthly.png')
plt.savefig(p1, dpi=150, bbox_inches='tight')
plt.show()
print(f"\n  Saved: {p1}")


# ─── Chart 2: Per-trade PnL timeline with monthly dividers ──────────────────

fig, ax = plt.subplots(figsize=(max(16, len(e2_trades) * 0.25), 6))

sorted_trades = sorted(e2_trades, key=lambda t: t['scan_date'])
trade_x = np.arange(len(sorted_trades))
trade_pnl = [t['pnl_eod'] for t in sorted_trades]
trade_colors = ['#27ae60' if p > 0 else '#e74c3c' for p in trade_pnl]

ax.bar(trade_x, trade_pnl, color=trade_colors, alpha=0.7, width=0.8)
ax.axhline(0, color='black', lw=1)

prev_month = None
for i, t in enumerate(sorted_trades):
    m = t['scan_date'][:7]
    if m != prev_month and prev_month is not None:
        ax.axvline(i - 0.5, color='gray', ls='--', alpha=0.5)
        ax.text(i - 0.5, ax.get_ylim()[1] * 0.95,
                MONTH_LABELS[MONTHS.index(m)], fontsize=8, ha='left', va='top',
                color='gray', fontweight='bold')
    prev_month = m

ax.set_xlabel(f'Trade # (chronological, {len(sorted_trades)} total)')
ax.set_ylabel('PnL %')
ax.set_title(f'Every Trade — Chronological Timeline\n'
             f'Green=winner, Red=loser, Gray lines=month boundaries',
             fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
p2 = os.path.join(BASE_DIR, 'diagnostic_timeline.png')
plt.savefig(p2, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Saved: {p2}")


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

results_dir = os.path.join(BASE_DIR, 'results')
os.makedirs(results_dir, exist_ok=True)

output = {
    'run_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
    'period': f'{BACKTEST_START} to {BACKTEST_END}',
    'n_trades': len(e2_trades),
    'overall': {
        'avg': round(base_m['avg'], 2), 'sharpe': round(base_m['sharpe'], 3),
        'wr': round(base_m['wr'], 3), 'total_dollars': round(base_dol),
    },
    'monthly': [{
        'month': m_lbl,
        'n': monthly[m_lbl]['metrics']['n'],
        'avg': round(monthly[m_lbl]['metrics']['avg'], 2),
        'median': round(monthly[m_lbl]['metrics']['med'], 2),
        'wr': round(monthly[m_lbl]['metrics']['wr'], 3),
        'sharpe': round(monthly[m_lbl]['metrics']['sharpe'], 3),
        'total_dol': round(monthly[m_lbl]['total_dol']),
        'n_win': monthly[m_lbl]['n_win'],
        'n_lose': monthly[m_lbl]['n_lose'],
        'best': {'ticker': monthly[m_lbl]['best']['ticker'],
                 'pnl': round(monthly[m_lbl]['best']['pnl_eod'], 1),
                 'date': monthly[m_lbl]['best']['scan_date']} if monthly[m_lbl]['best'] else None,
        'worst': {'ticker': monthly[m_lbl]['worst']['ticker'],
                  'pnl': round(monthly[m_lbl]['worst']['pnl_eod'], 1),
                  'date': monthly[m_lbl]['worst']['scan_date']} if monthly[m_lbl]['worst'] else None,
    } for m_lbl in MONTH_LABELS],
    'diagnosis': {
        'first_avg_below_10': first_avg_below_10,
        'first_wr_below_70': first_wr_below_70,
        'strong_months': strong_months,
        'weak_months': weak_months,
        'negative_months': negative_months,
    },
}

out_path = os.path.join(results_dir, 'diagnostic_monthly.json')
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\n  Saved: {out_path}")

print(f"\n{'='*80}")
print(f"  DONE — Monthly diagnostic complete")
print(f"{'='*80}")
