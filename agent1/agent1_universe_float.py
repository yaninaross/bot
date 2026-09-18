# %% Agent 1 — Universe Float Comparison
#
# Compare two universes for gap-fade SHORT (CS+SD, N=1, exit@17:00):
#
#   SOFT_FLOAT (current): hardcoded _FLOAT dict is soft filter — unknown tickers pass through
#   HARD_FLOAT (new):     ALL tickers must have Polygon weighted_shares_outstanding >= 10M
#                          no hardcoded dict — purely Polygon reference data
#
# Uses deployed config: CS+SD, N=1, exit@17:00 AH
# Colab-cell friendly. ~10-15 min runtime (API-bound).

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
PRICE_MIN      = 10.0
FLOAT_MIN      = 10_000_000
PRE_FILTER_N   = 15
TOP_N          = 1

PEAK_WINDOW    = 3
MAX_PEAK_BAR   = 12
FALLBACK_BAR   = 15

CAPITAL        = 100_000
EXIT_TIME      = '17:00'

POLYGON_API_KEY = os.environ["POLYGON_API_KEY"]
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

# Hardcoded _FLOAT dict — used only for SOFT_FLOAT universe
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
            time.sleep(5 * (2 ** attempt))
    r = requests.get(url, params=p, timeout=60)
    r.raise_for_status()
    return r.json()


# ═══════════════════════════════════════════════════════════════════════════════
# GROUPED DAILY CACHE
# ═══════════════════════════════════════════════════════════════════════════════

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
# ASYNC MINUTE BAR FETCH (RTH + AH to 20:00)
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
            rth = []
            for b in bars:
                ts = pd.Timestamp(b['t'], unit='ms', tz='UTC').tz_convert(ET)
                t_str = ts.strftime('%H:%M')
                if '09:30' <= t_str < '20:00':
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
# ASYNC TICKER REFERENCE FETCH (per-ticker — for CS type check)
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
# OPEN PATTERN CLASSIFICATION
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
# SCANNER — supports both float modes
# ═══════════════════════════════════════════════════════════════════════════════

def scan_day(dt_str):
    """Scan for gap-ups. No float filtering here — that happens post-reference fetch."""
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
        candidates.append({
            'ticker': tk, 'gap_pct': gap, 'prior_close': prev_c,
            'open_price': cur_p, 'scan_date': dt_str,
            'volume_today': today[tk]['v'],
        })

    candidates.sort(key=lambda x: -x['gap_pct'])

    nxt = _next_n_td(dt_str, 1)
    for s in candidates:
        for i in range(len(nxt)):
            s[f't{i+1}_date'] = nxt[i].strftime('%Y-%m-%d')
    return candidates


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

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
# ═══════════════════════════════════════════════════════════════════════════════
#
#  BACKTEST ENGINE
#
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest():
    t_total = time.time()

    print(f"\n{'='*90}")
    print(f"  AGENT 1 — UNIVERSE FLOAT COMPARISON")
    print(f"  {BACKTEST_START} → {BACKTEST_END}")
    print(f"  SOFT_FLOAT: hardcoded dict (soft — unknown tickers pass through)")
    print(f"  HARD_FLOAT: Polygon weighted_shares_outstanding >= {FLOAT_MIN/1e6:.0f}M (all must pass)")
    print(f"  Config: CS+SD, N={TOP_N}, exit@{EXIT_TIME}")
    print(f"{'='*90}")

    # ── Trading days ──
    d = datetime.strptime(BACKTEST_START, '%Y-%m-%d').date()
    end = datetime.strptime(BACKTEST_END, '%Y-%m-%d').date()
    dates = []
    while d <= end:
        if _is_td(d): dates.append(d.strftime('%Y-%m-%d'))
        d += timedelta(days=1)
    n_dates = len(dates)
    print(f"\n  {n_dates} trading days")

    # ── Phase 1: Scan (no float filter — collect all gap-ups) ──
    t1 = time.time()
    print(f"\n  Phase 1: Scanning for gap-ups (no float filter)...")
    all_candidates = {}
    total_cands = 0
    for i, dt in enumerate(dates):
        cands = scan_day(dt)
        all_candidates[dt] = cands
        total_cands += len(cands)
        if (i+1) % 30 == 0 or i == n_dates - 1:
            print(f"    [{i+1:>3}/{n_dates}] {dt}  |  {total_cands} total  |  {time.time()-t_total:.0f}s")
    print(f"  → {total_cands} total gap-up candidates")

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
        if done % 300 == 0 or done == len(pairs_list):
            print(f"    {done}/{len(pairs_list)}  |  {time.time()-t_total:.0f}s")
    print(f"  → Got bars for {len(all_bars)}/{len(all_pairs)} pairs")

    # ── Phase 3: Fetch ticker references (type + weighted_shares_outstanding) ──
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

    n_cs = sum(1 for tk in all_tickers if _ticker_ref.get(tk, {}).get('type') == 'CS')
    n_with_shares = sum(1 for tk in all_tickers
                        if (_ticker_ref.get(tk, {}).get('shares') or 0) >= FLOAT_MIN)
    print(f"  → {n_cs} CS out of {len(all_tickers)} tickers")
    print(f"  → {n_with_shares} tickers with weighted_shares_outstanding >= {FLOAT_MIN/1e6:.0f}M")

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

    pat_counts = defaultdict(int)
    for p in pattern_cache.values():
        pat_counts[p] += 1
    print(f"\n  Phase 4: Pattern classification")
    for pat in ['sustained_drop', 'drop_then_recover', 'mild_drop', 'flat_or_bounce', 'no_data']:
        cnt = pat_counts.get(pat, 0)
        if pattern_cache:
            print(f"    {pat:25s}  {cnt:>5d}  ({cnt/len(pattern_cache)*100:.1f}%)")

    # ── Phase 5: Fetch outcome dates ──
    t5 = time.time()
    outcome_dates = set()
    for cands in all_candidates.values():
        for c in cands[:PRE_FILTER_N]:
            outcome_dates.add(c['scan_date'])
            d = c.get('t1_date')
            if d: outcome_dates.add(d)
    to_fetch = sorted(od for od in outcome_dates if od not in _gd_cache)
    for od in to_fetch:
        try: fetch_grouped_daily(od)
        except: pass
    print(f"\n  Phase 5: Fetched {len(to_fetch)} additional dates")

    # ── Phase 6: Build trades for BOTH universes ──
    t6 = time.time()
    print(f"\n  Phase 6: Building CS+SD trades for both universes...")

    universes = {
        'SOFT_FLOAT': {
            'label': 'SOFT_FLOAT (current — unknown tickers pass)',
            'trades_daily': {},
        },
        'HARD_FLOAT': {
            'label': 'HARD_FLOAT (all must have Polygon float ≥ 10M)',
            'trades_daily': {},
        },
    }

    for dt in dates:
        cands = all_candidates.get(dt, [])

        # For each universe, apply its float filter, then CS+SD+entry, then TOP_N
        for univ_key in universes:
            qualified = []
            for c in cands[:PRE_FILTER_N]:
                tk = c['ticker']
                key = (tk, dt)
                ref = _ticker_ref.get(tk, {})

                # CS check — both universes require CS
                if ref.get('type') != 'CS':
                    continue

                # Float filter — differs by universe
                if univ_key == 'SOFT_FLOAT':
                    fs = _FLOAT.get(tk)
                    if fs is not None and fs < FLOAT_MIN:
                        continue
                elif univ_key == 'HARD_FLOAT':
                    polygon_shares = ref.get('shares') or 0
                    if polygon_shares < FLOAT_MIN:
                        continue

                # sustained_drop pattern
                pat = pattern_cache.get(key, 'no_data')
                if pat != 'sustained_drop':
                    continue

                # Entry detection
                entry = entry_cache.get(key)
                if entry is None:
                    continue

                tc = copy.deepcopy(c)
                tc['entry_price'] = entry['entry_price']
                tc['entry_time'] = entry['entry_time']
                tc['entry_type'] = entry['entry_type']
                tc['open_pattern'] = pat
                tc['universe'] = univ_key

                ep = tc['entry_price']
                polygon_shares = ref.get('shares') or 0
                tc['polygon_shares'] = polygon_shares

                # Intraday exit @ EXIT_TIME (SHORT: entry - exit)
                bars = all_bars.get(key, [])
                exit_price = None
                entry_t = tc['entry_time']
                for t_str, bar in bars:
                    if t_str > EXIT_TIME:
                        break
                    if t_str >= entry_t:
                        exit_price = bar['c']
                tc['pnl_exit'] = round((ep - exit_price) / ep * 100, 3) if exit_price is not None else None

                # T+1 close (for reference)
                td = _gd_cache.get(tc.get('t1_date', ''), {}).get(tk, {})
                tc['pnl_t1'] = round((ep - td['c']) / ep * 100, 3) if td.get('c') else None

                if tc['pnl_exit'] is not None:
                    qualified.append(tc)

            qualified.sort(key=lambda x: -x['gap_pct'])
            universes[univ_key]['trades_daily'][dt] = qualified[:TOP_N]

    for univ_key, udata in universes.items():
        total = sum(len(v) for v in udata['trades_daily'].values())
        active = sum(1 for v in udata['trades_daily'].values() if v)
        print(f"    {udata['label']:<55}  {total:>4} trades  {active:>4} active days")

    elapsed = time.time() - t_total
    print(f"\n  TOTAL: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    return universes, dates


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def compare_results(universes, dates):
    n_dates = len(dates)

    print(f"\n\n{'='*120}")
    print(f"  UNIVERSE FLOAT COMPARISON — CS+SD, N={TOP_N}, exit@{EXIT_TIME}")
    print(f"  {dates[0]} → {dates[-1]}  |  {n_dates} trading days  |  ${CAPITAL:,.0f} pool")
    print(f"{'='*120}")

    results = {}
    for univ_key, udata in universes.items():
        all_pnls = []
        daily_pnl = defaultdict(list)
        all_trades = []

        for dt in dates:
            for tc in udata['trades_daily'].get(dt, []):
                p = tc['pnl_exit']
                if p is not None:
                    all_pnls.append(p)
                    daily_pnl[dt].append(p)
                    all_trades.append(tc)

        m = compute_metrics(all_pnls, univ_key)

        cum_nc = 0.0
        equity_curve = [0.0]
        eq_dates = [datetime.strptime(dates[0], '%Y-%m-%d')]
        for dt in dates:
            dp = daily_pnl.get(dt, [])
            if dp:
                pos_size = CAPITAL / len(dp)
                cum_nc += sum(pos_size * (r / 100) for r in dp)
            equity_curve.append(cum_nc)
            eq_dates.append(datetime.strptime(dt, '%Y-%m-%d'))

        results[univ_key] = {
            'metrics': m, 'cum_nc': cum_nc, 'trades': all_trades,
            'daily_pnl': dict(daily_pnl), 'equity_curve': equity_curve,
            'eq_dates': eq_dates,
            'active_days': len(daily_pnl),
        }

    # ── Summary table ──
    headers = ['Universe', 'n', 'Active Days', 'Avg PnL', 'Median', 'Trim Mean',
               'WR%', 'Sharpe', 'PF', 'CI Low', 'CI Hi', 'Max Loss', 'Pool $']
    rows = []
    for univ_key in ['SOFT_FLOAT', 'HARD_FLOAT']:
        r = results[univ_key]
        m = r['metrics']
        rows.append([
            universes[univ_key]['label'][:50],
            m['n'],
            r['active_days'],
            f"{m['avg']:+.2f}%",
            f"{m['med']:+.2f}%",
            f"{m['trimmed_mean']:+.2f}%",
            f"{m['wr']*100:.0f}%",
            f"{m['sharpe']:.3f}",
            f"{m['pf']:.2f}" if m['pf'] < 999 else '∞',
            f"{m['ci_lo']:+.2f}",
            f"{m['ci_hi']:+.2f}",
            f"{m['max_loss']:+.1f}%",
            f"${r['cum_nc']:+,.0f}",
        ])
    print()
    print(tabulate(rows, headers=headers, tablefmt='simple', stralign='right'))

    # ── Delta analysis ──
    soft = results['SOFT_FLOAT']
    hard = results['HARD_FLOAT']
    sm, hm = soft['metrics'], hard['metrics']

    print(f"\n  ── DELTA (HARD_FLOAT − SOFT_FLOAT) ──")
    print(f"    Trades:      {hm['n'] - sm['n']:+d}")
    print(f"    Active days: {hard['active_days'] - soft['active_days']:+d}")
    print(f"    Avg PnL:     {hm['avg'] - sm['avg']:+.2f}%")
    print(f"    Sharpe:      {hm['sharpe'] - sm['sharpe']:+.3f}")
    print(f"    Win Rate:    {(hm['wr'] - sm['wr'])*100:+.1f}%")
    print(f"    Pool $:      ${hard['cum_nc'] - soft['cum_nc']:+,.0f}")
    print(f"    Max Loss:    {hm['max_loss'] - sm['max_loss']:+.1f}%")

    # ── Overlap analysis ──
    soft_keys = set((t['ticker'], t['scan_date']) for t in soft['trades'])
    hard_keys = set((t['ticker'], t['scan_date']) for t in hard['trades'])
    both = soft_keys & hard_keys
    soft_only = soft_keys - hard_keys
    hard_only = hard_keys - soft_keys

    print(f"\n  ── TRADE OVERLAP ──")
    print(f"    Both universes:   {len(both):>4}")
    print(f"    SOFT_FLOAT only:  {len(soft_only):>4}")
    print(f"    HARD_FLOAT only:  {len(hard_only):>4}")

    if soft_only:
        so_pnls = [t['pnl_exit'] for t in soft['trades']
                   if (t['ticker'], t['scan_date']) in soft_only and t['pnl_exit'] is not None]
        if so_pnls:
            so_m = compute_metrics(so_pnls)
            print(f"\n    SOFT_FLOAT-only trades (removed by hard filter):")
            print(f"      n={so_m['n']}  avg={so_m['avg']:+.2f}%  Sharpe={so_m['sharpe']:.3f}  "
                  f"WR={so_m['wr']*100:.0f}%  worst={so_m['max_loss']:+.1f}%")
            # Show specific tickers
            so_trades = [t for t in soft['trades'] if (t['ticker'], t['scan_date']) in soft_only]
            so_trades.sort(key=lambda x: x['pnl_exit'] or 0)
            print(f"      Worst 5:")
            for t in so_trades[:5]:
                shares = t.get('polygon_shares', 0)
                print(f"        {t['scan_date']} {t['ticker']:>6}  gap={t['gap_pct']*100:+.1f}%  "
                      f"pnl={t['pnl_exit']:+.2f}%  polygon_shares={shares:,.0f}")
            if len(so_trades) > 5:
                print(f"      Best 5:")
                for t in so_trades[-5:]:
                    shares = t.get('polygon_shares', 0)
                    print(f"        {t['scan_date']} {t['ticker']:>6}  gap={t['gap_pct']*100:+.1f}%  "
                          f"pnl={t['pnl_exit']:+.2f}%  polygon_shares={shares:,.0f}")

    if hard_only:
        ho_pnls = [t['pnl_exit'] for t in hard['trades']
                   if (t['ticker'], t['scan_date']) in hard_only and t['pnl_exit'] is not None]
        if ho_pnls:
            ho_m = compute_metrics(ho_pnls)
            print(f"\n    HARD_FLOAT-only trades (new — replaced soft-only):")
            print(f"      n={ho_m['n']}  avg={ho_m['avg']:+.2f}%  Sharpe={ho_m['sharpe']:.3f}  "
                  f"WR={ho_m['wr']*100:.0f}%  worst={ho_m['max_loss']:+.1f}%")
            ho_trades = [t for t in hard['trades'] if (t['ticker'], t['scan_date']) in hard_only]
            ho_trades.sort(key=lambda x: x['pnl_exit'] or 0)
            print(f"      Worst 5:")
            for t in ho_trades[:5]:
                shares = t.get('polygon_shares', 0)
                print(f"        {t['scan_date']} {t['ticker']:>6}  gap={t['gap_pct']*100:+.1f}%  "
                      f"pnl={t['pnl_exit']:+.2f}%  polygon_shares={shares:,.0f}")
            if len(ho_trades) > 5:
                print(f"      Best 5:")
                for t in ho_trades[-5:]:
                    shares = t.get('polygon_shares', 0)
                    print(f"        {t['scan_date']} {t['ticker']:>6}  gap={t['gap_pct']*100:+.1f}%  "
                          f"pnl={t['pnl_exit']:+.2f}%  polygon_shares={shares:,.0f}")

    # ── Monthly breakdown ──
    print(f"\n  ── MONTHLY COMPARISON ──")
    months = sorted(set(dt[:7] for dt in dates))
    headers_m = ['Month', 'SOFT n', 'SOFT Avg', 'SOFT $', 'HARD n', 'HARD Avg', 'HARD $', 'Δ $']
    rows_m = []
    for mo in months:
        sp = [p for dt, pnls in soft['daily_pnl'].items() if dt[:7] == mo for p in pnls]
        hp = [p for dt, pnls in hard['daily_pnl'].items() if dt[:7] == mo for p in pnls]
        s_m = compute_metrics(sp) if sp else {'avg': 0, 'n': 0}
        h_m = compute_metrics(hp) if hp else {'avg': 0, 'n': 0}
        s_dollar = sum(CAPITAL * (r / 100) for r in sp)
        h_dollar = sum(CAPITAL * (r / 100) for r in hp)
        rows_m.append([
            mo,
            s_m['n'], f"{s_m['avg']:+.2f}%", f"${s_dollar:+,.0f}",
            h_m['n'], f"{h_m['avg']:+.2f}%", f"${h_dollar:+,.0f}",
            f"${h_dollar - s_dollar:+,.0f}",
        ])
    print()
    print(tabulate(rows_m, headers=headers_m, tablefmt='simple', stralign='right'))

    # ── Verdict ──
    print(f"\n  ── VERDICT ──")
    if hm['sharpe'] > sm['sharpe'] and hm['ci_lo'] > 0:
        print(f"    ★ HARD_FLOAT WINS: +{hm['sharpe']-sm['sharpe']:.3f} Sharpe, "
              f"${hard['cum_nc']-soft['cum_nc']:+,.0f} dollars")
        if hm['max_loss'] > sm['max_loss']:
            print(f"      Better tail risk too: max loss {hm['max_loss']:+.1f}% vs {sm['max_loss']:+.1f}%")
    elif sm['sharpe'] > hm['sharpe']:
        print(f"    ★ SOFT_FLOAT (current) holds: +{sm['sharpe']-hm['sharpe']:.3f} Sharpe advantage")
    else:
        print(f"    ★ INCONCLUSIVE: similar Sharpe ({sm['sharpe']:.3f} vs {hm['sharpe']:.3f})")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_comparison(results, universes, dates):
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        BASE_DIR = os.getcwd()

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle(f'Agent 1 — Universe Float Comparison (CS+SD, N={TOP_N}, exit@{EXIT_TIME})\n'
                 f'{dates[0]} → {dates[-1]}  |  $100K pool',
                 fontsize=12, fontweight='bold')

    colors = {'SOFT_FLOAT': '#2980b9', 'HARD_FLOAT': '#e74c3c'}
    labels = {'SOFT_FLOAT': 'SOFT (current)', 'HARD_FLOAT': 'HARD (Polygon ≥10M)'}

    # Panel 1: Equity curves
    ax = axes[0]
    for univ_key in ['SOFT_FLOAT', 'HARD_FLOAT']:
        r = results[univ_key]
        m = r['metrics']
        ax.plot(r['eq_dates'], r['equity_curve'],
                color=colors[univ_key], linewidth=2,
                label=f'{labels[univ_key]}: ${r["cum_nc"]:+,.0f} Sh={m["sharpe"]:.3f}')
    ax.axhline(0, color='#333', lw=0.8, ls=':')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.set_title('Equity Curve', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Monthly $ difference
    ax = axes[1]
    months = sorted(set(dt[:7] for dt in dates))
    month_labels = [m[5:] for m in months]
    soft_monthly = []
    hard_monthly = []
    for mo in months:
        sp = [p for dt, pnls in results['SOFT_FLOAT']['daily_pnl'].items()
              if dt[:7] == mo for p in pnls]
        hp = [p for dt, pnls in results['HARD_FLOAT']['daily_pnl'].items()
              if dt[:7] == mo for p in pnls]
        soft_monthly.append(sum(CAPITAL * (r / 100) for r in sp))
        hard_monthly.append(sum(CAPITAL * (r / 100) for r in hp))

    x = np.arange(len(months))
    w = 0.35
    ax.bar(x - w/2, soft_monthly, w, color=colors['SOFT_FLOAT'], alpha=0.8, label='SOFT')
    ax.bar(x + w/2, hard_monthly, w, color=colors['HARD_FLOAT'], alpha=0.8, label='HARD')
    ax.set_xticks(x)
    ax.set_xticklabels(month_labels, fontsize=8)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.set_title('Monthly $ PnL', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')
    ax.axhline(0, color='#333', lw=0.8, ls=':')

    # Panel 3: PnL distribution
    ax = axes[2]
    for univ_key in ['SOFT_FLOAT', 'HARD_FLOAT']:
        pnls = [t['pnl_exit'] for t in results[univ_key]['trades'] if t['pnl_exit'] is not None]
        if pnls:
            ax.hist(pnls, bins=40, alpha=0.5, color=colors[univ_key],
                    label=f'{labels[univ_key]} (n={len(pnls)})', edgecolor='white')
    ax.axvline(0, color='#333', lw=1, ls=':')
    ax.set_xlabel('PnL (%)')
    ax.set_title('PnL Distribution', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    p = os.path.join(BASE_DIR, 'agent1_universe_float.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n  Saved: {p}")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

universes, dates = run_backtest()
results = compare_results(universes, dates)
plot_comparison(results, universes, dates)
