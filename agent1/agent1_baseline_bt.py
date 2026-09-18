# %% Agent 1 — Gap-Fade SHORT Backtest (CS+SD deployed strategy)
#
# Deployed strategy:
#   CS + sustained_drop filter, SOFT float, top 1 by gap, local_high entry, SHORT
#   Exit at 17:00 ET (1h after close, after-hours), PNL_CAP = -50%
#   No TP, no SL — hold to fixed exit time.
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

BACKTEST_START = '2026-01-02'
BACKTEST_END   = '2026-09-18'

AB_COMPARE     = False         # True = run OLD vs NEW float filter side by side

GAP_MIN_PCT    = 5.0
PRICE_MIN      = 10.0
FLOAT_MIN      = 10_000_000
TOP_N          = 1
PRE_FILTER_N   = 30

PEAK_WINDOW    = 3
MAX_PEAK_BAR   = 12
FALLBACK_BAR   = 15

EXIT_TIME      = '17:00'      # exit at 17:00 ET (1h after close, after-hours)
PNL_CAP        = -50.0        # clip catastrophic losses

CAPITAL        = 100_000
SCORING_TIME_ET = '09:20'     # 6:20 AM PST = 9:20 AM ET (premarket, matches live scanner)
PM_MIN_BARS    = 5            # min premarket bars to use PM price (filters thinly-traded tickers)

POLYGON_API_KEY = os.environ["POLYGON_API_KEY"]
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
            pm = []
            rth = []
            ah = []
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
    """Initial scan using grouped daily open — wider 3% net to catch PM movers."""
    if not _is_td(dt_str): return []
    prior_str = _prior_td(dt_str).strftime('%Y-%m-%d')
    prior = fetch_grouped_daily(prior_str)
    today = fetch_grouped_daily(dt_str)

    gap_min = 0.03  # 3% wide net — reranked by PM price later
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
    print(f"  AGENT 1 — Gap-Fade SHORT (CS+SD deployed strategy)")
    print(f"  {BACKTEST_START} → {BACKTEST_END}")
    print(f"  Universe: ALL US equities, gap >= {GAP_MIN_PCT}%, price >= ${PRICE_MIN}")
    pst_h = int(SCORING_TIME_ET.split(':')[0]) - 3
    pst_m = SCORING_TIME_ET.split(':')[1]
    print(f"  Gap source: PREMARKET bars at {SCORING_TIME_ET} ET ({pst_h}:{pst_m} AM PST)")
    print(f"  Filter: CS + sustained_drop + float >= {FLOAT_MIN/1e6:.0f}M, top {TOP_N} by gap")
    print(f"  Entry: local_high + fallback  |  Exit: {EXIT_TIME} ET  |  Direction: SHORT")
    print(f"  PNL_CAP: {PNL_CAP}%  |  Capital: ${CAPITAL:,.0f}/day, non-compounding")
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

    # ── Phase 2b: Rerank candidates by premarket price at SCORING_TIME_ET ──
    # Only use PM price for tickers with >= PM_MIN_BARS bars (filters thinly traded
    # tickers that wouldn't appear in the live scanner's snapshot API)
    print(f"\n  Phase 2b: Reranking by premarket gap at {SCORING_TIME_ET} ET "
          f"(min {PM_MIN_BARS} PM bars)...")
    gap_min_final = GAP_MIN_PCT / 100.0
    n_reranked = 0
    n_skipped_thin = 0
    for dt in dates:
        cands = all_candidates.get(dt, [])
        for c in cands:
            tk = c['ticker']
            key = (tk, dt)
            bar_data = all_bars.get(key)
            if bar_data is None:
                continue
            pm_bars = bar_data.get('pm', [])
            if not pm_bars:
                continue
            # Count PM bars up to scoring time
            pm_before = [(t, d) for t, d in pm_bars if t <= SCORING_TIME_ET]
            if len(pm_before) < PM_MIN_BARS:
                n_skipped_thin += 1
                continue
            pm_price = pm_before[-1][1]['c']
            if pm_price and pm_price > 0:
                prev_c = c['prior_close']
                c['gap_pct'] = (pm_price - prev_c) / prev_c
                c['pm_price'] = pm_price
                n_reranked += 1

        # Re-sort by PM gap and re-filter at 5%
        reranked = [c for c in cands if c['gap_pct'] >= gap_min_final]
        reranked.sort(key=lambda x: -x['gap_pct'])
        all_candidates[dt] = reranked

    print(f"  → Reranked {n_reranked} candidates using premarket prices "
          f"({n_skipped_thin} skipped: < {PM_MIN_BARS} PM bars)")

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
    for key, bar_data in all_bars.items():
        rth_bars = bar_data.get('rth', []) if isinstance(bar_data, dict) else bar_data
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

    elapsed = time.time() - t_total
    print(f"\n  Shared pipeline: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    return all_candidates, all_bars, pattern_cache, entry_cache, dates, timings, t_total


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6: Build trades (use_api_float toggles OLD vs NEW filter)
# ═══════════════════════════════════════════════════════════════════════════════

def build_trades(all_candidates, all_bars, pattern_cache, entry_cache, dates,
                 use_api_float=True, verbose=True):

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
        exit_price = None
        if bar_data:
            session_bars = bar_data.get('rth', []) + bar_data.get('ah', [])
            entry_t = tc['entry_time']
            for t_str, bar in session_bars:
                if t_str > EXIT_TIME:
                    break
                if t_str >= entry_t:
                    exit_price = bar['c']

        if exit_price is not None:
            pnl = round((ep - exit_price) / ep * 100, 3)
            tc['pnl_eod'] = max(pnl, PNL_CAP)
        else:
            t0d = _gd_cache.get(dt, {}).get(tk, {})
            if t0d.get('c'):
                pnl = round((ep - t0d['c']) / ep * 100, 3)
                tc['pnl_eod'] = max(pnl, PNL_CAP)
            else:
                tc['pnl_eod'] = None

        tc['exit_price'] = exit_price
        t1d = _gd_cache.get(tc.get('t1_date', ''), {}).get(tk, {})
        tc['pnl_t1'] = round((ep - t1d['c']) / ep * 100, 3) if t1d.get('c') else None
        t2d = _gd_cache.get(tc.get('t2_date', ''), {}).get(tk, {})
        tc['pnl_t2'] = round((ep - t2d['c']) / ep * 100, 3) if t2d.get('c') else None
        return tc

    trades = []
    cnt = {'lh': 0, 'fb': 0, 'ne': 0, 'nb': 0}
    n_api_filtered = 0
    daily_picks = {}

    for dt in dates:
        cands = all_candidates.get(dt, [])

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
            if use_api_float and tk not in _FLOAT:
                api_shares = ref.get('shares')
                if api_shares is not None and api_shares < FLOAT_MIN:
                    n_api_filtered += 1
                    continue
            cc = copy.deepcopy(c)
            cc['asset_type'] = at
            cc['open_pattern'] = pattern
            cs_qualified.append(cc)

        daily_picks[dt] = [c['ticker'] for c in cs_qualified[:TOP_N]]

        for c in cs_qualified[:TOP_N]:
            trade = _build_trade(c)
            if trade:
                cnt['lh' if trade['entry_type'] == 'local_high' else 'fb'] += 1
                trades.append(trade)
            elif (c['ticker'], c['scan_date']) not in all_bars:
                cnt['nb'] += 1
            else:
                cnt['ne'] += 1

    valid_trades = [t for t in trades if t.get('pnl_eod') is not None]

    if verbose:
        lbl = "NEW (API float)" if use_api_float else "OLD (dict only)"
        print(f"\n  {lbl}: {len(valid_trades)} trades  "
              f"(lh={cnt['lh']} fb={cnt['fb']} ne={cnt['ne']} nb={cnt['nb']}"
              f"{f' api_filtered={n_api_filtered}' if use_api_float else ''})")

    return valid_trades, daily_picks, n_api_filtered


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD RESULT
# ═══════════════════════════════════════════════════════════════════════════════

def build_result(trades, dates, label):
    pnls = [t['pnl_eod'] for t in trades]
    m = compute_metrics(pnls, label)

    daily_pnl = defaultdict(list)
    for t in trades:
        daily_pnl[t['scan_date']].append(t['pnl_eod'])

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

def print_results(res, dates):
    m = res['metrics']
    print(f"\n\n{'='*90}")
    print(f"  AGENT 1 — Gap-Fade SHORT CS+SD — {res['label']} — {dates[0]} → {dates[-1]}")
    print(f"  Exit: {EXIT_TIME} ET  |  PNL_CAP: {PNL_CAP}%  |  Capital: ${CAPITAL:,.0f}/day NC")
    print(f"{'='*90}")

    tbl = [
        ['Trades',        f"{m['n']}"],
        ['Active days',   f"{res['active_days']}"],
        ['Trades/day',    f"{res['tpd']:.1f}"],
        ['Avg PnL',       f"{m['avg']:+.2f}%"],
        ['Median PnL',    f"{m['med']:+.2f}%"],
        ['Trimmed Mean',  f"{m['trimmed_mean']:+.2f}%"],
        ['Win Rate',      f"{m['wr']*100:.0f}%"],
        ['Sharpe',        f"{m['sharpe']:.3f}"],
        ['Profit Factor', f"{m['pf']:.2f}"],
        ['95% CI',        f"[{m['ci_lo']:+.2f}, {m['ci_hi']:+.2f}]"],
        ['Total $ (NC)',  f"${res['cum_nc']:+,.0f}"],
        ['Total $ (CMP)', f"${res['final_capital']-CAPITAL:+,.0f}"],
        ['Max DD',        f"{res['max_dd']:.1f}%"],
    ]
    print()
    print(tabulate(tbl, headers=['Metric', 'Value'], tablefmt='simple', stralign='right'))

    print(f"\n  ── ENTRY TYPE ──")
    for lbl in ['local_high', 'fallback']:
        arr = [t['pnl_eod'] for t in res['trades'] if t['entry_type'] == lbl]
        if arr:
            a = np.array(arr)
            sh = np.mean(a) / np.std(a, ddof=1) if len(a) > 1 and np.std(a, ddof=1) > 0 else 0
            print(f"    {lbl:12s}  n={len(a):<4d}  avg={np.mean(a):+.2f}%  "
                  f"WR={float((a>0).mean())*100:.0f}%  Sharpe={sh:.3f}")

    print(f"\n  ── PNL BY HORIZON ──")
    for key, lbl in [('pnl_eod', 'EOD'), ('pnl_t1', 'T+1'), ('pnl_t2', 'T+2')]:
        vals = [t[key] for t in res['trades'] if t.get(key) is not None]
        if vals:
            a = np.array(vals)
            sh = np.mean(a) / np.std(a, ddof=1) if len(a) > 1 and np.std(a, ddof=1) > 0 else 0
            print(f"    {lbl:6s}  n={len(a):<4d}  avg={np.mean(a):+.2f}%  "
                  f"med={np.median(a):+.2f}%  WR={float((a>0).mean())*100:.0f}%  Sharpe={sh:.3f}")
    print()


def print_ab_comparison(res_old, res_new, dates, picks_old, picks_new):
    mo = res_old['metrics']
    mn = res_new['metrics']

    print(f"\n\n{'='*90}")
    print(f"  FLOAT FILTER A/B — {dates[0]} → {dates[-1]} ({len(dates)} trading days)")
    print(f"  OLD: CS + hardcoded _FLOAT only (unknown = pass)")
    print(f"  NEW: CS + API share_class_shares_outstanding fallback (matches live)")
    print(f"{'='*90}")

    headers = ['Metric', 'OLD (dict only)', 'NEW (live match)', 'Delta']
    rows = []
    def _row(name, vo, vn, fmt='+.2f', pct=False):
        so = f"{vo:{fmt.lstrip('+')}}" if not pct else f"{vo*100:.0f}%"
        sn = f"{vn:{fmt.lstrip('+')}}" if not pct else f"{vn*100:.0f}%"
        delta = vn - vo
        sd = f"{delta*100:+.0f}pp" if pct else f"{delta:{fmt}}"
        rows.append([name, so, sn, sd])

    _row('Trades', mo['n'], mn['n'], fmt='.0f')
    _row('Active days', res_old['active_days'], res_new['active_days'], fmt='.0f')
    _row('Avg PnL %', mo['avg'], mn['avg'])
    _row('Median PnL %', mo['med'], mn['med'])
    _row('Trimmed Mean %', mo['trimmed_mean'], mn['trimmed_mean'])
    _row('Win Rate', mo['wr'], mn['wr'], pct=True)
    _row('Sharpe', mo['sharpe'], mn['sharpe'], fmt='+.3f')
    _row('Profit Factor', mo['pf'], mn['pf'])
    _row('Total $ (NC)', res_old['cum_nc'], res_new['cum_nc'], fmt='+,.0f')
    _row('Max DD %', res_old['max_dd'], res_new['max_dd'])

    print()
    print(tabulate(rows, headers=headers, tablefmt='simple', stralign='right',
                   numalign='right'))

    # ── Days where picks differ ──
    diff_days = [dt for dt in dates
                 if picks_old.get(dt, []) != picks_new.get(dt, [])]
    n_diff = len(diff_days)
    print(f"\n  ── PICK DIFFERENCES ──")
    print(f"  Same pick: {len(dates) - n_diff} days  |  Different pick: {n_diff} days "
          f"({n_diff/len(dates)*100:.1f}%)")

    if diff_days:
        old_by_dt = {t['scan_date']: t for t in res_old['trades']}
        new_by_dt = {t['scan_date']: t for t in res_new['trades']}
        diff_rows = []
        old_diff_pnls, new_diff_pnls = [], []
        for dt in diff_days:
            ot = old_by_dt.get(dt)
            nt = new_by_dt.get(dt)
            otk = ot['ticker'] if ot else '-'
            ntk = nt['ticker'] if nt else '-'
            opnl = ot['pnl_eod'] if ot and ot.get('pnl_eod') is not None else None
            npnl = nt['pnl_eod'] if nt and nt.get('pnl_eod') is not None else None
            opnl_s = f"{opnl:+.2f}%" if opnl is not None else "N/A"
            npnl_s = f"{npnl:+.2f}%" if npnl is not None else "N/A"
            old_ref = _ticker_ref.get(otk, {})
            api_s = old_ref.get('shares')
            note = f" ({api_s/1e6:.1f}M)" if otk != '-' and otk not in _FLOAT and api_s is not None else ''
            diff_rows.append([dt, f"{otk}{note}", opnl_s, ntk, npnl_s])
            if opnl is not None: old_diff_pnls.append(opnl)
            if npnl is not None: new_diff_pnls.append(npnl)

        print()
        print(tabulate(diff_rows,
                       headers=['Date', 'OLD pick', 'OLD PnL', 'NEW pick', 'NEW PnL'],
                       tablefmt='simple', stralign='right'))

        if old_diff_pnls and new_diff_pnls:
            print(f"\n  On differing days:")
            print(f"    OLD avg: {np.mean(old_diff_pnls):+.2f}%  "
                  f"(n={len(old_diff_pnls)}, WR={np.mean(np.array(old_diff_pnls)>0)*100:.0f}%)")
            print(f"    NEW avg: {np.mean(new_diff_pnls):+.2f}%  "
                  f"(n={len(new_diff_pnls)}, WR={np.mean(np.array(new_diff_pnls)>0)*100:.0f}%)")
            print(f"    Δ avg:   {np.mean(new_diff_pnls) - np.mean(old_diff_pnls):+.2f}%")

    # ── Monthly breakdown ──
    print(f"\n  ── MONTHLY BREAKDOWN ──")
    months = sorted(set(dt[:7] for dt in dates))
    month_rows = []
    for mo_str in months:
        om = [t['pnl_eod'] for t in res_old['trades']
              if t['scan_date'].startswith(mo_str) and t.get('pnl_eod') is not None]
        nm = [t['pnl_eod'] for t in res_new['trades']
              if t['scan_date'].startswith(mo_str) and t.get('pnl_eod') is not None]
        md = sum(1 for dt in diff_days if dt.startswith(mo_str))
        month_rows.append([mo_str, len(om),
                          f"{np.mean(om):+.2f}%" if om else "—",
                          len(nm),
                          f"{np.mean(nm):+.2f}%" if nm else "—",
                          md])
    print()
    print(tabulate(month_rows,
                   headers=['Month', 'OLD n', 'OLD avg', 'NEW n', 'NEW avg', 'Diff days'],
                   tablefmt='simple', stralign='right'))
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results(res, dates, filename='agent1_baseline_bt.png'):
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        BASE_DIR = os.getcwd()

    color = '#2c3e50'
    m = res['metrics']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f'Agent 1 — Gap-Fade SHORT ({res["label"]}): {dates[0]} → {dates[-1]}\n'
        f'{m["n"]} trades  |  Avg={m["avg"]:+.2f}%  |  WR={m["wr"]*100:.0f}%  |  '
        f'Sharpe={m["sharpe"]:.3f}  |  ${res["cum_nc"]:+,.0f} NC',
        fontsize=10, fontweight='bold')

    ax = axes[0]
    ds = res['daily_stats']
    d_dates = [datetime.strptime(s['date'], '%Y-%m-%d') for s in ds]
    cum = [s['cum_pnl'] for s in ds]
    ax.plot(d_dates, cum, color=color, linewidth=2, label=f'${res["cum_nc"]:+,.0f}')
    ax.axhline(0, color='#333', lw=0.8, ls=':')
    ax.set_title('Cumulative P&L (non-compounding)')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    pnls = [t['pnl_eod'] for t in res['trades']]
    if pnls:
        ax.hist(pnls, bins=max(10, len(pnls)//2), color=color, alpha=0.6,
                label=f'Avg={np.mean(pnls):+.2f}%')
        ax.axvline(np.mean(pnls), color=color, ls='--', lw=2)
    ax.axvline(0, color='#333', lw=1.2)
    ax.set_xlabel('EOD PnL (%)')
    ax.set_title('PnL Distribution')
    ax.legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    p1 = os.path.join(BASE_DIR, filename)
    plt.savefig(p1, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n  Saved: {p1}")


def plot_ab_comparison(res_old, res_new, dates, picks_old, picks_new):
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        BASE_DIR = os.getcwd()

    c_old, c_new = '#e74c3c', '#2980b9'

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        f'Agent 1 — Float Filter A/B: {dates[0]} → {dates[-1]}\n'
        f'OLD (dict only) vs NEW (live match)  |  {len(dates)} trading days',
        fontsize=12, fontweight='bold')

    # 1. Cumulative P&L overlay
    ax = axes[0, 0]
    d_dates = [datetime.strptime(s['date'], '%Y-%m-%d') for s in res_old['daily_stats']]
    cum_old = [s['cum_pnl'] for s in res_old['daily_stats']]
    cum_new = [s['cum_pnl'] for s in res_new['daily_stats']]
    ax.plot(d_dates, cum_old, color=c_old, lw=1.5, alpha=0.8,
            label=f'OLD ${res_old["cum_nc"]:+,.0f}')
    ax.plot(d_dates, cum_new, color=c_new, lw=1.5, alpha=0.8,
            label=f'NEW ${res_new["cum_nc"]:+,.0f}')
    ax.axhline(0, color='#333', lw=0.8, ls=':')
    ax.set_title('Cumulative P&L (non-compounding)')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # 2. Cumulative delta
    ax = axes[0, 1]
    cum_delta = [n - o for o, n in zip(cum_old, cum_new)]
    ax.fill_between(d_dates, cum_delta, 0,
                    where=[d >= 0 for d in cum_delta], color=c_new, alpha=0.3)
    ax.fill_between(d_dates, cum_delta, 0,
                    where=[d < 0 for d in cum_delta], color=c_old, alpha=0.3)
    ax.plot(d_dates, cum_delta, color='#2c3e50', lw=1.5)
    ax.axhline(0, color='#333', lw=1)
    ax.set_title('Cumulative Δ (NEW − OLD)')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:+,.0f}'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.grid(True, alpha=0.3)

    # 3. PnL distributions
    ax = axes[1, 0]
    pnl_old = [t['pnl_eod'] for t in res_old['trades']]
    pnl_new = [t['pnl_eod'] for t in res_new['trades']]
    lo = min(min(pnl_old, default=0), min(pnl_new, default=0)) - 1
    hi = max(max(pnl_old, default=0), max(pnl_new, default=0)) + 1
    bins = np.linspace(lo, hi, 40)
    if pnl_old:
        ax.hist(pnl_old, bins=bins, color=c_old, alpha=0.5,
                label=f'OLD avg={np.mean(pnl_old):+.2f}%')
        ax.axvline(np.mean(pnl_old), color=c_old, ls='--', lw=2)
    if pnl_new:
        ax.hist(pnl_new, bins=bins, color=c_new, alpha=0.5,
                label=f'NEW avg={np.mean(pnl_new):+.2f}%')
        ax.axvline(np.mean(pnl_new), color=c_new, ls='--', lw=2)
    ax.axvline(0, color='#333', lw=1)
    ax.set_xlabel('EOD PnL (%)')
    ax.set_title('PnL Distribution')
    ax.legend(fontsize=9)

    # 4. Monthly avg PnL bars
    ax = axes[1, 1]
    months = sorted(set(dt[:7] for dt in dates))
    old_mo = [np.mean([t['pnl_eod'] for t in res_old['trades']
              if t['scan_date'].startswith(m) and t.get('pnl_eod') is not None] or [0])
              for m in months]
    new_mo = [np.mean([t['pnl_eod'] for t in res_new['trades']
              if t['scan_date'].startswith(m) and t.get('pnl_eod') is not None] or [0])
              for m in months]
    x = np.arange(len(months)); w = 0.35
    ax.bar(x - w/2, old_mo, w, color=c_old, alpha=0.7, label='OLD')
    ax.bar(x + w/2, new_mo, w, color=c_new, alpha=0.7, label='NEW')
    ax.set_xticks(x); ax.set_xticklabels([m[5:] for m in months])
    ax.axhline(0, color='#333', lw=0.8)
    ax.set_ylabel('Avg PnL %'); ax.set_title('Monthly Avg PnL')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    p1 = os.path.join(BASE_DIR, 'agent1_baseline_bt_ab.png')
    plt.savefig(p1, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n  Saved: {p1}")

    # Scatter plot for diff days
    diff_days = [dt for dt in dates
                 if picks_old.get(dt, []) != picks_new.get(dt, [])]
    if len(diff_days) >= 3:
        old_map = {t['scan_date']: t['pnl_eod'] for t in res_old['trades']
                   if t.get('pnl_eod') is not None}
        new_map = {t['scan_date']: t['pnl_eod'] for t in res_new['trades']
                   if t.get('pnl_eod') is not None}
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        for dt in diff_days:
            op, np_ = old_map.get(dt), new_map.get(dt)
            if op is not None and np_ is not None:
                ax2.scatter(op, np_, color=c_new if np_ > op else c_old,
                           alpha=0.6, s=40, edgecolors='white', lw=0.5)
        lims = ax2.get_xlim()
        ax2.plot(lims, lims, 'k--', alpha=0.3, lw=1)
        ax2.axhline(0, color='#999', lw=0.5); ax2.axvline(0, color='#999', lw=0.5)
        ax2.set_xlabel('OLD PnL %'); ax2.set_ylabel('NEW PnL %')
        ax2.set_title(f'Diff-Day PnL: OLD vs NEW ({len(diff_days)} days)\nAbove diagonal = NEW wins')
        ax2.set_aspect('equal', adjustable='datalim'); ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        p2 = os.path.join(BASE_DIR, 'agent1_baseline_bt_scatter.png')
        plt.savefig(p2, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  Saved: {p2}")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

all_candidates, all_bars, pattern_cache, entry_cache, dates, timings, t_total = run_backtest()

if AB_COMPARE:
    print(f"\n{'='*80}")
    print(f"  Phase 6: Building trades — OLD vs NEW")
    print(f"{'='*80}")
    trades_old, picks_old, _ = build_trades(
        all_candidates, all_bars, pattern_cache, entry_cache, dates, use_api_float=False)
    trades_new, picks_new, n_filt = build_trades(
        all_candidates, all_bars, pattern_cache, entry_cache, dates, use_api_float=True)

    res_old = build_result(trades_old, dates, 'OLD (dict only)')
    res_new = build_result(trades_new, dates, 'NEW (live match)')

    print_results(res_old, dates)
    print_results(res_new, dates)
    print_ab_comparison(res_old, res_new, dates, picks_old, picks_new)
    plot_ab_comparison(res_old, res_new, dates, picks_old, picks_new)
else:
    trades, picks, _ = build_trades(
        all_candidates, all_bars, pattern_cache, entry_cache, dates, use_api_float=True)
    result = build_result(trades, dates, 'CS+SD SHORT 17:00 exit')
    print_results(result, dates)
    plot_results(result, dates)

print(f"\n  Total runtime: {time.time() - t_total:.0f}s ({(time.time() - t_total)/60:.1f} min)")
