# %% Agent 1_2 — Project X SHORT Backtest (H006 + DEEP4 Proximity)
#
# 100% aligned with agent1_2.py live scanner logic:
#   Universe:  _FLOAT_SET only (float >= 5M, miss=EXCLUDE)
#   Gap:       >= 2%, <= 75%, vs prior close (no AH freshness, no re-rank)
#   Price:     >= $5.00
#   Asset:     CS only (Polygon reference, 8 non-CS types excluded)
#   Pattern:   sustained_drop (classified at bar 60)
#   Entry:     local_high (3-bar lookback, bars 3-72), proximity <= 3%
#              NO fallback — quality signals only
#   Exit:      17:00 ET (2:00 PM PT) — same-day intraday
#   Top N:     5 per day, ranked by gap %
#   Capital:   $100,000/signal, equal weight, non-compounding
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
# CONFIG — matched to agent1_2.py
# ═══════════════════════════════════════════════════════════════════════════════

BACKTEST_START = '2025-09-01'
BACKTEST_END   = '2026-10-10'

GAP_MIN_PCT    = 2.0           # 2% threshold (agent1_2: GAP_MIN=0.02)
GAP_MAX_PCT    = 75.0          # 75% cap (agent1_2: gap_pct > 0.75 excluded)
PRICE_MIN      = 5.0           # $5 minimum (agent1_2: PRICE_MIN=5.0)
FLOAT_MIN      = 5_000_000     # 5M shares (agent1_2: FLOAT_MIN_SHARES=5_000_000)
TOP_N          = 5             # 5 signals/day (agent1_2: TOP_N=5)
PRE_FILTER_N   = 40            # wider funnel (agent1_2: PRE_FILTER_N=40)

PEAK_WINDOW    = 3             # 3-bar lookback (agent1_2: PEAK_WINDOW=3)
MIN_ENTRY_BAR  = 3             # earliest entry bar (agent1_2: MIN_ENTRY_BAR=3)
MAX_PEAK_BAR   = 72            # search bars 3-72 (agent1_2: MAX_PEAK_BAR=72)
PROX_MAX       = 0.03          # within 3% of running high (agent1_2: PROX_MAX=0.03)

EXIT_TIME      = '17:00'       # 17:00 ET exit (agent1_2: EXIT_TIME_ET='17:00')
CAPITAL        = 100_000       # per-signal sizing (agent1_2: POSITION_SIZE=100_000)

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs")
_BASE = "https://api.polygon.io"
_TK_RE = re.compile(r'^[A-Z]{1,5}$')
ET = pytz.timezone('America/New_York')

_EXCLUDE_TK = {'SPY','QQQ','IWM','DIA','VXX','UVXY','SQQQ','TQQQ','SPXU','SPXL',
               'SDOW','UDOW','SDS','SSO','SH','LABU','LABD','SOXL','SOXS',
               'FNGU','FNGD','NUGT','DUST','JNUG','JDST','TNA','TZA',
               'UPRO','VIXY','SVXY','TVIX'}
_EXCLUDE_SUFFIX = ('W', 'WS', 'U', 'R')

_NON_CS_TYPES = {'ETF', 'ETV', 'ETN', 'ETP', 'ETS', 'ADRC', 'FUND', 'PFD'}

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
_FLOAT_SET = {tk for tk, fs in _FLOAT.items() if fs >= FLOAT_MIN}


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
    sem = asyncio.Semaphore(10)
    async with aiohttp.ClientSession() as session:
        tasks = [_fetch_bars_one(session, tk, dt, sem) for tk, dt in pairs]
        results = await asyncio.gather(*tasks)
    return {key: val for key, val in results if val is not None}

def fetch_minute_bars_batch(pairs):
    if not pairs: return {}
    return asyncio.run(_fetch_bars_batch(pairs))


# ═══════════════════════════════════════════════════════════════════════════════
# ASYNC TICKER REFERENCE FETCH (asset type — CS filter)
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
            }
        return tk, {}

async def _fetch_refs_batch(tickers):
    sem = asyncio.Semaphore(10)
    async with aiohttp.ClientSession() as session:
        tasks = [_fetch_ref_one(session, tk, sem) for tk in tickers]
        results = await asyncio.gather(*tasks)
    return {tk: val for tk, val in results}

def fetch_ticker_refs(tickers):
    uncached = [tk for tk in tickers if tk not in _ticker_ref]
    if uncached:
        batch = asyncio.run(_fetch_refs_batch(uncached))
        _ticker_ref.update(batch)


def _is_cs(tk):
    ref = _ticker_ref.get(tk, {})
    asset_type = ref.get('type', '')
    if not asset_type:
        return False
    return asset_type not in _NON_CS_TYPES


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN CLASSIFICATION — sustained_drop at bar 60
# (identical to agent1_2.py:classify_open_pattern)
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
# LOCAL HIGH + PROXIMITY FILTER
# (identical to agent1_2.py:find_local_high_with_proximity)
# No fallback — returns None if no qualifying local high found.
# ═══════════════════════════════════════════════════════════════════════════════

def find_local_high_with_proximity(rth_bars, window=PEAK_WINDOW,
                                    min_bar=MIN_ENTRY_BAR, max_bar=MAX_PEAK_BAR):
    if len(rth_bars) < window * 2 + 1:
        return None

    start = max(window, min_bar)
    for i in range(start, min(max_bar, len(rth_bars) - window)):
        ch = rth_bars[i][1]['h']
        if all(rth_bars[i - j][1]['h'] < ch and rth_bars[i + j][1]['h'] < ch
               for j in range(1, window + 1)):
            confirm_idx = i + window
            if confirm_idx >= len(rth_bars):
                continue
            entry_price = rth_bars[confirm_idx][1]['c']
            entry_time = rth_bars[confirm_idx][0]

            running_high = max(b[1]['h'] for b in rth_bars[:confirm_idx + 1])
            proximity = (running_high - entry_price) / running_high if running_high > 0 else 1.0
            drop_from_open = (rth_bars[0][1]['o'] - entry_price) / rth_bars[0][1]['o'] if rth_bars[0][1]['o'] > 0 else 0

            if proximity > PROX_MAX:
                continue

            return {
                'entry_price': round(entry_price, 4),
                'entry_time': entry_time,
                'entry_type': 'local_high',
                'peak_bar': i,
                'proximity': round(proximity, 4),
                'running_high': round(running_high, 4),
                'drop_from_open': round(drop_from_open, 4),
            }

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# EXIT PRICE — identical to agent1_2.py:get_exit_price
# ═══════════════════════════════════════════════════════════════════════════════

def get_exit_price(bar_data, exit_time=EXIT_TIME):
    ah = bar_data.get('ah', [])
    rth = bar_data.get('rth', [])

    if exit_time == '17:00' and ah:
        last_ah = None
        for t_str, bd in ah:
            if t_str <= '17:00':
                last_ah = bd['c']
        if last_ah is not None:
            return last_ah

    if ah:
        return ah[-1][1]['c']
    if rth:
        return rth[-1][1]['c']
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# SCANNER — aligned with agent1_2.py:compute_gaps
# No AH freshness, no gap erosion, no scan-time re-rank.
# Gap = daily open vs prior close. Float must be in _FLOAT_SET (miss=EXCLUDE).
# ═══════════════════════════════════════════════════════════════════════════════

def scan_day(dt_str):
    if not _is_td(dt_str): return []
    prior_str = _prior_td(dt_str).strftime('%Y-%m-%d')
    prior = fetch_grouped_daily(prior_str)
    today = fetch_grouped_daily(dt_str)

    gap_min = GAP_MIN_PCT / 100.0
    gap_max = GAP_MAX_PCT / 100.0
    candidates = []
    for tk in today:
        if tk in _EXCLUDE_TK: continue
        if any(tk.endswith(s) for s in _EXCLUDE_SUFFIX) and len(tk) > 2: continue
        if tk not in _FLOAT_SET: continue
        if tk not in prior: continue

        prev_c = prior[tk]['c']
        if prev_c <= 0: continue
        cur_o = today[tk]['o']
        if cur_o <= 0 or cur_o < PRICE_MIN: continue

        gap = (cur_o - prev_c) / prev_c
        if gap < gap_min: continue
        if gap > gap_max: continue

        candidates.append({
            'ticker': tk, 'gap_pct': gap, 'prior_close': prev_c,
            'open_price': cur_o, 'scan_date': dt_str,
            'volume_today': today[tk].get('v', 0),
        })

    candidates.sort(key=lambda x: -x['gap_pct'])
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
    print(f"  AGENT 1_2 — Project X SHORT (H006 + DEEP4 Proximity)")
    print(f"  {BACKTEST_START} → {BACKTEST_END}")
    print(f"  Universe: _FLOAT_SET only (float >= {FLOAT_MIN/1e6:.0f}M, miss=EXCLUDE)")
    print(f"  Gap: >= {GAP_MIN_PCT}%, <= {GAP_MAX_PCT}%, vs prior close")
    print(f"  Price: >= ${PRICE_MIN:.2f}  |  Asset: CS only (8 non-CS types excluded)")
    print(f"  Pattern: sustained_drop required (classified at bar 60)")
    print(f"  Entry: local_high (3-bar, bars {MIN_ENTRY_BAR}-{MAX_PEAK_BAR}), prox <= {PROX_MAX*100:.0f}%")
    print(f"  NO fallback — quality signals only")
    print(f"  Exit: {EXIT_TIME} ET (2:00 PM PT)  |  Direction: SHORT")
    print(f"  Capital: ${CAPITAL:,.0f}/signal, top {TOP_N}/day, non-compounding")
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
    print(f"\n  Phase 1: Scanning for gap-ups (_FLOAT_SET universe)...")
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

    # ── Phase 3: Fetch ticker reference data (CS filter) ──
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
    n_cs = sum(1 for tk in all_tickers if _is_cs(tk))
    n_non_cs = len(all_tickers) - n_cs
    print(f"  → {n_cs} CS, {n_non_cs} non-CS excluded")

    # ── Phase 4: Classify patterns + find entries ──
    t4 = time.time()
    pattern_cache = {}
    entry_cache = {}
    for key, bar_data in all_bars.items():
        rth_bars = bar_data.get('rth', []) if isinstance(bar_data, dict) else bar_data
        pattern_cache[key] = classify_open_pattern(rth_bars)
        if rth_bars and pattern_cache[key] == 'sustained_drop':
            entry_cache[key] = find_local_high_with_proximity(rth_bars)
        else:
            entry_cache[key] = None
    timings['classify'] = time.time() - t4

    pat_counts = defaultdict(int)
    for p in pattern_cache.values():
        pat_counts[p] += 1
    print(f"\n  Phase 4: Pattern classification")
    for pat in ['sustained_drop', 'drop_then_recover', 'mild_drop', 'flat_or_bounce', 'no_data']:
        ct = pat_counts.get(pat, 0)
        pct = ct / len(pattern_cache) * 100 if pattern_cache else 0
        print(f"    {pat:25s}  {ct:>5d}  ({pct:.1f}%)")

    elapsed = time.time() - t_total
    print(f"\n  Pipeline complete: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    return all_candidates, all_bars, pattern_cache, entry_cache, dates, timings, t_total


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD TRADES
# Mirrors agent1_2.py:run_scan Phase B logic exactly:
#   1. CS filter (8 non-CS types excluded)
#   2. sustained_drop pattern required
#   3. local_high with proximity <= 3%, no fallback
#   4. Top N by gap, ranked
# ═══════════════════════════════════════════════════════════════════════════════

def build_trades(all_candidates, all_bars, pattern_cache, entry_cache, dates,
                 verbose=True):

    trades = []
    cnt = {'signal': 0, 'no_entry': 0, 'not_sd': 0, 'not_cs': 0, 'no_bars': 0}
    daily_picks = {}
    funnel_log = []

    for dt in dates:
        cands = all_candidates.get(dt, [])
        day_signals = []

        for c in cands[:PRE_FILTER_N]:
            tk = c['ticker']

            if not _is_cs(tk):
                cnt['not_cs'] += 1
                continue

            pattern = pattern_cache.get((tk, dt), 'no_data')

            if pattern != 'sustained_drop':
                cnt['not_sd'] += 1
                continue

            entry = entry_cache.get((tk, dt))
            if entry is None:
                cnt['no_entry'] += 1
                continue

            tc = copy.deepcopy(c)
            tc['entry_price'] = entry['entry_price']
            tc['entry_time'] = entry['entry_time']
            tc['entry_type'] = entry['entry_type']
            tc['peak_bar'] = entry['peak_bar']
            tc['proximity'] = entry['proximity']
            tc['drop_from_open'] = entry['drop_from_open']
            tc['open_pattern'] = pattern

            bar_data = all_bars.get((tk, dt))
            exit_price = get_exit_price(bar_data) if bar_data else None

            ep = tc['entry_price']
            if exit_price is not None:
                tc['pnl_eod'] = round((ep - exit_price) / ep * 100, 3)
            else:
                t0d = _gd_cache.get(dt, {}).get(tk, {})
                if t0d.get('c'):
                    tc['pnl_eod'] = round((ep - t0d['c']) / ep * 100, 3)
                else:
                    tc['pnl_eod'] = None

            tc['exit_price'] = exit_price
            day_signals.append(tc)

        day_signals.sort(key=lambda x: -x['gap_pct'])
        selected = day_signals[:TOP_N]
        for i, s in enumerate(selected):
            s['rank'] = i + 1
        daily_picks[dt] = [s['ticker'] for s in selected]
        trades.extend(selected)
        cnt['signal'] += len(selected)

        n_sd = sum(1 for c in cands[:PRE_FILTER_N]
                   if pattern_cache.get((c['ticker'], dt)) == 'sustained_drop'
                   and _is_cs(c['ticker']))
        funnel_log.append((dt, len(cands), n_sd, len(selected)))

    valid_trades = [t for t in trades if t.get('pnl_eod') is not None]

    if verbose:
        print(f"\n  Trades: {len(valid_trades)} signals")
        print(f"    Funnel: {cnt['not_cs']} non-CS, {cnt['not_sd']} not sustained_drop, "
              f"{cnt['no_entry']} no qualifying entry, {cnt['signal']} fired")

    return valid_trades, daily_picks, funnel_log


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD RESULT
# ═══════════════════════════════════════════════════════════════════════════════

def build_result(trades, dates, label):
    pnls = [t['pnl_eod'] for t in trades]
    m = compute_metrics(pnls, label)

    daily_pnl = defaultdict(list)
    for t in trades:
        daily_pnl[t['scan_date']].append(t['pnl_eod'])

    cum_nc = 0.0
    daily_stats = []
    for dt in dates:
        dp = daily_pnl.get(dt, [])
        if dp:
            n = len(dp)
            day_dollar = sum(CAPITAL * (r / 100) for r in dp)
        else:
            day_dollar = 0.0
            n = 0
        cum_nc += day_dollar
        daily_stats.append({'date': dt, 'n_trades': n,
                            'day_pnl_dollar': day_dollar, 'cum_pnl': cum_nc})

    equity = [CAPITAL]
    for ds in daily_stats:
        equity.append(equity[-1] + ds['day_pnl_dollar'])

    eq_arr = np.array(equity)
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / peak * 100
    max_dd = float(dd.min())
    active_days = sum(1 for ds in daily_stats if ds['n_trades'] > 0)
    tpd = len(trades) / active_days if active_days > 0 else 0

    # Monthly breakdown
    monthly = defaultdict(list)
    for t in trades:
        mo = t['scan_date'][:7]
        monthly[mo].append(t['pnl_eod'])

    return {
        'label': label, 'trades': trades, 'metrics': m,
        'equity': equity, 'daily_stats': daily_stats,
        'max_dd': max_dd, 'cum_nc': cum_nc,
        'active_days': active_days, 'tpd': tpd,
        'monthly': monthly,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

def print_results(res, dates):
    m = res['metrics']
    print(f"\n\n{'='*90}")
    print(f"  AGENT 1_2 — Project X SHORT — {res['label']}")
    print(f"  {dates[0]} → {dates[-1]}")
    print(f"  Gap >= {GAP_MIN_PCT}%, <= {GAP_MAX_PCT}%  |  Float >= {FLOAT_MIN/1e6:.0f}M (miss=EXCL)")
    print(f"  Pattern: sustained_drop  |  Entry: LH prox <= {PROX_MAX*100:.0f}%, no FB")
    print(f"  Exit: {EXIT_TIME} ET  |  Capital: ${CAPITAL:,.0f}/signal NC")
    print(f"{'='*90}")

    tbl = [
        ['Trades',        f"{m['n']}"],
        ['Active days',   f"{res['active_days']}"],
        ['Trades/day',    f"{res['tpd']:.2f}"],
        ['Avg PnL',       f"{m['avg']:+.2f}%"],
        ['Median PnL',    f"{m['med']:+.2f}%"],
        ['Trimmed Mean',  f"{m['trimmed_mean']:+.2f}%"],
        ['Win Rate',      f"{m['wr']*100:.0f}%"],
        ['Sharpe',        f"{m['sharpe']:.3f}"],
        ['Profit Factor', f"{m['pf']:.2f}"],
        ['95% CI',        f"[{m['ci_lo']:+.2f}, {m['ci_hi']:+.2f}]"],
        ['Total $ (NC)',  f"${res['cum_nc']:+,.0f}"],
        ['Max DD',        f"{res['max_dd']:.1f}%"],
    ]
    print()
    print(tabulate(tbl, headers=['Metric', 'Value'], tablefmt='simple', stralign='right'))

    # ── Monthly breakdown ──
    print(f"\n  ── MONTHLY BREAKDOWN ──")
    monthly = res['monthly']
    mo_rows = []
    for mo in sorted(monthly.keys()):
        arr = np.array(monthly[mo])
        sh = np.mean(arr) / np.std(arr, ddof=1) if len(arr) > 1 and np.std(arr, ddof=1) > 0 else 0
        mo_rows.append([
            mo, len(arr), f"{np.mean(arr):+.2f}%", f"{np.median(arr):+.2f}%",
            f"{float((arr>0).mean())*100:.0f}%", f"{sh:.3f}",
            f"${CAPITAL * np.sum(arr) / 100:+,.0f}"
        ])
    print()
    print(tabulate(mo_rows, headers=['Month', 'N', 'Avg', 'Med', 'WR', 'Sharpe', '$PnL'],
                   tablefmt='simple', stralign='right'))

    # ── Proximity distribution ──
    print(f"\n  ── PROXIMITY DISTRIBUTION ──")
    proxs = [t.get('proximity', 0) for t in res['trades'] if t.get('proximity') is not None]
    if proxs:
        pa = np.array(proxs) * 100
        print(f"    Mean: {np.mean(pa):.1f}%  |  Med: {np.median(pa):.1f}%  |  "
              f"Min: {np.min(pa):.1f}%  |  Max: {np.max(pa):.1f}%")
        for lo, hi in [(0, 1), (1, 2), (2, 3)]:
            mask = (pa >= lo) & (pa < hi)
            n_bin = mask.sum()
            if n_bin > 0:
                bin_pnls = np.array([t['pnl_eod'] for t, p in zip(res['trades'], proxs)
                                     if lo <= p*100 < hi])
                if len(bin_pnls) > 0:
                    print(f"    [{lo}-{hi}%): n={n_bin}  avg={np.mean(bin_pnls):+.2f}%  "
                          f"WR={float((bin_pnls>0).mean())*100:.0f}%")

    # ── Trade log ──
    print(f"\n  ── TRADE LOG (last 30) ──")
    recent = res['trades'][-30:]
    log_rows = []
    for t in recent:
        log_rows.append([
            t['scan_date'], t['ticker'],
            f"+{t['gap_pct']*100:.1f}%",
            f"${t['entry_price']:.2f}",
            t['entry_time'],
            f"{t.get('proximity',0)*100:.1f}%",
            f"{t['pnl_eod']:+.2f}%",
        ])
    print()
    print(tabulate(log_rows, headers=['Date', 'Ticker', 'Gap', 'Entry', 'Time', 'Prox', 'PnL'],
                   tablefmt='simple', stralign='right'))
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results(res, dates, filename='agent1_2_baseline_bt.png'):
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        BASE_DIR = os.getcwd()

    m = res['metrics']
    color_main = '#1a5276'
    color_pos = '#27ae60'
    color_neg = '#c0392b'

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f'Agent 1_2 — Project X SHORT (H006 + DEEP4)\n'
        f'{dates[0]} → {dates[-1]}  |  {m["n"]} trades  |  Avg={m["avg"]:+.2f}%  |  '
        f'WR={m["wr"]*100:.0f}%  |  Sharpe={m["sharpe"]:.3f}  |  ${res["cum_nc"]:+,.0f} NC',
        fontsize=11, fontweight='bold')

    # 1. Cumulative P&L
    ax = axes[0, 0]
    ds = res['daily_stats']
    d_dates = [datetime.strptime(s['date'], '%Y-%m-%d') for s in ds]
    cum = [s['cum_pnl'] for s in ds]
    ax.plot(d_dates, cum, color=color_main, linewidth=2)
    ax.fill_between(d_dates, cum, 0, where=[c >= 0 for c in cum],
                    color=color_pos, alpha=0.1)
    ax.fill_between(d_dates, cum, 0, where=[c < 0 for c in cum],
                    color=color_neg, alpha=0.1)
    ax.axhline(0, color='#333', lw=0.8, ls=':')
    ax.set_title('Cumulative P&L (non-compounding)')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.grid(True, alpha=0.3)

    # 2. PnL Distribution
    ax = axes[0, 1]
    pnls = [t['pnl_eod'] for t in res['trades']]
    if pnls:
        colors = [color_pos if p > 0 else color_neg for p in pnls]
        ax.hist(pnls, bins=max(10, len(pnls)//3), color=color_main, alpha=0.6,
                edgecolor='white', linewidth=0.5)
        ax.axvline(np.mean(pnls), color=color_main, ls='--', lw=2,
                   label=f'Avg={np.mean(pnls):+.2f}%')
        ax.axvline(np.median(pnls), color=color_pos, ls=':', lw=2,
                   label=f'Med={np.median(pnls):+.2f}%')
    ax.axvline(0, color='#333', lw=1.2)
    ax.set_xlabel('EOD PnL (%)')
    ax.set_title('PnL Distribution')
    ax.legend(fontsize=9)

    # 3. Monthly bar chart
    ax = axes[1, 0]
    monthly = res['monthly']
    months = sorted(monthly.keys())
    mo_avgs = [np.mean(monthly[mo]) for mo in months]
    mo_colors = [color_pos if a > 0 else color_neg for a in mo_avgs]
    ax.bar(range(len(months)), mo_avgs, color=mo_colors, alpha=0.7, edgecolor='white')
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels([m[-5:] for m in months], rotation=45, fontsize=8)
    ax.axhline(0, color='#333', lw=0.8, ls=':')
    ax.set_ylabel('Avg PnL %')
    ax.set_title('Monthly Average PnL')
    ax.grid(True, alpha=0.3, axis='y')
    for i, (mo, avg) in enumerate(zip(months, mo_avgs)):
        n = len(monthly[mo])
        ax.text(i, avg + 0.1 if avg >= 0 else avg - 0.3,
                f'n={n}', ha='center', fontsize=7, color='#555')

    # 4. Proximity vs PnL scatter
    ax = axes[1, 1]
    proxs = [t.get('proximity', 0) * 100 for t in res['trades']]
    pnl_vals = [t['pnl_eod'] for t in res['trades']]
    scatter_colors = [color_pos if p > 0 else color_neg for p in pnl_vals]
    ax.scatter(proxs, pnl_vals, c=scatter_colors, alpha=0.5, s=30, edgecolors='white', linewidth=0.3)
    ax.axhline(0, color='#333', lw=0.8, ls=':')
    ax.axvline(PROX_MAX * 100, color=color_neg, ls='--', lw=1, alpha=0.5, label=f'Max prox {PROX_MAX*100:.0f}%')
    ax.set_xlabel('Proximity to running high (%)')
    ax.set_ylabel('PnL (%)')
    ax.set_title('Proximity vs PnL')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    p = os.path.join(BASE_DIR, filename)
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n  Saved: {p}")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

all_candidates, all_bars, pattern_cache, entry_cache, dates, timings, t_total = run_backtest()

trades, daily_picks, funnel_log = build_trades(
    all_candidates, all_bars, pattern_cache, entry_cache, dates)

res = build_result(trades, dates, 'H006 + DEEP4 Proximity')
print_results(res, dates)
plot_results(res, dates)

# ── Funnel summary by month ──
print(f"\n  ── FUNNEL BY MONTH ──")
mo_funnel = defaultdict(lambda: [0, 0, 0])
for dt, n_gap, n_sd, n_sig in funnel_log:
    mo = dt[:7]
    mo_funnel[mo][0] += n_gap
    mo_funnel[mo][1] += n_sd
    mo_funnel[mo][2] += n_sig

funnel_rows = []
for mo in sorted(mo_funnel.keys()):
    g, s, sig = mo_funnel[mo]
    funnel_rows.append([mo, g, s, sig, f"{sig/s*100:.0f}%" if s > 0 else '0%'])
print()
print(tabulate(funnel_rows, headers=['Month', 'Gap-ups', 'Sust.Drop', 'Signals', 'SD→Sig%'],
               tablefmt='simple', stralign='right'))

print(f"\n  Total runtime: {time.time() - t_total:.0f}s ({(time.time() - t_total)/60:.1f} min)")
