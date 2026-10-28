# %% Agent 1 v2 — Multi-Strategy SHORT Gap-Fade Scanner
#
# Runs all 5 top strategies on the same candidate pool.
# Run between 9:33-9:41 AM ET (6:33-6:41 PST).
#
# Strategies (ranked by Post-Aug Sharpe):
#   1. C5: PM+Red      N=1  (PostSh 0.948, 7 trades, +14.64% avg)
#   2. C3+C5           N=1  (PostSh 0.849, 6 trades, +14.34% avg)
#   3. C7: PMTrend<0   N=1  (PostSh 0.801, 7 trades, +12.87% avg)
#   4. C9: 2BarDrop    N=2  (PostSh 0.628, 12 trades, +8.33% avg)
#   5. C3: SkipT6      N=3  (PostSh 0.343, 34 trades, +3.55% avg)
#
# Entry: local high (3-bar lookback + 3-bar confirmation, bars 3-11).
#        No fallback for filtered strategies — local high or skip.
# Exit:  17:00 ET same day.
# SL:    None (C5 worst = -3.12%, natural tail protection)
#
# Colab-cell friendly. Config vars at top.

import os, sys, time, re, asyncio
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

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from google.cloud import bigquery as bq_mod
except ImportError:
    bq_mod = None

try:
    from google.colab import auth
    try: auth.authenticate_user()
    except: pass
except: pass


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit before running
# ═══════════════════════════════════════════════════════════════════════════════

SCAN_TIMESTAMP = None       # PST — 'YYYY-MM-DD HH:MM' or None for live
DRY_RUN = False             # True = print only, skip nothing

GAP_MIN_PCT    = 3.0        # minimum gap %
PRICE_MIN      = 10.0       # minimum price $
FLOAT_MIN      = 10_000_000
PRE_FILTER_N   = 30         # walk top 30 by gap for CS check
GAP_CONSUMED_MAX = 0.50     # skip if >50% of gap eroded

SCAN_TIME_ET   = '09:31'
EXIT_TIME_ET   = '17:00'

MIN_RTH_BARS   = 2          # min RTH bars with volume > 0 by scan time
MIN_OPEN_VOL   = 5_000      # min shares in bar 0 (09:30)

PEAK_WINDOW    = 3
MAX_PEAK_BAR   = 12         # bars 3-11 (inclusive peak), entry at peak+3
FALLBACK_BAR   = None       # no fallback for filtered strategies

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs")

GCP_PROJECT = os.environ.get("GCP_PROJECT", "")
if not GCP_PROJECT:
    try:
        import google.auth
        _, GCP_PROJECT = google.auth.default()
    except Exception:
        pass
BQ_TABLE = f"{GCP_PROJECT}.scoring_logs.agent1_v2_scan_log" if GCP_PROJECT else ""
_BASE = "https://api.polygon.io"
_TK_RE = re.compile(r'^[A-Z]{1,5}$')

ET = pytz.timezone('America/New_York')
PT = pytz.timezone('America/Los_Angeles')

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
# STRATEGY DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

STRATEGIES = [
    {
        'id': 'C5',
        'name': 'C5: PM+Red',
        'optimal_n': 1,
        'post_sharpe': 0.948,
        'post_avg': 14.64,
        'post_trades': 7,
        'worst_loss': -3.12,
        'description': (
            'STRONGEST FILTER — highest Sharpe. Requires two pre-market '
            'confirmations: (1) PM retrace >= 8% — meaning the stock '
            'pulled back at least 8% from its PM high after 8:30 ET, '
            'showing sellers are active, and (2) bar0 is red (close < open '
            'at 9:30-9:31 ET) — the gap is already fading at the open.'
        ),
        'setup': (
            'Look for: Stock gapped up 3%+ but PM retrace shows selling '
            'pressure. First RTH candle is RED. This confirms the gap is '
            'fading immediately. Enter SHORT at local high confirmation '
            '(bars 3-11). Natural tail protection: worst loss only -3.12%.'
        ),
    },
    {
        'id': 'C3+C5',
        'name': 'C3+C5: SkipT6+PM+Red',
        'optimal_n': 1,
        'post_sharpe': 0.849,
        'post_avg': 14.34,
        'post_trades': 6,
        'worst_loss': -3.12,
        'description': (
            'Same as C5 but ALSO skips T6 gap tier (15-30% gaps). Slightly '
            'more conservative — avoids the extreme movers where PM+Red '
            'might not be enough. All C3+C5 trades are a subset of C5. '
            'C5 catches one extra winner (GRML +16.43%) that C3+C5 skips.'
        ),
        'setup': (
            'Same as C5 (PM retrace >= 8% + bar0 red) but SKIP if gap is '
            '15-30%. If C5 pick has a 15-30% gap, this strategy sits out. '
            'Use as a confirmation layer: if C3+C5 agrees with C5, higher '
            'confidence in the trade.'
        ),
    },
    {
        'id': 'C7',
        'name': 'C7: PMTrend<0',
        'optimal_n': 1,
        'post_sharpe': 0.801,
        'post_avg': 12.87,
        'post_trades': 7,
        'worst_loss': -3.12,
        'description': (
            'Independent PM signal — checks if PM close < PM open (after '
            '8:30 ET). The stock gapped up but the pre-market trend is '
            'NEGATIVE. Sellers are winning in PM. Different signal from C5 '
            '(retrace vs trend), so picks can diverge.'
        ),
        'setup': (
            'Look for: Stock gapped up 3%+ but between 8:30-9:30 ET the '
            'price TRENDED DOWN (last PM close < first PM open). The gap '
            'enthusiasm is fading before the market even opens. Enter SHORT '
            'at local high. This catches stocks C5 might miss if bar0 is green.'
        ),
    },
    {
        'id': 'C9',
        'name': 'C9: 2BarDrop',
        'optimal_n': 2,
        'post_sharpe': 0.628,
        'post_avg': 8.33,
        'post_trades': 12,
        'worst_loss': -3.18,
        'description': (
            'Early RTH confirmation — requires bar1 close < bar0 open. '
            'The stock opened, tried to hold, but by the end of bar 1 '
            '(9:32 ET) it is already below the opening price. More trades '
            'than C5/C7 because the filter is less restrictive.'
        ),
        'setup': (
            'Look for: After bar 1 closes (9:32 ET), check if bar1_close < '
            'bar0_open. If yes, the gap is already failing within 2 minutes. '
            'N=2 optimal — take the top 2 gap stocks that pass this filter. '
            'Enter SHORT at local high (bars 3-11).'
        ),
    },
    {
        'id': 'C3',
        'name': 'C3: SkipT6',
        'optimal_n': 3,
        'post_sharpe': 0.343,
        'post_avg': 3.55,
        'post_trades': 34,
        'worst_loss': -24.89,
        'description': (
            'WIDEST NET — only filter is skipping 15-30% gap tier (T6). '
            'These extreme gaps tend to be event-driven (M&A, FDA) and '
            'don\'t fade reliably. N=3 optimal. Most trades but lowest Sharpe. '
            'WARNING: worst loss -24.89% — needs position sizing discipline.'
        ),
        'setup': (
            'Look for: Any stock with 3%+ gap that is NOT in the 15-30% '
            'range. No PM filter, no bar confirmation. Pure gap-fade on '
            'the moderate tier. N=3 — take top 3 by gap size. Enter SHORT '
            'at local high. RISK: no tail protection, size accordingly.'
        ),
    },
]


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


# ═══════════════════════════════════════════════════════════════════════════════
# POLYGON API
# ═══════════════════════════════════════════════════════════════════════════════

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


def fetch_grouped_daily(dt_str):
    data = _pg(f"{_BASE}/v2/aggs/grouped/locale/us/market/stocks/{dt_str}",
               {"adjusted": "true", "include_otc": "false"})
    out = {}
    for r in data.get('results', []):
        tk = r.get('T', '')
        if not _TK_RE.match(tk): continue
        out[tk] = {'o': float(r.get('o', 0)), 'h': float(r.get('h', 0)),
                   'l': float(r.get('l', 0)), 'c': float(r.get('c', 0)),
                   'v': int(r.get('v', 0))}
    return out


def fetch_snapshot_all():
    url = f"{_BASE}/v2/snapshot/locale/us/markets/stocks/tickers"
    data = _pg(url, {"include_otc": "false"})
    tickers = data.get('tickers', [])
    out = {}
    for t in tickers:
        tk = t.get('ticker', '')
        if not _TK_RE.match(tk): continue
        prev_c = t.get('prevDay', {}).get('c')
        pm = t.get('min', {})
        pm_price = float(pm['c']) if pm and pm.get('c') and pm['c'] > 0 else None
        if pm_price is None and prev_c:
            pm_price = float(prev_c)
        if pm_price and pm_price > 0:
            out[tk] = {'price': pm_price, 'prev_close': float(prev_c) if prev_c else None}
    return out


async def _fetch_bars_one(session, tk, dt, sem):
    url = f"{_BASE}/v2/aggs/ticker/{tk}/range/1/minute/{dt}/{dt}"
    params = {"adjusted": "true", "sort": "asc", "limit": "1000", "apiKey": POLYGON_API_KEY}
    async with sem:
        for attempt in range(3):
            try:
                async with session.get(url, params=params,
                                       timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(2 ** (attempt + 1))
                        continue
                    if resp.status != 200:
                        return tk, None
                    data = await resp.json()
            except Exception:
                await asyncio.sleep(1)
                continue
            bars = data.get('results', [])
            if not bars:
                return tk, None
            pm, rth, ah = [], [], []
            for b in bars:
                ts = pd.Timestamp(b['t'], unit='ms', tz='UTC').tz_convert(ET)
                t_str = ts.strftime('%H:%M')
                bar_d = {'o': float(b['o']), 'h': float(b['h']),
                         'l': float(b['l']), 'c': float(b['c']), 'v': int(b['v'])}
                if '04:00' <= t_str < '09:30':
                    pm.append((t_str, bar_d))
                elif '09:30' <= t_str < '16:00':
                    rth.append((t_str, bar_d))
                elif '16:00' <= t_str <= '20:00':
                    ah.append((t_str, bar_d))
            return tk, {'pm': pm, 'rth': rth, 'ah': ah}
        return tk, None


async def _fetch_bars_batch(tickers, dt):
    sem = asyncio.Semaphore(15)
    async with aiohttp.ClientSession() as session:
        tasks = [_fetch_bars_one(session, tk, dt, sem) for tk in tickers]
        results = await asyncio.gather(*tasks)
    return {tk: data for tk, data in results if data is not None}


def fetch_minute_bars(tickers, dt):
    if not tickers: return {}
    return asyncio.run(_fetch_bars_batch(tickers, dt))


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
            return tk, {'type': res.get('type'), 'name': res.get('name', '')}
        return tk, {}


async def _fetch_refs_batch(tickers):
    sem = asyncio.Semaphore(15)
    async with aiohttp.ClientSession() as session:
        tasks = [_fetch_ref_one(session, tk, sem) for tk in tickers]
        results = await asyncio.gather(*tasks)
    return {tk: val for tk, val in results}


def fetch_ticker_refs(tickers):
    if not tickers: return {}
    return asyncio.run(_fetch_refs_batch(tickers))


def get_price_at_T(bar_data, T):
    if bar_data is None: return None
    all_bars = (bar_data.get('pm') or []) + (bar_data.get('rth') or [])
    best = None
    for t_str, bd in all_bars:
        if t_str <= T:
            best = bd['c']
        else:
            break
    return best


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY FILTERS
# ═══════════════════════════════════════════════════════════════════════════════

def check_c5(candidate, bar_data):
    """PM retrace >= 8% AND bar0 red."""
    if bar_data is None: return False
    pm_bars = bar_data.get('pm', [])
    rth_bars = bar_data.get('rth', [])
    prior_close = candidate.get('prior_close', 0)

    pm_after_830 = [(t, b) for t, b in pm_bars if t >= '08:30']
    if pm_after_830 and prior_close > 0:
        pm_high = max(b['h'] for _, b in pm_after_830)
        last_pm_close = pm_after_830[-1][1]['c']
        denom = pm_high - prior_close
        pm_retrace = (pm_high - last_pm_close) / denom if denom > 0 else 0
        if pm_retrace < 0.08:
            return False
    else:
        return False

    if rth_bars:
        bar0 = rth_bars[0][1]
        if bar0['c'] > bar0['o']:
            return False
    else:
        return False

    return True


def check_c3_c5(candidate, bar_data):
    """C5 filters + skip T6 gap tier (15-30%)."""
    gap = candidate.get('gap_pct', 0)
    if 0.15 <= gap < 0.30:
        return False
    return check_c5(candidate, bar_data)


def check_c7(candidate, bar_data):
    """PM close < PM open (after 8:30 ET)."""
    if bar_data is None: return False
    pm_bars = bar_data.get('pm', [])
    pm_after_830 = [(t, b) for t, b in pm_bars if t >= '08:30']
    if len(pm_after_830) >= 2:
        first_pm = pm_after_830[0][1]['o']
        last_pm = pm_after_830[-1][1]['c']
        pm_trend = (last_pm - first_pm) / first_pm if first_pm > 0 else 0
        return pm_trend < 0
    return False


def check_c9(candidate, bar_data):
    """bar1 close < bar0 open (2-bar drop)."""
    if bar_data is None: return False
    rth_bars = bar_data.get('rth', [])
    if len(rth_bars) >= 2:
        open_p = rth_bars[0][1]['o']
        bar1_close = rth_bars[1][1]['c']
        drop = (bar1_close - open_p) / open_p if open_p > 0 else 0
        return drop < 0
    return False


def check_c3(candidate, bar_data):
    """Skip T6 gap tier (15-30%) only."""
    gap = candidate.get('gap_pct', 0)
    return not (0.15 <= gap < 0.30)


FILTER_FNS = {
    'C5': check_c5,
    'C3+C5': check_c3_c5,
    'C7': check_c7,
    'C9': check_c9,
    'C3': check_c3,
}


# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL HIGH DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def find_local_high(rth_bars, window=PEAK_WINDOW, max_peak_bar=MAX_PEAK_BAR,
                    min_entry_bar=0):
    if len(rth_bars) < window * 2 + 1:
        return None
    start = max(window, min_entry_bar - window)
    for i in range(start, min(max_peak_bar, len(rth_bars) - window)):
        ch = rth_bars[i][1]['h']
        if all(rth_bars[i - j][1]['h'] < ch and rth_bars[i + j][1]['h'] < ch
               for j in range(1, window + 1)):
            ci = i + window
            if ci < len(rth_bars):
                return {
                    'entry_price': rth_bars[ci][1]['c'],
                    'entry_time': rth_bars[ci][0],
                    'entry_type': 'local_high',
                    'peak_bar': i,
                    'peak_time': rth_bars[i][0],
                }
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SCANNER
# ═══════════════════════════════════════════════════════════════════════════════

t_start = time.time()

if SCAN_TIMESTAMP is not None:
    parts = SCAN_TIMESTAMP.strip().split(' ', 1)
    scan_date = parts[0]
    mode = 'historical'
else:
    scan_date = datetime.now(ET).strftime('%Y-%m-%d')
    mode = 'live'

if not _is_td(scan_date):
    print(f"\n  {scan_date} is not a trading day. Exiting.")
    sys.exit(0) if mode != 'live' else None

prior_date = _prior_td(scan_date)
prior_str = prior_date.strftime('%Y-%m-%d')

now_et_dt = datetime.now(ET)
now_et = now_et_dt.strftime('%H:%M:%S ET')
now_pst = datetime.now(PT).strftime('%H:%M:%S PST')

now_et_hhmm = now_et_dt.strftime('%H:%M')
if mode == 'historical':
    scan_phase = 'HISTORICAL'
elif now_et_hhmm < '09:30':
    scan_phase = 'PRE-MARKET'
elif now_et_hhmm < '09:33':
    scan_phase = 'MARKET OPEN — bars forming'
elif now_et_hhmm < '09:44':
    scan_phase = 'ENTRY WINDOW — local highs may confirm'
else:
    scan_phase = 'POST-WINDOW — entries should be resolved'

print(f"\n{'='*80}")
print(f"  AGENT 1 v2 — MULTI-STRATEGY SHORT SCANNER")
print(f"  Date: {scan_date}  |  Mode: {mode}  |  Time: {now_pst}")
print(f"  Prior: {prior_str}  |  Exit: {EXIT_TIME_ET} ET")
print(f"  Phase: {scan_phase}")
print(f"{'='*80}")


# ── Step 1: Fetch prior closes ──────────────────────────────────────────────
print(f"\n  [1] Fetching prior closes ({prior_str})...")
prior_data = fetch_grouped_daily(prior_str)
print(f"      {len(prior_data)} tickers")

# ── Step 2: Fetch current prices ────────────────────────────────────────────
print(f"  [2] Fetching current prices...")
if mode == 'live':
    current_data = fetch_snapshot_all()
    print(f"      {len(current_data)} tickers (snapshot)")
else:
    current_data = fetch_grouped_daily(scan_date)
    print(f"      {len(current_data)} tickers (grouped daily)")

# SPY gap
spy_gap = None
if 'SPY' in prior_data and 'SPY' in current_data:
    spy_prev = prior_data['SPY']['c']
    spy_cur = current_data['SPY'].get('o', current_data['SPY'].get('price', 0))
    if spy_prev > 0 and spy_cur > 0:
        spy_gap = (spy_cur - spy_prev) / spy_prev * 100
        print(f"      SPY gap: {spy_gap:+.2f}%")


# ── Step 3: Gap filter ──────────────────────────────────────────────────────
print(f"  [3] Computing gaps (>= {GAP_MIN_PCT}%, >= ${PRICE_MIN})...")
gap_min = GAP_MIN_PCT / 100.0
candidates = []
for tk in current_data:
    if tk in _EXCLUDE_TK: continue
    if any(tk.endswith(s) for s in _EXCLUDE_SUFFIX) and len(tk) > 2: continue
    if tk not in prior_data: continue
    prev_c = prior_data[tk]['c']
    if prev_c <= 0: continue
    cur_p = current_data[tk].get('o', current_data[tk].get('price', 0))
    if cur_p <= 0 or cur_p < PRICE_MIN: continue
    gap = (cur_p - prev_c) / prev_c
    if gap < gap_min: continue
    fs = _FLOAT.get(tk)
    if fs is not None and fs < FLOAT_MIN: continue
    candidates.append({
        'ticker': tk, 'gap_pct': gap, 'prior_close': prev_c,
        'open_price': cur_p, 'float_shares': fs,
    })
candidates.sort(key=lambda x: -x['gap_pct'])
print(f"      {len(candidates)} candidates after gap+price+float")


# ── Step 4: AH reference + gap erosion ──────────────────────────────────────
print(f"  [4] Fetching minute bars (prior AH + scan day)...")
top_tickers = [c['ticker'] for c in candidates[:PRE_FILTER_N]]

prior_bars = fetch_minute_bars(top_tickers, prior_str) if top_tickers else {}
scan_bars = fetch_minute_bars(top_tickers, scan_date) if top_tickers else {}
print(f"      Prior AH: {len(prior_bars)}/{len(top_tickers)}  |  Scan day: {len(scan_bars)}/{len(top_tickers)}")

n_stale = 0
n_eroded = 0
qualified = []

for c in candidates[:PRE_FILTER_N]:
    tk = c['ticker']

    # AH reference price
    pb = prior_bars.get(tk)
    if pb:
        ah_bars = pb.get('ah', [])
        if ah_bars:
            ah_ref = ah_bars[-1][1]['c']
            c['ah_ref_price'] = ah_ref
            fresh_gap = (c['open_price'] - ah_ref) / ah_ref
            c['gap_pct'] = fresh_gap
            if fresh_gap < gap_min:
                n_stale += 1
                continue

    # Re-rank by gap at SCAN_TIME_ET
    sb = scan_bars.get(tk)
    if sb:
        c['bar_data'] = sb
        price_T = get_price_at_T(sb, SCAN_TIME_ET)
        if price_T is not None and mode == 'historical':
            ref_price = c.get('ah_ref_price', c['prior_close'])
            gap_T = (price_T - ref_price) / ref_price
            if gap_T < gap_min: continue
            if price_T < PRICE_MIN: continue
            c['gap_pct'] = gap_T
            c['scan_price'] = price_T

        # Gap erosion
        rth = sb.get('rth', [])
        if rth:
            daily_open = rth[0][1]['o']
            ref_price = c.get('ah_ref_price', c['prior_close'])
            scan_rth = [b for b in rth if b[0] <= SCAN_TIME_ET]
            gap_range = daily_open - ref_price
            if scan_rth and gap_range > 0:
                low_to_scan = min(b[1]['l'] for b in scan_rth)
                gap_consumed = (daily_open - low_to_scan) / gap_range
                c['gap_consumed'] = gap_consumed
                if gap_consumed > GAP_CONSUMED_MAX:
                    n_eroded += 1
                    continue

    qualified.append(c)

qualified.sort(key=lambda x: -x['gap_pct'])
print(f"      After AH+erosion: {len(qualified)} ({n_stale} stale, {n_eroded} eroded)")


# ── Step 5: CS type check ───────────────────────────────────────────────────
print(f"  [5] CS type check for top {min(PRE_FILTER_N, len(qualified))}...")
refs = fetch_ticker_refs([c['ticker'] for c in qualified[:PRE_FILTER_N]])
cs_pool = []
n_non_cs = 0
for c in qualified[:PRE_FILTER_N]:
    tk = c['ticker']
    ref = refs.get(tk, {})
    at = ref.get('type', 'UNK')
    c['asset_type'] = at
    c['name'] = ref.get('name', '')
    if at != 'CS':
        n_non_cs += 1
        continue
    cs_pool.append(c)
print(f"      {len(cs_pool)} CS stocks ({n_non_cs} non-CS excluded)")


# ── Step 5b: Liquidity filter (skip during pre-market — no RTH bars yet) ──
is_pre_entry = scan_phase in ('PRE-MARKET', 'MARKET OPEN — bars forming')
if is_pre_entry:
    print(f"  [5b] Liquidity: skipped (pre-market, no RTH bars yet)")
else:
    n_illiquid = 0
    liquid_pool = []
    for c in cs_pool:
        tk = c['ticker']
        bd = c.get('bar_data') or scan_bars.get(tk)
        if bd:
            rth = bd.get('rth', [])
            scan_rth = [(t, b) for t, b in rth if t <= SCAN_TIME_ET]
            bars_with_vol = sum(1 for _, b in scan_rth if b['v'] > 0)
            bar0_vol = rth[0][1]['v'] if rth else 0
            if bars_with_vol < MIN_RTH_BARS or bar0_vol < MIN_OPEN_VOL:
                n_illiquid += 1
                continue
        liquid_pool.append(c)
    cs_pool = liquid_pool
    print(f"  [5b] Liquidity: {n_illiquid} excluded (need {MIN_RTH_BARS} bars, {MIN_OPEN_VOL:,} vol)  →  {len(cs_pool)} remain")


# ── Step 6: Apply strategy filters ──────────────────────────────────────────
print(f"  [6] Applying 5 strategy filters...")

strategy_picks = {}

for strat in STRATEGIES:
    sid = strat['id']
    opt_n = strat['optimal_n']
    check_fn = FILTER_FNS[sid]
    show_n = max(opt_n, 2)

    picks = []
    for c in cs_pool:
        tk = c['ticker']
        bd = c.get('bar_data') or scan_bars.get(tk)
        if check_fn(c, bd):
            entry = None
            rth = bd.get('rth', []) if bd else []
            if rth and len(rth) >= PEAK_WINDOW * 2 + 1:
                entry = find_local_high(rth)
            picks.append({
                'ticker': tk,
                'gap_pct': c['gap_pct'],
                'open_price': c['open_price'],
                'prior_close': c['prior_close'],
                'float_shares': c.get('float_shares'),
                'name': c.get('name', ''),
                'entry': entry,
                'bar_data': bd,
            })
            if len(picks) >= show_n:
                break

    strategy_picks[sid] = picks
    status = f"{len(picks)} picks"
    if picks:
        tks = ', '.join(p['ticker'] for p in picks)
        status += f" — {tks}"
    print(f"      {strat['name']:25s}  N={opt_n}  →  {status}")


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

runtime = time.time() - t_start
print(f"\n  Pipeline: {runtime:.1f}s")

# Deduplicated ticker counts
all_unique = {}
for strat in STRATEGIES:
    for p in strategy_picks[strat['id']]:
        tk = p['ticker']
        if tk not in all_unique:
            all_unique[tk] = p

confirmed_tks = {tk for tk, p in all_unique.items() if p.get('entry')}
candidate_tks = {tk for tk, p in all_unique.items() if not p.get('entry')}
n_confirmed = len(confirmed_tks)
n_candidate = len(candidate_tks)
n_unique = len(all_unique)

ticker_strats = defaultdict(list)
for strat in STRATEGIES:
    for p in strategy_picks[strat['id']]:
        ticker_strats[p['ticker']].append(strat['id'])

any_picks = n_unique > 0


# ── RESULT SUMMARY (first thing you see) ──────────────────────────────────────

print(f"\n\n{'='*90}")
print(f"  RESULT — {scan_date}  |  {scan_phase}")
print(f"{'='*90}")

if not any_picks:
    if is_pre_entry:
        print(f"\n  No PM-filter candidates. C5/C9 need RTH bars — re-run after 09:32 ET.")
    else:
        print(f"\n  No qualifying stocks today. No action needed.")
elif n_confirmed > 0 and n_candidate == 0:
    print(f"\n  LOCAL HIGH: {n_confirmed} CONFIRMED")
    for tk in sorted(confirmed_tks):
        p = all_unique[tk]
        e = p['entry']
        strats = ', '.join(ticker_strats[tk])
        print(f"    {tk:6s}  gap={p['gap_pct']*100:+.1f}%  "
              f"entry=${e['entry_price']:.2f} @ {e['entry_time']} ET  "
              f"[{strats}]")
elif n_confirmed > 0:
    print(f"\n  LOCAL HIGH: {n_confirmed} CONFIRMED  |  {n_candidate} candidate pending")
    for tk in sorted(confirmed_tks):
        p = all_unique[tk]
        e = p['entry']
        strats = ', '.join(ticker_strats[tk])
        print(f"    CONFIRMED  {tk:6s}  gap={p['gap_pct']*100:+.1f}%  "
              f"entry=${e['entry_price']:.2f} @ {e['entry_time']} ET  "
              f"[{strats}]")
    for tk in sorted(candidate_tks):
        p = all_unique[tk]
        strats = ', '.join(ticker_strats[tk])
        print(f"    CANDIDATE  {tk:6s}  gap={p['gap_pct']*100:+.1f}%  "
              f"open=${p['open_price']:.2f}  — watch for local high  "
              f"[{strats}]")
else:
    print(f"\n  LOCAL HIGH: NONE confirmed")
    print(f"  {n_unique} unique candidate{'s' if n_unique > 1 else ''} — DO NOT enter yet")
    for tk in sorted(candidate_tks):
        p = all_unique[tk]
        strats = ', '.join(ticker_strats[tk])
        print(f"    {tk:6s}  gap={p['gap_pct']*100:+.1f}%  "
              f"open=${p['open_price']:.2f}  [{strats}]")

if is_pre_entry and any_picks:
    print(f"\n  C5/C3+C5/C9 not evaluated yet (need RTH bars).")
    print(f"  Re-run at 06:32 PST for full picture.")

if any_picks:
    print(f"\n  Exit: {EXIT_TIME_ET} ET  |  No SL for C5/C7  |  C3 size with discipline")


# ── STRATEGY DETAIL ───────────────────────────────────────────────────────────

print(f"\n\n{'='*90}")
print(f"  STRATEGY DETAIL")
print(f"{'='*90}")

for strat in STRATEGIES:
    sid = strat['id']
    picks = strategy_picks[sid]
    opt_n = strat['optimal_n']

    print(f"\n  {'─'*86}")
    print(f"  ┌ {strat['name']}  |  N={opt_n}  |  "
          f"PostSh={strat['post_sharpe']:.3f}  Avg={strat['post_avg']:+.2f}%  "
          f"Worst={strat['worst_loss']:+.2f}%")
    print(f"  │ {strat['setup']}")

    if not picks:
        needs_rth = sid in ('C5', 'C3+C5', 'C9')
        if is_pre_entry and needs_rth:
            print(f"  │ → Needs RTH bars. Re-run after 09:32 ET.")
        else:
            print(f"  │ → No qualifying stocks.")
        print(f"  └{'─'*85}")
        continue

    for i, p in enumerate(picks):
        rank = i + 1
        is_primary = rank <= opt_n
        rank_label = '' if is_primary else f'  (backup — N={opt_n} optimal)'

        fs = p.get('float_shares')
        fs_str = f"{fs/1e6:.0f}M" if fs and fs < 1e9 else (f"{fs/1e9:.1f}B" if fs else 'n/a')

        entry = p.get('entry')
        if entry:
            status_tag = 'CONFIRMED'
            entry_str = (f"SHORT @ ${entry['entry_price']:.2f} "
                         f"(peak bar {entry['peak_bar']} → entry {entry['entry_time']} ET)")
        else:
            status_tag = 'CANDIDATE'
            entry_str = "watch for local high (bars 3-11)"

        bd = p.get('bar_data')
        pm_info = ''
        if bd:
            pm_bars = bd.get('pm', [])
            pm_830 = [(t, b) for t, b in pm_bars if t >= '08:30']
            if pm_830:
                pm_high = max(b['h'] for _, b in pm_830)
                pm_last = pm_830[-1][1]['c']
                prior_c = p['prior_close']
                denom = pm_high - prior_c
                pm_ret = (pm_high - pm_last) / denom if denom > 0 else 0
                first_pm_o = pm_830[0][1]['o']
                pm_trend = (pm_last - first_pm_o) / first_pm_o if first_pm_o > 0 else 0
                pm_info = f"PM ret={pm_ret:.2f}  PM trend={pm_trend:+.3f}"

            rth = bd.get('rth', [])
            if rth:
                bar0 = rth[0][1]
                bar0_color = 'RED' if bar0['c'] < bar0['o'] else 'GREEN'
                pm_info += f"  bar0={bar0_color}"
                if len(rth) >= 2:
                    bar1 = rth[1][1]
                    drop_2b = (bar1['c'] - bar0['o']) / bar0['o'] * 100 if bar0['o'] > 0 else 0
                    pm_info += f"  2bar={drop_2b:+.1f}%"

        print(f"  │ [{status_tag}] #{rank}  {p['ticker']:6s}  "
              f"gap={p['gap_pct']*100:+.1f}%  ${p['open_price']:.2f}  "
              f"float={fs_str}{rank_label}")
        print(f"  │   → {entry_str}")
        if pm_info:
            print(f"  │   {pm_info}")

    print(f"  └{'─'*85}")


# ── CROSS-STRATEGY AGREEMENT ─────────────────────────────────────────────────

if len(ticker_strats) > 1:
    print(f"\n\n{'='*90}")
    print(f"  CROSS-STRATEGY AGREEMENT")
    print(f"{'='*90}")

    sorted_tickers = sorted(ticker_strats.items(), key=lambda x: -len(x[1]))
    rows = []
    for tk, strats in sorted_tickers:
        n_strats = len(strats)
        strat_str = ', '.join(strats)
        p = all_unique[tk]
        gap_str = f"+{p['gap_pct']*100:.1f}%"
        status = 'CONFIRMED' if p.get('entry') else 'CANDIDATE'
        conf = 'HIGH' if n_strats >= 4 else ('MED' if n_strats >= 2 else 'LOW')
        rows.append([tk, gap_str, n_strats, conf, status, strat_str])

    print()
    print(tabulate(rows,
                   headers=['Ticker', 'Gap', '#Strats', 'Conf', 'Status', 'Strategies'],
                   tablefmt='simple', stralign='left'))

    top_tk, top_strats = sorted_tickers[0]
    print(f"\n  → Best: {top_tk} in {len(top_strats)} strategies ({', '.join(top_strats)})")

print(f"\n  Runtime: {runtime:.1f}s")
print(f"{'='*90}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# BIGQUERY LOGGING — candidates + strategy picks
# ═══════════════════════════════════════════════════════════════════════════════

def log_scan_to_bq():
    if bq_mod is None:
        print("  [BQ] google-cloud-bigquery not installed — skipping log")
        return
    if not GCP_PROJECT:
        print("  [BQ] GCP_PROJECT not set — skipping log")
        return

    run_ts = datetime.now(ET).isoformat()
    rows = []

    for c in cs_pool:
        tk = c['ticker']
        rows.append({
            'run_timestamp': run_ts,
            'scan_date': scan_date,
            'scan_phase': scan_phase,
            'row_type': 'CANDIDATE',
            'ticker': tk,
            'name': c.get('name', ''),
            'gap_pct': round(c['gap_pct'] * 100, 2),
            'open_price': round(c['open_price'], 2),
            'prior_close': round(c['prior_close'], 2),
            'float_shares': c.get('float_shares'),
            'gap_consumed': round(c.get('gap_consumed', 0), 3),
            'strategy_id': None,
            'strategy_rank': None,
            'entry_price': None,
            'entry_time': None,
            'entry_type': None,
            'peak_bar': None,
            'pm_retrace': None,
            'pm_trend': None,
            'bar0_color': None,
            'bar2_drop_pct': None,
            'local_high_confirmed': False,
        })

    for strat in STRATEGIES:
        sid = strat['id']
        for rank, p in enumerate(strategy_picks.get(sid, []), 1):
            tk = p['ticker']
            entry = p.get('entry')
            bd = p.get('bar_data')

            pm_retrace_val = None
            pm_trend_val = None
            bar0_color = None
            bar2_drop = None

            if bd:
                pm_bars = bd.get('pm', [])
                pm_830 = [(t, b) for t, b in pm_bars if t >= '08:30']
                if pm_830:
                    pm_high = max(b['h'] for _, b in pm_830)
                    pm_last = pm_830[-1][1]['c']
                    prior_c = p.get('prior_close', 0)
                    denom = pm_high - prior_c
                    if denom > 0:
                        pm_retrace_val = round((pm_high - pm_last) / denom, 3)
                    first_pm_o = pm_830[0][1]['o']
                    if first_pm_o > 0:
                        pm_trend_val = round((pm_last - first_pm_o) / first_pm_o, 4)

                rth = bd.get('rth', [])
                if rth:
                    b0 = rth[0][1]
                    bar0_color = 'RED' if b0['c'] < b0['o'] else 'GREEN'
                    if len(rth) >= 2:
                        bar2_drop = round((rth[1][1]['c'] - b0['o']) / b0['o'] * 100, 2) if b0['o'] > 0 else None

            rows.append({
                'run_timestamp': run_ts,
                'scan_date': scan_date,
                'scan_phase': scan_phase,
                'row_type': 'PICK',
                'ticker': tk,
                'name': p.get('name', ''),
                'gap_pct': round(p['gap_pct'] * 100, 2),
                'open_price': round(p['open_price'], 2),
                'prior_close': round(p['prior_close'], 2),
                'float_shares': p.get('float_shares'),
                'gap_consumed': None,
                'strategy_id': sid,
                'strategy_rank': rank,
                'entry_price': round(entry['entry_price'], 2) if entry else None,
                'entry_time': entry['entry_time'] if entry else None,
                'entry_type': entry['entry_type'] if entry else None,
                'peak_bar': entry['peak_bar'] if entry else None,
                'pm_retrace': pm_retrace_val,
                'pm_trend': pm_trend_val,
                'bar0_color': bar0_color,
                'bar2_drop_pct': bar2_drop,
                'local_high_confirmed': entry is not None,
            })

    if not rows:
        print("  [BQ] No rows to log")
        return

    client = bq_mod.Client()

    create_sql = f"""
    CREATE TABLE IF NOT EXISTS `{BQ_TABLE}` (
        run_timestamp        TIMESTAMP,
        scan_date            DATE,
        scan_phase           STRING,
        row_type             STRING,
        ticker               STRING,
        name                 STRING,
        gap_pct              FLOAT64,
        open_price           FLOAT64,
        prior_close          FLOAT64,
        float_shares         INT64,
        gap_consumed         FLOAT64,
        strategy_id          STRING,
        strategy_rank        INT64,
        entry_price          FLOAT64,
        entry_time           STRING,
        entry_type           STRING,
        peak_bar             INT64,
        pm_retrace           FLOAT64,
        pm_trend             FLOAT64,
        bar0_color           STRING,
        bar2_drop_pct        FLOAT64,
        local_high_confirmed BOOL
    )"""
    try:
        client.query(create_sql).result()
    except Exception as e:
        print(f"  [BQ] Table create failed: {e}")

    try:
        errors = client.insert_rows_json(BQ_TABLE, rows)
        if errors:
            print(f"  [BQ] Insert errors: {errors[:3]}")
        else:
            n_cand = sum(1 for r in rows if r['row_type'] == 'CANDIDATE')
            n_pick = sum(1 for r in rows if r['row_type'] == 'PICK')
            print(f"  [BQ] Logged {n_cand} candidates + {n_pick} picks to {BQ_TABLE}")
    except Exception as e:
        print(f"  [BQ] Insert failed: {e}")


log_scan_to_bq()
