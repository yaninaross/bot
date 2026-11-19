# %% Agent 1 — V8.2-S Official SHORT Gap-Fade Bot (Choice 1: Max Profit)
#
# ═══════════════════════════════════════════════════════════════════════════════
# V8.2-S OFFICIAL LIVE RULES (12.5M SPLIT-PROOF ABLATION WINNER — CHOICE 1)
# ═══════════════════════════════════════════════════════════════════════════════
#   1. Easy-to-Borrow Pre-Check (IBKR Live Short Feed + Structural Fallback):
#      Excludes HTB stocks (<5K shares or >30% borrow fee) and auto-promotes next ETB.
#   2. Pre-Open News / Catalyst Veto (<= 06:30 PST, Zero Look-Ahead):
#      Blocks M&A / Buyout / Takeover on ALL gaps and Analyst Upgrades on <15% gaps.
#   3. No Secondary Bounce Re-Shorts (_B2 Disabled):
#      Short each qualified stock ONCE in the morning (06:33–06:44 PST).
#   4. 15%–30% Gap Unlock (Bearish Peak->Entry T&Q Required):
#      Allows 15%–30% gaps into watchlist, confirming ONLY when Peak->Entry order flow
#      shows heavy institutional selling (MSYN >= +0.35 AND Block_OFI <= -0.15).
#   5. Small Gaps (<15%) — Choice 1 (Max Profit):
#      Executes directly on price reversal (no <15% T&Q veto; pass --picky for Choice 2).
#   6. Risk & Exits (PST):
#      - Hard Stop: -10.0% from entry price (NO trailing profit lock — lets winners run!).
#      - Active-Slot Sizing: min(1/k, 50%) of SHORT sleeve (Top 3 max slots).
#      - 08:30 PST: Cover ALL <15% gap shorts & $50+ local_high spikes.
#      - 09:30 PST: Cover <$50 local_high spikes (>=15% gaps).
#      - 11:00 PST: Cover ALL remaining bar0_flush shorts (100% FLAT).
#
# RECOMMENDED INTRADAY RUN SCHEDULE (ALL TIMES IN PST):
#   1. 06:31:15 PST — Locks top 3 ETB + Catalyst-clean candidates (Watchlist).
#   2. 06:33–06:44 PST — Run every 1–2 min to catch Bar0 Flush (06:33) or 3-Bar Local High.
#   3. 06:44:30 PST — Morning scan DONE! No midday re-scans needed; manage -10% stop & exits.
# ═══════════════════════════════════════════════════════════════════════════════

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
# CONFIG — Official V8.2-S (Choice 1: Max Profit)
# ═══════════════════════════════════════════════════════════════════════════════

STRATEGY_VERSION = "V8.2-S"
SCAN_TIMESTAMP = os.environ.get("SCAN_TIMESTAMP", None)  # PST — 'YYYY-MM-DD HH:MM' or None for live
ENABLE_TQ_LT15_GATE    = os.environ.get("ENABLE_TQ_LT15_GATE", "0") == "1"  # False = Choice 1 (Max Profit); True (--picky) = Choice 2
SHOW_LEGACY_STRATEGIES = False
for _arg in sys.argv[1:]:
    if _arg.startswith("--date="):
        _d_val = _arg.split("=", 1)[1].strip()
        SCAN_TIMESTAMP = _d_val if " " in _d_val else (None if _d_val == datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y-%m-%d") else f"{_d_val} 13:05")
    elif _arg == "--picky":
        ENABLE_TQ_LT15_GATE = True
        STRATEGY_VERSION = "V8.2-S-Picky"
    elif _arg == "--legacy":
        SHOW_LEGACY_STRATEGIES = True
DRY_RUN = False             # True = print only, skip nothing

GAP_MIN_PCT    = 3.0        # minimum gap %
PRICE_MIN      = 10.0       # minimum price $
PRICE_MAX      = 100.0      # maximum open price $ (eliminates large-cap PEAD & tail risk)
FLOAT_MIN      = 10_000_000
PRE_FILTER_N   = 30         # walk top 30 CS stocks by gap
GAP_CONSUMED_MAX = 0.50     # skip if >50% of gap eroded

SCAN_TIME_ET      = '09:31'
EXIT_TIME_INST_ET = '11:30' # 08:30 PST — <15% gap shorts & $50+ local_high spikes
EXIT_TIME_LH_ET   = '12:30' # 09:30 PST — <$50 local_high morning spikes (>=15% gaps)
EXIT_TIME_ET      = '14:00' # 11:00 PST — bar0_flush >=15% gaps (Hard Flat)

MIN_RTH_BARS   = 2          # min RTH bars with volume > 0 by scan time
MIN_OPEN_VOL   = 5_000      # min shares in bar 0 (09:30)

PEAK_WINDOW    = 3
MAX_PEAK_BAR   = 12         # bars 3-11 (inclusive peak), entry at peak+3
MAX_PEAK_TIME_ET  = '09:41' # latest clock time for peak bar (09:30 + 11m)
MAX_ENTRY_TIME_ET = '09:44' # latest clock time for confirmation bar (09:41 + 3m)
MIN_PULLBACK_PCT  = 0.5     # min pullback % from peak high to confirmation bar close
MAX_MISSING_BARS_MOD_GAP = 1    # max missing 1-min bars before entry when gap < 15%
MAX_OPEN_RUNUP_MOD_GAP   = 0.07 # max post-open runup (7%) at entry when gap < 15%
MOD_GAP_PEAK_STOP_BUFFER = None # Disabled (uses clean -10.0% hard stop from entry)
MOD_GAP_MAX_PRICE        = None # Disabled
MOD_GAP_MAX_PRIOR_DVOL   = None # Disabled
USE_PRIORITY_TIER_RANKING    = True       # Rank Tier-1 (>=30% Parabolic + <15% Retail) ahead of Tier-2
TIER1_MOD_GAP_MAX_PRICE      = 25.0       # Tier-1 retail cap price threshold for <15% gaps
TIER1_MOD_GAP_MAX_PRIOR_DVOL = 25_000_000 # Tier-1 retail cap prior-day $Vol threshold for <15% gaps
FALLBACK_BAR   = None       # no fallback for filtered strategies

# V8.2-S Risk & Alpha Gates (12.5M Split-Proof Ablation Locked)
SHORT_HARD_STOP_PCT          = 0.10       # -10.0% hard stop from entry (broadest Sharpe plateau; NO -5% tight stop)
ENABLE_TRAILING_PROFIT_LOCK  = False      # Explicitly OFF (blanket +5% profit lock lost -55.7% Net over 12.5M!)
ENABLE_PREOPEN_CATALYST_VETO = True       # Blocks M&A/Buyout (all gaps) & Analyst Upgrades (<15% gaps) before open
TQ_15_30_MIN_MSYN            = 0.35       # 15-30% gap unlock requires Bearish Peak->Entry MSYN >= +0.35
TQ_15_30_MAX_BLK_OFI         = -0.15      # 15-30% gap unlock requires Institutional Block_OFI <= -0.15

# Broker Short-Locate Pre-Check & Automatic ETB Fallback (IBKR Live Short Feed + Optional Alpaca + Manual Override)
ENABLE_SHORT_LOCATE_ETB_FILTER = True     # Pre-checks live IBKR shortable shares & borrow fee; auto-promotes next ETB stock if HTB
MIN_ETB_SHARES                 = 5_000    # Minimum available shortable shares at IBKR to qualify as ETB
MAX_BORROW_FEE_PCT             = 30.0     # Maximum annualized borrow fee (%) to qualify as ETB
HTB_BLACKLIST_TODAY            = [
    s.strip().upper() for s in os.environ.get("HTB_BLACKLIST_TODAY", "").split(",") if s.strip()
]
for _arg in sys.argv[1:]:
    if _arg.startswith("--htb="):
        HTB_BLACKLIST_TODAY.extend([s.strip().upper() for s in _arg.split("=", 1)[1].split(",") if s.strip()])

# High-Elasticity Structural Gate (disabled by default: hard-filtering cuts active days and reduces 12M Total PnL)
ENABLE_HIGH_ELASTICITY_GATE    = False
MIN_ELASTICITY_GAP_DEC         = 0.075
HIGH_PRICE_ELASTICITY_MIN      = 60.0
HIGH_PRICE_MIN_GAP_DEC         = 0.18
MEGA_ADV_ELASTICITY_MIN        = 100_000_000
MEGA_ADV_MIN_PRICE             = 25.0
MEGA_ADV_MIN_GAP_DEC           = 0.12

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
_SPAC_KEYWORDS  = ('ACQUISITION', 'BLANK CHECK', 'SPAC')

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
        'id': 'V8.2',
        'name': 'V8.2-S: Official SHORT Winner (Choice 1)',
        'base_filter': 'V82',
        'optimal_n': 3,
        'sizing_mode': 'active_cap50',
        'allow_bar0_flush': True,
        'continue_on_failed_pb': True,
        'hybrid_exit_lt15_1130': True,
        'addon_bounce_reshort': False,
        'addon_1100_winner': False,
        'unlock_15_30_tq': True,
        'post_sharpe': 1.76,
        'post_avg': 0.77,
        'post_trades': 233,
        'worst_loss': -10.0,
        'description': (
            'OFFICIAL V8.2-S SHORT BOT (Choice 1 — Audit-Clean Split-Proof 12.5M: +86.7% Sum Net, '
            '$100K -> $212K standalone, Sharpe 1.76, MaxDD -26.5%; Combined 50/50 unleveraged cash '
            'with V8.2-L LONG Bot: $100K -> $361K, MaxDD -13.9%, 12/13 green months, net of -0.75% '
            'slippage & real IBKR borrow fees). Features: IBKR ETB Locate + Pre-Open Catalyst Veto '
            '(M&A/Buyout & <15% Upgrades) + NO _B2 Re-Shorts + 15-30% Gap Unlock (only when bearish '
            'Peak->Entry T&Q confirms MSYN>=+0.35 & BlkOFI<=-0.15) + -10% Hard Stop (NO Profit Lock).'
        ),
        'setup': (
            'Morning (06:33-06:44 PST): $10-$100 ETB + Catalyst-clean common stock. '
            'Short on 06:33 PST Bar0 Flush or 3-bar local high (PB>=0.5%). '
            '15-30% gaps require bearish Peak->Entry T&Q (MSYN>=+0.35 & BlkOFI<=-0.15). '
            'Hard Stop: -10.0% (NO trailing profit lock, NO _B2 re-shorts). '
            'Exits (PST): 08:30 (<15% gap or $50+ LH) | 09:30 (<$50 LH >=15%) | 11:00 (bar0_flush >=15%).'
        ),
    },
]

if SHOW_LEGACY_STRATEGIES:
    STRATEGIES.extend([
        {
            'id': 'C2',
            'name': 'Legacy C2: Skip 15-30% + _B2 Bounce',
            'base_filter': 'S1',
            'optimal_n': 3,
            'sizing_mode': 'active_cap50',
            'allow_bar0_flush': True,
            'continue_on_failed_pb': True,
            'hybrid_exit_lt15_1130': True,
            'addon_bounce_reshort': True,
            'addon_1100_winner': False,
            'unlock_15_30_tq': False,
            'post_sharpe': 0.35,
            'post_avg': 0.25,
            'post_trades': 349,
            'worst_loss': -10.0,
            'description': 'Legacy C2 comparison (with _B2 bounce re-shorts & 15-30% blocked).',
            'setup': 'Legacy C2 reference only (enabled via --legacy flag).',
        },
    ])


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
        day_o = t.get('day', {}).get('o')
        pm = t.get('min', {})
        pm_price = float(pm['c']) if pm and pm.get('c') and pm['c'] > 0 else None
        open_p = float(day_o) if day_o and float(day_o) > 0 else pm_price
        if open_p is None and prev_c:
            open_p = float(prev_c)
        if open_p and open_p > 0:
            out[tk] = {
                'o': open_p,
                'price': pm_price if pm_price else open_p,
                'prev_close': float(prev_c) if prev_c else None,
            }
    return out


async def _fetch_bars_one(session, tk, dt, sem, max_time_et=None):
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
                if max_time_et is not None and t_str > max_time_et:
                    continue
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


import ssl
try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL_CTX = ssl.create_default_context()


async def _fetch_bars_batch(tickers, dt, max_time_et=None):
    sem = asyncio.Semaphore(15)
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=_SSL_CTX)) as session:
        tasks = [_fetch_bars_one(session, tk, dt, sem, max_time_et=max_time_et) for tk in tickers]
        results = await asyncio.gather(*tasks)
    return {tk: data for tk, data in results if data is not None}


def fetch_minute_bars(tickers, dt, max_time_et=None):
    if not tickers: return {}
    return asyncio.run(_fetch_bars_batch(tickers, dt, max_time_et=max_time_et))


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
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=_SSL_CTX)) as session:
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
    """PM retrace >= 8% AND bar0 red (strictly within SCAN_TIME_ET)."""
    if bar_data is None: return False
    pm_bars = bar_data.get('pm', [])
    rth_bars = bar_data.get('rth', [])
    prior_close = candidate.get('prior_close', 0)

    pm_after_830 = [(t, b) for t, b in pm_bars if '08:30' <= t < '09:30']
    if pm_after_830 and prior_close > 0:
        pm_high = max(b['h'] for _, b in pm_after_830)
        last_pm_close = pm_after_830[-1][1]['c']
        denom = pm_high - prior_close
        pm_retrace = (pm_high - last_pm_close) / denom if denom > 0 else 0
        if pm_retrace < 0.08:
            return False
    else:
        return False

    scan_rth = [(t, b) for t, b in rth_bars if t <= SCAN_TIME_ET]
    if scan_rth:
        bar0 = scan_rth[0][1]
        if bar0['c'] > bar0['o']:
            return False
    else:
        return False

    return True


def check_c3_c5(candidate, bar_data):
    """C5 filters + skip T6 gap tier (15-30% on either scan gap or raw open gap)."""
    if not check_c3(candidate, bar_data):
        return False
    return check_c5(candidate, bar_data)


def check_c7(candidate, bar_data):
    """PM close < PM open (after 8:30 ET)."""
    if bar_data is None: return False
    pm_bars = bar_data.get('pm', [])
    pm_after_830 = [(t, b) for t, b in pm_bars if '08:30' <= t < '09:30']
    if len(pm_after_830) >= 2:
        first_pm = pm_after_830[0][1]['o']
        last_pm = pm_after_830[-1][1]['c']
        pm_trend = (last_pm - first_pm) / first_pm if first_pm > 0 else 0
        return pm_trend < 0
    return False


def check_c9(candidate, bar_data):
    """bar1 close < bar0 open (2-bar drop strictly within SCAN_TIME_ET)."""
    if bar_data is None: return False
    rth_bars = bar_data.get('rth', [])
    scan_rth = [(t, b) for t, b in rth_bars if t <= SCAN_TIME_ET]
    if len(scan_rth) >= 2:
        open_p = scan_rth[0][1]['o']
        bar1_close = scan_rth[1][1]['c']
        drop = (bar1_close - open_p) / open_p if open_p > 0 else 0
        return drop < 0
    return False


def check_c3(candidate, bar_data):
    """Skip T6 gap tier (15-30%) on both scan gap and raw open gap."""
    gap = candidate.get('gap_pct', 0)
    raw_gap = candidate.get('raw_open_gap', gap)
    if (0.15 <= gap < 0.30) or (0.15 <= raw_gap < 0.30):
        return False
    return True


def _passes_high_elasticity_gate(candidate):
    """High-Elasticity Structural Gate (24M Walk-Forward Validated across 534 T&Q Stocks):
    Filters out low-elasticity institutional earnings drifters where short mean-reversion upside is capped
    at ~1-2% while institutional PEAD continuation risk is -8%+ (e.g., NXT $83.41 +5.2% gap -> -8.82%,
    FORM $81.03 +12.9% gap -> -2.27%, SIGI $98.09 +7.0% gap):
      1. Gap < 7.5% (MIN_ELASTICITY_GAP_DEC)
      2. Open Price >= $60 with Gap < 18.0%
      3. Open Price >= $25 AND Prior-Day $Vol >= $100M/d with Gap < 12.0% (preserves <$25 retail/crypto spikers like RIOT)
    """
    if not ENABLE_HIGH_ELASTICITY_GATE:
        return True, ""
    gap = candidate.get('gap_pct', 0.0)
    op  = candidate.get('open_price', 0.0)
    adv = candidate.get('prior_dvol') or 0.0
    if gap < MIN_ELASTICITY_GAP_DEC:
        return False, f"Gap+{gap*100:.1f}%<7.5%"
    if op >= HIGH_PRICE_ELASTICITY_MIN and gap < HIGH_PRICE_MIN_GAP_DEC:
        return False, f"Px${op:.0f}>=$60&Gap+{gap*100:.1f}%<18%"
    if op >= MEGA_ADV_MIN_PRICE and adv >= MEGA_ADV_ELASTICITY_MIN and gap < MEGA_ADV_MIN_GAP_DEC:
        return False, f"ADV${adv/1e6:.0f}M>=$100M&Gap+{gap*100:.1f}%<12%"
    return True, ""


def _passes_mod_gap_retail_check(candidate):
    """Out-of-sample validated structural guard:
    1. Exclude SPACs / Blank Check companies ('ACQUISITION', 'BLANK CHECK', 'SPAC' in name).
    2. Enforce High-Elasticity Structural Gate (blocks low-elasticity institutional drifters).
    """
    nm = (candidate.get('name') or '').upper()
    if any(w in nm for w in _SPAC_KEYWORDS):
        return False
    ok_el, _ = _passes_high_elasticity_gate(candidate)
    if not ok_el:
        return False
    gap = candidate.get('gap_pct', 0)
    if gap < 0.15:
        if MOD_GAP_MAX_PRICE is not None and candidate.get('open_price', 0) > MOD_GAP_MAX_PRICE:
            return False
        prior_dvol = candidate.get('prior_dvol')
        if MOD_GAP_MAX_PRIOR_DVOL is not None and prior_dvol is not None and prior_dvol >= MOD_GAP_MAX_PRIOR_DVOL:
            return False
    return True


def _is_tier1_setup(candidate):
    """Priority Tier 1: Parabolic gaps (>=30%) or Retail Small-Cap Moderate gaps (<15%, Price<=$25, Prior $Vol<$25M).
    Priority Tier 2: 15-30% T&Q-unlocked gaps & Institutional/Mid-Cap Moderate gaps (<15%, Price>$25 or Prior $Vol>=$25M).
    """
    gap = candidate.get('gap_pct', 0)
    if gap >= 0.30:
        return True
    if gap < 0.15:
        op = candidate.get('open_price', 0)
        dvol = candidate.get('prior_dvol')
        if op <= TIER1_MOD_GAP_MAX_PRICE and (dvol is None or dvol < TIER1_MOD_GAP_MAX_PRIOR_DVOL):
            return True
    return False


def check_v82(candidate, bar_data):
    """V8.2-S Official Filter: Price $10-$100, No SPACs, allows 15-30% gaps into watchlist
    (which are strictly gated at trigger time by Bearish Peak->Entry T&Q: MSYN >= +0.35 & BlkOFI <= -0.15).
    """
    opx = candidate.get('open_price', 0)
    if not (PRICE_MIN <= opx <= PRICE_MAX):
        return False
    return _passes_mod_gap_retail_check(candidate)


def check_s6(candidate, bar_data):
    """Price <= $100, Skip 12-30% gap zone, No SPACs, and Retail Small-Cap (<$25, <$25M prior $Vol) on <15% gaps."""
    if candidate.get('open_price', 0) > PRICE_MAX:
        return False
    gap = candidate.get('gap_pct', 0)
    raw_gap = candidate.get('raw_open_gap', gap)
    if (0.12 <= gap < 0.30) or (0.12 <= raw_gap < 0.30):
        return False
    return _passes_mod_gap_retail_check(candidate)


def check_s1(candidate, bar_data):
    """Price <= $100, Skip 15-30% T6 gap tier, No SPACs, and Retail Small-Cap (<$25, <$25M prior $Vol) on <15% gaps."""
    if candidate.get('open_price', 0) > PRICE_MAX:
        return False
    gap = candidate.get('gap_pct', 0)
    raw_gap = candidate.get('raw_open_gap', gap)
    if (0.15 <= gap < 0.30) or (0.15 <= raw_gap < 0.30):
        return False
    return _passes_mod_gap_retail_check(candidate)


def check_s12(candidate, bar_data):
    """Same filter as S1 (Price <= $100, Skip 15-30% T6), evaluated at optimal_n = 2."""
    return check_s1(candidate, bar_data)


FILTER_FNS = {
    'V82': check_v82,
    'S6': check_s6,
    'S1': check_s1,
    'S12': check_s12,
    'C5': check_c5,
    'C3+C5': check_c3_c5,
    'C7': check_c7,
    'C9': check_c9,
    'C3': check_c3,
}


# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL HIGH & HYBRID INTRADAY ADD-ON DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def find_local_high(rth_bars, window=PEAK_WINDOW, max_peak_bar=MAX_PEAK_BAR,
                    min_entry_bar=0, min_pullback_pct=MIN_PULLBACK_PCT,
                    open_price=None, gap_pct=None, ref_price=None,
                    allow_bar0_flush=True, continue_on_failed_pb=True,
                    tweezer_tip_guard=False):
    if len(rth_bars) < window * 2 + 1:
        # Allow 09:33 ET bar0_flush to confirm as soon as 4 RTH bars (09:30..09:33) exist
        if not (allow_bar0_flush and len(rth_bars) >= 4):
            return None

    # U4: Bar 0 Opening Flush (09:30 red-bar peak confirmed at 09:33 ET bar 3)
    if allow_bar0_flush and len(rth_bars) >= 4:
        b0 = rth_bars[0][1]
        h0 = b0['h']
        if b0['c'] <= b0['o'] and all(rth_bars[j][1]['h'] < h0 for j in range(1, 4)):
            ci = 3
            ep = rth_bars[ci][1]['c']
            et = rth_bars[ci][0]
            pb_pct = (h0 - ep) / h0 * 100.0 if h0 > 0 else 0.0
            drop_open_pct = (open_price - ep) / open_price * 100.0 if (open_price and open_price > 0) else 0.0
            gap_range = (open_price - ref_price) if (open_price and ref_price and open_price > ref_price) else 0.0
            gap_ret = (ep - ref_price) / gap_range if gap_range > 0 else 0.0
            req_pb = min_pullback_pct if min_pullback_pct is not None else 0.0
            if pb_pct >= req_pb and drop_open_pct >= 1.0 and gap_ret >= 0.35:
                mod_ok = True
                if gap_pct is not None and gap_pct < 0.15:
                    clock_idx = (int(et[:2]) - 9) * 60 + (int(et[3:5]) - 30)
                    if (clock_idx - ci) > MAX_MISSING_BARS_MOD_GAP:
                        mod_ok = False
                    if open_price and open_price > 0 and (ep - open_price) / open_price >= MAX_OPEN_RUNUP_MOD_GAP:
                        mod_ok = False
                if mod_ok:
                    return {
                        'entry_price': ep,
                        'entry_time': et,
                        'entry_bar_idx': ci,
                        'entry_type': 'bar0_flush',
                        'peak_bar': 0,
                        'peak_time': rth_bars[0][0],
                        'peak_high': h0,
                        'pullback_pct': round(pb_pct, 2),
                    }

    if len(rth_bars) < window * 2 + 1:
        return None

    start = max(window, min_entry_bar - window)
    for i in range(start, min(max_peak_bar, len(rth_bars) - window)):
        if rth_bars[i][0] > MAX_PEAK_TIME_ET:
            break
        ch = rth_bars[i][1]['h']
        if all(rth_bars[i - j][1]['h'] < ch and rth_bars[i + j][1]['h'] < ch
               for j in range(1, window + 1)):
            ci = i + window
            if ci < len(rth_bars) and rth_bars[ci][0] <= MAX_ENTRY_TIME_ET:
                ep = rth_bars[ci][1]['c']
                et = rth_bars[ci][0]
                pb_pct = (ch - ep) / ch * 100.0 if ch > 0 else 0.0
                if min_pullback_pct is not None and pb_pct < min_pullback_pct:
                    if continue_on_failed_pb:
                        continue
                    return None
                # Option 1: Tweezer-Aware Tip Guard (disabled by default after 12M OOS audit)
                if tweezer_tip_guard:
                    max_prev3 = max(rth_bars[i - j][1]['h'] for j in range(1, window + 1))
                    max_prev23 = max(rth_bars[i - j][1]['h'] for j in (2, 3))
                    tip1 = (ch - max_prev3) / ch * 100.0 if ch > 0 else 0.0
                    tip23 = (ch - max_prev23) / ch * 100.0 if ch > 0 else 0.0
                    if not ((tip1 >= 0.15) or (tip23 >= 0.50 and pb_pct >= 3.0)):
                        if continue_on_failed_pb:
                            continue
                        return None
                if gap_pct is not None and gap_pct < 0.15:
                    clock_idx = (int(et[:2]) - 9) * 60 + (int(et[3:5]) - 30)
                    if (clock_idx - ci) > MAX_MISSING_BARS_MOD_GAP:
                        return None
                    if open_price and open_price > 0 and (ep - open_price) / open_price >= MAX_OPEN_RUNUP_MOD_GAP:
                        return None
                stop_price = (
                    round(ch * (1.0 + MOD_GAP_PEAK_STOP_BUFFER), 2)
                    if (MOD_GAP_PEAK_STOP_BUFFER is not None and gap_pct is not None and gap_pct < 0.15)
                    else None
                )
                return {
                    'entry_price': ep,
                    'entry_time': et,
                    'entry_bar_idx': ci,
                    'entry_type': 'local_high',
                    'peak_bar': i,
                    'peak_time': rth_bars[i][0],
                    'peak_high': ch,
                    'pullback_pct': round(pb_pct, 2),
                    'stop_price': stop_price,
                }
            return None
    return None


def find_secondary_bounce_reshort(rth_bars, primary_entry, min_mfe_pct=3.0, min_bounce_pct=1.5, min_profit_pct=0.5):
    """U7b: Check if a confirmed morning trade reaches >=3% MFE, then forms a 3-bar bounce local high (09:55-13:30 ET) while still in profit."""
    if not primary_entry or not rth_bars:
        return None
    ep1 = primary_entry['entry_price']
    entry_idx = primary_entry.get('entry_bar_idx', primary_entry['peak_bar'] + PEAK_WINDOW)
    start_idx = max(entry_idx + 4, 25)
    running_low = ep1
    for k in range(entry_idx + 1, min(start_idx, len(rth_bars))):
        if rth_bars[k][1]['l'] < running_low:
            running_low = rth_bars[k][1]['l']

    window = PEAK_WINDOW
    for i in range(start_idx, min(len(rth_bars) - window, 240)):
        t_str = rth_bars[i][0]
        if t_str < '09:55':
            if rth_bars[i][1]['l'] < running_low:
                running_low = rth_bars[i][1]['l']
            continue
        if t_str > '13:30':
            break
        mfe_pct = (ep1 - running_low) / ep1 * 100.0 if ep1 > 0 else 0.0
        ch = rth_bars[i][1]['h']
        if mfe_pct >= min_mfe_pct:
            bounce_pct = (ch - running_low) / running_low * 100.0 if running_low > 0 else 0.0
            if bounce_pct >= min_bounce_pct:
                if all(rth_bars[i - j][1]['h'] < ch and rth_bars[i + j][1]['h'] < ch for j in range(1, window + 1)):
                    ci = i + window
                    ep2 = rth_bars[ci][1]['c']
                    pb2 = (ch - ep2) / ch * 100.0 if ch > 0 else 0.0
                    cur_profit = (ep1 - ep2) / ep1 * 100.0 if ep1 > 0 else 0.0
                    if pb2 >= 0.4 and cur_profit >= min_profit_pct and rth_bars[ci][0] <= '13:35':
                        return {
                            'entry_price': ep2,
                            'entry_time': rth_bars[ci][0],
                            'entry_bar_idx': ci,
                            'entry_type': 'bounce_reshort',
                            'peak_bar': i,
                            'peak_time': t_str,
                            'peak_high': ch,
                            'pullback_pct': round(pb2, 2),
                            'mfe_before_bounce_pct': round(mfe_pct, 2),
                            'bounce_from_low_pct': round(bounce_pct, 2),
                        }
        if rth_bars[i][1]['l'] < running_low:
            running_low = rth_bars[i][1]['l']
    return None


def find_1100_winner_addon(rth_bars, primary_entry):
    """U7a: Check at 11:00 ET if a confirmed morning trade is up >= +3.0% with <= 40% bounce off its morning low."""
    if not primary_entry or not rth_bars:
        return None
    ep1 = primary_entry['entry_price']
    bars_to_1100 = [(t, b) for t, b in rth_bars if primary_entry['entry_time'] < t <= '11:00']
    if len(bars_to_1100) < 5 or bars_to_1100[-1][0] < '10:59':
        return None
    p_1100 = bars_to_1100[-1][1]['c']
    t_1100 = bars_to_1100[-1][0]
    low_to_1100 = min(b['l'] for _, b in bars_to_1100)
    max_drop = ep1 - low_to_1100
    cur_pnl_1100 = (ep1 - p_1100) / ep1 * 100.0 if ep1 > 0 else 0.0
    bounce_frac = (p_1100 - low_to_1100) / max_drop if max_drop > 0 else 1.0
    if cur_pnl_1100 >= 3.0 and bounce_frac <= 0.40:
        return {
            'entry_price': p_1100,
            'entry_time': t_1100,
            'entry_bar_idx': 90,
            'entry_type': 'winner_1100',
            'peak_bar': primary_entry['peak_bar'],
            'peak_time': primary_entry['peak_time'],
            'peak_high': primary_entry['peak_high'],
            'pullback_pct': round(cur_pnl_1100, 2),
            'bounce_frac_pct': round(bounce_frac * 100.0, 1),
        }
    return None


def get_target_exit_et(open_price, gap_pct, entry_type, hybrid_exit_lt15_1130=False):
    """
    Rule #3 Signal-Adaptive 3-Tier RTH Exit Schedule (returns ET 'HH:MM'; PST = ET - 3h):
      - Tier 1 (08:30 PST / '11:30' ET): $50+ `local_high` morning spikes, or C2 `<15%` gap morning trades
      - Tier 2 (09:30 PST / '12:30' ET): `< $50` `local_high` morning spikes
      - Tier 3 (11:00 PST / '14:00' ET — Hard Flat): `bar0_flush` collapses (except C2 `<15%`), `_B2` bounces, `_W11` pyramids
    """
    if entry_type == 'local_high':
        if (open_price is not None and open_price >= 50.0) or (hybrid_exit_lt15_1130 and gap_pct is not None and gap_pct < 0.15):
            return EXIT_TIME_INST_ET
        return EXIT_TIME_LH_ET
    if entry_type == 'bar0_flush':
        if hybrid_exit_lt15_1130 and gap_pct is not None and gap_pct < 0.15:
            return EXIT_TIME_INST_ET
        return EXIT_TIME_ET
    if entry_type in ('bounce_reshort', 'winner_1100'):
        return EXIT_TIME_ET
    if (open_price is not None and open_price >= 50.0) or (hybrid_exit_lt15_1130 and gap_pct is not None and gap_pct < 0.15):
        return EXIT_TIME_INST_ET
    return EXIT_TIME_LH_ET


def get_exit_reason_pst(open_price, gap_pct, entry_type, hybrid_exit_lt15_1130=False):
    """Human-readable Rule #3 tier rationale for the Operator Playbook."""
    if entry_type == 'local_high':
        if open_price is not None and open_price >= 50.0:
            return 'Rule #3 Tier 1: $50+ institutional local_high -> 08:30 PST'
        if hybrid_exit_lt15_1130 and gap_pct is not None and gap_pct < 0.15:
            return 'Rule #3 C2 <15% gap local_high -> 08:30 PST'
        return 'Rule #3 Tier 2: <$50 local_high morning spike -> 09:30 PST'
    if entry_type == 'bar0_flush':
        if hybrid_exit_lt15_1130 and gap_pct is not None and gap_pct < 0.15:
            return 'Rule #3 C2 <15% gap bar0_flush -> 08:30 PST'
        return 'Rule #3 Tier 3: bar0_flush multi-hour collapse -> 11:00 PST'
    if entry_type == 'bounce_reshort':
        return 'Rule #3 Tier 3: Midday _B2 bounce re-short -> 11:00 PST'
    if entry_type == 'winner_1100':
        return 'Rule #3 Tier 3: 08:00 PST _W11 pyramid -> 11:00 PST'
    return 'Rule #3 RTH Schedule (08:30 / 09:30 / 11:00 PST)'


def get_intraday_trade_telemetry(bar_data, entry, exit_time_target=EXIT_TIME_ET):
    """Compute live RTH price, unrealized/realized PnL %, MFE drop %, stop-out state, and add-on arming status.
    Strictly uses Regular Trading Hours (rth) bars — never after-hours (ah) dark-pool prints.
    """
    if not bar_data or not entry:
        return None
    rth = bar_data.get('rth', [])
    ep = entry['entry_price']
    et = entry['entry_time']
    stop_price = entry.get('stop_price')
    if ep <= 0 or not rth:
        return None

    post_bars = [(t, b) for t, b in rth if t > et]
    if not post_bars:
        return {
            'cur_price': ep, 'cur_time': et, 'pnl_pct': 0.0,
            'exited': False, 'stopped_out': False, 'stop_price': stop_price,
            'mfe_pct': 0.0, 'bounce_armed': False,
        }

    # Moderate-Gap (<15% gap local_high) Hard Stop & Late-Entry Blocker: Peak + 1.70%
    stopped_out = False
    stop_hit_time = None
    stop_fill_price = None
    if stop_price is not None:
        for t, b in post_bars:
            if t > exit_time_target:
                break
            if b['h'] >= stop_price:
                stopped_out = True
                stop_hit_time = t
                stop_fill_price = round(max(stop_price, b['o']), 2)
                break

    if stopped_out:
        cur_time = stop_hit_time
        cur_price = stop_fill_price
        exited = True
        rth_midday = [(t, b) for t, b in rth if et < t <= stop_hit_time]
    else:
        latest_time = rth[-1][0]
        bars_to_exit = [(t, b) for t, b in rth if et <= t <= exit_time_target]
        exited = (latest_time >= exit_time_target) and (len(bars_to_exit) > 0)
        ref_bars = bars_to_exit if (exited and bars_to_exit) else post_bars
        cur_time, cur_bar = ref_bars[-1]
        cur_price = cur_bar['c']
        rth_midday = [(t, b) for t, b in rth if et < t <= '13:35']

    pnl_pct = (ep - cur_price) / ep * 100.0
    low_midday = min((b['l'] for _, b in rth_midday), default=ep)
    mfe_pct = max(0.0, (ep - low_midday) / ep * 100.0)
    bounce_armed = (mfe_pct >= 3.0) and (not stopped_out)

    return {
        'cur_price': cur_price,
        'cur_time': cur_time,
        'pnl_pct': round(pnl_pct, 2),
        'exited': exited,
        'stopped_out': stopped_out,
        'stop_price': stop_price,
        'mfe_pct': round(mfe_pct, 2),
        'bounce_armed': bounce_armed,
    }


def _et_to_pst(hhmm_et):
    """Convert an 'HH:MM' ET string (e.g. '09:39', '11:30', '12:30', '14:00') to 'HH:MM' PST."""
    if not hhmm_et or ':' not in str(hhmm_et):
        return str(hhmm_et)
    try:
        h, m = map(int, str(hhmm_et).split(':')[:2])
        return f"{(h - 3) % 24:02d}:{m:02d}"
    except Exception:
        return str(hhmm_et)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SCANNER
# ═══════════════════════════════════════════════════════════════════════════════

t_start = time.time()

sim_cutoff_et = None
if SCAN_TIMESTAMP is not None:
    parts = SCAN_TIMESTAMP.strip().split(' ', 1)
    scan_date = parts[0]
    mode = 'historical'
    if len(parts) > 1 and parts[1].strip():
        try:
            dt_pt = PT.localize(datetime.strptime(f"{scan_date} {parts[1].strip()}", "%Y-%m-%d %H:%M"))
            sim_cutoff_et = dt_pt.astimezone(ET).strftime("%H:%M")
        except ValueError:
            sim_cutoff_et = None
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

now_et_hhmm = sim_cutoff_et if sim_cutoff_et is not None else now_et_dt.strftime('%H:%M')
now_pst_hhmm = _et_to_pst(now_et_hhmm)
if mode == 'historical' and sim_cutoff_et is None:
    scan_phase = 'HISTORICAL'
elif now_et_hhmm < '09:30':
    scan_phase = 'PRE-MARKET (< 06:30 PST)'
elif now_et_hhmm < '09:33':
    scan_phase = 'MARKET OPEN — bars forming (06:30–06:32 PST)'
elif now_et_hhmm < '09:45':
    scan_phase = 'ENTRY WINDOW — morning entries confirming (06:33–06:44 PST)'
elif now_et_hhmm <= '14:00':
    scan_phase = 'HOLD / EXIT WINDOW — morning entries resolved; manage -10% stop & scheduled exits (06:45–11:00 PST)'
else:
    scan_phase = 'POST-WINDOW — all morning entries & RTH exits resolved (> 11:00 PST)'

is_pre_entry = scan_phase.startswith('PRE-MARKET') or scan_phase.startswith('MARKET OPEN')

print(f"\n{'='*86}")
print(f"  AGENT 1 — {STRATEGY_VERSION} SHORT BOT ({'Choice 2: Picky <15% T&Q' if ENABLE_TQ_LT15_GATE else 'Choice 1: Max Profit — Official Live Winner'})")
print(f"  Date: {scan_date}  |  Mode: {mode}  |  Time: {now_pst}")
print(f"  Prior: {prior_str}  |  Hard Stop: -{SHORT_HARD_STOP_PCT*100:.0f}% (NO Profit Lock)  |  NO _B2 Re-Shorts")
print(f"  Exits (PST): 08:30 (<15% gap / $50+ LH) | 09:30 (<$50 LH >=15%) | 11:00 (bar0_flush >=15%)")
print(f"  Phase: {scan_phase}")
print(f"{'='*86}")


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
    prev_v = prior_data[tk].get('v', 0)
    cur_p = current_data[tk].get('o', current_data[tk].get('price', 0))
    if cur_p <= 0 or cur_p < PRICE_MIN: continue
    gap = (cur_p - prev_c) / prev_c
    if gap < gap_min: continue
    fs = _FLOAT.get(tk)
    if fs is not None and fs < FLOAT_MIN: continue
    candidates.append({
        'ticker': tk, 'gap_pct': gap, 'raw_open_gap': gap,
        'prior_close': prev_c, 'prior_dvol': prev_c * prev_v,
        'open_price': cur_p, 'float_shares': fs,
    })
candidates.sort(key=lambda x: -x['gap_pct'])
print(f"      {len(candidates)} candidates after gap+price+float")


# ── Step 4: CS type check (filter out ETFs/ETNs/SPACs BEFORE taking top PRE_FILTER_N) ──
print(f"  [4] CS type check across gap candidates (selecting top {PRE_FILTER_N} CS stocks)...")
cs_candidates = []
n_non_cs = 0
n_checked = 0
for batch_start in range(0, len(candidates), 40):
    batch = candidates[batch_start:batch_start + 40]
    refs = fetch_ticker_refs([c['ticker'] for c in batch])
    for c in batch:
        n_checked += 1
        tk = c['ticker']
        ref = refs.get(tk, {})
        at = ref.get('type', 'UNK')
        nm = ref.get('name', '') or ''
        c['asset_type'] = at
        c['name'] = nm
        if at != 'CS' or any(w in nm.upper() for w in _SPAC_KEYWORDS):
            n_non_cs += 1
            continue
        cs_candidates.append(c)
        if len(cs_candidates) >= PRE_FILTER_N:
            break
    if len(cs_candidates) >= PRE_FILTER_N:
        break
print(f"      {len(cs_candidates)} CS stocks selected (checked {n_checked}, excluded {n_non_cs} ETFs/SPACs/non-CS)")


# ── Step 5: AH reference + gap erosion ──────────────────────────────────────
print(f"  [5] Fetching minute bars (prior AH + scan day) for {len(cs_candidates)} CS stocks...")
top_tickers = [c['ticker'] for c in cs_candidates]

prior_bars = fetch_minute_bars(top_tickers, prior_str) if top_tickers else {}
scan_bars = fetch_minute_bars(top_tickers, scan_date, max_time_et=sim_cutoff_et) if top_tickers else {}
print(f"      Prior AH: {len(prior_bars)}/{len(top_tickers)}  |  Scan day: {len(scan_bars)}/{len(top_tickers)}")

n_stale = 0
n_eroded = 0
qualified = []

for c in cs_candidates:
    tk = c['ticker']

    # Ensure open_price uses official 09:30 RTH open bar when available
    sb = scan_bars.get(tk)
    if sb and sb.get('rth'):
        c['open_price'] = sb['rth'][0][1]['o']
        if c['prior_close'] > 0:
            c['raw_open_gap'] = (c['open_price'] - c['prior_close']) / c['prior_close']

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

    # Re-rank by gap at SCAN_TIME_ET (locked in both live and historical once RTH begins)
    if sb:
        c['bar_data'] = sb
        price_T = get_price_at_T(sb, SCAN_TIME_ET)
        if price_T is not None and scan_phase != 'PRE-MARKET':
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
cs_pool = qualified
print(f"      After AH+erosion: {len(cs_pool)} CS stocks ({n_stale} stale, {n_eroded} eroded)")


# ── Step 5b: Liquidity filter (skip during pre-market — no RTH bars yet) ──
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

    # ── Step 5c: Broker Short-Locate Pre-Check & Automatic ETB Fallback (Colab-proof) ──
    try:
        import sys as _sys_loc
        _agent1_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.path.join(os.getcwd(), 'agent1')
        if _agent1_dir not in _sys_loc.path:
            _sys_loc.path.insert(0, _agent1_dir)
        from short_locate_etb import evaluate_short_locate
    except Exception:
        _INLINE_IBKR_CACHE = {}
        def _load_inline_ibkr():
            if _INLINE_IBKR_CACHE:
                return _INLINE_IBKR_CACHE
            try:
                import urllib.request as _ureq
                raw_txt = _ureq.urlopen("ftp://shortstock:@ftp2.interactivebrokers.com/usa.txt", timeout=12).read().decode("utf-8", errors="ignore")
                for _ln in raw_txt.splitlines():
                    if not _ln or _ln.startswith("#"):
                        continue
                    _p = _ln.split("|")
                    if len(_p) < 8:
                        continue
                    _sym = _p[0].strip().upper()
                    _fee = float(_p[6].strip()) if _p[6].strip() and _p[6].strip() != "NA" else 99.0
                    _av_s = _p[7].strip().replace(">", "").replace("<", "")
                    _av = int(float(_av_s)) if _av_s and _av_s != "NA" else 0
                    _INLINE_IBKR_CACHE[_sym] = {"fee_rate": _fee, "avail_shares": _av}
            except Exception:
                pass
            return _INLINE_IBKR_CACHE

        def evaluate_short_locate(ticker, open_price=20.0, prior_dvol=15e6, gap_pct_dec=0.10,
                                  manual_htb_blacklist=None, min_avail_shares=5000, max_borrow_fee=30.0):
            btk = str(ticker).split("_")[0].strip().upper()
            if manual_htb_blacklist and btk in {s.strip().upper() for s in manual_htb_blacklist}:
                return {"is_etb": False, "tier": "HTB_REJECT", "badge": "HTB-MANUAL", "reason": "Manual HTB blacklist"}
            imap = _load_inline_ibkr()
            info = imap.get(btk)
            if info is not None:
                av, fee = int(info["avail_shares"]), float(info["fee_rate"])
                av_s = f"{av//1000}K" if av < 1_000_000 else f"{av/1_000_000:.1f}M"
                if av < min_avail_shares or fee > max_borrow_fee:
                    return {"is_etb": False, "tier": "HTB_REJECT", "badge": f"HTB({fee:.0f}%/{av_s})", "reason": "IBKR fee/avail fail"}
                if gap_pct_dec >= 0.35 and (prior_dvol or 0) < 10e6 and av < 20000:
                    return {"is_etb": False, "tier": "HTB_REJECT", "badge": f"HTB-SQZ({av_s})", "reason": "Parabolic low-avail"}
                return {"is_etb": True, "tier": "ETB_LIVE", "badge": f"ETB({fee:.1f}%/{av_s})", "reason": "IBKR FTP verified"}
            if (prior_dvol or 0) >= 20e6 and (open_price or 0) >= 15.0:
                return {"is_etb": True, "tier": "ETB_STRUCTURAL", "badge": "ETB-LIQ", "reason": "Structural $20M/$15 fallback"}
            return {"is_etb": False, "tier": "HTB_REJECT", "badge": "HTB-UNLISTED", "reason": "Missing from IBKR & <$20M ADV"}

    if evaluate_short_locate is not None:
        etb_pool = []
        htb_excluded = []
        for c in cs_pool:
            loc_res = evaluate_short_locate(
                c['ticker'],
                open_price=c.get('open_price', 20.0),
                prior_dvol=c.get('prior_dvol', 15e6),
                gap_pct_dec=c.get('gap_pct', 0.10),
                manual_htb_blacklist=HTB_BLACKLIST_TODAY,
                min_avail_shares=MIN_ETB_SHARES,
                max_borrow_fee=MAX_BORROW_FEE_PCT,
            )
            c['locate_info'] = loc_res
            c['locate_badge'] = loc_res['badge']
            if ENABLE_SHORT_LOCATE_ETB_FILTER and not DRY_RUN and not loc_res['is_etb']:
                htb_excluded.append(f"{c['ticker']}({loc_res['badge']})")
                continue
            etb_pool.append(c)
        cs_pool = etb_pool
        htb_str = f": {', '.join(htb_excluded)}" if htb_excluded else ""
        print(f"  [5c] Short-Locate & Auto-ETB Fallback: {len(htb_excluded)} HTB excluded{htb_str}  →  {len(cs_pool)} ETB remain")

    # ── Step 5d: Pre-Open News / Catalyst Veto (Zero Look-Ahead <= 09:30 ET) ──
    # Blocks M&A / Buyout / Takeover on ALL gaps and Analyst Upgrades on <15% gaps (unless dilution present).
    if ENABLE_PREOPEN_CATALYST_VETO and cs_pool:
        from concurrent.futures import ThreadPoolExecutor as _CatTPE
        _t0_et = pd.Timestamp(f"{scan_date} 09:30:00", tz="America/New_York")
        _pub_lte = _t0_et.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
        _pub_gte = (_t0_et - pd.Timedelta(days=3)).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")

        def _check_preopen_cat(c_item):
            tk_c = c_item['ticker']
            is_ma, is_upg, is_dil = False, False, False
            try:
                r = requests.get(
                    "https://api.polygon.io/v2/reference/news",
                    params={
                        "ticker": tk_c, "published_utc.lte": _pub_lte, "published_utc.gte": _pub_gte,
                        "limit": 10, "apiKey": POLYGON_API_KEY,
                    },
                    timeout=8,
                )
                if r.status_code == 200:
                    txt = " | ".join(
                        (x.get("title") or "") + " " + (x.get("description") or "")
                        for x in r.json().get("results", [])
                    ).upper()
                    is_dil = any(w in txt for w in (
                        "PUBLIC OFFERING", "DIRECT OFFERING", "CONVERTIBLE NOTE", "WARRANT", "DILUTION", "SHELF REGISTRATION"
                    ))
                    is_ma = (not is_dil) and any(w in txt for w in (
                        "TO BE ACQUIRED", "DEFINITIVE AGREEMENT TO ACQUIRE", "BUYOUT", "TAKEOVER",
                        "STRATEGIC STAKE", "DIVESTITURE", "SELL EUROPE BUSINESS", "GOING PRIVATE", "MERGER AGREEMENT"
                    ))
                    is_upg = (not is_dil) and any(w in txt for w in (
                        "UPGRADES TO BUY", "UPGRADED TO BUY", "UPGRADES TO OUTPERFORM", "UPGRADED TO OUTPERFORM",
                        "RAISES PRICE TARGET", "INITIATES WITH BUY", "OVERWEIGHT"
                    ))
            except Exception:
                pass
            lbl = "M&A/Buyout" if is_ma else ("Upgrade" if is_upg else ("Dilution" if is_dil else "Clean"))
            return {"is_ma": is_ma, "is_upg": is_upg, "is_dil": is_dil, "lbl": lbl}

        with _CatTPE(max_workers=12) as _ex_cat:
            _cat_results = list(_ex_cat.map(_check_preopen_cat, cs_pool))

        cat_pool = []
        cat_excluded = []
        for c, cf in zip(cs_pool, _cat_results):
            c['catalyst_info'] = cf
            c['catalyst_badge'] = cf['lbl']
            gap_c = c.get('gap_pct', 0.0)
            if not DRY_RUN and (cf['is_ma'] or (gap_c < 0.15 and cf['is_upg'])):
                cat_excluded.append(f"{c['ticker']}({cf['lbl']})")
                continue
            cat_pool.append(c)
        cs_pool = cat_pool
        cat_str = f": {', '.join(cat_excluded)}" if cat_excluded else ""
        print(f"  [5d] Pre-Open Catalyst Veto (<=06:30 PST): {len(cat_excluded)} excluded{cat_str}  →  {len(cs_pool)} clean remain")

    # ── Step 5e: High-Elasticity Structural Gate (disabled by default) ────────
    if ENABLE_HIGH_ELASTICITY_GATE:
        elast_pool = []
        elast_excluded = []
        for c in cs_pool:
            ok_el, reason_el = _passes_high_elasticity_gate(c)
            if not ok_el and not DRY_RUN:
                elast_excluded.append(f"{c['ticker']}({reason_el})")
                continue
            elast_pool.append(c)
        cs_pool = elast_pool
        el_str = f": {', '.join(elast_excluded)}" if elast_excluded else ""
        print(f"  [5e] High-Elasticity Gate: {len(elast_excluded)} low-elasticity excluded{el_str}  →  {len(cs_pool)} remain")



# ── Step 6: Apply V8.2-S strategy filters + morning confirmation ────────────
print(f"  [6] Applying {len(STRATEGIES)} strategy ({', '.join(s['id'] for s in STRATEGIES)}){' with Priority Tier Ranking (T1 before T2)' if USE_PRIORITY_TIER_RANKING else ''}...")

if USE_PRIORITY_TIER_RANKING:
    cs_pool = sorted(cs_pool, key=lambda c: (1 if _is_tier1_setup(c) else 0, c.get('gap_pct', 0)), reverse=True)

strategy_picks = {}

_TQ_SHORT_MEMO = {}

def _fetch_live_short_tq(tk, dt_str, cutoff_et="09:40", start_et="09:30"):
    key = (tk, dt_str, start_et, cutoff_et)
    if key in _TQ_SHORT_MEMO:
        return _TQ_SHORT_MEMO[key]
    ts_start = int(pd.Timestamp(f"{dt_str} {start_et}:00", tz="America/New_York").timestamp() * 1e9)
    ts_end   = int(pd.Timestamp(f"{dt_str} {cutoff_et}:59", tz="America/New_York").timestamp() * 1e9)
    trades, quotes = [], []
    try:
        p = {"timestamp.gte": ts_start, "timestamp.lte": ts_end, "limit": 50000, "sort": "timestamp", "order": "asc", "apiKey": POLYGON_API_KEY}
        r_t = requests.get(f"https://api.polygon.io/v3/trades/{tk}", params=p, timeout=10)
        if r_t.status_code == 200:
            trades = r_t.json().get("results", [])
        r_q = requests.get(f"https://api.polygon.io/v3/quotes/{tk}", params=p, timeout=10)
        if r_q.status_code == 200:
            quotes = r_q.json().get("results", [])
    except Exception:
        pass
    if len(trades) < 10:
        res = {"lr_ofi": 0.0, "block_ofi": 0.0, "odd_ofi": 0.0, "msyn": 0.0}
        _TQ_SHORT_MEMO[key] = res
        return res
    t_ts = np.array([int(x.get("sip_timestamp") or x.get("participant_timestamp") or 0) for x in trades], dtype=np.int64)
    t_px = np.array([float(x.get("price", 0.0)) for x in trades], dtype=float)
    t_sz = np.array([float(x.get("size", 0.0)) for x in trades], dtype=float)
    signs, ask_step_dn = np.sign(np.diff(t_px, prepend=t_px[0])), 0.0
    if len(quotes) >= 5:
        q_ts  = np.array([int(x.get("sip_timestamp") or x.get("participant_timestamp") or 0) for x in quotes], dtype=np.int64)
        q_bid = np.array([float(x.get("bid_price", 0.0)) for x in quotes], dtype=float)
        q_ask = np.array([float(x.get("ask_price", 0.0)) for x in quotes], dtype=float)
        vq = (q_bid > 0) & (q_ask > 0) & (q_ask >= q_bid)
        if np.any(vq):
            q_ts, q_bid, q_ask = q_ts[vq], q_bid[vq], q_ask[vq]
            q_mid = 0.5 * (q_bid + q_ask)
            mid_at_t = q_mid[np.clip(np.searchsorted(q_ts, t_ts, side="right") - 1, 0, len(q_mid) - 1)]
            signs = np.where(t_px > mid_at_t, 1.0, np.where(t_px < mid_at_t, -1.0, 0.0))
            ad = np.diff(q_ask)
            ask_step_dn = float((np.sum(ad < -1e-4) - np.sum(ad > 1e-4)) / max(len(q_ask), 1))
    for i in range(1, len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i - 1] if signs[i - 1] != 0 else 1.0
    tot_vol, signed_vol = max(float(np.sum(t_sz)), 1.0), t_sz * signs
    lr_ofi = float(np.sum(signed_vol) / tot_vol)
    odd_m, blk_m = t_sz < 100, (t_sz >= 1000) | ((t_sz * t_px) >= 20000.0)
    odd_ofi = float(np.sum(signed_vol[odd_m]) / max(float(np.sum(t_sz[odd_m])), 1.0)) if np.any(odd_m) else 0.0
    blk_ofi = float(np.sum(signed_vol[blk_m]) / max(float(np.sum(t_sz[blk_m])), 1.0)) if np.any(blk_m) else 0.0
    msyn = float(np.clip(-1.25 * lr_ofi - 0.85 * blk_ofi + 0.65 * odd_ofi + 1.10 * ask_step_dn, -2.5, 2.5))
    res = {"lr_ofi": lr_ofi, "block_ofi": blk_ofi, "odd_ofi": odd_ofi, "msyn": msyn}
    _TQ_SHORT_MEMO[key] = res
    return res

for strat in STRATEGIES:
    sid = strat['id']
    opt_n = strat['optimal_n']
    base_filter_id = strat.get('base_filter', sid)
    check_fn = FILTER_FNS[base_filter_id]
    allow_bar0 = strat.get('allow_bar0_flush', True)
    cont_pb = strat.get('continue_on_failed_pb', True)
    hyb_1130 = strat.get('hybrid_exit_lt15_1130', False)
    addon_bounce = strat.get('addon_bounce_reshort', False)
    addon_1100 = strat.get('addon_1100_winner', False)
    unlock_15_30_tq = strat.get('unlock_15_30_tq', False)
    show_n = max(opt_n, 2)

    picks = []
    for c in cs_pool:
        tk = c['ticker']
        bd = c.get('bar_data') or scan_bars.get(tk)
        if check_fn(c, bd):
            entry = None
            tq_reject_reason = None
            rth = bd.get('rth', []) if bd else []
            ref_p = c.get('ah_ref_price', c.get('prior_close'))
            if rth and (len(rth) >= PEAK_WINDOW * 2 + 1 or (allow_bar0 and len(rth) >= 4)):
                entry = find_local_high(
                    rth,
                    open_price=c.get('open_price'),
                    gap_pct=c.get('gap_pct'),
                    ref_price=ref_p,
                    allow_bar0_flush=allow_bar0,
                    continue_on_failed_pb=cont_pb,
                )
            gap_v = c.get('gap_pct', 0.0)
            raw_g = c.get('raw_open_gap', gap_v)
            is_15_30_gap = (0.15 <= gap_v < 0.30) or (0.15 <= raw_g < 0.30)
            if entry is not None and not DRY_RUN:
                tq_start_et = entry.get('peak_time', '09:30') if entry['entry_type'] == 'local_high' else '09:30'
                need_tq = (unlock_15_30_tq and is_15_30_gap) or (ENABLE_TQ_LT15_GATE and gap_v < 0.15)
                if need_tq:
                    tq_info = _fetch_live_short_tq(tk, scan_date, entry['entry_time'], start_et=tq_start_et)
                    entry['tq_msyn'] = tq_info['msyn']
                    entry['tq_lr_ofi'] = tq_info['lr_ofi']
                    entry['tq_blk_ofi'] = tq_info['block_ofi']
                    entry['tq_odd_ofi'] = tq_info['odd_ofi']
                    if is_15_30_gap and unlock_15_30_tq:
                        if not (tq_info['msyn'] >= TQ_15_30_MIN_MSYN and tq_info['block_ofi'] <= TQ_15_30_MAX_BLK_OFI):
                            tq_reject_reason = (
                                f"15-30% Gap T&Q Gate [{tq_start_et}->{entry['entry_time']}] "
                                f"(need MSYN>={TQ_15_30_MIN_MSYN:+.2f} & BlkOFI<={TQ_15_30_MAX_BLK_OFI:+.2f}; "
                                f"got MSYN={tq_info['msyn']:+.2f}, BlkOFI={tq_info['block_ofi']:+.2f})"
                            )
                            entry = None
                    elif ENABLE_TQ_LT15_GATE and gap_v < 0.15:
                        if entry['entry_type'] == 'local_high' and tq_info['msyn'] < 0.0:
                            tq_reject_reason = (
                                f"T&Q Bullish Pullback Buying [{tq_start_et}->{entry['entry_time']}] "
                                f"(MSYN={tq_info['msyn']:+.2f}, BlkOFI={tq_info['block_ofi']:+.2f}, OFI={tq_info['lr_ofi']:+.2f})"
                            )
                            entry = None
                        elif entry['entry_type'] == 'bar0_flush' and tq_info['msyn'] < -0.50:
                            tq_reject_reason = (
                                f"T&Q Bullish Absorption (MSYN={tq_info['msyn']:+.2f}, "
                                f"BlkOFI={tq_info['block_ofi']:+.2f}, OFI={tq_info['lr_ofi']:+.2f})"
                            )
                            entry = None
            if entry is not None and SHORT_HARD_STOP_PCT is not None and entry.get('stop_price') is None:
                entry['stop_price'] = round(entry['entry_price'] * (1.0 + SHORT_HARD_STOP_PCT), 2)
            rank_num = len(picks) + 1
            exit_target = get_target_exit_et(
                c.get('open_price'), c.get('gap_pct'),
                entry['entry_type'] if entry else None,
                hybrid_exit_lt15_1130=hyb_1130,
            )
            picks.append({
                'ticker': tk,
                'gap_pct': c['gap_pct'],
                'open_price': c['open_price'],
                'prior_close': c['prior_close'],
                'prior_dvol': c.get('prior_dvol'),
                'priority_tier': 'T1' if _is_tier1_setup(c) else 'T2',
                'float_shares': c.get('float_shares'),
                'locate_badge': c.get('locate_badge', 'ETB'),
                'locate_info': c.get('locate_info', {}),
                'catalyst_badge': c.get('catalyst_badge', 'Clean'),
                'gap_consumed': c.get('gap_consumed'),
                'name': c.get('name', ''),
                'entry': entry,
                'tq_reject_reason': tq_reject_reason,
                'exit_time': exit_target,
                'bar_data': bd,
                'rank': rank_num,
                'is_primary': rank_num <= opt_n,
                'is_addon': False,
            })
            if len(picks) >= show_n:
                break

    # Count confirmed primary morning trades & check idle slots for intraday add-ons
    confirmed_primary = [p for p in picks if p['is_primary'] and p.get('entry') is not None]
    n_active_slots = len(confirmed_primary)

    # U7b: Midday Bounce Re-Short into idle slots (09:55-13:35 ET)
    if addon_bounce and n_active_slots < opt_n:
        for p in list(confirmed_primary):
            if n_active_slots >= opt_n:
                break
            bd = p.get('bar_data')
            rth = bd.get('rth', []) if bd else []
            ent2 = find_secondary_bounce_reshort(rth, p['entry'])
            if ent2 is not None:
                n_active_slots += 1
                picks.append({
                    'ticker': f"{p['ticker']}_B2",
                    'base_ticker': p['ticker'],
                    'gap_pct': p['gap_pct'],
                    'open_price': p['open_price'],
                    'prior_close': p['prior_close'],
                    'float_shares': p.get('float_shares'),
                    'gap_consumed': p.get('gap_consumed'),
                    'name': p.get('name', ''),
                    'entry': ent2,
                    'exit_time': EXIT_TIME_ET,
                    'bar_data': bd,
                    'rank': len(picks) + 1,
                    'is_primary': True,
                    'is_addon': True,
                })

    # U7a: 11:00 ET Winner Pyramid Add-On into idle slots (C3)
    if addon_1100 and n_active_slots < opt_n:
        for p in list(confirmed_primary):
            if n_active_slots >= opt_n:
                break
            bd = p.get('bar_data')
            rth = bd.get('rth', []) if bd else []
            ent_w11 = find_1100_winner_addon(rth, p['entry'])
            if ent_w11 is not None:
                n_active_slots += 1
                picks.append({
                    'ticker': f"{p['ticker']}_W11",
                    'base_ticker': p['ticker'],
                    'gap_pct': p['gap_pct'],
                    'open_price': p['open_price'],
                    'prior_close': p['prior_close'],
                    'float_shares': p.get('float_shares'),
                    'gap_consumed': p.get('gap_consumed'),
                    'name': p.get('name', ''),
                    'entry': ent_w11,
                    'exit_time': EXIT_TIME_ET,
                    'bar_data': bd,
                    'rank': len(picks) + 1,
                    'is_primary': True,
                    'is_addon': True,
                })

    # Compute active_cap50 slot weight for each confirmed primary trade
    n_morning_confirmed = len(confirmed_primary)
    slot_pct = round(min(1.0 / n_active_slots, 0.50) * 100.0, 1) if n_active_slots > 0 else 50.0
    for p in picks:
        p['slot_pct'] = slot_pct if (p['is_primary'] and p.get('entry')) else None
        p['idle_slots_remaining'] = max(0, opt_n - n_active_slots)
        p['n_morning_confirmed'] = n_morning_confirmed
        p['n_active_slots'] = n_active_slots
        p['telemetry'] = get_intraday_trade_telemetry(
            p.get('bar_data'), p.get('entry'), p.get('exit_time', EXIT_TIME_ET)
        ) if p.get('entry') else None

    strategy_picks[sid] = picks
    status = f"{len(picks)} picks ({n_active_slots}/{opt_n} active slots)"
    if picks:
        tks = ', '.join(
            f"{p['ticker']}*" if p.get('entry') else (f"{p['ticker']}" if p['is_primary'] else f"{p['ticker']}(backup)")
            for p in picks
        )
        status += f" — {tks}"
    print(f"      {strat['name']:38s}  N={opt_n}  →  {status}")


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

runtime = time.time() - t_start
print(f"\n  Pipeline: {runtime:.1f}s")

# Deduplicated ticker counts (tracking primary vs backup per strategy)
all_unique = {}
ticker_strats = defaultdict(list)
ticker_primary_strats = defaultdict(list)
for strat in STRATEGIES:
    sid = strat['id']
    for p in strategy_picks[sid]:
        tk = p['ticker']
        if tk not in all_unique:
            all_unique[tk] = p
        if p.get('is_primary', True):
            ticker_strats[tk].append(sid)
            ticker_primary_strats[tk].append(sid)
        else:
            ticker_strats[tk].append(f"{sid}(backup)")

confirmed_tks = {tk for tk, p in all_unique.items() if p.get('entry')}
candidate_tks = {tk for tk, p in all_unique.items() if not p.get('entry')}
n_confirmed = len(confirmed_tks)
n_candidate = len(candidate_tks)
n_unique = len(all_unique)

any_picks = n_unique > 0


# ── RESULT SUMMARY (first thing you see) ──────────────────────────────────────

print(f"\n\n{'='*95}")
print(f"  RESULT — {scan_date}  |  {scan_phase}")
print(f"{'='*95}")

is_post_window = (
    scan_phase.startswith('HOLD / EXIT WINDOW')
    or scan_phase.startswith('MIDDAY ADD-ON WINDOW')
    or scan_phase.startswith('POST-WINDOW')
    or scan_phase == 'HISTORICAL'
)

def _format_telem_short(p):
    tel = p.get('telemetry')
    if not tel:
        return ""
    cur_pst = _et_to_pst(tel['cur_time'])
    if tel.get('stopped_out'):
        return f"  [STOPPED_OUT@{cur_pst} PST: ${tel['cur_price']:.2f} ({tel['pnl_pct']:+.2f}%) — DO NOT ENTER / COVER]"
    state_lbl = f"COVERED@{cur_pst} PST" if tel['exited'] else f"LIVE@{cur_pst} PST"
    return f"  [{state_lbl}: ${tel['cur_price']:.2f} ({tel['pnl_pct']:+.2f}% | MFE +{tel['mfe_pct']:.1f}%)]"

if not any_picks:
    if is_pre_entry:
        print(f"\n  No qualifying candidates yet — re-run after 06:32 PST.")
    else:
        print(f"\n  No qualifying stocks today. No action needed.")
elif n_confirmed > 0 and n_candidate == 0:
    print(f"\n  CONFIRMED TRADES: {n_confirmed}")
    for tk in sorted(confirmed_tks):
        p = all_unique[tk]
        e = p['entry']
        tel = p.get('telemetry') or {}
        strats = ', '.join(ticker_strats[tk])
        role = 'ADD-ON LEG' if p.get('is_addon') else ('PRIMARY' if ticker_primary_strats[tk] else 'BACKUP ONLY')
        sz_str = f"size={p['slot_pct']:.1f}%" if p.get('slot_pct') else ""
        ex_pst = _et_to_pst(p.get('exit_time', EXIT_TIME_ET))
        stop_s = f"  stop=${e['stop_price']:.2f}" if e.get('stop_price') is not None else ""
        ex_str = f"exit={ex_pst} PST{stop_s}"
        ent_pst = _et_to_pst(e['entry_time'])
        tel_str = _format_telem_short(p)
        tag_lbl = 'STOPPED  ' if tel.get('stopped_out') else 'CONFIRMED'
        print(f"    {tag_lbl}  {tk:8s}  gap={p['gap_pct']*100:+.1f}%  "
              f"entry=${e['entry_price']:.2f} @ {ent_pst} PST ({e['entry_type']})  "
              f"{sz_str}  {ex_str}  [{strats}]  ({role}){tel_str}")
elif n_confirmed > 0:
    pending_label = f"{n_candidate} skipped (no morning trigger)" if is_post_window else f"{n_candidate} candidate pending"
    print(f"\n  CONFIRMED TRADES: {n_confirmed}  |  {pending_label}")
    for tk in sorted(confirmed_tks):
        p = all_unique[tk]
        e = p['entry']
        tel = p.get('telemetry') or {}
        strats = ', '.join(ticker_strats[tk])
        role = 'ADD-ON LEG' if p.get('is_addon') else ('PRIMARY' if ticker_primary_strats[tk] else 'BACKUP ONLY')
        sz_str = f"size={p['slot_pct']:.1f}%" if p.get('slot_pct') else ""
        ex_pst = _et_to_pst(p.get('exit_time', EXIT_TIME_ET))
        stop_s = f"  stop=${e['stop_price']:.2f}" if e.get('stop_price') is not None else ""
        ex_str = f"exit={ex_pst} PST{stop_s}"
        ent_pst = _et_to_pst(e['entry_time'])
        tel_str = _format_telem_short(p)
        tag_lbl = 'STOPPED  ' if tel.get('stopped_out') else 'CONFIRMED'
        print(f"    {tag_lbl}  {tk:8s}  gap={p['gap_pct']*100:+.1f}%  "
              f"entry=${e['entry_price']:.2f} @ {ent_pst} PST ({e['entry_type']})  "
              f"{sz_str}  {ex_str}  [{strats}]  ({role}){tel_str}")
    for tk in sorted(candidate_tks):
        p = all_unique[tk]
        strats = ', '.join(ticker_strats[tk])
        role = 'PRIMARY' if ticker_primary_strats[tk] else 'BACKUP ONLY'
        tag = 'SKIPPED  ' if is_post_window else 'CANDIDATE'
        msg = 'no morning trigger by 06:44 PST — skip' if is_post_window else 'watch for Bar0 Flush (06:33 PST) or Local High (<=06:44 PST)'
        print(f"    {tag}  {tk:8s}  gap={p['gap_pct']*100:+.1f}%  "
              f"open=${p['open_price']:.2f}  — {msg}  "
              f"[{strats}]  ({role})")
else:
    print(f"\n  CONFIRMED TRADES: NONE")
    msg = 'no morning trigger by 06:44 PST — no trades today' if is_post_window else 'DO NOT enter yet'
    print(f"  {n_unique} unique candidate{'s' if n_unique > 1 else ''} — {msg}")
    for tk in sorted(candidate_tks):
        p = all_unique[tk]
        strats = ', '.join(ticker_strats[tk])
        role = 'PRIMARY' if ticker_primary_strats[tk] else 'BACKUP ONLY'
        print(f"    {tk:8s}  gap={p['gap_pct']*100:+.1f}%  "
              f"open=${p['open_price']:.2f}  [{strats}]  ({role})")

if is_pre_entry and any_picks:
    print(f"\n  Entries require RTH bars (06:33–06:44 PST). Re-run after 06:33 PST.")

if any_picks:
    stop_lbl = f"-{SHORT_HARD_STOP_PCT*100:.0f}% Hard Stop (NO Trailing Profit Lock)" if SHORT_HARD_STOP_PCT is not None else "None"
    tq_lbl = "15–30% Bearish T&Q + <15% T&Q (Choice 2 Picky)" if ENABLE_TQ_LT15_GATE else "15–30% Bearish T&Q Only (Choice 1 Max Profit)"
    print(f"\n  Version: {STRATEGY_VERSION}  |  Sizing: Active Slot min(1/k, 50%)  |  Price: $10–${PRICE_MAX:.0f}  |  Stop: {stop_lbl}")
    print(f"  Gates: IBKR ETB + Pre-Open Catalyst Veto + NO _B2 Re-Shorts + {tq_lbl}")


# ── OPERATOR PLAYBOOK — WHAT TO DO RIGHT NOW & WHEN TO RUN (PST) ─────────────

print(f"\n\n{'='*95}")
print(f"  OPERATOR PLAYBOOK — {STRATEGY_VERSION} — WHAT TO DO RIGHT NOW & WHEN TO RUN (PST)")
print(f"{'='*95}")

# 1. Immediate Action Items
print(f"\n  [A] IMMEDIATE ACTION ITEMS (Current Clock: {now_pst_hhmm} PST):")
if is_pre_entry:
    print(f"      1. DO NOT SHORT YET. Market is in pre-entry phase ({scan_phase}).")
    if any_picks:
        cand_list = ', '.join(sorted(candidate_tks))
        print(f"      2. Keep watchlist ready: {cand_list}.")
    print(f"      3. Re-run this script at 06:33:10 PST to check for Bar 0 Opening Flush.")
elif not any_picks:
    print(f"      1. NO ACTION TODAY. Zero stocks passed the ETB + Catalyst + gap + price + liquidity filters.")
    print(f"      2. You are done for the day — no need to re-run this script.")
else:
    step_idx = 1
    if n_confirmed > 0:
        # Sort confirmed trades by entry time then ticker
        sorted_conf = sorted(confirmed_tks, key=lambda k: (all_unique[k]['entry']['entry_time'], k))
        for tk in sorted_conf:
            p = all_unique[tk]
            e = p['entry']
            tel = p.get('telemetry') or {}
            base_tk = p.get('base_ticker', tk)
            strats = ', '.join(ticker_strats[tk])
            sz = p.get('slot_pct') or 50.0
            ex_pst = _et_to_pst(p.get('exit_time', EXIT_TIME_ET))
            ent_pst = _et_to_pst(e['entry_time'])
            etype = e['entry_type']
            leg_desc = f"ADD-ON SHORT ({etype}) on {base_tk}" if p.get('is_addon') else f"PRIMARY SHORT ({etype}) on {base_tk}"
            ex_reason = get_exit_reason_pst(
                p.get('open_price'), p.get('gap_pct'), etype,
                hybrid_exit_lt15_1130=('V8.2' in ticker_strats[tk] or ('C2' in ticker_strats[tk] and len(ticker_strats[tk]) == 1)),
            )
            sl_pct_lbl = f"-{SHORT_HARD_STOP_PCT*100:.0f}%" if SHORT_HARD_STOP_PCT is not None else "Stop"

            if tel.get('stopped_out'):
                cur_pst = _et_to_pst(tel['cur_time'])
                print(f"      {step_idx}. [{tk}] {leg_desc} [{strats}]:")
                print(f"         -> STOPPED OUT ({sl_pct_lbl} Hard Stop ${e['stop_price']:.2f} Hit @ {cur_pst} PST): "
                      f"Triggered @ ${e['entry_price']:.2f} ({ent_pst} PST, peak=${e['peak_high']:.2f}) | "
                      f"Covered @ ${tel['cur_price']:.2f} | Locked PnL: {tel['pnl_pct']:+.2f}% (Size: {sz:.1f}%)")
                print(f"         -> WARNING: DO NOT ENTER LATE! If still short {base_tk}, COVER IMMEDIATELY.")
            elif tel.get('exited'):
                cur_pst = _et_to_pst(tel['cur_time'])
                stop_ok_s = f" ({sl_pct_lbl} Stop ${e['stop_price']:.2f} held)" if e.get('stop_price') is not None else ""
                print(f"      {step_idx}. [{tk}] {leg_desc} [{strats}]:")
                print(f"         -> COMPLETED: Entered @ ${e['entry_price']:.2f} ({ent_pst} PST){stop_ok_s} | "
                      f"Covered @ ${tel['cur_price']:.2f} ({cur_pst} PST — {ex_reason}) | "
                      f"Locked PnL: {tel['pnl_pct']:+.2f}% (Size: {sz:.1f}%)")
            else:
                live_s = f" | Now ${tel['cur_price']:.2f} ({tel['pnl_pct']:+.2f}%, MFE +{tel['mfe_pct']:.1f}%)" if tel else ""
                stop_line = (
                    f" | {sl_pct_lbl} Hard Stop: ${e['stop_price']:.2f} (+{SHORT_HARD_STOP_PCT*100:.0f}% above ${e['entry_price']:.2f} entry — NO trailing profit lock)"
                    if e.get('stop_price') is not None else ""
                )
                print(f"      {step_idx}. [{tk}] {leg_desc} [{strats}]:")
                print(f"         -> ACTION: Ensure SHORT is open on {base_tk} (triggered @ ${e['entry_price']:.2f} at {ent_pst} PST){live_s}{stop_line}")
                print(f"         -> SIZE & EXIT: Allocate {sz:.1f}% of SHORT sleeve. "
                      f"Hold and COVER at {ex_pst} PST ({ex_reason}).")
            step_idx += 1

    if not is_post_window and n_candidate > 0:
        pend_list = ', '.join(sorted(candidate_tks))
        print(f"      {step_idx}. PENDING MORNING CANDIDATES ({pend_list}):")
        print(f"         -> DO NOT SHORT YET. Re-run every 1-2 minutes until 06:44:30 PST.")
        step_idx += 1
    elif is_post_window and n_confirmed == 0:
        print(f"      1. NO TRADES TRIGGERED BY 06:44 PST. All morning candidates failed confirmation.")
        print(f"      2. DO NOT SHORT anything today. Scanning is complete for {scan_date}.")

# 2. Next Run Guidance
print(f"\n  [B] WHEN & HOW OFTEN TO RUN THIS SCRIPT NEXT (PST):")
any_addon_strat = any(s.get('addon_bounce_reshort') or s.get('addon_1100_winner') for s in STRATEGIES)
if now_et_hhmm < '09:45':
    print(f"      • RIGHT NOW (Morning Window 06:33–06:44 PST):")
    print(f"        Run every 1–2 minutes (or at 06:33, 06:37, 06:41, and 06:44:30 PST) to catch morning triggers.")
elif not any_addon_strat:
    print(f"      • MORNING WINDOW CLOSED (V8.2-S has NO midday _B2 re-shorts):")
    if n_confirmed > 0:
        print(f"        {n_confirmed} morning short(s) active. No more scans needed today!")
        print(f"        Keep your -10% hard stop active and cover at your scheduled exit (08:30 / 09:30 / 11:00 PST).")
    else:
        print(f"        0 trades triggered by 06:44 PST. You are DONE for the day!")
elif now_et_hhmm <= '13:35':
    print(f"      • LEGACY MIDDAY ADD-ON MODE ACTIVE (06:55–10:35 PST).")
else:
    print(f"      • ENTRY WINDOW CLOSED (Now past 10:35 PST). Cover any remaining open shorts by 11:00 PST (Hard Flat).")

# 3. Quick Reference Daily Schedule (PST)
print(f"\n  [C] DAILY {STRATEGY_VERSION} SHORT RUN & EXIT SCHEDULE (PST TIME ZONE):")
print(f"      ┌──────────────────┬──────────────────────┬───────────────────────────────────────────────────────────┐")
print(f"      │ Window (PST)     │ How Often to Run     │ What the Script Checks & What You Do                      │")
print(f"      ├──────────────────┼──────────────────────┼───────────────────────────────────────────────────────────┤")
print(f"      │ 06:31–06:32 PST  │ Once at 06:31:15 PST │ Locks top 3 ETB + Catalyst-clean candidates (Watchlist)   │")
print(f"      │ 06:33–06:44 PST  │ Every 1–2 minutes    │ Confirms Bar0 Flush (06:33) & 3-Bar Local High shorts     │")
print(f"      │ 06:44:30 PST     │ Morning Scan DONE    │ No _B2 re-shorts! Place -10% hard stop & let winners run  │")
print(f"      │ 08:30 PST        │ Exit Checkpoint #1   │ Cover ALL <15% gap shorts & $50+ local_high spikes        │")
print(f"      │ 09:30 PST        │ Exit Checkpoint #2   │ Cover <$50 local_high spikes (>=15% gaps)                 │")
print(f"      │ 11:00 PST        │ Exit Checkpoint #3   │ Cover ALL remaining bar0_flush shorts (100% FLAT)         │")
print(f"      └──────────────────┴──────────────────────┴───────────────────────────────────────────────────────────┘")


# ── STRATEGY DETAIL ───────────────────────────────────────────────────────────

print(f"\n\n{'='*95}")
print(f"  STRATEGY DETAIL")
print(f"{'='*95}")

for strat in STRATEGIES:
    sid = strat['id']
    picks = strategy_picks[sid]
    opt_n = strat['optimal_n']

    print(f"\n  {'─'*90}")
    print(f"  ┌ {strat['name']}  |  N={opt_n}  |  "
          f"12.5M Sh={strat['post_sharpe']:.2f}  AvgNet={strat['post_avg']:+.2f}%  "
          f"HardStop={strat['worst_loss']:+.1f}%")
    print(f"  │ {strat['setup']}")

    if not picks:
        print(f"  │ → No qualifying stocks.")
        print(f"  └{'─'*89}")
        continue

    for i, p in enumerate(picks):
        rank = i + 1
        is_primary = p.get('is_primary', rank <= opt_n)
        rank_label = '  (ADD-ON LEG into idle slot)' if p.get('is_addon') else ('' if is_primary else f'  (backup — N={opt_n} optimal)')

        fs = p.get('float_shares')
        fs_str = f"{fs/1e6:.0f}M" if fs and fs < 1e9 else (f"{fs/1e9:.1f}B" if fs else 'n/a')

        entry = p.get('entry')
        tel = p.get('telemetry')
        tel_inline = ""
        if tel:
            cur_pst = _et_to_pst(tel['cur_time'])
            if tel.get('stopped_out'):
                st_lbl = f"STOPPED OUT @ {cur_pst} PST"
            else:
                st_lbl = f"Covered @ {cur_pst} PST" if tel['exited'] else f"Live @ {cur_pst} PST"
            tel_inline = f" | {st_lbl}: ${tel['cur_price']:.2f} ({tel['pnl_pct']:+.2f}%)"

        if entry:
            status_tag = 'STOPPED' if (tel and tel.get('stopped_out')) else 'CONFIRMED'
            etype = entry.get('entry_type', 'local_high')
            sz_s = f" | Size: {p['slot_pct']:.1f}%" if p.get('slot_pct') else ""
            ex_pst = _et_to_pst(p.get('exit_time', EXIT_TIME_ET))
            stop_s = f" | Stop: ${entry['stop_price']:.2f}" if entry.get('stop_price') is not None else ""
            ex_s = f" | Exit: {ex_pst} PST{stop_s}"
            ent_pst = _et_to_pst(e['entry_time']) if (e := entry) else ""
            if etype == 'bounce_reshort':
                entry_str = (f"SHORT ADD-ON (Bounce Re-Short) @ ${entry['entry_price']:.2f} @ {ent_pst} PST "
                             f"(MFE +{entry.get('mfe_before_bounce_pct', 0):.1f}%, bounce +{entry.get('bounce_from_low_pct', 0):.1f}%, "
                             f"pb -{entry.get('pullback_pct', 0):.2f}%){sz_s}{ex_s}{tel_inline}")
            elif etype == 'winner_1100':
                entry_str = (f"SHORT ADD-ON (08:00 Pyramid) @ ${entry['entry_price']:.2f} @ {ent_pst} PST "
                             f"(08:00 PnL +{entry.get('pullback_pct', 0):.2f}%, bounce {entry.get('bounce_frac_pct', 0):.1f}% of drop){sz_s}{ex_s}{tel_inline}")
            else:
                pb_str = f", pullback -{entry.get('pullback_pct', 0):.2f}%" if entry.get('pullback_pct') is not None else ""
                tq_s = (f" | T&Q MSYN={entry['tq_msyn']:+.2f}, BlkOFI={entry['tq_blk_ofi']:+.2f}" if 'tq_msyn' in entry else "")
                entry_str = (f"SHORT ({etype.upper()}) @ ${entry['entry_price']:.2f} "
                             f"(peak bar {entry['peak_bar']} @ ${entry['peak_high']:.2f}{pb_str} → entry {ent_pst} PST){sz_s}{ex_s}{tq_s}{tel_inline}")
        elif is_post_window:
            status_tag = 'SKIPPED'
            if p.get('tq_reject_reason'):
                entry_str = f"blocked by {p['tq_reject_reason']} — skip"
            else:
                entry_str = "no morning trigger (Bar0 Flush or Local High >=0.5% PB) by 06:44 PST — skip"
        else:
            status_tag = 'CANDIDATE'
            entry_str = "watch for Bar0 Flush (06:33 PST) or Local High (bars 3-11, >=0.5% PB)"

        bd = p.get('bar_data')
        pm_info = ''
        if bd and not p.get('is_addon'):
            pm_bars = bd.get('pm', [])
            pm_830 = [(t, b) for t, b in pm_bars if '08:30' <= t < '09:30']
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
            scan_rth = [(t, b) for t, b in rth if t <= SCAN_TIME_ET]
            if scan_rth:
                bar0 = scan_rth[0][1]
                bar0_color = 'RED' if bar0['c'] <= bar0['o'] else 'GREEN'
                pm_info += f"  bar0={bar0_color}"
                if len(scan_rth) >= 2:
                    bar1 = scan_rth[1][1]
                    drop_2b = (bar1['c'] - bar0['o']) / bar0['o'] * 100 if bar0['o'] > 0 else 0
                    pm_info += f"  2bar={drop_2b:+.1f}%"

        loc_b = p.get('locate_badge', 'ETB')
        cat_b = p.get('catalyst_badge', 'Clean')
        print(f"  │ [{status_tag}] #{rank}  {p['ticker']:8s}  "
              f"gap={p['gap_pct']*100:+.1f}%  ${p['open_price']:.2f}  "
              f"float={fs_str}  locate={loc_b}  cat={cat_b}{rank_label}")
        print(f"  │   → {entry_str}")
        if pm_info:
            print(f"  │   {pm_info}")

    if picks:
        idle_rem = picks[0].get('idle_slots_remaining', 0)
        n_morn = picks[0].get('n_morning_confirmed', 0)
        if not (strat.get('addon_bounce_reshort') or strat.get('addon_1100_winner')):
            print(f"  │   [Active Slots: {n_morn}/{opt_n} morning short(s) | _B2 Midday Re-Shorts OFF (V8.2-S)]")
        elif n_morn == 0 and is_post_window:
            print(f"  │   [Idle Slots: {idle_rem}/{opt_n} unused — 0 morning triggers confirmed]")
        elif idle_rem == 0:
            print(f"  │   [Slots Full: {opt_n}/{opt_n} active slots filled]")
        elif n_morn >= 1 and idle_rem > 0:
            print(f"  │   [Idle Slots: {idle_rem}/{opt_n} open — legacy midday add-on watch active]")
    print(f"  └{'─'*89}")


# ── CROSS-STRATEGY AGREEMENT (only when multiple strategies active via --legacy) ─

if len(STRATEGIES) > 1 and len(ticker_strats) > 1:
    print(f"\n\n{'='*95}")
    print(f"  CROSS-STRATEGY AGREEMENT")
    print(f"{'='*95}")

    sorted_tickers = sorted(
        ticker_strats.items(),
        key=lambda x: (-int(bool(all_unique[x[0]].get('entry'))), -len(ticker_primary_strats[x[0]]), -len(x[1]))
    )
    rows = []
    for tk, strats in sorted_tickers:
        n_prim = len(ticker_primary_strats[tk])
        strat_str = ', '.join(strats)
        p = all_unique[tk]
        gap_str = f"+{p['gap_pct']*100:.1f}%"
        e = p.get('entry')
        tel = p.get('telemetry') or {}
        if e and tel.get('stopped_out'):
            status = f"STOPPED OUT @ {_et_to_pst(tel['cur_time'])} PST (${tel['cur_price']:.2f})"
        elif e:
            status = f"CONFIRMED ({e['entry_type']} @ {_et_to_pst(e['entry_time'])} PST)"
        else:
            status = 'SKIPPED' if is_post_window else 'CANDIDATE'
        conf = 'HIGH' if n_prim >= 3 else ('MED' if n_prim == 2 else ('LOW' if n_prim == 1 else 'BACKUP'))
        rows.append([tk, gap_str, f"{n_prim} ({len(strats)} tot)", conf, status, strat_str])

    print()
    print(tabulate(rows,
                   headers=['Ticker', 'Gap', '#Primary', 'Conf', 'Status', 'Strategies'],
                   tablefmt='simple', stralign='left'))

    top_tk, top_strats = sorted_tickers[0]
    print(f"\n  → Best: {top_tk} in {len(ticker_primary_strats[top_tk])} primary strategies ({', '.join(top_strats)})")

print(f"\n  Runtime: {runtime:.1f}s")
print(f"{'='*95}\n")


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
                pm_830 = [(t, b) for t, b in pm_bars if '08:30' <= t < '09:30']
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
                scan_rth = [(t, b) for t, b in rth if t <= SCAN_TIME_ET]
                if scan_rth:
                    b0 = scan_rth[0][1]
                    bar0_color = 'RED' if b0['c'] <= b0['o'] else 'GREEN'
                    if len(scan_rth) >= 2:
                        bar2_drop = round((scan_rth[1][1]['c'] - b0['o']) / b0['o'] * 100, 2) if b0['o'] > 0 else None

            gc = p.get('gap_consumed')
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
                'gap_consumed': round(gc, 3) if gc is not None else None,
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


if not DRY_RUN:
    log_scan_to_bq()
