# %% Agent 3 — Long Strategy Scanner (Pension Account)
# Two LONG strategies — no shorting, no leverage:
#   A6: Gap-down sustained rise → buy at local low, hold T+1
#   C1: Multi-day reversal → prior 5d decline + gap-up, buy open, hold T+3
#
# Usage:
#   MODE = 'scan'            → morning scanner (default)
#   MODE = 'fill'            → fill T+1/T+2/T+3 outcomes after close
#   MODE = 'report'          → performance summary
#   SCAN_DATE = '2026-09-15' → historical (omit for live/today)
#   DRY_RUN = True           → skip DB writes
#
# Run every morning at 6:15 AM PST (9:15 AM EST)
# Colab-cell friendly.

import os, sys, json, time, re, logging, sqlite3, asyncio
from datetime import date, datetime, timedelta
from pathlib import Path
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
    from google.colab import auth
    try: auth.authenticate_user()
    except: pass
except: pass


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

MODE = 'scan'                # 'scan' | 'fill' | 'report'
SCAN_DATE = None             # None = today, 'YYYY-MM-DD' = historical
DRY_RUN = False
REPORT_WEEKS = 4

POLYGON_API_KEY = os.environ["POLYGON_API_KEY"]
_BASE = "https://api.polygon.io"

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

DB_PATH = os.path.join(BASE_DIR, 'data', 'agent3.db')
FLOAT_CACHE_PATH = os.path.join(BASE_DIR, 'data', 'float_cache.json')
LOG_PATH = os.path.join(BASE_DIR, 'logs', 'agent3.log')

# A6 config — gap-down sustained rise LONG
A6_GAP_MIN_PCT     = 5.0       # gap-DOWN >= 5%
A6_PRICE_MIN       = 10.0
A6_TOP_N           = 5         # top 5 by abs(gap_pct)
A6_PRE_FILTER_N    = 15
A6_HOLD_DAYS       = 1         # T+1 (E1/E1b: Sharpe 0.600 vs T+2 0.458; pre-market 08:00 peak 0.703)
LOW_WINDOW         = 3
MAX_LOW_BAR        = 12
FALLBACK_BAR       = 15
SKIP_BOUNCE_PCT    = 0.15

# C1 config — multi-day reversal LONG
C1_GAP_MIN_PCT     = 3.0       # gap-UP >= 3%
C1_GAP_MAX_PCT     = 50.0      # gap cap (exclude splits/corp actions)
C1_PRICE_MIN       = 20.0
C1_DECLINE_5D_MIN  = 0.15      # prior 5d decline >= 15%
C1_HOLD_DAYS       = 3         # T+3

FLOAT_MIN          = 10_000_000
CAPITAL            = 100_000
PM_BAR_WORKERS     = 10

_TK_RE = re.compile(r'^[A-Z]{1,5}$')
ET = pytz.timezone('America/New_York')
PT = pytz.timezone('America/Los_Angeles')

_W = 120

_EXCLUDE_TK = {'SPY','QQQ','IWM','DIA','VXX','UVXY','SQQQ','TQQQ','SPXU','SPXL',
               'SDOW','UDOW','SDS','SSO','SH','LABU','LABD','SOXL','SOXS',
               'FNGU','FNGD','NUGT','DUST','JNUG','JDST','TNA','TZA',
               'UPRO','VIXY','SVXY','TVIX'}
_EXCLUDE_SUFFIX = ('W', 'WS', 'U', 'R')

_NYSE_HOLIDAYS = {
    '2025-01-01','2025-01-20','2025-02-17','2025-04-18','2025-05-26',
    '2025-06-19','2025-07-04','2025-09-01','2025-11-27','2025-12-25',
    '2026-01-01','2026-01-19','2026-02-16','2026-04-03','2026-05-25',
    '2026-06-19','2026-07-03','2026-09-07','2026-11-26','2026-12-25',
    '2027-01-01','2027-01-18','2027-02-15','2027-03-26','2027-05-31',
    '2027-06-18','2027-07-05','2027-09-06','2027-11-25','2027-12-24',
    '2028-01-17','2028-02-21','2028-04-14','2028-05-29',
    '2028-06-19','2028-07-04','2028-09-04','2028-11-23','2028-12-25',
    '2029-01-01','2029-01-15','2029-02-19','2029-03-30','2029-05-28',
    '2029-06-19','2029-07-04','2029-09-03','2029-11-22','2029-12-25',
    '2030-01-01','2030-01-21','2030-02-18','2030-04-19','2030-05-27',
    '2030-06-19','2030-07-04','2030-09-02','2030-11-28','2030-12-25',
}

_EMBEDDED_FLOAT = {
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
_FLOAT_UNIVERSE = {tk for tk, fs in _EMBEDDED_FLOAT.items() if fs >= FLOAT_MIN}


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    fmt = '%(asctime)s | %(levelname)s | %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for h in root.handlers[:]:
        root.removeHandler(h)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(ch)
    fh = logging.FileHandler(LOG_PATH)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(fh)
    return logging.getLogger('agent3')

log = setup_logging()


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

def init_db(db_path=DB_PATH):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")

    conn.executescript("""
    CREATE TABLE IF NOT EXISTS daily_scans (
        scan_date        TEXT PRIMARY KEY,
        run_timestamp    TEXT,
        mode             TEXT,
        n_a6_universe    INTEGER,
        n_c1_universe    INTEGER,
        n_a6_selected    INTEGER,
        n_c1_selected    INTEGER,
        spy_gap_pct      REAL,
        runtime_seconds  REAL
    );

    CREATE TABLE IF NOT EXISTS a6_candidates (
        candidate_id     TEXT PRIMARY KEY,
        scan_date        TEXT,
        ticker           TEXT,
        gap_pct          REAL,
        prior_close      REAL,
        open_price       REAL,
        float_shares     REAL,
        asset_type       TEXT,
        open_pattern     TEXT,
        entry_type       TEXT,
        entry_price      REAL,
        entry_time       TEXT,
        rank_by_gap      INTEGER,
        passed_cs        INTEGER,
        passed_pattern   INTEGER,
        passed_entry     INTEGER,
        selected         INTEGER
    );

    CREATE TABLE IF NOT EXISTS c1_candidates (
        candidate_id     TEXT PRIMARY KEY,
        scan_date        TEXT,
        ticker           TEXT,
        gap_pct          REAL,
        prior_close      REAL,
        open_price       REAL,
        float_shares     REAL,
        asset_type       TEXT,
        prior_5d_return  REAL,
        close_5d_ago     REAL,
        rank_by_gap      INTEGER,
        passed_cs        INTEGER,
        selected         INTEGER
    );

    CREATE TABLE IF NOT EXISTS selected_trades (
        trade_id         TEXT PRIMARY KEY,
        scan_date        TEXT,
        strategy         TEXT,
        ticker           TEXT,
        selection_rank   INTEGER,
        gap_pct          REAL,
        prior_close      REAL,
        open_price       REAL,
        entry_price      REAL,
        entry_time       TEXT,
        entry_type       TEXT,
        float_shares     REAL,
        open_pattern     TEXT,
        prior_5d_return  REAL,
        t1_date          TEXT,
        t2_date          TEXT,
        t3_date          TEXT,
        t0_close         REAL DEFAULT NULL,
        t1_close         REAL DEFAULT NULL,
        t2_close         REAL DEFAULT NULL,
        t3_close         REAL DEFAULT NULL,
        pnl_eod          REAL DEFAULT NULL,
        pnl_t1           REAL DEFAULT NULL,
        pnl_t2           REAL DEFAULT NULL,
        pnl_t3           REAL DEFAULT NULL,
        actually_traded  INTEGER DEFAULT 0,
        actual_pnl       REAL DEFAULT NULL,
        notes            TEXT DEFAULT NULL
    );

    CREATE TABLE IF NOT EXISTS trading_calendar (
        date             TEXT PRIMARY KEY,
        is_trading_day   INTEGER
    );

    CREATE INDEX IF NOT EXISTS idx_a6_cand_date ON a6_candidates(scan_date);
    CREATE INDEX IF NOT EXISTS idx_c1_cand_date ON c1_candidates(scan_date);
    CREATE INDEX IF NOT EXISTS idx_trades_date ON selected_trades(scan_date);
    CREATE INDEX IF NOT EXISTS idx_trades_strat ON selected_trades(strategy);
    CREATE INDEX IF NOT EXISTS idx_trades_exit ON selected_trades(t2_date);
    """)

    existing = conn.execute("SELECT COUNT(*) FROM trading_calendar").fetchone()[0]
    if existing == 0:
        log.info("Populating trading calendar 2025-2030...")
        rows = []
        d = date(2025, 1, 1)
        end_d = date(2030, 12, 31)
        while d <= end_d:
            ds = d.strftime('%Y-%m-%d')
            is_td = 1 if d.weekday() < 5 and ds not in _NYSE_HOLIDAYS else 0
            rows.append((ds, is_td))
            d += timedelta(days=1)
        conn.executemany("INSERT OR IGNORE INTO trading_calendar VALUES (?,?)", rows)
        conn.commit()
        log.info(f"  Calendar: {len(rows)} days, {sum(r[1] for r in rows)} trading days")

    conn.close()
    return db_path


# ═══════════════════════════════════════════════════════════════════════════════
# FLOAT CACHE
# ═══════════════════════════════════════════════════════════════════════════════

def load_float_cache(path=FLOAT_CACHE_PATH):
    cache = dict(_EMBEDDED_FLOAT)
    if os.path.exists(path):
        with open(path) as f:
            file_cache = json.load(f)
        for tk, fs in file_cache.items():
            if isinstance(fs, (int, float)) and fs > 0:
                cache[tk] = int(fs)
        log.info(f"Float cache: {len(cache)} tickers ({len(file_cache)} from file)")
    else:
        log.info(f"Float cache: {len(cache)} tickers (embedded only)")
    return cache


def save_float_cache(cache, path=FLOAT_CACHE_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(cache, f, indent=1)


# ═══════════════════════════════════════════════════════════════════════════════
# TRADING CALENDAR HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _get_conn():
    return sqlite3.connect(DB_PATH)


def is_trading_day(d):
    ds = d if isinstance(d, str) else d.strftime('%Y-%m-%d')
    conn = _get_conn()
    row = conn.execute("SELECT is_trading_day FROM trading_calendar WHERE date=?", (ds,)).fetchone()
    conn.close()
    if row is None:
        dt = datetime.strptime(ds, '%Y-%m-%d').date() if isinstance(ds, str) else ds
        return dt.weekday() < 5 and ds not in _NYSE_HOLIDAYS
    return bool(row[0])


def get_prior_trading_date(target_date):
    if isinstance(target_date, str):
        target_date = datetime.strptime(target_date, '%Y-%m-%d').date()
    d = target_date - timedelta(days=1)
    for _ in range(10):
        if is_trading_day(d): return d
        d -= timedelta(days=1)
    raise ValueError(f"No trading day found before {target_date}")


def get_nth_prior_td(target_date, n):
    if isinstance(target_date, str):
        target_date = datetime.strptime(target_date, '%Y-%m-%d').date()
    count = 0
    d = target_date - timedelta(days=1)
    for _ in range(60):
        if is_trading_day(d):
            count += 1
            if count >= n: return d
        d -= timedelta(days=1)
    raise ValueError(f"Couldn't find {n}th prior TD from {target_date}")


def get_next_n_trading_dates(from_date, n=3):
    if isinstance(from_date, str):
        from_date = datetime.strptime(from_date, '%Y-%m-%d').date()
    results = []
    d = from_date + timedelta(days=1)
    for _ in range(30):
        if is_trading_day(d):
            results.append(d)
            if len(results) >= n: break
        d += timedelta(days=1)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# POLYGON — API HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _pg(url, params=None):
    p = dict(params or {}); p['apiKey'] = POLYGON_API_KEY
    for attempt in range(5):
        try:
            r = requests.get(url, params=p, timeout=30)
            if r.status_code == 429:
                wait = 12 * (attempt + 1)
                log.warning(f"Rate limited (429), waiting {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            wait = 5 * (2 ** attempt)
            log.debug(f"  [retry {attempt+1}/5] waiting {wait}s...")
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
    log.info(f"Grouped daily ({dt_str}): {len(out)} tickers")
    return out


def fetch_snapshot_all():
    """Live snapshot — returns dict keyed by ticker with 'o' set to current/premarket price."""
    data = _pg(f"{_BASE}/v2/snapshot/locale/us/markets/stocks/tickers",
               {"include_otc": "false"})
    out = {}
    for t in data.get('tickers', []):
        tk = t.get('ticker', '')
        if not _TK_RE.match(tk): continue
        day = t.get('day', {})
        prev = t.get('prevDay', {})
        cur_price = day.get('o') or day.get('c') or 0
        if cur_price <= 0:
            mn = t.get('min', {})
            cur_price = mn.get('c') or mn.get('o') or 0
        if cur_price <= 0:
            continue
        out[tk] = {'o': float(cur_price), 'h': float(day.get('h', 0)),
                   'l': float(day.get('l', 0)), 'c': float(day.get('c', 0)),
                   'v': int(day.get('v', 0))}
    log.info(f"Snapshot (live): {len(out)} tickers with price data")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# ASYNC MINUTE BAR FETCH (RTH only — for A6 pattern + entry)
# ═══════════════════════════════════════════════════════════════════════════════

async def _fetch_bars_one(session, tk, dt, sem):
    url = f"{_BASE}/v2/aggs/ticker/{tk}/range/1/minute/{dt}/{dt}"
    params = {"adjusted": "true", "sort": "asc", "limit": "500", "apiKey": POLYGON_API_KEY}
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
                if '09:30' <= t_str < '16:00':
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
# GAP-DOWN PATTERN CLASSIFICATION (A6)
# ═══════════════════════════════════════════════════════════════════════════════

def classify_gap_down_pattern(rth_bars):
    if len(rth_bars) < 60:
        return 'no_data'
    open_price = rth_bars[0][1]['o']
    if open_price <= 0:
        return 'no_data'
    max_high_30m = max(b[1]['h'] for b in rth_bars[:30])
    close_bar60 = rth_bars[59][1]['c']
    rise_30m = (max_high_30m - open_price) / abs(open_price)
    recovery_denom = max_high_30m - open_price
    recovery = (close_bar60 - open_price) / recovery_denom if recovery_denom > 0 else 0
    if rise_30m > 0.03 and recovery > 0.50:
        return 'sustained_rise'
    elif rise_30m > 0.03 and recovery <= 0.50:
        return 'rise_then_drop'
    elif rise_30m <= 0.01:
        return 'continued_drop'
    else:
        return 'mild_rise'


# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL LOW DETECTION (A6 entry)
# ═══════════════════════════════════════════════════════════════════════════════

def find_local_low(bars, window=LOW_WINDOW, max_low_bar=MAX_LOW_BAR,
                   fallback_bar=FALLBACK_BAR):
    if len(bars) < window * 2 + 1:
        return None
    open_price = bars[0][1]['o']
    for i in range(window, min(max_low_bar, len(bars) - window)):
        cl = bars[i][1]['l']
        if all(bars[i - j][1]['l'] > cl and bars[i + j][1]['l'] > cl
               for j in range(1, window + 1)):
            ci = i + window
            if ci < len(bars):
                return {
                    'entry_price': bars[ci][1]['c'],
                    'entry_time': bars[ci][0],
                    'entry_type': 'local_low',
                    'low_bar': i,
                }
    if fallback_bar is not None and fallback_bar < len(bars):
        fb_price = bars[fallback_bar][1]['c']
        if open_price > 0 and (fb_price - open_price) / open_price > SKIP_BOUNCE_PCT:
            return None
        return {
            'entry_price': fb_price,
            'entry_time': bars[fallback_bar][0],
            'entry_type': 'fallback',
            'low_bar': None,
        }
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# DB WRITE — SCAN RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

def write_scan_to_db(scan_date_str, a6_all, c1_all, a6_selected, c1_selected,
                     mode, runtime, spy_gap):
    conn = _get_conn()

    conn.execute("""INSERT OR REPLACE INTO daily_scans VALUES (?,?,?,?,?,?,?,?,?)""",
                 (scan_date_str, datetime.now(pytz.UTC).isoformat(), mode,
                  len(a6_all), len(c1_all), len(a6_selected), len(c1_selected),
                  spy_gap, runtime))

    for c in a6_all:
        cid = f"A6_{scan_date_str}_{c['ticker']}"
        conn.execute("""INSERT OR REPLACE INTO a6_candidates VALUES
            (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (cid, scan_date_str, c['ticker'], c['gap_pct'], c['prior_close'],
             c['open_price'], c.get('float_shares'), c.get('asset_type'),
             c.get('open_pattern'), c.get('entry_type'), c.get('entry_price'),
             c.get('entry_time'), c.get('rank_by_gap'),
             int(c.get('passed_cs', False)), int(c.get('passed_pattern', False)),
             int(c.get('passed_entry', False)), int(c.get('selected', False))))

    for c in c1_all:
        cid = f"C1_{scan_date_str}_{c['ticker']}"
        conn.execute("""INSERT OR REPLACE INTO c1_candidates VALUES
            (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (cid, scan_date_str, c['ticker'], c['gap_pct'], c['prior_close'],
             c['open_price'], c.get('float_shares'), c.get('asset_type'),
             c.get('prior_5d_return'), c.get('close_5d_ago'), c.get('rank_by_gap'),
             int(c.get('passed_cs', False)), int(c.get('selected', False))))

    nxt = get_next_n_trading_dates(scan_date_str, max(A6_HOLD_DAYS, C1_HOLD_DAYS))
    t_dates = [d.strftime('%Y-%m-%d') for d in nxt]

    for s in a6_selected:
        tid = f"A6_{scan_date_str}_{s['ticker']}"
        conn.execute("""INSERT OR REPLACE INTO selected_trades
            (trade_id, scan_date, strategy, ticker, selection_rank, gap_pct,
             prior_close, open_price, entry_price, entry_time, entry_type,
             float_shares, open_pattern, prior_5d_return,
             t1_date, t2_date, t3_date)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (tid, scan_date_str, 'A6', s['ticker'], s.get('selection_rank'),
             s['gap_pct'], s['prior_close'], s['open_price'],
             s.get('entry_price'), s.get('entry_time'), s.get('entry_type'),
             s.get('float_shares'), s.get('open_pattern'), None,
             t_dates[0] if len(t_dates) > 0 else None,
             t_dates[1] if len(t_dates) > 1 else None,
             t_dates[2] if len(t_dates) > 2 else None))

    for s in c1_selected:
        tid = f"C1_{scan_date_str}_{s['ticker']}"
        conn.execute("""INSERT OR REPLACE INTO selected_trades
            (trade_id, scan_date, strategy, ticker, selection_rank, gap_pct,
             prior_close, open_price, entry_price, entry_time, entry_type,
             float_shares, open_pattern, prior_5d_return,
             t1_date, t2_date, t3_date)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (tid, scan_date_str, 'C1', s['ticker'], s.get('selection_rank'),
             s['gap_pct'], s['prior_close'], s['open_price'],
             s['open_price'], '09:30', 'open',
             s.get('float_shares'), 'c1_reversal', s.get('prior_5d_return'),
             t_dates[0] if len(t_dates) > 0 else None,
             t_dates[1] if len(t_dates) > 1 else None,
             t_dates[2] if len(t_dates) > 2 else None))

    conn.commit()
    conn.close()
    log.info(f"DB write: A6={len(a6_selected)}, C1={len(c1_selected)} selected trades")


# ═══════════════════════════════════════════════════════════════════════════════
# PRINT WATCHLISTS
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt_float(fs):
    if fs is None: return 'n/a'
    if fs >= 1e9: return f"{fs/1e9:.1f}B"
    return f"{fs/1e6:.0f}M"


def print_watchlists(a6_selected, c1_selected, scan_date_str):
    nxt = get_next_n_trading_dates(scan_date_str, max(A6_HOLD_DAYS, C1_HOLD_DAYS))
    t_strs = [d.strftime('%Y-%m-%d') for d in nxt]
    now_str = datetime.now(PT).strftime('%H:%M:%S PST')

    print(f"\n  {'='*74}")
    print(f"  |{'AGENT 3 — LONG WATCHLIST':^72}|")
    print(f"  |{'Pension account — no shorting, no leverage':^72}|")
    print(f"  |{'':^72}|")
    print(f"  |  Date: {scan_date_str:<20}  Run: {now_str:<30} |")
    print(f"  {'='*74}")

    # ── A6: Gap-Down Sustained Rise ──
    print(f"\n  {'─'*74}")
    print(f"  A6 — GAP-DOWN SUSTAINED RISE  (LONG)")
    print(f"  Strategy: BUY at local low (~bar 6-12), HOLD to T+{A6_HOLD_DAYS} close ({t_strs[A6_HOLD_DAYS-1] if len(t_strs) >= A6_HOLD_DAYS else 'TBD'})")
    print(f"  Filter:   gap-DOWN >= {A6_GAP_MIN_PCT}%, CS only, sustained_rise pattern, local_low entry")
    print(f"  Universe: _FLOAT ({len(_FLOAT_UNIVERSE)} tickers, float >= 10M)")
    print(f"  E1b note: Pre-mkt @08:00 Sharpe 0.703 WR 79% | Close @16:00 Sharpe 0.600")
    print(f"  {'─'*74}")

    if not a6_selected:
        print(f"\n  No A6 candidates today.\n")
    else:
        rows = []
        for s in a6_selected:
            rows.append([
                f"#{s.get('selection_rank', '?')}",
                s['ticker'],
                f"{s['gap_pct']*100:+.1f}%",
                f"${s['open_price']:.2f}",
                f"${s.get('entry_price', 0):.2f}" if s.get('entry_price') else 'TBD',
                s.get('entry_time', 'TBD'),
                s.get('open_pattern', '?'),
                _fmt_float(s.get('float_shares')),
            ])
        print()
        print(tabulate(rows,
                       headers=['Rank', 'Ticker', 'Gap%', 'Open', 'Entry', 'Time', 'Pattern', 'Float'],
                       tablefmt='simple', colalign=('left', 'left', 'right', 'right', 'right', 'left', 'left', 'right')))

        for s in a6_selected:
            print(f"\n  #{s.get('selection_rank', '?')} {s['ticker']}  —  LONG")
            print(f"     Gap: {s['gap_pct']*100:+.1f}%  |  Prior close: ${s['prior_close']:.2f}  |  Open: ${s['open_price']:.2f}")
            if s.get('entry_price'):
                print(f"     Entry: ${s['entry_price']:.2f} at {s.get('entry_time', '?')} ({s.get('entry_type', '?')})")
                print(f"     Exit: T+{A6_HOLD_DAYS} close ({t_strs[A6_HOLD_DAYS-1] if len(t_strs) >= A6_HOLD_DAYS else 'TBD'})")
            else:
                print(f"     Entry: Watch for local low (3-bar window, bars 3-12)")
                print(f"     Exit: T+{A6_HOLD_DAYS} close ({t_strs[A6_HOLD_DAYS-1] if len(t_strs) >= A6_HOLD_DAYS else 'TBD'})")

    # ── C1: Multi-Day Reversal ──
    print(f"\n  {'─'*74}")
    print(f"  C1 — MULTI-DAY REVERSAL  (LONG)")
    print(f"  Strategy: BUY at market open (9:30 AM), HOLD to T+{C1_HOLD_DAYS} close ({t_strs[C1_HOLD_DAYS-1] if len(t_strs) >= C1_HOLD_DAYS else 'TBD'})")
    print(f"  Filter:   prior 5d decline >= {C1_DECLINE_5D_MIN*100:.0f}%, gap-UP {C1_GAP_MIN_PCT}%-{C1_GAP_MAX_PCT}%, CS only, price >= ${C1_PRICE_MIN:.0f}")
    print(f"  Universe: _FLOAT ({len(_FLOAT_UNIVERSE)} tickers, float >= 10M)")
    print(f"  {'─'*74}")

    if not c1_selected:
        print(f"\n  No C1 candidates today.\n")
    else:
        rows = []
        for s in c1_selected:
            p5d = s.get('prior_5d_return')
            p5d_str = f"{p5d*100:+.1f}%" if p5d is not None else 'n/a'
            rows.append([
                f"#{s.get('selection_rank', '?')}",
                s['ticker'],
                f"+{s['gap_pct']*100:.1f}%",
                p5d_str,
                f"${s['open_price']:.2f}",
                _fmt_float(s.get('float_shares')),
            ])
        print()
        print(tabulate(rows,
                       headers=['Rank', 'Ticker', 'Gap%', '5d Decline', 'Open/Entry', 'Float'],
                       tablefmt='simple', colalign=('left', 'left', 'right', 'right', 'right', 'right')))

        for s in c1_selected:
            p5d = s.get('prior_5d_return')
            print(f"\n  #{s.get('selection_rank', '?')} {s['ticker']}  —  LONG")
            print(f"     Gap-UP: +{s['gap_pct']*100:.1f}%  |  Prior close: ${s['prior_close']:.2f}")
            print(f"     5d decline: {p5d*100:+.1f}%" if p5d is not None else "     5d decline: n/a")
            print(f"     Entry: BUY at open ${s['open_price']:.2f}")
            print(f"     Exit: T+{C1_HOLD_DAYS} close ({t_strs[C1_HOLD_DAYS-1] if len(t_strs) >= C1_HOLD_DAYS else 'TBD'})")

    print()


# ═══════════════════════════════════════════════════════════════════════════════
# FILL OUTCOMES
# ═══════════════════════════════════════════════════════════════════════════════

def run_fill_outcomes(fill_date_str=None):
    if fill_date_str is None:
        fill_date_str = datetime.now(ET).strftime('%Y-%m-%d')

    log.info(f"  FILL OUTCOMES — data available through {fill_date_str}")

    conn = _get_conn()
    trades = conn.execute("""
        SELECT trade_id, scan_date, strategy, ticker, entry_price,
               t1_date, t2_date, t3_date,
               t0_close, t1_close, t2_close, t3_close
        FROM selected_trades
        WHERE t0_close IS NULL OR t1_close IS NULL
              OR (strategy='A6' AND t2_close IS NULL)
              OR (strategy='C1' AND t3_close IS NULL)
    """).fetchall()

    if not trades:
        log.info("  No trades need filling.")
        conn.close()
        return

    log.info(f"  {len(trades)} trades need outcome data")

    dates_needed = set()
    for t in trades:
        dates_needed.add(t[1])  # scan_date = T0
        for i in range(5, 8):   # t1_date, t2_date, t3_date
            if t[i]: dates_needed.add(t[i])

    daily_data = {}
    for dt in sorted(dates_needed):
        if dt > fill_date_str: continue
        try:
            daily_data[dt] = fetch_grouped_daily(dt)
            time.sleep(0.05)
        except Exception as e:
            log.warning(f"  Failed to fetch {dt}: {e}")

    float_cache = load_float_cache()
    float_updated = False
    n_filled = 0

    for t in trades:
        tid, scan_date, strategy, ticker, entry_price = t[:5]
        t1_date, t2_date, t3_date = t[5], t[6], t[7]
        t0_close, t1_close, t2_close, t3_close = t[8], t[9], t[10], t[11]

        if entry_price is None or entry_price <= 0:
            continue

        updates = {}

        if t0_close is None and scan_date in daily_data:
            dd = daily_data[scan_date].get(ticker, {})
            if dd.get('c'):
                updates['t0_close'] = dd['c']
                updates['pnl_eod'] = round((dd['c'] - entry_price) / entry_price * 100, 4)

        if t1_close is None and t1_date and t1_date in daily_data:
            dd = daily_data[t1_date].get(ticker, {})
            if dd.get('c'):
                updates['t1_close'] = dd['c']
                updates['pnl_t1'] = round((dd['c'] - entry_price) / entry_price * 100, 4)

        if t2_close is None and t2_date and t2_date in daily_data:
            dd = daily_data[t2_date].get(ticker, {})
            if dd.get('c'):
                updates['t2_close'] = dd['c']
                updates['pnl_t2'] = round((dd['c'] - entry_price) / entry_price * 100, 4)

        if t3_close is None and t3_date and t3_date in daily_data:
            dd = daily_data[t3_date].get(ticker, {})
            if dd.get('c'):
                updates['t3_close'] = dd['c']
                updates['pnl_t3'] = round((dd['c'] - entry_price) / entry_price * 100, 4)

        if updates:
            set_clause = ', '.join(f"{k}=?" for k in updates.keys())
            conn.execute(f"UPDATE selected_trades SET {set_clause} WHERE trade_id=?",
                         list(updates.values()) + [tid])
            n_filled += 1
            pnl_str = ', '.join(f"{k}={v:+.2f}%" if 'pnl' in k else f"{k}={v:.2f}"
                                for k, v in updates.items())
            log.info(f"  {tid}: {pnl_str}")

        if ticker not in float_cache:
            try:
                url = f"{_BASE}/v3/reference/tickers/{ticker}"
                data = _pg(url)
                so = data.get('results', {}).get('share_class_shares_outstanding')
                if so and so > 0:
                    float_cache[ticker] = int(so)
                    float_updated = True
                time.sleep(0.05)
            except Exception:
                pass

    conn.commit()
    conn.close()
    log.info(f"  Filled {n_filled}/{len(trades)} trades")

    if float_updated:
        save_float_cache(float_cache)


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def run_report(weeks=REPORT_WEEKS):
    conn = _get_conn()
    cutoff = (datetime.now() - timedelta(weeks=weeks)).strftime('%Y-%m-%d')

    scans = conn.execute("""
        SELECT COUNT(*), SUM(n_a6_selected), SUM(n_c1_selected)
        FROM daily_scans WHERE scan_date >= ?
    """, (cutoff,)).fetchone()

    trades = conn.execute("""
        SELECT trade_id, scan_date, strategy, ticker, gap_pct, entry_price,
               pnl_eod, pnl_t1, pnl_t2, pnl_t3, actually_traded
        FROM selected_trades WHERE scan_date >= ?
        ORDER BY scan_date
    """, (cutoff,)).fetchall()
    conn.close()

    n_scans = scans[0] or 0
    n_a6 = scans[1] or 0
    n_c1 = scans[2] or 0

    print(f"\n  {'='*74}")
    print(f"  |{'AGENT 3 — PERFORMANCE REPORT':^72}|")
    print(f"  {'='*74}")
    print(f"\n  Period: Last {weeks} weeks (cutoff: {cutoff})")
    print(f"  Total scans:     {n_scans}")
    print(f"  Total A6:        {n_a6}")
    print(f"  Total C1:        {n_c1}")

    for strat, pnl_col, hold in [('A6', 'pnl_t1', A6_HOLD_DAYS), ('C1', 'pnl_t3', C1_HOLD_DAYS)]:
        strat_trades = [t for t in trades if t[2] == strat]
        pnl_idx = 7 if strat == 'A6' else 9  # pnl_t1 or pnl_t3
        pnls = [t[pnl_idx] for t in strat_trades if t[pnl_idx] is not None]

        print(f"\n  ── {strat} — T+{hold} Performance ──")
        if not pnls:
            print(f"  No T+{hold} outcomes available yet.")
            continue

        a = np.array(pnls)
        n = len(a)
        avg = float(np.mean(a)); std = float(np.std(a, ddof=1)) if n > 1 else 0
        wr = float((a > 0).mean())
        sharpe = avg / std if std > 0 else 0
        se = std / np.sqrt(n)

        print(f"  Trades:     {n}")
        print(f"  Avg PnL:    {avg:+.2f}%")
        print(f"  Median:     {float(np.median(a)):+.2f}%")
        print(f"  Win Rate:   {wr*100:.0f}%")
        print(f"  Sharpe:     {sharpe:.3f}")
        print(f"  CI 95%:     [{avg-1.96*se:+.2f}%, {avg+1.96*se:+.2f}%]")
        print(f"  Max loss:   {float(a.min()):+.2f}%")
        print(f"  Max gain:   {float(a.max()):+.2f}%")

    if trades:
        print(f"\n  ── Trade Log ──")
        rows = []
        for t in trades:
            _, dt, strat, tk, gap, ep, p_eod, p_t1, p_t2, p_t3, traded = t
            primary_pnl = p_t2 if strat == 'A6' else p_t3
            rows.append([
                dt, strat, tk,
                f"{gap*100:+.1f}%" if gap else '—',
                f"${ep:.2f}" if ep else '—',
                f"{p_eod:+.2f}%" if p_eod is not None else '—',
                f"{primary_pnl:+.2f}%" if primary_pnl is not None else '—',
                '✓' if traded else '',
            ])
        print()
        print(tabulate(rows,
                       headers=['Date', 'Strat', 'Ticker', 'Gap', 'Entry', 'EOD', 'Exit PnL', 'Traded'],
                       tablefmt='simple'))
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
#
#  MAIN SCANNER
#
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

def run_scan(scan_date_str=None, dry_run=False):
    t_start = time.time()

    if scan_date_str is None:
        scan_date_str = datetime.now(ET).strftime('%Y-%m-%d')
        mode = 'live'
    else:
        mode = 'historical'

    log.info(f"{'='*_W}")
    log.info(f"  AGENT 3 — Long Strategy Scanner (Pension)")
    log.info(f"  Date: {scan_date_str}  |  Mode: {mode}  |  Dry run: {dry_run}")
    log.info(f"  A6: gap-DOWN >= {A6_GAP_MIN_PCT}%, sustained_rise, local_low only, T+{A6_HOLD_DAYS}")
    log.info(f"  C1: 5d decline >= {C1_DECLINE_5D_MIN*100:.0f}%, gap-UP {C1_GAP_MIN_PCT}%-{C1_GAP_MAX_PCT}%, open entry, T+{C1_HOLD_DAYS}")
    log.info(f"  Universe: {len(_FLOAT_UNIVERSE)} tickers (float >= 10M)")
    log.info(f"{'='*_W}")

    if not is_trading_day(scan_date_str):
        log.info(f"  {scan_date_str} is not a trading day. Exiting.")
        return

    if not dry_run:
        log.info(f"\n  ── Step 0: Fill outcomes for open trades ──")
        run_fill_outcomes(fill_date_str=scan_date_str)

    init_db()
    float_cache = load_float_cache()

    # ── Step 1: Fetch prior day + lookback closes ──
    prior_date = get_prior_trading_date(scan_date_str)
    prior_str = prior_date.strftime('%Y-%m-%d')
    log.info(f"  Prior trading day: {prior_str}")

    t1 = time.time()
    prior_data = fetch_grouped_daily(prior_str)
    today_data = fetch_snapshot_all() if mode == 'live' else fetch_grouped_daily(scan_date_str)

    # C1 lookback: need close from 5 TDs ago
    close_5d_date = get_nth_prior_td(prior_str, 4)
    close_5d_str = close_5d_date.strftime('%Y-%m-%d')
    close_5d_data = fetch_grouped_daily(close_5d_str)
    log.info(f"  [1] Daily data: prior={prior_str}, 5d_ago={close_5d_str}  ({time.time()-t1:.1f}s)")

    # SPY gap
    spy_gap = None
    if 'SPY' in prior_data and 'SPY' in today_data:
        spy_prev = prior_data['SPY']['c']
        spy_cur = today_data['SPY']['o']
        if spy_prev > 0 and spy_cur > 0:
            spy_gap = round((spy_cur - spy_prev) / spy_prev * 100, 3)
            log.info(f"  SPY gap: {spy_gap:+.2f}%")

    # ── Step 2: Scan A6 candidates (gap-DOWN from _FLOAT_UNIVERSE) ──
    t2 = time.time()
    gap_min_a6 = A6_GAP_MIN_PCT / 100.0
    a6_all = []
    for tk in _FLOAT_UNIVERSE:
        if tk in _EXCLUDE_TK: continue
        if any(tk.endswith(s) for s in _EXCLUDE_SUFFIX) and len(tk) > 2: continue
        if tk not in prior_data or tk not in today_data: continue
        prev_c = prior_data[tk]['c']
        if prev_c <= 0: continue
        cur_o = today_data[tk]['o']
        if cur_o <= 0 or cur_o < A6_PRICE_MIN: continue
        gap = (cur_o - prev_c) / prev_c
        if gap > -gap_min_a6: continue

        fs = float_cache.get(tk, _EMBEDDED_FLOAT.get(tk))
        a6_all.append({
            'ticker': tk, 'gap_pct': gap, 'prior_close': prev_c,
            'open_price': cur_o, 'float_shares': fs,
            'passed_cs': False, 'passed_pattern': False, 'passed_entry': False,
            'selected': False,
        })

    a6_all.sort(key=lambda x: x['gap_pct'])  # most negative first
    for i, c in enumerate(a6_all):
        c['rank_by_gap'] = i + 1
    log.info(f"  [2] A6 scan: {len(a6_all)} gap-DOWN candidates  ({time.time()-t2:.1f}s)")

    # ── Step 3: Scan C1 candidates (gap-UP + prior 5d decline from _FLOAT_UNIVERSE) ──
    t3 = time.time()
    gap_min_c1 = C1_GAP_MIN_PCT / 100.0
    gap_max_c1 = C1_GAP_MAX_PCT / 100.0
    c1_all = []
    for tk in _FLOAT_UNIVERSE:
        if tk in _EXCLUDE_TK: continue
        if any(tk.endswith(s) for s in _EXCLUDE_SUFFIX) and len(tk) > 2: continue
        if tk not in prior_data or tk not in today_data: continue
        prev_c = prior_data[tk]['c']
        if prev_c <= 0: continue
        cur_o = today_data[tk]['o']
        if cur_o <= 0 or cur_o < C1_PRICE_MIN: continue
        gap = (cur_o - prev_c) / prev_c
        if gap < gap_min_c1 or gap > gap_max_c1: continue

        close_5d_ago = close_5d_data.get(tk, {}).get('c')
        if not close_5d_ago or close_5d_ago <= 0: continue
        prior_5d_return = (prev_c - close_5d_ago) / close_5d_ago
        if prior_5d_return > -C1_DECLINE_5D_MIN: continue

        fs = float_cache.get(tk, _EMBEDDED_FLOAT.get(tk))
        at = _ticker_ref.get(tk, {}).get('type')

        c1_all.append({
            'ticker': tk, 'gap_pct': gap, 'prior_close': prev_c,
            'open_price': cur_o, 'float_shares': fs,
            'prior_5d_return': prior_5d_return, 'close_5d_ago': close_5d_ago,
            'asset_type': at, 'passed_cs': False, 'selected': False,
        })

    c1_all.sort(key=lambda x: -x['gap_pct'])  # largest gap-up first
    for i, c in enumerate(c1_all):
        c['rank_by_gap'] = i + 1
    log.info(f"  [3] C1 scan: {len(c1_all)} reversal candidates  ({time.time()-t3:.1f}s)")

    # ── Step 4: Fetch minute bars for A6 (top PRE_FILTER_N) ──
    t4 = time.time()
    a6_bar_pairs = [(c['ticker'], scan_date_str) for c in a6_all[:A6_PRE_FILTER_N]]
    a6_bars = fetch_minute_bars_batch(a6_bar_pairs) if a6_bar_pairs else {}
    log.info(f"  [4] A6 minute bars: {len(a6_bars)}/{len(a6_bar_pairs)} fetched  ({time.time()-t4:.1f}s)")

    # ── Step 5: Fetch ticker refs for CS filter ──
    t5 = time.time()
    ref_tickers = set()
    for c in a6_all[:A6_PRE_FILTER_N]:
        ref_tickers.add(c['ticker'])
    for c in c1_all:
        ref_tickers.add(c['ticker'])
    uncached_refs = [tk for tk in ref_tickers if tk not in _ticker_ref]
    if uncached_refs:
        fetch_ticker_refs(uncached_refs)
    log.info(f"  [5] Ticker refs: {len(ref_tickers)} checked  ({time.time()-t5:.1f}s)")

    # ── Step 6: Classify A6 patterns + entries, build selected lists ──
    t6 = time.time()

    a6_selected = []
    for c in a6_all[:A6_PRE_FILTER_N]:
        tk = c['ticker']
        at = _ticker_ref.get(tk, {}).get('type', 'UNK')
        c['asset_type'] = at
        if at != 'CS': continue
        c['passed_cs'] = True

        key = (tk, scan_date_str)
        rth = a6_bars.get(key)
        if not rth or len(rth) < 60:
            c['open_pattern'] = 'no_data'
            continue
        pat = classify_gap_down_pattern(rth)
        c['open_pattern'] = pat
        if pat != 'sustained_rise': continue
        c['passed_pattern'] = True

        entry = find_local_low(rth)
        if entry is None or entry['entry_type'] != 'local_low':
            continue
        c['passed_entry'] = True
        c['entry_price'] = entry['entry_price']
        c['entry_time'] = entry['entry_time']
        c['entry_type'] = entry['entry_type']
        c['selected'] = True
        a6_selected.append(c)

        if len(a6_selected) >= A6_TOP_N:
            break

    for i, s in enumerate(a6_selected):
        s['selection_rank'] = i + 1

    # C1 selection: CS filter, then all qualifying
    c1_selected = []
    for c in c1_all:
        tk = c['ticker']
        at = _ticker_ref.get(tk, {}).get('type', 'UNK')
        c['asset_type'] = at
        if at != 'CS': continue
        c['passed_cs'] = True
        c['selected'] = True
        c1_selected.append(c)

    for i, s in enumerate(c1_selected):
        s['selection_rank'] = i + 1

    log.info(f"  [6] Selection: A6={len(a6_selected)} (from {len(a6_all)}), "
             f"C1={len(c1_selected)} (from {len(c1_all)})  ({time.time()-t6:.1f}s)")

    # ── Step 7: Write to DB ──
    runtime = time.time() - t_start
    if not dry_run:
        write_scan_to_db(scan_date_str, a6_all, c1_all, a6_selected, c1_selected,
                         mode, runtime, spy_gap)
    else:
        log.info("  DRY RUN — skipping DB write")

    # ── Step 8: Print watchlists ──
    print_watchlists(a6_selected, c1_selected, scan_date_str)

    # ── Summary ──
    log.info(f"\n  Funnel A6: {len(a6_all)} gap-DOWN → "
             f"{sum(1 for c in a6_all if c.get('passed_cs'))} CS → "
             f"{sum(1 for c in a6_all if c.get('passed_pattern'))} sustained_rise → "
             f"{len(a6_selected)} local_low selected")
    log.info(f"  Funnel C1: {len(c1_all)} reversal → "
             f"{sum(1 for c in c1_all if c.get('passed_cs'))} CS → "
             f"{len(c1_selected)} selected")
    log.info(f"  Total runtime: {runtime:.1f}s")

    return a6_selected, c1_selected


# ═══════════════════════════════════════════════════════════════════════════════
# DISPATCH
# ═══════════════════════════════════════════════════════════════════════════════

init_db()

if MODE == 'scan':
    run_scan(scan_date_str=SCAN_DATE, dry_run=DRY_RUN)
elif MODE == 'fill':
    run_fill_outcomes(fill_date_str=SCAN_DATE)
elif MODE == 'report':
    run_report(weeks=REPORT_WEEKS)
else:
    print(f"Unknown MODE: {MODE}. Use 'scan', 'fill', or 'report'.")
