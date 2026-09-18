# %% Agent 1 — Technical Gap-Fade Scanner
# Single script: scanner + outcome filler + performance report
# Run in notebook every morning at 6:15 AM PST (9:15 AM EST)
#
# E2 optimal strategy: N=1 CS+SD, SHORT at local high, exit at 17:00 ET (2 PM PST)
# Shows top 3 candidates for safety; trade #1 only.
# Backtest: $2.34M on $100K pool, Sharpe 0.589, +14.62% avg, 78% WR (174 days)
#
# Usage:
#   MODE = 'scan'           → morning scanner (default)
#   MODE = 'fill'           → fill T0/T1/T2 outcomes after close
#   MODE = 'report'         → performance summary
#   SCAN_DATE = '2026-09-08' → historical backtest (omit for live/today)
#   DRY_RUN = True          → skip DB writes

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
# CONFIG — edit these cell vars before running
# ═══════════════════════════════════════════════════════════════════════════════

MODE = 'scan'                # 'scan' | 'fill' | 'report'
                             # 'scan' does fill-then-scan automatically
SCAN_DATE = None             # None = today (live), 'YYYY-MM-DD' = historical
DRY_RUN = False              # True = skip DB writes
REPORT_WEEKS = 4             # for MODE='report'

POLYGON_API_KEY = os.environ["POLYGON_API_KEY"]
POLYGON_BASE = "https://api.polygon.io"

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

DB_PATH = os.path.join(BASE_DIR, 'data', 'agent1.db')
FLOAT_CACHE_PATH = os.path.join(BASE_DIR, 'data', 'float_cache.json')
LOG_PATH = os.path.join(BASE_DIR, 'logs', 'agent1.log')

GAP_MIN = 0.05               # 5% minimum gap
PRICE_MIN = 10.0             # $10 minimum price
FLOAT_MIN_SHARES = 10_000_000
TOP_N = 3
PM_BAR_WORKERS = 10
TICKER_REGEX = re.compile(r'^[A-Z]{1,5}$')
# Exclude SPAC units/warrants/rights (ending in U, W, WS, R) and test tickers
_EXCLUDED_SUFFIXES = ('W', 'WS', 'U', 'R')
_EXCLUDED_TICKERS = {'SPY', 'QQQ', 'IWM', 'DIA', 'VXX', 'UVXY', 'SQQQ', 'TQQQ',
                     'SPXU', 'SPXL', 'SDOW', 'UDOW', 'SDS', 'SSO', 'SH',
                     'LABU', 'LABD', 'SOXL', 'SOXS', 'FNGU', 'FNGD',
                     'NUGT', 'DUST', 'JNUG', 'JDST', 'TNA', 'TZA',
                     'UPRO', 'VIXY', 'SVXY', 'TVIX'}

ET = pytz.timezone('America/New_York')
PT = pytz.timezone('America/Los_Angeles')

_W = 120

# NYSE holidays 2026-2030 (New Year, MLK, Presidents, Good Friday, Memorial,
# Juneteenth, Independence, Labor, Thanksgiving, Christmas + early closes ignored)
_NYSE_HOLIDAYS = {
    # 2026
    '2026-01-01', '2026-01-19', '2026-02-16', '2026-04-03', '2026-05-25',
    '2026-06-19', '2026-07-03', '2026-09-07', '2026-11-26', '2026-12-25',
    # 2027
    '2027-01-01', '2027-01-18', '2027-02-15', '2027-03-26', '2027-05-31',
    '2027-06-18', '2027-07-05', '2027-09-06', '2027-11-25', '2027-12-24',
    # 2028
    '2028-01-17', '2028-02-21', '2028-04-14', '2028-05-29',
    '2028-06-19', '2028-07-04', '2028-09-04', '2028-11-23', '2028-12-25',
    # 2029
    '2029-01-01', '2029-01-15', '2029-02-19', '2029-03-30', '2029-05-28',
    '2029-06-19', '2029-07-04', '2029-09-03', '2029-11-22', '2029-12-25',
    # 2030
    '2030-01-01', '2030-01-21', '2030-02-18', '2030-04-19', '2030-05-27',
    '2030-06-19', '2030-07-04', '2030-09-02', '2030-11-28', '2030-12-25',
}

_EMBEDDED_FLOAT = {
    'AAPL':14594180000, 'AARD':21884158, 'ABM':58580923, 'ACHR':770023800, 'ACMR':64657388,
    'ACV':10396028, 'ACVA':169807980, 'ADBE':397500000, 'AEHR':32620450, 'AENT':50979630,
    'AEON':49882790, 'AGCO':70031729, 'AGRZ':22573405, 'AHMA':10965000, 'ALAB':173485104,
    'AMAT':793597443, 'AMD':1632475042, 'AME':229203002, 'APA':350351510, 'APLD':291469112,
    'ARBE':122649743, 'ARM':1068078760, 'ASAN':159441754, 'ASO':62018371, 'ASTS':299789305,
    'ATEC':154451076, 'ATHE':18208368, 'ATI':136170383, 'AVAV':50822615, 'AXGN':53761513,
    'AXTA':214019997, 'AXTI':65573212, 'BABA':2485623615, 'BBIO':195487474, 'BBNX':44989688,
    'BE':294527346, 'BEAM':103287665, 'BHVN':151043781, 'BJDX':4468051, 'BLZE':61900000,
    'BNC':41173850, 'BNKK':8000940, 'BOW':32943005, 'BRNX':720888, 'BRZE':112631669,
    'BTBT':360633503, 'CAN':749194858, 'CAT':459674889, 'CC':150498995, 'CCB':15286327,
    'CIFR':415030722, 'CLSK':256817073, 'COIN':222803032, 'COLA':4494439, 'COO':195030630,
    'CORZ':321340441, 'CPRT':925811482, 'CRDO':187951918, 'CRMT':8663493, 'CSR':16794151,
    'CVX':1975771274, 'DBGI':933509, 'DELL':324873640, 'DFNS':1663806, 'DKS':65355923,
    'DNOW':180785891, 'DOV':134680225, 'DPRO':34375971, 'DSS':10042518, 'EAT':41765010,
    'EDHL':5959275, 'ELOG':12832000, 'ET':3443299601, 'ETN':388400000, 'EVRG':230567172,
    'EXEL':247781692, 'FCEL':79954196, 'FCX':1436017523, 'FEIM':11177530, 'FIZZ':93615302,
    'FLNC':143163588, 'FLWS':37030262, 'GDDY':126647704, 'GENC':12338845, 'GLXG':5601515,
    'GME':448691257, 'GMEX':903642, 'GOOS':46656876, 'GRI':2186115, 'HBM':444144760,
    'HCAI':7629942, 'HGTY':102070523, 'HOFT':10741650, 'HOOD':790630234, 'HPE':1327464779,
    'HPQ':901790969, 'HTLD':77332611, 'HUT':123259468, 'HXHX':8950000, 'IBTA':20197545,
    'IESC':19924356, 'IMO':481577721, 'INBX':14717660, 'INDP':133242324, 'INFQ':225357052,
    'INTC':5044000000, 'INV':84612657, 'IONQ':381002314, 'IP':529569895, 'IPDN':14450323,
    'IREN':394058648, 'KEEL':617573212, 'KPTI':22681460, 'KR':612647282, 'KTOS':187721327,
    'LASE':50177716, 'LEU':19233658, 'LOVE':14638418, 'LULU':105594064, 'LUNR':173231343,
    'M':263037954, 'MARA':386299297, 'MAS':197187430, 'MCFT':16279890, 'MDCX':63180610,
    'META':2205128509, 'MGN':16250000, 'MOBX':16847921, 'MRNA':399235889, 'MSTR':364585501,
    'MU':1129393151, 'MVIS':23788853, 'MXL':90693191, 'NBIS':219465088, 'NEE':2085803189,
    'NEGG':20973423, 'NNE':53698146, 'NTHI':25936365, 'NTLA':140126693, 'NVDA':24100000000,
    'NVS':2034819509, 'OCGN':339110401, 'OKLO':186017650, 'ONDS':571454656, 'OPTT':270138823,
    'OPTX':40279878, 'ORCL':2880471000, 'OTLK':243447554, 'PAVS':856897, 'PCLA':9613805,
    'PDEX':3203334, 'PDSB':55971338, 'PGY':72137266, 'PL':340354193, 'PLNT':75216965,
    'PLTR':2300713329, 'PLUG':1397195278, 'PMTS':11530669, 'PRLD':79803485, 'QBTS':368835324,
    'QNTM':3816937, 'RDDT':146103200, 'RDW':249992609, 'REGN':101137842, 'RGTI':333768747,
    'RH':18900969, 'RIOT':375258935, 'RKLB':598350482, 'RLMD':106669846, 'ROIV':722323015,
    'RTX':1347758144, 'SCCO':834331706, 'SDRL':62541443, 'SHMD':60958903, 'SHOE':27151308,
    'SIG':39329783, 'SKYX':135430994, 'SLS':201945709, 'SLXN':5755872, 'SMCI':656965384,
    'SMR':410389522, 'SNDK':146419001, 'SNX':79958752, 'SNYR':15079956, 'SOUN':411576753,
    'SRPT':105627021, 'SSL':647761670, 'SSNC':234654996, 'STRL':30588518, 'STX':226644518,
    'SURG':52908423, 'SWKS':150472782, 'SWRD':211149963, 'SWVL':9964344, 'TEN':30127603,
    'TNON':1032578, 'TPST':16539717, 'TRUG':1903708, 'TSLA':3949547394, 'TTAN':82746425,
    'TTD':426771565, 'TYRA':59676939, 'UBER':2042560121, 'UNFI':60519140, 'UPST':97313046,
    'URI':62242489, 'VAC':34395320, 'VICR':34389014, 'VOYG':58217592, 'VRT':384988173,
    'VSME':2750784, 'VST':335635195, 'WDH':279111455, 'WULF':498968677, 'YARW':2669746,
    'ZEO':41703489, 'ZUMZ':16872215,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 3.1  LOGGING
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

    return logging.getLogger('agent1')

log = setup_logging()


# ═══════════════════════════════════════════════════════════════════════════════
# 3.2  DATABASE
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
        n_gap_universe   INTEGER,
        n_t4_t5          INTEGER,
        n_after_price    INTEGER,
        n_after_float    INTEGER,
        n_after_retrace  INTEGER,
        n_selected       INTEGER,
        spy_gap_pct      REAL,
        runtime_seconds  REAL
    );

    CREATE TABLE IF NOT EXISTS scan_candidates (
        candidate_id     TEXT PRIMARY KEY,
        scan_date        TEXT,
        ticker           TEXT,
        gap_pct          REAL,
        tier             TEXT,
        prior_close      REAL,
        open_price       REAL,
        float_shares     REAL,
        pm_high          REAL,
        pm_last          REAL,
        pm_retrace_pct   REAL,
        passed_price     INTEGER,
        passed_float     INTEGER,
        passed_retrace   INTEGER,
        passed_all       INTEGER,
        rank_by_gap      INTEGER,
        selected         INTEGER
    );

    CREATE TABLE IF NOT EXISTS selected_trades (
        trade_id         TEXT PRIMARY KEY,
        scan_date        TEXT,
        ticker           TEXT,
        selection_rank   INTEGER,
        tier             TEXT,
        gap_pct          REAL,
        prior_close      REAL,
        open_price       REAL,
        pm_retrace_pct   REAL,
        float_shares     REAL,
        t1_date          TEXT,
        t2_date          TEXT,
        entry_price      REAL DEFAULT NULL,
        entry_time       TEXT DEFAULT NULL,
        entry_type       TEXT DEFAULT NULL,
        slippage_pct     REAL DEFAULT NULL,
        borrow_rate_pct  REAL DEFAULT NULL,
        t0_close         REAL DEFAULT NULL,
        t1_close         REAL DEFAULT NULL,
        t2_close         REAL DEFAULT NULL,
        pnl_30m          REAL DEFAULT NULL,
        pnl_60m          REAL DEFAULT NULL,
        pnl_eod          REAL DEFAULT NULL,
        pnl_t1           REAL DEFAULT NULL,
        pnl_t2           REAL DEFAULT NULL,
        actually_traded  INTEGER DEFAULT 0,
        actual_pnl       REAL DEFAULT NULL,
        notes            TEXT DEFAULT NULL
    );

    CREATE TABLE IF NOT EXISTS trading_calendar (
        date             TEXT PRIMARY KEY,
        is_trading_day   INTEGER
    );

    CREATE INDEX IF NOT EXISTS idx_candidates_date ON scan_candidates(scan_date);
    CREATE INDEX IF NOT EXISTS idx_trades_date ON selected_trades(scan_date);
    CREATE INDEX IF NOT EXISTS idx_trades_t2 ON selected_trades(t2_date);
    """)

    # populate trading calendar 2026-01-01 through 2030-12-31
    existing = conn.execute("SELECT COUNT(*) FROM trading_calendar").fetchone()[0]
    if existing == 0:
        log.info("Populating trading calendar 2026-2030...")
        rows = []
        d = date(2026, 1, 1)
        end = date(2030, 12, 31)
        while d <= end:
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
# 3.3  FLOAT CACHE
# ═══════════════════════════════════════════════════════════════════════════════

def load_float_cache(path=FLOAT_CACHE_PATH):
    cache = dict(_EMBEDDED_FLOAT)
    if os.path.exists(path):
        with open(path) as f:
            file_cache = json.load(f)
        for tk, fs in file_cache.items():
            if isinstance(fs, (int, float)) and fs > 0:
                cache[tk] = int(fs)
        log.info(f"Float cache: {len(cache)} tickers ({len(file_cache)} from file, {len(_EMBEDDED_FLOAT)} embedded)")
    else:
        log.info(f"Float cache: {len(cache)} tickers (embedded only, no file at {path})")
    return cache


def save_float_cache(cache, path=FLOAT_CACHE_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(cache, f, indent=1)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.4  TRADING CALENDAR HELPERS
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
        if is_trading_day(d):
            return d
        d -= timedelta(days=1)
    raise ValueError(f"No trading day found in 10 days before {target_date}")


def get_next_n_trading_dates(from_date, n=2):
    if isinstance(from_date, str):
        from_date = datetime.strptime(from_date, '%Y-%m-%d').date()
    results = []
    d = from_date + timedelta(days=1)
    for _ in range(30):
        if is_trading_day(d):
            results.append(d)
            if len(results) >= n:
                break
        d += timedelta(days=1)
    return results


def get_t1_t2_dates(scan_date):
    nxt = get_next_n_trading_dates(scan_date, 2)
    t1 = nxt[0].strftime('%Y-%m-%d') if len(nxt) >= 1 else None
    t2 = nxt[1].strftime('%Y-%m-%d') if len(nxt) >= 2 else None
    return t1, t2


# ═══════════════════════════════════════════════════════════════════════════════
# 3.5  POLYGON — GROUPED DAILY
# ═══════════════════════════════════════════════════════════════════════════════

def _polygon_get(url, params=None):
    if params is None:
        params = {}
    params['apiKey'] = POLYGON_API_KEY
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code == 429:
        log.warning("Rate limited (429), waiting 12s...")
        time.sleep(12)
        resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_grouped_daily(date_str):
    url = f"{POLYGON_BASE}/v2/aggs/grouped/locale/us/market/stocks/{date_str}"
    data = _polygon_get(url, {"adjusted": "true", "include_otc": "false"})
    results = data.get('results', [])
    out = {}
    for r in results:
        tk = r.get('T', '')
        if not TICKER_REGEX.match(tk):
            continue
        out[tk] = {
            'open': float(r.get('o', 0)),
            'high': float(r.get('h', 0)),
            'low': float(r.get('l', 0)),
            'close': float(r.get('c', 0)),
            'volume': int(r.get('v', 0)),
            'vwap': float(r.get('vw', 0)),
        }
    log.info(f"Grouped daily ({date_str}): {len(out)} tickers fetched")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 3.6  POLYGON — SNAPSHOT (LIVE MODE)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_snapshot_all():
    url = f"{POLYGON_BASE}/v2/snapshot/locale/us/markets/stocks/tickers"
    data = _polygon_get(url, {"include_otc": "false"})
    tickers = data.get('tickers', [])
    out = {}
    for t in tickers:
        tk = t.get('ticker', '')
        if not TICKER_REGEX.match(tk):
            continue
        pm_price = None
        if t.get('todaysChange') is not None and t.get('prevDay', {}).get('c'):
            prev_c = t['prevDay']['c']
        else:
            prev_c = None

        # premarket price
        pm = t.get('min', {})
        if pm and pm.get('c') and pm['c'] > 0:
            pm_price = float(pm['c'])

        if pm_price is None and prev_c:
            pm_price = float(prev_c)

        if pm_price and pm_price > 0:
            out[tk] = {
                'price': pm_price,
                'prev_close': float(prev_c) if prev_c else None,
            }
    log.info(f"Snapshot: {len(out)} tickers with price data")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 3.7  GAP COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_gaps(prior_closes, current_prices, mode='historical'):
    candidates = []
    n_total = 0
    for tk in current_prices:
        if tk not in prior_closes:
            continue
        if tk in _EXCLUDED_TICKERS:
            continue
        if any(tk.endswith(s) for s in _EXCLUDED_SUFFIXES) and len(tk) > 2:
            continue
        prev_c = prior_closes[tk].get('close', 0) if isinstance(prior_closes[tk], dict) else prior_closes[tk]
        if prev_c <= 0:
            continue

        if mode == 'live':
            cur_p = current_prices[tk].get('price', 0)
        else:
            cur_p = current_prices[tk].get('open', 0)
        if cur_p <= 0:
            continue

        n_total += 1
        gap_pct = (cur_p - prev_c) / prev_c

        if gap_pct < GAP_MIN:
            continue
        if cur_p < PRICE_MIN:
            continue

        tier = 'T4' if gap_pct <= 0.08 else 'T5'
        candidates.append({
            'ticker': tk,
            'gap_pct': round(gap_pct, 6),
            'tier': tier,
            'prior_close': round(prev_c, 4),
            'open_price': round(cur_p, 4),
            'passed_price': True,
            'passed_float': True,
            'passed_retrace': True,
            'passed_all': True,
        })

    candidates.sort(key=lambda x: -x['gap_pct'])
    n_gap = len(candidates)
    log.info(f"Gap filter: {n_total} tickers checked, {n_gap} with gap >= {GAP_MIN*100:.0f}% and price >= ${PRICE_MIN}")
    return candidates


# ═══════════════════════════════════════════════════════════════════════════════
# 3.8  FLOAT FILTER
# ═══════════════════════════════════════════════════════════════════════════════

def apply_float_filter(candidates, float_cache):
    n_excluded = 0
    n_miss = 0
    for c in candidates:
        tk = c['ticker']
        fs = float_cache.get(tk)
        c['float_shares'] = fs
        if fs is not None and fs < FLOAT_MIN_SHARES:
            c['passed_float'] = False
            c['passed_all'] = False
            n_excluded += 1
        else:
            c['passed_float'] = True
            if fs is None:
                n_miss += 1
    log.info(f"Float filter: {n_excluded} excluded (<{FLOAT_MIN_SHARES/1e6:.0f}M shares), "
             f"{n_miss} not in cache (included)")
    return candidates


# ═══════════════════════════════════════════════════════════════════════════════
# 3.9  ASYNC PM BAR FETCH
# ═══════════════════════════════════════════════════════════════════════════════

async def _fetch_one_pm(session, ticker, date_str, semaphore):
    url = f"{POLYGON_BASE}/v2/aggs/ticker/{ticker}/range/1/minute/{date_str}/{date_str}"
    params = {
        "adjusted": "true", "sort": "asc", "limit": "500",
        "apiKey": POLYGON_API_KEY
    }

    async with semaphore:
        for attempt in range(3):
            try:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 429:
                        wait = 2 ** (attempt + 1)
                        log.debug(f"  {ticker}: 429, backoff {wait}s")
                        await asyncio.sleep(wait)
                        continue
                    if resp.status != 200:
                        return ticker, None
                    data = await resp.json()
            except Exception as e:
                log.debug(f"  {ticker}: error {e}, attempt {attempt+1}")
                await asyncio.sleep(1)
                continue

            bars = data.get('results', [])
            if not bars:
                return ticker, None

            pm_bars = []
            for b in bars:
                ts = pd.Timestamp(b['t'], unit='ms', tz='UTC').tz_convert(ET)
                t_str = ts.strftime('%H:%M')
                if '04:00' <= t_str < '09:30':
                    pm_bars.append({
                        'time': t_str,
                        'o': float(b.get('o', 0)),
                        'h': float(b.get('h', 0)),
                        'l': float(b.get('l', 0)),
                        'c': float(b.get('c', 0)),
                        'v': int(b.get('v', 0)),
                    })

            if not pm_bars:
                return ticker, {'pm_open': None, 'pm_high': None, 'pm_last': None, 'pm_retrace_pct': None}

            pm_open = pm_bars[0]['o']
            pm_high = max(b['h'] for b in pm_bars)
            pm_last = pm_bars[-1]['c']
            total_move = pm_high - pm_open
            retrace = (pm_high - pm_last) / total_move if total_move > 0 else 0.0

            return ticker, {
                'pm_open': round(pm_open, 4),
                'pm_high': round(pm_high, 4),
                'pm_last': round(pm_last, 4),
                'pm_retrace_pct': round(retrace, 4),
            }
        return ticker, None


async def _fetch_all_pm(tickers, date_str):
    semaphore = asyncio.Semaphore(PM_BAR_WORKERS)
    async with aiohttp.ClientSession() as session:
        tasks = [_fetch_one_pm(session, tk, date_str, semaphore) for tk in tickers]
        results = await asyncio.gather(*tasks)
    return {tk: data for tk, data in results if data is not None}


def fetch_pm_bars_async(tickers, date_str):
    if not tickers:
        return {}
    t0 = time.time()
    try:
        asyncio.get_running_loop()
        nest_asyncio.apply()
    except RuntimeError:
        pass
    pm_data = asyncio.run(_fetch_all_pm(tickers, date_str))
    elapsed = time.time() - t0
    n_ok = sum(1 for v in pm_data.values() if v.get('pm_retrace_pct') is not None)
    log.info(f"PM bars: {n_ok}/{len(tickers)} fetched in {elapsed:.1f}s")
    if elapsed > 120:
        log.warning(f"PM bar fetch took {elapsed:.0f}s — consider reducing universe or increasing workers")
    return pm_data


# ═══════════════════════════════════════════════════════════════════════════════
# 3.10  RETRACE FILTER
# ═══════════════════════════════════════════════════════════════════════════════

def apply_retrace_filter(candidates, pm_data):
    n_excluded = 0
    for c in candidates:
        if not c['passed_all']:
            continue
        tk = c['ticker']
        pm = pm_data.get(tk, {})
        retrace = pm.get('pm_retrace_pct')
        c['pm_high'] = pm.get('pm_high')
        c['pm_last'] = pm.get('pm_last')
        c['pm_retrace_pct'] = retrace

        if retrace is not None and retrace > PM_RETRACE_MAX:
            c['passed_retrace'] = False
            c['passed_all'] = False
            n_excluded += 1
        else:
            c['passed_retrace'] = True

    log.info(f"Retrace filter: {n_excluded} excluded (retrace > {PM_RETRACE_MAX})")
    return candidates


# ═══════════════════════════════════════════════════════════════════════════════
# 3.11  SELECT TOP N
# ═══════════════════════════════════════════════════════════════════════════════

def _is_common_stock(ticker):
    """Check Polygon reference API — returns True for type='CS' (common stock)."""
    try:
        data = _polygon_get(f"{POLYGON_BASE}/v3/reference/tickers/{ticker}")
        tk_type = data.get('results', {}).get('type', '')
        if tk_type != 'CS':
            log.info(f"  {ticker}: type='{tk_type}' — NOT common stock, skipping")
            return False
        return True
    except Exception as e:
        log.warning(f"  {ticker}: type check failed ({e}), including by default")
        return True


def select_top_n(candidates):
    qualified = [c for c in candidates if c.get('passed_all', False)]
    qualified.sort(key=lambda x: -x['gap_pct'])

    for i, c in enumerate(qualified):
        c['rank_by_gap'] = i + 1

    selected = []
    for c in qualified:
        if len(selected) >= TOP_N:
            break
        if not _is_common_stock(c['ticker']):
            c['passed_all'] = False
            continue
        c['selection_rank'] = len(selected) + 1
        c['selected'] = True
        selected.append(c)
        time.sleep(0.05)

    for c in candidates:
        if 'selected' not in c:
            c['selected'] = False
        if 'rank_by_gap' not in c:
            c['rank_by_gap'] = None

    log.info(f"Selection: {len(qualified)} qualified → top {min(TOP_N, len(selected))} selected (CS verified)")
    return selected


# ═══════════════════════════════════════════════════════════════════════════════
# 3.12  DB WRITE
# ═══════════════════════════════════════════════════════════════════════════════

def write_to_db(scan_date_str, candidates, selected, mode, runtime,
                spy_gap_pct, n_gap_universe):
    conn = _get_conn()

    n_t4_t5 = len(candidates)
    n_after_price = sum(1 for c in candidates if c.get('passed_price', True))
    n_after_float = sum(1 for c in candidates if c.get('passed_float', True))
    n_after_retrace = sum(1 for c in candidates if c.get('passed_all', False))

    conn.execute("""INSERT OR REPLACE INTO daily_scans VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                 (scan_date_str, datetime.now(pytz.UTC).isoformat(), mode,
                  n_gap_universe, n_t4_t5, n_after_price, n_after_float,
                  n_after_retrace, len(selected), spy_gap_pct, runtime))

    for c in candidates:
        cid = f"{scan_date_str}_{c['ticker']}"
        conn.execute("""INSERT OR REPLACE INTO scan_candidates VALUES
            (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (cid, scan_date_str, c['ticker'], c['gap_pct'], c['tier'],
             c['prior_close'], c['open_price'], c.get('float_shares'),
             c.get('pm_high'), c.get('pm_last'), c.get('pm_retrace_pct'),
             int(c.get('passed_price', True)), int(c.get('passed_float', True)),
             int(c.get('passed_retrace', True)), int(c.get('passed_all', False)),
             c.get('rank_by_gap'), int(c.get('selected', False))))

    t1, t2 = get_t1_t2_dates(scan_date_str)
    for s in selected:
        tid = f"{scan_date_str}_{s['ticker']}"
        conn.execute("""INSERT OR REPLACE INTO selected_trades
            (trade_id, scan_date, ticker, selection_rank, tier, gap_pct,
             prior_close, open_price, pm_retrace_pct, float_shares, t1_date, t2_date)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (tid, scan_date_str, s['ticker'], s['selection_rank'], s['tier'],
             s['gap_pct'], s['prior_close'], s['open_price'],
             s.get('pm_retrace_pct'), s.get('float_shares'), t1, t2))

    conn.commit()
    conn.close()
    log.info(f"DB write: {len(candidates)} candidates, {len(selected)} selected trades")


# ═══════════════════════════════════════════════════════════════════════════════
# 3.13  PRINT WATCHLIST
# ═══════════════════════════════════════════════════════════════════════════════

def print_watchlist(selected, scan_date_str):
    t1, t2 = get_t1_t2_dates(scan_date_str)
    now_str = datetime.now(PT).strftime('%H:%M:%S PST')

    print(f"\n  {'='*62}")
    print(f"  |{'AGENT 1 WATCHLIST':^60}|")
    print(f"  |{'':^60}|")
    print(f"  |  Date: {scan_date_str:<20}  Run: {now_str:<22} |")
    print(f"  |  Exit: 17:00 ET (2:00 PM PST)   Trade: #1 only       |")
    print(f"  {'='*62}")

    if not selected:
        print(f"\n  No qualifying trades found.\n")
        return

    rows = []
    for s in selected:
        fs = s.get('float_shares')
        if fs is not None:
            if fs >= 1e9:
                fs_str = f"{fs/1e9:.1f}B"
            else:
                fs_str = f"{fs/1e6:.0f}M"
        else:
            fs_str = 'n/a'

        retrace = s.get('pm_retrace_pct')
        ret_str = f"{retrace:.2f}" if retrace is not None else 'n/a'

        rows.append([
            f"#{s['selection_rank']}",
            s['ticker'],
            f"+{s['gap_pct']*100:.1f}%",
            s['tier'],
            ret_str,
            f"${s['open_price']:.2f}",
            fs_str,
        ])

    print()
    print(tabulate(rows,
                   headers=['Rank', 'Ticker', 'Gap%', 'Tier', 'PM Retrace', 'Price', 'Float'],
                   tablefmt='simple', colalign=('left', 'left', 'right', 'center', 'right', 'right', 'right')))

    print(f"\n  Strategy: SHORT #1 at local high, exit at 17:00 ET (2:00 PM PST)")
    print(f"  Action:   Open chart for {selected[0]['ticker']} (top pick)")
    print(f"            Watch for local high (3-bar window, ~7-8 min after open)")
    print(f"            Enter SHORT on confirmation, close at 2 PM PST (after-hours)")
    print(f"            #{2}-{len(selected)} shown for context only")
    print()

    # per-ticker detail
    for s in selected:
        retrace = s.get('pm_retrace_pct')
        print(f"  #{s['selection_rank']} {s['ticker']}")
        print(f"     Gap: +{s['gap_pct']*100:.1f}%  |  Prior close: ${s['prior_close']:.2f}  |  PM last: ${s['open_price']:.2f}")
        if retrace is not None:
            if retrace < 0.2:
                pm_note = "held (bullish PM, caution)"
            elif retrace <= 0.5:
                pm_note = "sweet spot (partial retrace)"
            elif retrace <= 0.8:
                pm_note = "moderate retrace"
            else:
                pm_note = "full retrace"
            print(f"     PM retrace: {retrace:.2f} — {pm_note}")
        else:
            print(f"     PM retrace: n/a (no PM bars)")
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# FILL OUTCOMES (MODE='fill')
# ═══════════════════════════════════════════════════════════════════════════════

def run_fill_outcomes(fill_date_str=None):
    """Fill T0/T1/T2 closing prices for open trades.
    fill_date_str = the latest date for which closing data is available.
    At 6:15 AM, yesterday's close is available via grouped daily.
    So we use get_prior_trading_date(fill_date_str) if fill_date_str is today."""
    if fill_date_str is None:
        fill_date_str = datetime.now(ET).strftime('%Y-%m-%d')

    log.info(f"  FILL OUTCOMES — data available through {fill_date_str}")

    conn = _get_conn()
    trades = conn.execute("""
        SELECT trade_id, scan_date, ticker, t1_date, t2_date, open_price,
               t0_close, t1_close, t2_close
        FROM selected_trades
        WHERE t0_close IS NULL OR t1_close IS NULL OR t2_close IS NULL
    """).fetchall()

    if not trades:
        log.info("  No trades need filling.")
        conn.close()
        return

    log.info(f"  {len(trades)} trades need outcome data")

    dates_needed = set()
    for t in trades:
        dates_needed.add(t[1])  # scan_date = T0
        if t[3]: dates_needed.add(t[3])  # t1_date
        if t[4]: dates_needed.add(t[4])  # t2_date

    daily_data = {}
    for dt in dates_needed:
        if dt > fill_date_str:
            continue
        try:
            daily_data[dt] = fetch_grouped_daily(dt)
            time.sleep(0.05)
        except Exception as e:
            log.warning(f"  Failed to fetch {dt}: {e}")

    float_cache = load_float_cache()
    float_updated = False
    n_filled = 0

    for t in trades:
        tid, scan_date, ticker, t1_date, t2_date, open_price = t[:6]
        t0_close, t1_close, t2_close = t[6], t[7], t[8]

        updates = {}

        if t0_close is None and scan_date in daily_data:
            dd = daily_data[scan_date].get(ticker, {})
            if dd.get('close'):
                updates['t0_close'] = dd['close']
                updates['pnl_eod'] = round((open_price - dd['close']) / open_price * 100, 4)

        if t1_close is None and t1_date and t1_date in daily_data:
            dd = daily_data[t1_date].get(ticker, {})
            if dd.get('close'):
                updates['t1_close'] = dd['close']
                updates['pnl_t1'] = round((open_price - dd['close']) / open_price * 100, 4)

        if t2_close is None and t2_date and t2_date in daily_data:
            dd = daily_data[t2_date].get(ticker, {})
            if dd.get('close'):
                updates['t2_close'] = dd['close']
                updates['pnl_t2'] = round((open_price - dd['close']) / open_price * 100, 4)

        if updates:
            set_clause = ', '.join(f"{k}=?" for k in updates.keys())
            conn.execute(f"UPDATE selected_trades SET {set_clause} WHERE trade_id=?",
                         list(updates.values()) + [tid])
            n_filled += 1
            pnl_str = ', '.join(f"{k}={v:+.2f}%" if 'pnl' in k else f"{k}={v:.2f}"
                                for k, v in updates.items())
            log.info(f"  {tid}: {pnl_str}")

        # expand float cache for new tickers
        if ticker not in float_cache:
            try:
                url = f"{POLYGON_BASE}/v3/reference/tickers/{ticker}"
                data = _polygon_get(url)
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
        log.info(f"  Float cache updated: {len(float_cache)} tickers")


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT (MODE='report')
# ═══════════════════════════════════════════════════════════════════════════════

def run_report(weeks=REPORT_WEEKS):
    conn = _get_conn()
    cutoff = (datetime.now() - timedelta(weeks=weeks)).strftime('%Y-%m-%d')

    scans = conn.execute("""
        SELECT COUNT(*), SUM(n_selected), AVG(n_selected), AVG(runtime_seconds)
        FROM daily_scans WHERE scan_date >= ?
    """, (cutoff,)).fetchone()

    trades = conn.execute("""
        SELECT ticker, scan_date, tier, gap_pct, open_price,
               pnl_eod, pnl_t1, pnl_t2
        FROM selected_trades WHERE scan_date >= ?
        ORDER BY scan_date
    """, (cutoff,)).fetchall()

    conn.close()

    n_scans = scans[0] or 0
    n_total = scans[1] or 0
    avg_per_day = scans[2] or 0
    avg_runtime = scans[3] or 0

    pnl_t2 = [t[7] for t in trades if t[7] is not None]
    pnl_eod = [t[5] for t in trades if t[5] is not None]

    print(f"\n  {'='*62}")
    print(f"  |{'AGENT 1 PERFORMANCE SUMMARY':^60}|")
    print(f"  {'='*62}")
    print(f"\n  Period: Last {weeks} weeks (cutoff: {cutoff})")
    print(f"  Total scans:        {n_scans}")
    print(f"  Total selected:     {n_total} (avg {avg_per_day:.1f}/day)")
    print(f"  Avg runtime:        {avg_runtime:.1f}s")

    if pnl_t2:
        arr = np.array(pnl_t2)
        n = len(arr)
        avg = float(np.mean(arr))
        med = float(np.median(arr))
        std = float(np.std(arr, ddof=1)) if n > 1 else 0
        se = std / np.sqrt(n)
        wr = float((arr > 0).mean())
        sharpe = avg / std if std > 0 else 0
        ci_lo = avg - 1.96 * se
        ci_hi = avg + 1.96 * se

        best_idx = int(np.argmax(arr))
        worst_idx = int(np.argmin(arr))
        best_trade = trades[[i for i, t in enumerate(trades) if t[7] is not None][best_idx]]
        worst_trade = trades[[i for i, t in enumerate(trades) if t[7] is not None][worst_idx]]

        t4_pnl = [t[7] for t in trades if t[7] is not None and t[2] == 'T4']
        t5_pnl = [t[7] for t in trades if t[7] is not None and t[2] == 'T5']

        print(f"\n  ── T+2 Performance ({n} trades) ──")
        print(f"  Avg PnL T+2:        {avg:+.2f}%")
        print(f"  Median PnL T+2:     {med:+.2f}%")
        print(f"  Std Dev:            {std:.2f}%")
        print(f"  Win Rate:           {wr*100:.0f}%")
        print(f"  Sharpe (proxy):     {sharpe:.3f}")
        print(f"  CI 95%:             [{ci_lo:+.2f}%, {ci_hi:+.2f}%]")
        print(f"  CI > 0:             {'YES' if ci_lo > 0 else 'NO'}")
        print(f"  Best trade:         {best_trade[0]} {best_trade[7]:+.1f}% ({best_trade[1]})")
        print(f"  Worst trade:        {worst_trade[0]} {worst_trade[7]:+.1f}% ({worst_trade[1]})")
        if t4_pnl:
            print(f"  T4 (5-8%):          {np.mean(t4_pnl):+.2f}% avg (n={len(t4_pnl)})")
        if t5_pnl:
            print(f"  T5 (>8%):           {np.mean(t5_pnl):+.2f}% avg (n={len(t5_pnl)})")
    else:
        print(f"\n  No T+2 outcomes available yet.")

    if pnl_eod:
        arr_eod = np.array(pnl_eod)
        print(f"\n  ── EOD Performance ({len(arr_eod)} trades) ──")
        print(f"  Avg PnL EOD:        {np.mean(arr_eod):+.2f}%")
        print(f"  Win Rate EOD:       {(arr_eod > 0).mean()*100:.0f}%")

    # per-trade table
    if trades:
        print(f"\n  ── Trade Log ──")
        rows = []
        for t in trades:
            tk, dt, tier, gap, price, pnl_e, pnl_1, pnl_2 = t
            rows.append([
                dt, tk, tier, f"+{gap*100:.1f}%", f"${price:.2f}",
                f"{pnl_e:+.2f}%" if pnl_e is not None else '—',
                f"{pnl_1:+.2f}%" if pnl_1 is not None else '—',
                f"{pnl_2:+.2f}%" if pnl_2 is not None else '—',
            ])
        print()
        print(tabulate(rows,
                       headers=['Date', 'Ticker', 'Tier', 'Gap', 'Price', 'EOD', 'T+1', 'T+2'],
                       tablefmt='simple'))
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — SCANNER (MODE='scan')
# ═══════════════════════════════════════════════════════════════════════════════

def run_scan(scan_date_str=None, dry_run=False):
    t_start = time.time()

    if scan_date_str is None:
        scan_date_str = datetime.now(ET).strftime('%Y-%m-%d')
        mode = 'live'
    else:
        mode = 'historical'

    log.info(f"{'='*_W}")
    log.info(f"  AGENT 1 — Technical Gap-Fade Scanner")
    log.info(f"  Date: {scan_date_str}  |  Mode: {mode}  |  Dry run: {dry_run}")
    log.info(f"{'='*_W}")

    if not is_trading_day(scan_date_str):
        log.info(f"  {scan_date_str} is not a trading day. Exiting.")
        return

    # Step 0: fill outcomes for any open trades before scanning
    if not dry_run:
        log.info(f"\n  ── Step 0: Fill outcomes for open trades ──")
        run_fill_outcomes(fill_date_str=scan_date_str)

    if mode == 'historical':
        log.warning("HISTORICAL MODE: using open price as PM proxy. Gap calculations are approximate.")

    # 1. DB + float cache
    init_db()
    float_cache = load_float_cache()

    # 2. Prior trading day
    prior_date = get_prior_trading_date(scan_date_str)
    prior_str = prior_date.strftime('%Y-%m-%d')
    log.info(f"  Prior trading day: {prior_str}")

    # 3. Fetch prior closes
    t1 = time.time()
    prior_data = fetch_grouped_daily(prior_str)
    log.info(f"  [1] Grouped daily: {time.time()-t1:.1f}s")

    # 4. Fetch current prices
    t2 = time.time()
    if mode == 'live':
        current_data = fetch_snapshot_all()
    else:
        current_data = fetch_grouped_daily(scan_date_str)
    log.info(f"  [2] Current prices: {time.time()-t2:.1f}s")

    # SPY gap
    spy_gap = None
    if 'SPY' in prior_data and 'SPY' in current_data:
        spy_prev = prior_data['SPY']['close']
        spy_cur = current_data['SPY'].get('open', current_data['SPY'].get('price', 0))
        if spy_prev > 0 and spy_cur > 0:
            spy_gap = round((spy_cur - spy_prev) / spy_prev * 100, 3)
            log.info(f"  SPY gap: {spy_gap:+.2f}%")

    # 5. Compute gaps
    t3 = time.time()
    candidates = compute_gaps(prior_data, current_data, mode)
    n_gap_universe = len(candidates)
    log.info(f"  [3] Gap computation: {time.time()-t3:.1f}s")

    # 6. Float filter
    candidates = apply_float_filter(candidates, float_cache)

    # 7. Select top N
    selected = select_top_n(candidates)

    # 10. Write to DB
    runtime = time.time() - t_start
    if not dry_run:
        write_to_db(scan_date_str, candidates, selected, mode, runtime,
                    spy_gap, n_gap_universe)
    else:
        log.info("  DRY RUN — skipping DB write")

    # 11. Print watchlist
    print_watchlist(selected, scan_date_str)

    # 12. Summary
    n_qualified = sum(1 for c in candidates if c.get('passed_all', False))
    log.info(f"  Funnel: {n_gap_universe} gap >= 5% → "
             f"{sum(1 for c in candidates if c.get('passed_float', True))} after float → "
             f"{len(selected)} selected")
    log.info(f"  Total runtime: {runtime:.1f}s")

    return selected


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
