# %% Agent 1 — Technical Gap-Fade Scanner (Dual Variant)
# Single script: scanner + outcome filler + performance report
#
# Two variants, shared pipeline:
#   E2-A  Agent 1:     CS only, scan 6:31 PST, local_high bars 3-11, fb 15
#   E2-B  Agent 1-SD:  CS + sustained_drop (confirmed 7:30 PST),
#                       local_high bars 60-72, fb 75
#
# Usage:
#   MODE = 'scan'                       → morning scanner (default)
#   MODE = 'fill'                       → fill T0/T1/T2 outcomes after close
#   MODE = 'report'                     → performance summary
#   SCAN_TIMESTAMP = '2026-09-11 06:31' → time machine (PST, minute granularity)
#   SCAN_TIMESTAMP = '2026-09-11 07:30' → runs BOTH E2-A and E2-B
#   SCAN_TIMESTAMP = None               → live (uses current time)
#   DRY_RUN = True                      → skip DB writes
#
# Time logic (PST):
#   E2-A runs at any time (no gate)
#   >= 07:30 → E2-B also runs (reads E2-A pool, checks sustained_drop)
#
# Gap freshness: gap is computed vs last AH price on prior trading day,
# not vs prior close. Filters out M&A / stale gaps where the move
# happened in after-hours and the stock is already at equilibrium.

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
SCAN_TIMESTAMP = None  # PST — 'YYYY-MM-DD HH:MM' or None for live
DRY_RUN = False              # True = skip DB writes
REPORT_WEEKS = 4             # for MODE='report'

E2A_READY_PST = (6, 31)      # E2-A available at 6:31 AM PST
E2B_READY_PST = (7, 30)      # E2-B available at 7:30 AM PST

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs")
POLYGON_BASE = "https://api.polygon.io"

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

DB_PATH = os.path.join(BASE_DIR, 'data', 'agent1.db')
LOG_PATH = os.path.join(BASE_DIR, 'logs', 'agent1.log')

GAP_MIN = 0.03               # 3% minimum gap
PRICE_MIN = 10.0             # $10 minimum price
FLOAT_MIN_SHARES = 10_000_000
TOP_N = 5
PRE_FILTER_N = 30            # walk top 30 by gap, take first CS (E2-A) or CS+SD (E2-B)
STOP_LOSS_PCT = 0.50
GAP_CONSUMED_MAX = 0.50     # skip if >50% of gap eroded by scan time (low vs open)
EXIT_TIME_ET = '17:00'       # exit at 17:00 ET (2:00 PM PST)
SCAN_TIME_ET = '09:31'       # 6:31 AM PST — rank candidates by gap at this bar

PEAK_WINDOW = 3
E2A_MAX_PEAK_BAR = 12        # search bars 3-11
E2A_FALLBACK_BAR = 15        # fallback at 09:45 ET
E2B_MIN_ENTRY_BAR = 60       # SD entry starts bar 60 (10:30 ET)
E2B_MAX_PEAK_BAR = 72        # search bars 60-71
E2B_FALLBACK_BAR = 75        # fallback at 10:45 ET

BAR_WORKERS = 10
TICKER_REGEX = re.compile(r'^[A-Z]{1,5}$')
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

    return logging.getLogger('agent1')

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
        variant          TEXT,
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

    CREATE TABLE IF NOT EXISTS scan_pool (
        pool_id          TEXT PRIMARY KEY,
        scan_date        TEXT,
        pool_rank        INTEGER,
        ticker           TEXT,
        gap_pct          REAL,
        prior_close      REAL,
        scan_price       REAL,
        float_shares     REAL,
        passed_cs        INTEGER DEFAULT NULL
    );

    CREATE TABLE IF NOT EXISTS selected_trades (
        trade_id         TEXT PRIMARY KEY,
        scan_date        TEXT,
        variant          TEXT,
        ticker           TEXT,
        selection_rank   INTEGER,
        tier             TEXT,
        gap_pct          REAL,
        prior_close      REAL,
        open_price       REAL,
        pm_retrace_pct   REAL,
        float_shares     REAL,
        open_pattern     TEXT DEFAULT NULL,
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
    CREATE INDEX IF NOT EXISTS idx_pool_date ON scan_pool(scan_date);
    CREATE INDEX IF NOT EXISTS idx_trades_date ON selected_trades(scan_date);
    CREATE INDEX IF NOT EXISTS idx_trades_variant ON selected_trades(variant);
    CREATE INDEX IF NOT EXISTS idx_trades_t2 ON selected_trades(t2_date);
    """)

    # migrate existing tables: add columns if missing
    for col, tbl, default in [
        ('variant', 'selected_trades', "'e2a'"),
        ('variant', 'daily_scans', "'e2a'"),
        ('open_pattern', 'selected_trades', "NULL"),
    ]:
        try:
            conn.execute(f"SELECT {col} FROM {tbl} LIMIT 1")
        except sqlite3.OperationalError:
            conn.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} TEXT DEFAULT {default}")
            conn.commit()
            log.info(f"  Migrated: added {col} to {tbl}")

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
# POLYGON — GROUPED DAILY
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
# POLYGON — SNAPSHOT (LIVE MODE)
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
# GAP COMPUTATION
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
# FLOAT FILTER — frozen embedded dict only, miss=INCLUDE, hit+<10M=EXCLUDE
# ═══════════════════════════════════════════════════════════════════════════════

def apply_float_filter(candidates):
    n_excluded = 0
    n_miss = 0
    for c in candidates:
        tk = c['ticker']
        fs = _EMBEDDED_FLOAT.get(tk)
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
             f"{n_miss} not in dict (included)")
    return candidates


# ═══════════════════════════════════════════════════════════════════════════════
# ASYNC MINUTE BAR FETCH
# ═══════════════════════════════════════════════════════════════════════════════

async def _fetch_bars_one(session, ticker, date_str, semaphore):
    url = f"{POLYGON_BASE}/v2/aggs/ticker/{ticker}/range/1/minute/{date_str}/{date_str}"
    params = {
        "adjusted": "true", "sort": "asc", "limit": "1000",
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
            rth_bars = []
            ah_bars = []
            for b in bars:
                ts = pd.Timestamp(b['t'], unit='ms', tz='UTC').tz_convert(ET)
                t_str = ts.strftime('%H:%M')
                bar_d = {
                    'o': float(b.get('o', 0)),
                    'h': float(b.get('h', 0)),
                    'l': float(b.get('l', 0)),
                    'c': float(b.get('c', 0)),
                    'v': int(b.get('v', 0)),
                }
                if '04:00' <= t_str < '09:30':
                    pm_bars.append((t_str, bar_d))
                elif '09:30' <= t_str < '16:00':
                    rth_bars.append((t_str, bar_d))
                elif '16:00' <= t_str <= '20:00':
                    ah_bars.append((t_str, bar_d))

            # PM summary for backwards compat
            pm_summary = {'pm_open': None, 'pm_high': None, 'pm_last': None, 'pm_retrace_pct': None}
            if pm_bars:
                pm_open = pm_bars[0][1]['o']
                pm_high = max(b[1]['h'] for b in pm_bars)
                pm_last = pm_bars[-1][1]['c']
                total_move = pm_high - pm_open
                retrace = (pm_high - pm_last) / total_move if total_move > 0 else 0.0
                pm_summary = {
                    'pm_open': round(pm_open, 4),
                    'pm_high': round(pm_high, 4),
                    'pm_last': round(pm_last, 4),
                    'pm_retrace_pct': round(retrace, 4),
                }

            return ticker, {
                'pm': pm_bars,
                'rth': rth_bars,
                'ah': ah_bars,
                'pm_summary': pm_summary,
            }
        return ticker, None


async def _fetch_bars_batch(tickers, date_str):
    semaphore = asyncio.Semaphore(BAR_WORKERS)
    async with aiohttp.ClientSession() as session:
        tasks = [_fetch_bars_one(session, tk, date_str, semaphore) for tk in tickers]
        results = await asyncio.gather(*tasks)
    return {tk: data for tk, data in results if data is not None}


def fetch_minute_bars(tickers, date_str):
    if not tickers:
        return {}
    t0 = time.time()
    try:
        asyncio.get_running_loop()
        nest_asyncio.apply()
    except RuntimeError:
        pass
    bar_data = asyncio.run(_fetch_bars_batch(tickers, date_str))
    elapsed = time.time() - t0
    n_ok = sum(1 for v in bar_data.values() if v.get('rth'))
    log.info(f"Minute bars: {n_ok}/{len(tickers)} fetched in {elapsed:.1f}s")
    return bar_data


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN CLASSIFICATION (sustained_drop check for E2-B)
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

def find_local_high(bars, window=PEAK_WINDOW, max_peak_bar=E2A_MAX_PEAK_BAR,
                    fallback_bar=E2A_FALLBACK_BAR, min_entry_bar=0):
    if len(bars) < window * 2 + 1:
        return None
    start = max(window, min_entry_bar - window)
    for i in range(start, min(max_peak_bar, len(bars) - window)):
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
    fb = max(fallback_bar, min_entry_bar) if fallback_bar is not None else None
    if fb is not None and fb < len(bars):
        return {
            'entry_price': bars[fb][1]['c'],
            'entry_time': bars[fb][0],
            'entry_type': 'fallback',
            'peak_bar': None,
        }
    return None


def get_price_at_T(bar_data, T):
    if bar_data is None:
        return None
    all_bars = (bar_data.get('pm') or []) + (bar_data.get('rth') or [])
    best = None
    for t_str, bd in all_bars:
        if t_str <= T:
            best = bd['c']
        else:
            break
    return best


# ═══════════════════════════════════════════════════════════════════════════════
# AFTER-HOURS REFERENCE PRICE (gap freshness check)
# ═══════════════════════════════════════════════════════════════════════════════

def get_ah_reference_prices(tickers, prior_date_str):
    """Fetch prior day minute bars and return last AH price for each ticker.
    AH = 16:00-20:00 ET. If no AH data, ticker is omitted from result."""
    bar_data = fetch_minute_bars(tickers, prior_date_str)
    ah_prices = {}
    for tk, data in bar_data.items():
        ah_bars = data.get('ah', [])
        if ah_bars:
            ah_prices[tk] = ah_bars[-1][1]['c']
    log.info(f"  AH reference: {len(ah_prices)}/{len(tickers)} tickers have AH data on {prior_date_str}")
    return ah_prices


# ═══════════════════════════════════════════════════════════════════════════════
# CS TYPE CHECK
# ═══════════════════════════════════════════════════════════════════════════════

def _is_common_stock(ticker):
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


# ═══════════════════════════════════════════════════════════════════════════════
# SELECT TOP N — walk PRE_FILTER_N candidates, CS-verify, take first TOP_N
# ═══════════════════════════════════════════════════════════════════════════════

def select_top_n(candidates):
    qualified = [c for c in candidates if c.get('passed_all', False)]
    qualified.sort(key=lambda x: -x['gap_pct'])

    for i, c in enumerate(qualified):
        c['rank_by_gap'] = i + 1

    selected = []
    for c in qualified[:PRE_FILTER_N]:
        if len(selected) >= TOP_N:
            break
        if not _is_common_stock(c['ticker']):
            c['passed_all'] = False
            c['passed_cs'] = False
            continue
        c['passed_cs'] = True
        c['selection_rank'] = len(selected) + 1
        c['selected'] = True
        selected.append(c)
        time.sleep(0.05)

    for c in candidates:
        if 'selected' not in c:
            c['selected'] = False
        if 'rank_by_gap' not in c:
            c['rank_by_gap'] = None

    log.info(f"Selection: {len(qualified)} qualified → walked top {min(PRE_FILTER_N, len(qualified))}"
             f" → {len(selected)} selected (CS verified)")
    return selected


# ═══════════════════════════════════════════════════════════════════════════════
# PERSIST / LOAD SCAN POOL (6:31 top-30 for E2-B reuse)
# ═══════════════════════════════════════════════════════════════════════════════

def persist_scan_pool(scan_date_str, candidates):
    # Pool = all candidates that pass gap+price+float (before CS check)
    qualified = [c for c in candidates if c.get('passed_float', True)]
    qualified.sort(key=lambda x: -x['gap_pct'])
    pool = qualified[:PRE_FILTER_N]

    conn = _get_conn()
    conn.execute("DELETE FROM scan_pool WHERE scan_date=?", (scan_date_str,))
    for i, c in enumerate(pool):
        pid = f"{scan_date_str}_{i+1}_{c['ticker']}"
        cs_flag = None
        if 'passed_cs' in c:
            cs_flag = 1 if c['passed_cs'] else 0
        conn.execute("""INSERT OR REPLACE INTO scan_pool
            (pool_id, scan_date, pool_rank, ticker, gap_pct, prior_close,
             scan_price, float_shares, passed_cs)
            VALUES (?,?,?,?,?,?,?,?,?)""",
            (pid, scan_date_str, i + 1, c['ticker'], c['gap_pct'],
             c['prior_close'], c['open_price'], c.get('float_shares'), cs_flag))

    conn.commit()
    conn.close()
    log.info(f"Scan pool: persisted {len(pool)} candidates for {scan_date_str}")
    return pool


def load_scan_pool(scan_date_str):
    conn = _get_conn()
    rows = conn.execute("""
        SELECT pool_rank, ticker, gap_pct, prior_close, scan_price,
               float_shares, passed_cs
        FROM scan_pool WHERE scan_date=?
        ORDER BY pool_rank
    """, (scan_date_str,)).fetchall()
    conn.close()

    if not rows:
        log.warning(f"No scan pool found for {scan_date_str}. Run E2-A first.")
        return []

    pool = []
    for r in rows:
        pool.append({
            'pool_rank': r[0],
            'ticker': r[1],
            'gap_pct': r[2],
            'prior_close': r[3],
            'open_price': r[4],
            'scan_price': r[4],
            'float_shares': r[5],
            'passed_cs': r[6],
            'tier': 'T4' if r[2] <= 0.08 else 'T5',
            'passed_price': True,
            'passed_float': True,
            'passed_all': True,
        })
    log.info(f"Scan pool: loaded {len(pool)} candidates for {scan_date_str}")
    return pool


# ═══════════════════════════════════════════════════════════════════════════════
# DB WRITE
# ═══════════════════════════════════════════════════════════════════════════════

def write_to_db(scan_date_str, candidates, selected, mode, runtime,
                spy_gap_pct, n_gap_universe, variant='e2a'):
    conn = _get_conn()

    n_t4_t5 = len(candidates)
    n_after_price = sum(1 for c in candidates if c.get('passed_price', True))
    n_after_float = sum(1 for c in candidates if c.get('passed_float', True))
    n_after_retrace = sum(1 for c in candidates if c.get('passed_all', False))

    conn.execute("""INSERT OR REPLACE INTO daily_scans
        (scan_date, run_timestamp, mode, variant, n_gap_universe, n_t4_t5,
         n_after_price, n_after_float, n_after_retrace, n_selected,
         spy_gap_pct, runtime_seconds)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (scan_date_str, datetime.now(pytz.UTC).isoformat(), mode, variant,
         n_gap_universe, n_t4_t5, n_after_price, n_after_float,
         n_after_retrace, len(selected), spy_gap_pct, runtime))

    if variant == 'e2a':
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
        tid = f"{scan_date_str}_{variant}_{s['ticker']}"
        conn.execute("""INSERT OR REPLACE INTO selected_trades
            (trade_id, scan_date, variant, ticker, selection_rank, tier, gap_pct,
             prior_close, open_price, pm_retrace_pct, float_shares, open_pattern,
             t1_date, t2_date)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (tid, scan_date_str, variant, s['ticker'], s['selection_rank'],
             s['tier'], s['gap_pct'], s['prior_close'], s['open_price'],
             s.get('pm_retrace_pct'), s.get('float_shares'),
             s.get('open_pattern'), t1, t2))

    conn.commit()
    conn.close()
    log.info(f"DB write ({variant}): {len(candidates)} candidates, {len(selected)} selected trades")


# ═══════════════════════════════════════════════════════════════════════════════
# PRINT WATCHLIST
# ═══════════════════════════════════════════════════════════════════════════════

_VARIANT_LABELS = {
    'e2a': 'AGENT 1 (E2-A)',
    'e2b': 'AGENT 1-SD (E2-B)',
}

def print_watchlist(selected, scan_date_str, variant='e2a'):
    t1, t2 = get_t1_t2_dates(scan_date_str)
    now_str = datetime.now(PT).strftime('%H:%M:%S PST')
    label = _VARIANT_LABELS.get(variant, variant.upper())

    print(f"\n  {'='*62}")
    print(f"  |{label + ' WATCHLIST':^60}|")
    print(f"  |{'':^60}|")
    print(f"  |  Date: {scan_date_str:<20}  Run: {now_str:<22} |")
    print(f"  |  Exit: {EXIT_TIME_ET} ET (2:00 PM PST)                      |")
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

        pat = s.get('open_pattern', '')
        pat_str = pat if pat else ''

        rows.append([
            f"#{s['selection_rank']}",
            s['ticker'],
            f"+{s['gap_pct']*100:.1f}%",
            s['tier'],
            ret_str,
            f"${s['open_price']:.2f}",
            fs_str,
            pat_str,
        ])

    print()
    print(tabulate(rows,
                   headers=['Rank', 'Ticker', 'Gap%', 'Tier', 'PM Ret', 'Price', 'Float', 'Pattern'],
                   tablefmt='simple', colalign=('left', 'left', 'right', 'center',
                                                 'right', 'right', 'right', 'left')))

    print(f"\n  Strategy: SHORT — {label}")
    if variant == 'e2a':
        print(f"  Entry:    Watch for local high (3-bar window, bars 3-11 = ~09:33-09:41 ET)")
        print(f"            Fallback at bar 15 close (09:45 ET) if no local high found")
    else:
        print(f"  Entry:    Watch for local high (3-bar window, bars 60-72 = ~10:30-10:42 ET)")
        print(f"            Fallback at bar 75 close (10:45 ET) if no local high found")
    print(f"  Exit:     {EXIT_TIME_ET} ET (2:00 PM PST)")
    print(f"  SL:       entry × 1.50 (disaster cap)")
    print()

    for s in selected:
        retrace = s.get('pm_retrace_pct')
        sl_price = s['open_price'] * (1 + STOP_LOSS_PCT)
        print(f"  #{s['selection_rank']} {s['ticker']}")
        print(f"     Gap: +{s['gap_pct']*100:.1f}%  |  Prior close: ${s['prior_close']:.2f}  |  Open: ${s['open_price']:.2f}")
        print(f"     Stop Loss: ${sl_price:.2f}  (entry × 1.50 — disaster cap)")
        print(f"     Exit:      {EXIT_TIME_ET} ET (2:00 PM PST)")
        if s.get('open_pattern'):
            print(f"     Pattern:   {s['open_pattern']}")
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

    conn.commit()
    conn.close()
    log.info(f"  Filled {n_filled}/{len(trades)} trades")


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
        SELECT ticker, scan_date, variant, tier, gap_pct, open_price,
               pnl_eod, pnl_t1, pnl_t2
        FROM selected_trades WHERE scan_date >= ?
        ORDER BY scan_date, variant
    """, (cutoff,)).fetchall()

    conn.close()

    n_scans = scans[0] or 0
    n_total = scans[1] or 0
    avg_per_day = scans[2] or 0
    avg_runtime = scans[3] or 0

    pnl_eod = [t[6] for t in trades if t[6] is not None]

    print(f"\n  {'='*62}")
    print(f"  |{'AGENT 1 PERFORMANCE SUMMARY':^60}|")
    print(f"  {'='*62}")
    print(f"\n  Period: Last {weeks} weeks (cutoff: {cutoff})")
    print(f"  Total scans:        {n_scans}")
    print(f"  Total selected:     {n_total} (avg {avg_per_day:.1f}/day)")
    print(f"  Avg runtime:        {avg_runtime:.1f}s")

    for v_label, v_code in [('E2-A (Agent 1)', 'e2a'), ('E2-B (Agent 1-SD)', 'e2b')]:
        v_trades = [t for t in trades if (t[2] or 'e2a') == v_code]
        v_pnl_eod = [t[6] for t in v_trades if t[6] is not None]

        if not v_pnl_eod:
            continue

        arr = np.array(v_pnl_eod)
        n = len(arr)
        avg = float(np.mean(arr))
        med = float(np.median(arr))
        std = float(np.std(arr, ddof=1)) if n > 1 else 0
        se = std / np.sqrt(n)
        wr = float((arr > 0).mean())
        sharpe = avg / std if std > 0 else 0
        ci_lo = avg - 1.96 * se
        ci_hi = avg + 1.96 * se

        print(f"\n  ── {v_label} — EOD Performance ({n} trades) ──")
        print(f"  Avg PnL EOD:        {avg:+.2f}%")
        print(f"  Median PnL EOD:     {med:+.2f}%")
        print(f"  Std Dev:            {std:.2f}%")
        print(f"  Win Rate:           {wr*100:.0f}%")
        print(f"  Sharpe (proxy):     {sharpe:.3f}")
        print(f"  CI 95%:             [{ci_lo:+.2f}%, {ci_hi:+.2f}%]")

        t4_pnl = [t[6] for t in v_trades if t[6] is not None and t[3] == 'T4']
        t5_pnl = [t[6] for t in v_trades if t[6] is not None and t[3] == 'T5']
        if t4_pnl:
            print(f"  T4 (3-8%):          {np.mean(t4_pnl):+.2f}% avg (n={len(t4_pnl)})")
        if t5_pnl:
            print(f"  T5 (>8%):           {np.mean(t5_pnl):+.2f}% avg (n={len(t5_pnl)})")

    # per-trade table
    if trades:
        print(f"\n  ── Trade Log ──")
        rows = []
        for t in trades:
            tk, dt, var, tier, gap, price, pnl_e, pnl_1, pnl_2 = t
            var_str = var or 'e2a'
            rows.append([
                dt, var_str, tk, tier, f"+{gap*100:.1f}%", f"${price:.2f}",
                f"{pnl_e:+.2f}%" if pnl_e is not None else '—',
                f"{pnl_1:+.2f}%" if pnl_1 is not None else '—',
                f"{pnl_2:+.2f}%" if pnl_2 is not None else '—',
            ])
        print()
        print(tabulate(rows,
                       headers=['Date', 'Var', 'Ticker', 'Tier', 'Gap', 'Price', 'EOD', 'T+1', 'T+2'],
                       tablefmt='simple'))
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# SCANNER — E2-A (run at 6:31 PST / 9:31 ET)
# ═══════════════════════════════════════════════════════════════════════════════

def run_scan_e2a(scan_date_str=None, dry_run=False):
    t_start = time.time()

    if scan_date_str is None:
        scan_date_str = datetime.now(ET).strftime('%Y-%m-%d')
        mode = 'live'
    else:
        mode = 'historical'

    log.info(f"{'='*_W}")
    log.info(f"  AGENT 1 — E2-A Scanner (CS only, 6:31 PST)")
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

    # 1. DB init
    init_db()

    # 2. Prior trading day (24-hour gap: always use immediately prior session)
    scan_dt = datetime.strptime(scan_date_str, '%Y-%m-%d').date()
    prior_date = get_prior_trading_date(scan_date_str)
    prior_str = prior_date.strftime('%Y-%m-%d')
    cal_days = (scan_dt - prior_date).days
    day_name = scan_dt.strftime('%A')
    log.info(f"  Scan day: {day_name} ({scan_date_str})")
    log.info(f"  Prior trading day: {prior_str} ({cal_days} calendar day(s) ago)")
    if cal_days > 4:
        log.warning(f"  ⚠ Gap spans {cal_days} calendar days (holiday stretch) — gap may be stale")

    # 3. Fetch prior closes
    t1 = time.time()
    prior_data = fetch_grouped_daily(prior_str)
    log.info(f"  [1] Grouped daily: {time.time()-t1:.1f}s")

    # 4. Fetch current prices (snapshot for live, grouped daily for historical)
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

    # 5. Compute gaps (3% min, $10 price min)
    t3 = time.time()
    candidates = compute_gaps(prior_data, current_data, mode)
    n_gap_universe = len(candidates)
    log.info(f"  [3] Gap computation: {time.time()-t3:.1f}s")

    # 6. Float filter (frozen embedded dict, miss=INCLUDE)
    candidates = apply_float_filter(candidates)

    # 6a. Fetch prior day AH bars — gap = price vs last 16h traded price, not prior close
    qualified = [c for c in candidates if c.get('passed_float', True)]
    qualified.sort(key=lambda x: -x['gap_pct'])
    ah_tickers = [c['ticker'] for c in qualified[:PRE_FILTER_N]]
    ah_prices = {}
    if ah_tickers:
        t_ah = time.time()
        log.info(f"  [3b] Fetching prior day AH bars for {len(ah_tickers)} candidates...")
        ah_prices = get_ah_reference_prices(ah_tickers, prior_str)
        log.info(f"  [3b] AH reference: {time.time()-t_ah:.1f}s")

        n_stale = 0
        for c in candidates:
            tk = c['ticker']
            if tk in ah_prices:
                ah_ref = ah_prices[tk]
                c['ah_ref_price'] = round(ah_ref, 4)
                old_gap = c['gap_pct']
                fresh_gap = (c['open_price'] - ah_ref) / ah_ref
                c['gap_pct_daily'] = old_gap
                c['gap_pct'] = round(fresh_gap, 6)
                if fresh_gap < GAP_MIN:
                    c['passed_all'] = False
                    n_stale += 1
                    log.info(f"    {tk}: STALE gap — daily +{old_gap*100:.1f}% but AH ref ${ah_ref:.2f} "
                             f"→ fresh gap {fresh_gap*100:+.1f}% < {GAP_MIN*100:.0f}%")
        if n_stale > 0:
            log.info(f"  AH filter: {n_stale} candidates excluded (gap happened in prior AH, not fresh)")

    # 6b. Fetch scan-day minute bars, re-rank by gap at SCAN_TIME_ET, check gap erosion
    qualified = [c for c in candidates if c.get('passed_all', False)]
    qualified.sort(key=lambda x: -x['gap_pct'])
    fetch_tickers = [c['ticker'] for c in qualified[:PRE_FILTER_N]]
    if fetch_tickers:
        t4 = time.time()
        log.info(f"  [4] Fetching scan-day minute bars for {len(fetch_tickers)} candidates...")
        bar_cache = fetch_minute_bars(fetch_tickers, scan_date_str)
        log.info(f"  [4] Minute bars: {time.time()-t4:.1f}s")

        n_reranked = 0
        n_dropped = 0
        n_eroded = 0
        reranked = []
        for c in candidates:
            tk = c['ticker']
            if not c.get('passed_all', False):
                continue
            if tk not in bar_cache:
                n_dropped += 1
                continue
            price_T = get_price_at_T(bar_cache[tk], SCAN_TIME_ET)
            if price_T is None:
                n_dropped += 1
                continue
            ref_price = c.get('ah_ref_price', c['prior_close'])

            if mode == 'historical':
                gap_T = (price_T - ref_price) / ref_price
                if gap_T < GAP_MIN:
                    continue
                if price_T < PRICE_MIN:
                    continue
                c['gap_pct'] = round(gap_T, 6)
                c['open_price'] = round(price_T, 4)

            # Gap erosion: if intraday low consumed >50% of gap by scan time, skip
            rth = bar_cache[tk].get('rth', [])
            if rth:
                daily_open = rth[0][1]['o']
                scan_bars = [b for b in rth if b[0] <= SCAN_TIME_ET]
                gap_range = daily_open - ref_price
                if scan_bars and gap_range > 0:
                    low_to_scan = min(b[1]['l'] for b in scan_bars)
                    gap_consumed = (daily_open - low_to_scan) / gap_range
                    c['gap_consumed'] = round(gap_consumed, 4)
                    if gap_consumed > GAP_CONSUMED_MAX:
                        n_eroded += 1
                        log.info(f"    {tk}: gap eroded {gap_consumed*100:.0f}% consumed "
                                 f"(open ${daily_open:.2f} → low ${low_to_scan:.2f}, "
                                 f"ref ${ref_price:.2f})")
                        continue

            reranked.append(c)
            n_reranked += 1

        reranked.sort(key=lambda x: -x['gap_pct'])
        candidates = reranked
        n_gap_universe = len(candidates)
        drop_msg = f"{n_dropped} no bar"
        erode_msg = f"{n_eroded} eroded" if n_eroded else ""
        log.info(f"  Re-ranked {n_reranked} by gap at {SCAN_TIME_ET} ET "
                 f"({', '.join(m for m in [drop_msg, erode_msg] if m)})")

    # 7. Select top N (walk top 30, CS-verify, take first)
    selected = select_top_n(candidates)

    # 8. Persist scan pool (top 30 after gap+price+float for E2-B reuse)
    if not dry_run:
        persist_scan_pool(scan_date_str, candidates)

    # 9. Write to DB
    runtime = time.time() - t_start
    if not dry_run:
        write_to_db(scan_date_str, candidates, selected, mode, runtime,
                    spy_gap, n_gap_universe, variant='e2a')
    else:
        log.info("  DRY RUN — skipping DB write")

    # 10. Print watchlist
    print_watchlist(selected, scan_date_str, variant='e2a')

    # 11. Summary
    n_qualified = sum(1 for c in candidates if c.get('passed_all', False))
    log.info(f"  Funnel: {n_gap_universe} gap >= {GAP_MIN*100:.0f}% → "
             f"{sum(1 for c in candidates if c.get('passed_float', True))} after float → "
             f"{len(selected)} selected")
    log.info(f"  Total runtime: {runtime:.1f}s")

    return selected


# ═══════════════════════════════════════════════════════════════════════════════
# SCANNER — E2-B (run at 7:30 PST / 10:30 ET)
# Reads persisted 6:31 pool, fetches RTH bars, checks SD, selects
# ═══════════════════════════════════════════════════════════════════════════════

def run_scan_e2b(scan_date_str=None, dry_run=False):
    t_start = time.time()

    if scan_date_str is None:
        scan_date_str = datetime.now(ET).strftime('%Y-%m-%d')
        mode = 'live'
    else:
        mode = 'historical'

    log.info(f"{'='*_W}")
    log.info(f"  AGENT 1-SD — E2-B Scanner (CS + sustained_drop, 7:30 PST)")
    log.info(f"  Date: {scan_date_str}  |  Mode: {mode}  |  Dry run: {dry_run}")
    log.info(f"{'='*_W}")

    if not is_trading_day(scan_date_str):
        log.info(f"  {scan_date_str} is not a trading day. Exiting.")
        return

    init_db()

    # 1. Load persisted 6:31 scan pool
    pool = load_scan_pool(scan_date_str)
    if not pool:
        log.error(f"  No scan pool for {scan_date_str}. Run with timestamp >= 06:31 PST first.")
        return

    # 2. Fetch minute bars for pool candidates (need 60+ RTH bars for SD classification)
    tickers = [c['ticker'] for c in pool]
    log.info(f"  Fetching minute bars for {len(tickers)} pool candidates...")
    bar_data = fetch_minute_bars(tickers, scan_date_str)

    # 3. Classify patterns + check SD + CS + find entry
    n_sd = 0
    n_no_sd = 0
    n_no_data = 0
    n_no_cs = 0

    selected = []
    for c in pool:
        if len(selected) >= TOP_N:
            break

        tk = c['ticker']
        bars = bar_data.get(tk)
        if bars is None:
            n_no_data += 1
            continue

        rth_bars = bars.get('rth', [])
        pattern = classify_open_pattern(rth_bars)
        c['open_pattern'] = pattern

        if pattern != 'sustained_drop':
            n_no_sd += 1
            continue

        # CS check (use cached result from E2-A if available)
        if c.get('passed_cs') == 0:
            n_no_cs += 1
            continue
        if c.get('passed_cs') is None:
            if not _is_common_stock(tk):
                n_no_cs += 1
                continue
            time.sleep(0.05)

        n_sd += 1

        # Attach PM summary
        pm_summary = bars.get('pm_summary', {})
        c['pm_high'] = pm_summary.get('pm_high')
        c['pm_last'] = pm_summary.get('pm_last')
        c['pm_retrace_pct'] = pm_summary.get('pm_retrace_pct')

        c['selection_rank'] = len(selected) + 1
        c['selected'] = True
        selected.append(c)

    log.info(f"  SD filter: {n_sd} sustained_drop (selected), "
             f"{n_no_sd} other pattern, {n_no_cs} non-CS, {n_no_data} no bar data")

    # 4. Write to DB
    runtime = time.time() - t_start
    spy_gap = None
    if not dry_run:
        write_to_db(scan_date_str, pool, selected, mode, runtime,
                    spy_gap, len(pool), variant='e2b')
    else:
        log.info("  DRY RUN — skipping DB write")

    # 5. Print watchlist
    print_watchlist(selected, scan_date_str, variant='e2b')

    # 6. Summary
    log.info(f"  Funnel: {len(pool)} pool → {n_sd} SD+CS → {len(selected)} selected")
    log.info(f"  Total runtime: {runtime:.1f}s")

    # 7. Print pattern breakdown for all pool candidates
    pat_counts = defaultdict(int)
    for c in pool:
        pat_counts[c.get('open_pattern', 'not_checked')] += 1
    if pat_counts:
        print(f"\n  Pattern breakdown (pool of {len(pool)}):")
        for pat in ['sustained_drop', 'drop_then_recover', 'mild_drop', 'flat_or_bounce', 'no_data', 'not_checked']:
            if pat_counts.get(pat, 0) > 0:
                print(f"    {pat:25s}  {pat_counts[pat]:>3d}")
        print()

    return selected


# ═══════════════════════════════════════════════════════════════════════════════
# DISPATCH
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_scan_timestamp(ts):
    if ts is None:
        now_pst = datetime.now(PT)
        return now_pst.strftime('%Y-%m-%d'), now_pst.hour, now_pst.minute
    ts = ts.strip()
    if ' ' in ts:
        date_part, time_part = ts.split(' ', 1)
        hh, mm = int(time_part.split(':')[0]), int(time_part.split(':')[1])
    else:
        date_part = ts
        hh, mm = 23, 59
    return date_part, hh, mm


init_db()

if MODE == 'scan':
    scan_date_str, pst_hour, pst_min = _parse_scan_timestamp(SCAN_TIMESTAMP)
    pst_time = (pst_hour, pst_min)
    is_live = SCAN_TIMESTAMP is None

    print(f"\n  {'='*62}")
    print(f"  |{'TIME MACHINE':^60}|")
    print(f"  |{'':^60}|")
    ts_display = f"{scan_date_str}  {pst_hour:02d}:{pst_min:02d} PST"
    if is_live:
        ts_display += '  (LIVE)'
    print(f"  |  {ts_display:<58}|")
    print(f"  {'='*62}")

    run_e2b = pst_time >= E2B_READY_PST

    # E2-A always runs (no time gate — data uses SCAN_TIME_ET regardless)
    scan_date_arg = scan_date_str if not is_live else None
    run_scan_e2a(scan_date_str=scan_date_arg, dry_run=DRY_RUN)

    if run_e2b:
        print(f"\n  {'─'*62}")
        print(f"  E2-B (Agent 1-SD) — sustained_drop confirmed by 10:30 ET")
        print(f"  {'─'*62}\n")
        run_scan_e2b(scan_date_str=scan_date_arg, dry_run=DRY_RUN)
    else:
        print(f"\n  {'─'*62}")
        print(f"  E2-B (Agent 1-SD): TOO EARLY — need 7:30 PST (10:30 ET)")
        print(f"  Current: {pst_hour:02d}:{pst_min:02d} PST. SD requires 60 bars of RTH data.")
        print(f"  Re-run at 07:30 PST or later to get E2-B output.")
        print(f"  {'─'*62}\n")

elif MODE == 'fill':
    scan_date_str, _, _ = _parse_scan_timestamp(SCAN_TIMESTAMP)
    fill_date = scan_date_str if SCAN_TIMESTAMP else None
    run_fill_outcomes(fill_date_str=fill_date)
elif MODE == 'report':
    run_report(weeks=REPORT_WEEKS)
else:
    print(f"Unknown MODE: {MODE}. Use 'scan', 'fill', or 'report'.")
