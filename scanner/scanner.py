# %% Agent 2 — Combined Morning Scanner
# Merges two signal sources:
#   Source A: Polygon full US equity universe (technical gap signal)
#   Source B: BigQuery LLM scanner output (catalyst signal)
# Selects top 2 SHORT candidates, outputs unified watchlist.
#
# Usage (Colab cell or CLI):
#   MODE = 'scan'              → morning scanner (default)
#   MODE = 'fill'              → fill T0/T1/T2 outcomes
#   MODE = 'report'            → performance summary
#   MODE = 'report_analysis'   → weekly analysis queries
#   SCAN_DATE = None           → today (live)
#   SCAN_DATE = '2026-09-08'   → historical
#   DRY_RUN = True             → skip DB writes

import os, sys, json, time, re, logging, sqlite3, asyncio, argparse
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
    from google.cloud import bigquery
except ImportError:
    import subprocess; subprocess.check_call([sys.executable, "-m", "pip", "install", "google-cloud-bigquery", "--quiet", "--break-system-packages"])
    from google.cloud import bigquery

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

MODE = 'scan'
SCAN_DATE = None
DRY_RUN = False
REPORT_WEEKS = 4

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs")
POLYGON_BASE = "https://api.polygon.io"

BQ_PROJECT = os.environ.get("BIGQUERY_PROJECT_ID", "yanina-everything")
BQ_PREMARKET_TABLE = os.environ.get("BIGQUERY_PREMARKET_TABLE",
                                     "ys-time-series.scoring_logs.premarket_scanner")
BQ_DISCOVERY_TABLE = os.environ.get("BIGQUERY_DISCOVERY_TABLE",
                                     "ys-time-series.scoring_logs.discovery_scanner")

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

DB_PATH = os.environ.get("DB_PATH", os.path.join(BASE_DIR, 'data', 'scanner.db'))
FLOAT_CACHE_PATH = os.environ.get("FLOAT_CACHE_PATH",
                                   os.path.join(BASE_DIR, 'data', 'float_cache.json'))
LOG_PATH = os.path.join(BASE_DIR, 'logs', 'scanner.log')

# Gap filters
GAP_MIN_SCAN = 0.015       # 1.5% — pull PM bars above this
GAP_MIN_AGENT1 = 0.050     # 5% — pure gap signal threshold
PRICE_MIN = 10.0
FLOAT_MIN = 10_000_000
PM_RETRACE_MAX = 0.80

# Selection
TARGET_TRADES = 2
PM_BAR_WORKERS = 10

# Catalyst tier assignments
CATALYST_T1 = {
    'offering', 'secondary_offering', 'atm_offering', 'convertible_note',
    'bought_deal', 'shelf_registration',
    'fda_approval_primary', 'fda_approval_secondary',
    'fda_fast_track', 'fda_breakthrough',
    'clinical_trial_positive',
    'regulatory', 'government_contract',
}
CATALYST_T2 = {
    'initiation_buy', 'upgrade', 'pt_raise', 'consensus_revision_up',
    'sector_call_positive',
    'partnership_announced', 'contract_win_large', 'product_launch',
    'institutional_13d', 'index_addition',
    'analyst',
}
CATALYST_T3 = {
    'initiation_neutral', 'pt_cut', 'contract_win_small',
    'insider_buy_large', 'insider', 'insider_buy_small',
    'revenue_guidance_raise',
    'peer_earnings_halo', 'licensing_deal', 'distribution_agreement',
    'sympathy_move', 'macro',
}
CATALYST_EXCLUDE = {
    'earnings', 'earnings_beat_guidance', 'earnings_beat_no_guide',
    'earnings_inline', 'earnings_miss',
    'acquisition_target', 'merger_of_equals',
    'breaking', 'short_squeeze', 'gamma_squeeze',
    'technical_breakout',
}

# Scoring weights
WEIGHT_GAP = 0.50
WEIGHT_CATALYST = 0.35
WEIGHT_RETRACE = 0.15

ET = pytz.timezone('America/New_York')
PT = pytz.timezone('America/Los_Angeles')

TICKER_REGEX = re.compile(r'^[A-Z]{1,5}$')
_EXCLUDED_SUFFIXES = ('W', 'WS', 'U', 'R')
_EXCLUDED_TICKERS = {
    'SPY', 'QQQ', 'IWM', 'DIA', 'VXX', 'UVXY', 'SQQQ', 'TQQQ',
    'SPXU', 'SPXL', 'SDOW', 'UDOW', 'SDS', 'SSO', 'SH',
    'LABU', 'LABD', 'SOXL', 'SOXS', 'FNGU', 'FNGD',
    'NUGT', 'DUST', 'JNUG', 'JDST', 'TNA', 'TZA',
    'UPRO', 'VIXY', 'SVXY', 'TVIX',
}

_NYSE_HOLIDAYS = {
    '2026-01-01', '2026-01-19', '2026-02-16', '2026-04-03', '2026-05-25',
    '2026-06-19', '2026-07-03', '2026-09-07', '2026-11-26', '2026-12-25',
    '2027-01-01', '2027-01-18', '2027-02-15', '2027-03-26', '2027-05-31',
    '2027-06-18', '2027-07-05', '2027-09-06', '2027-11-25', '2027-12-24',
    '2028-01-17', '2028-02-21', '2028-04-14', '2028-05-29',
    '2028-06-19', '2028-07-04', '2028-09-04', '2028-11-23', '2028-12-25',
    '2029-01-01', '2029-01-15', '2029-02-19', '2029-03-30', '2029-05-28',
    '2029-06-19', '2029-07-04', '2029-09-03', '2029-11-22', '2029-12-25',
    '2030-01-01', '2030-01-21', '2030-02-18', '2030-04-19', '2030-05-27',
    '2030-06-19', '2030-07-04', '2030-09-02', '2030-11-28', '2030-12-25',
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
    return logging.getLogger('scanner')

log = setup_logging()


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

def init_db(db_path=DB_PATH):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")

    conn.executescript("""
    CREATE TABLE IF NOT EXISTS daily_runs (
        run_date          TEXT PRIMARY KEY,
        run_timestamp     TEXT,
        mode              TEXT,
        runtime_seconds   REAL,
        n_polygon_total   INTEGER,
        n_polygon_gap     INTEGER,
        n_polygon_price   INTEGER,
        n_polygon_float   INTEGER,
        n_polygon_retrace INTEGER,
        n_bq_premarket    INTEGER,
        n_bq_discovery    INTEGER,
        n_bq_total        INTEGER,
        n_both_sources    INTEGER,
        n_source_a_only   INTEGER,
        n_source_b_only   INTEGER,
        n_selected        INTEGER,
        spy_gap_pct       REAL,
        spy_prev_close    REAL
    );

    CREATE TABLE IF NOT EXISTS candidates (
        candidate_id      TEXT PRIMARY KEY,
        run_date          TEXT,
        ticker            TEXT,
        in_polygon        INTEGER,
        in_bq             INTEGER,
        bq_source         TEXT,
        gap_pct           REAL,
        gap_direction     TEXT DEFAULT 'up',
        prior_close       REAL,
        open_price        REAL,
        gap_tier          TEXT,
        pm_high           REAL,
        pm_last           REAL,
        pm_retrace_pct    REAL,
        float_shares      REAL,
        market_cap_proxy  REAL,
        headline          TEXT,
        news_timestamp    TEXT,
        news_age_hrs      REAL,
        catalyst_type     TEXT,
        subcategory       TEXT,
        llm_direction     TEXT,
        llm_confidence    REAL,
        llm_expected_move REAL,
        llm_remaining_move REAL,
        llm_move_exhausted INTEGER,
        gate_reason       TEXT,
        catalyst_tier     TEXT,
        passed_gap        INTEGER,
        passed_price      INTEGER,
        passed_float      INTEGER,
        passed_retrace    INTEGER,
        passed_catalyst   INTEGER,
        passed_all        INTEGER,
        filter_fail       TEXT,
        gap_score         REAL,
        catalyst_score    REAL,
        retrace_score     REAL,
        combined_score    REAL,
        signal_source     TEXT,
        rank_overall      INTEGER,
        selected          INTEGER DEFAULT 0,
        selection_rank    INTEGER,
        t1_date           TEXT,
        t2_date           TEXT,
        t0_close          REAL,
        t1_close          REAL,
        t2_close          REAL,
        pnl_eod           REAL,
        pnl_t1            REAL,
        pnl_t2            REAL,
        log_timestamp     TEXT
    );

    CREATE TABLE IF NOT EXISTS trades (
        trade_id          TEXT PRIMARY KEY,
        run_date          TEXT,
        ticker            TEXT,
        selection_rank    INTEGER,
        signal_source     TEXT,
        gap_pct           REAL,
        gap_tier          TEXT,
        catalyst_tier     TEXT,
        subcategory       TEXT,
        combined_score    REAL,
        pm_retrace_pct    REAL,
        t1_date           TEXT,
        t2_date           TEXT,
        entry_price       REAL DEFAULT NULL,
        entry_time        TEXT DEFAULT NULL,
        entry_type        TEXT DEFAULT NULL,
        skip_reason       TEXT DEFAULT NULL,
        slippage_pct      REAL DEFAULT NULL,
        borrow_rate_pct   REAL DEFAULT NULL,
        position_size_usd REAL DEFAULT NULL,
        t0_close          REAL DEFAULT NULL,
        t1_close          REAL DEFAULT NULL,
        t2_close          REAL DEFAULT NULL,
        pnl_30m           REAL DEFAULT NULL,
        pnl_60m           REAL DEFAULT NULL,
        pnl_eod           REAL DEFAULT NULL,
        pnl_t1            REAL DEFAULT NULL,
        pnl_t2            REAL DEFAULT NULL,
        actually_traded   INTEGER DEFAULT 0,
        actual_pnl        REAL DEFAULT NULL,
        exit_reason       TEXT DEFAULT NULL,
        notes             TEXT DEFAULT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_candidates_date ON candidates(run_date);
    CREATE INDEX IF NOT EXISTS idx_candidates_selected ON candidates(selected);
    CREATE INDEX IF NOT EXISTS idx_trades_t2 ON trades(t2_date);
    CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(run_date);
    """)
    conn.close()
    return db_path


def _get_conn():
    return sqlite3.connect(DB_PATH)


# ═══════════════════════════════════════════════════════════════════════════════
# TRADING CALENDAR
# ═══════════════════════════════════════════════════════════════════════════════

def is_trading_day(d):
    ds = d if isinstance(d, str) else d.strftime('%Y-%m-%d')
    dt = datetime.strptime(ds, '%Y-%m-%d').date() if isinstance(ds, str) else ds
    return dt.weekday() < 5 and ds not in _NYSE_HOLIDAYS


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


def get_t1_t2_dates(scan_date_str):
    nxt = get_next_n_trading_dates(scan_date_str, 2)
    t1 = nxt[0].strftime('%Y-%m-%d') if len(nxt) >= 1 else None
    t2 = nxt[1].strftime('%Y-%m-%d') if len(nxt) >= 2 else None
    return t1, t2


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
        log.info(f"Float cache: {len(cache)} tickers ({len(file_cache)} file + {len(_EMBEDDED_FLOAT)} embedded)")
    else:
        log.info(f"Float cache: {len(cache)} tickers (embedded only)")
    return cache


def save_float_cache(cache, path=FLOAT_CACHE_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    serializable = {k: v for k, v in cache.items() if v is not None}
    with open(path, 'w') as f:
        json.dump(serializable, f, indent=1)


# ═══════════════════════════════════════════════════════════════════════════════
# POLYGON API
# ═══════════════════════════════════════════════════════════════════════════════

def _polygon_get(url, params=None):
    if params is None:
        params = {}
    params['apiKey'] = POLYGON_API_KEY
    for attempt in range(5):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                wait = 12 * (attempt + 1)
                log.warning(f"Rate limited (429), waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            wait = 5 * (2 ** attempt)
            log.warning(f"Connection error, retry {attempt+1}/5, waiting {wait}s...")
            time.sleep(wait)
    resp = requests.get(url, params=params, timeout=60)
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
        }
    log.info(f"Grouped daily ({date_str}): {len(out)} tickers")
    return out


def fetch_snapshot_all():
    url = f"{POLYGON_BASE}/v2/snapshot/locale/us/markets/stocks/tickers"
    data = _polygon_get(url, {"include_otc": "false"})
    tickers = data.get('tickers', [])
    out = {}
    for t in tickers:
        tk = t.get('ticker', '')
        if not TICKER_REGEX.match(tk):
            continue
        prev_c = (t.get('prevDay') or {}).get('c')
        pm_p = (t.get('min') or {}).get('c')
        if not pm_p or pm_p <= 0:
            pm_p = prev_c
        if pm_p and pm_p > 0 and prev_c and prev_c > 0:
            out[tk] = {
                'price': float(pm_p),
                'prev_close': float(prev_c),
            }
    log.info(f"Snapshot: {len(out)} tickers with price data")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE A — POLYGON GAP COMPUTATION
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

        if gap_pct < GAP_MIN_SCAN:
            continue

        candidates.append({
            'ticker': tk,
            'gap_pct': round(gap_pct, 6),
            'prior_close': round(prev_c, 4),
            'open_price': round(cur_p, 4),
        })

    candidates.sort(key=lambda x: -x['gap_pct'])
    log.info(f"Gap filter: {n_total} tickers checked, "
             f"{len(candidates)} with gap >= {GAP_MIN_SCAN*100:.1f}%")
    return candidates, n_total


# ═══════════════════════════════════════════════════════════════════════════════
# ASYNC PM BAR FETCH
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
                async with session.get(url, params=params,
                                       timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(2 ** (attempt + 1))
                        continue
                    if resp.status != 200:
                        return ticker, None
                    data = await resp.json()
            except Exception:
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
                        'o': float(b.get('o', 0)),
                        'h': float(b.get('h', 0)),
                        'l': float(b.get('l', 0)),
                        'c': float(b.get('c', 0)),
                        'v': int(b.get('v', 0)),
                    })

            if not pm_bars:
                return ticker, {'pm_high': None, 'pm_last': None, 'pm_retrace_pct': None}

            pm_open = pm_bars[0]['o']
            pm_high = max(b['h'] for b in pm_bars)
            pm_last = pm_bars[-1]['c']
            total_move = pm_high - pm_open
            retrace = (pm_high - pm_last) / total_move if total_move > 0 else 0.0

            return ticker, {
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
    pm_data = asyncio.run(_fetch_all_pm(tickers, date_str))
    elapsed = time.time() - t0
    n_ok = sum(1 for v in pm_data.values() if v.get('pm_retrace_pct') is not None)
    log.info(f"PM bars: {n_ok}/{len(tickers)} fetched in {elapsed:.1f}s")
    return pm_data


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE B — BIGQUERY LLM SCANNER SIGNALS
# ═══════════════════════════════════════════════════════════════════════════════

def query_bq(date_str):
    try:
        bq_client = bigquery.Client(project=BQ_PROJECT)
    except Exception as e:
        log.error(f"BQ client init failed: {e}")
        return [], []

    premarket = []
    discovery = []

    # Premarket scanner
    try:
        q_pm = f"""
        SELECT
            ticker, direction,
            confidence, expected_move_pct,
            remaining_move_pct, move_is_exhausted,
            catalyst_type, catalyst, headline, news_time,
            gate_reason, selected, rank_in_direction,
            scanner_version
        FROM `{BQ_PREMARKET_TABLE}`
        WHERE DATE(scan_date) = '{date_str}'
        ORDER BY confidence DESC
        """
        rows = list(bq_client.query(q_pm).result())
        for r in rows:
            premarket.append({
                'ticker': r['ticker'],
                'bq_source': 'premarket',
                'llm_direction': r['direction'],
                'llm_confidence': r['confidence'] or 0,
                'llm_expected_move': r['expected_move_pct'] or 0,
                'llm_remaining_move': r.get('remaining_move_pct') or 0,
                'llm_move_exhausted': 1 if r.get('move_is_exhausted') else 0,
                'catalyst_type': r['catalyst_type'] or '',
                'catalyst': r.get('catalyst') or '',
                'headline': r.get('headline') or '',
                'news_time': r.get('news_time') or '',
                'gate_reason': r.get('gate_reason') or '',
                'bq_selected': r.get('selected', False),
                'bq_rank': r.get('rank_in_direction') or 0,
            })
        log.info(f"BQ premarket ({date_str}): {len(premarket)} signals")
    except Exception as e:
        log.error(f"BQ premarket query failed: {e}")

    # Discovery scanner
    try:
        q_disc = f"""
        SELECT
            ticker, direction,
            confidence, expected_move_pct,
            remaining_move_pct, move_is_exhausted,
            catalyst_type, catalyst, headline, news_time,
            gate_reason, selected, rank_in_direction,
            scanner_version
        FROM `{BQ_DISCOVERY_TABLE}`
        WHERE DATE(scan_date) = '{date_str}'
        ORDER BY confidence DESC
        """
        rows = list(bq_client.query(q_disc).result())
        for r in rows:
            discovery.append({
                'ticker': r['ticker'],
                'bq_source': 'discovery',
                'llm_direction': r['direction'],
                'llm_confidence': r['confidence'] or 0,
                'llm_expected_move': r['expected_move_pct'] or 0,
                'llm_remaining_move': r.get('remaining_move_pct') or 0,
                'llm_move_exhausted': 1 if r.get('move_is_exhausted') else 0,
                'catalyst_type': r['catalyst_type'] or '',
                'catalyst': r.get('catalyst') or '',
                'headline': r.get('headline') or '',
                'news_time': r.get('news_time') or '',
                'gate_reason': r.get('gate_reason') or '',
                'bq_selected': r.get('selected', False),
                'bq_rank': r.get('rank_in_direction') or 0,
            })
        log.info(f"BQ discovery ({date_str}): {len(discovery)} signals")
    except Exception as e:
        log.error(f"BQ discovery query failed: {e}")

    return premarket, discovery


# ═══════════════════════════════════════════════════════════════════════════════
# MERGE SOURCES
# ═══════════════════════════════════════════════════════════════════════════════

def merge_sources(polygon_candidates, bq_premarket, bq_discovery):
    polygon_map = {c['ticker']: c for c in polygon_candidates}

    # BQ: premarket first, discovery fills in if not present
    bq_map = {}
    for s in bq_premarket:
        tk = s['ticker']
        if tk not in bq_map:
            bq_map[tk] = s
    for s in bq_discovery:
        tk = s['ticker']
        if tk not in bq_map:
            bq_map[tk] = s

    all_tickers = set(polygon_map.keys()) | set(bq_map.keys())

    merged = []
    n_both = n_a_only = n_b_only = 0

    for tk in all_tickers:
        in_poly = tk in polygon_map
        in_bq = tk in bq_map

        if in_poly and in_bq:
            signal_source = 'both'
            n_both += 1
        elif in_poly:
            signal_source = 'gap_only'
            n_a_only += 1
        else:
            signal_source = 'catalyst_only'
            n_b_only += 1

        entry = {
            'ticker': tk,
            'in_polygon': 1 if in_poly else 0,
            'in_bq': 1 if in_bq else 0,
            'signal_source': signal_source,
        }

        # Gap data — prefer Polygon
        if in_poly:
            pg = polygon_map[tk]
            entry['gap_pct'] = pg['gap_pct']
            entry['prior_close'] = pg['prior_close']
            entry['open_price'] = pg['open_price']
        elif in_bq:
            # BQ doesn't store gap_pct directly; use expected_move as proxy
            bqs = bq_map[tk]
            entry['gap_pct'] = abs(bqs['llm_expected_move']) / 100.0 if bqs['llm_expected_move'] else 0
            entry['prior_close'] = None
            entry['open_price'] = None

        # Gap direction
        entry['gap_direction'] = 'up' if entry.get('gap_pct', 0) >= 0 else 'down'

        # BQ/LLM data
        if in_bq:
            bqs = bq_map[tk]
            entry['bq_source'] = bqs['bq_source']
            entry['headline'] = bqs.get('headline')
            entry['news_time'] = bqs.get('news_time')
            entry['catalyst_type'] = bqs.get('catalyst_type', '')
            entry['subcategory'] = bqs.get('catalyst_type', '')
            entry['llm_direction'] = bqs.get('llm_direction')
            entry['llm_confidence'] = bqs.get('llm_confidence', 0)
            entry['llm_expected_move'] = bqs.get('llm_expected_move', 0)
            entry['llm_remaining_move'] = bqs.get('llm_remaining_move', 0)
            entry['llm_move_exhausted'] = bqs.get('llm_move_exhausted', 0)
            entry['gate_reason'] = bqs.get('gate_reason', '')
        else:
            entry['bq_source'] = None
            entry['headline'] = None
            entry['news_time'] = None
            entry['catalyst_type'] = None
            entry['subcategory'] = None
            entry['llm_direction'] = None
            entry['llm_confidence'] = None
            entry['llm_expected_move'] = None
            entry['llm_remaining_move'] = None
            entry['llm_move_exhausted'] = None
            entry['gate_reason'] = None

        merged.append(entry)

    merged.sort(key=lambda x: -(x.get('gap_pct') or 0))

    log.info(f"Merge: {n_both} both, {n_a_only} gap-only, {n_b_only} catalyst-only "
             f"= {len(merged)} total")

    return merged, n_both, n_a_only, n_b_only


# ═══════════════════════════════════════════════════════════════════════════════
# CATALYST TIER ASSIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════

def assign_catalyst_tier(catalyst_type):
    if not catalyst_type:
        return None
    ct = catalyst_type.lower().strip()
    if ct in CATALYST_T1:
        return 'T1'
    if ct in CATALYST_T2:
        return 'T2'
    if ct in CATALYST_T3:
        return 'T3'
    if ct in CATALYST_EXCLUDE:
        return 'EXCLUDE'
    return 'T2'


# ═══════════════════════════════════════════════════════════════════════════════
# APPLY FILTERS
# ═══════════════════════════════════════════════════════════════════════════════

def apply_filters(merged, pm_data, float_cache):
    for c in merged:
        tk = c['ticker']
        gap = c.get('gap_pct', 0) or 0

        # PM bar data
        pm = pm_data.get(tk, {})
        c['pm_high'] = pm.get('pm_high')
        c['pm_last'] = pm.get('pm_last')
        c['pm_retrace_pct'] = pm.get('pm_retrace_pct')

        # Float
        fs = float_cache.get(tk)
        c['float_shares'] = fs
        if fs and c.get('open_price'):
            c['market_cap_proxy'] = round(c['open_price'] * fs)
        else:
            c['market_cap_proxy'] = None

        # Gap tier
        if gap >= 0.20:
            c['gap_tier'] = 'T5+'
        elif gap >= 0.08:
            c['gap_tier'] = 'T5'
        elif gap >= 0.05:
            c['gap_tier'] = 'T4'
        elif gap >= 0.015:
            c['gap_tier'] = 'T3'
        else:
            c['gap_tier'] = 'none'

        # Catalyst tier
        c['catalyst_tier'] = assign_catalyst_tier(c.get('catalyst_type'))

        # ── Filter chain ──
        c['passed_gap'] = 1 if gap >= GAP_MIN_SCAN else 0
        c['passed_price'] = 1 if (c.get('open_price') or 0) >= PRICE_MIN else 0
        c['passed_float'] = 1 if (fs is None or fs >= FLOAT_MIN) else 0

        retrace = c.get('pm_retrace_pct')
        c['passed_retrace'] = 1 if (retrace is None or retrace <= PM_RETRACE_MAX) else 0

        # Catalyst filter (BQ tickers only)
        if c['in_bq']:
            ct = c.get('catalyst_tier')
            gr = c.get('gate_reason') or ''
            if ct == 'EXCLUDE':
                c['passed_catalyst'] = 0
            elif gr and not gr.startswith('MOMENTUM_FLIP'):
                c['passed_catalyst'] = 0
            elif c.get('llm_direction') not in ('SHORT', None):
                # BQ flagged as LONG or NEUTRAL → skip for SHORT agent
                c['passed_catalyst'] = 0
            else:
                c['passed_catalyst'] = 1
        else:
            c['passed_catalyst'] = None

        # ── passed_all decision ──
        filter_fail = None
        src = c['signal_source']

        if not c['passed_gap']:
            filter_fail = 'gap_too_small'
        elif not c['passed_price'] and src != 'catalyst_only':
            filter_fail = 'price_too_low'
        elif not c['passed_float'] and src != 'catalyst_only':
            filter_fail = 'float_too_low'
        elif not c['passed_retrace'] and src != 'catalyst_only':
            filter_fail = 'pm_retrace_too_high'
        elif src == 'gap_only' and gap < GAP_MIN_AGENT1:
            filter_fail = 'gap_below_agent1_threshold'
        elif src in ('both', 'catalyst_only') and c['passed_catalyst'] == 0:
            ct = c.get('catalyst_tier')
            if ct == 'EXCLUDE':
                filter_fail = 'catalyst_excluded'
            elif c.get('llm_direction') == 'LONG':
                filter_fail = 'llm_direction_long'
            elif c.get('llm_direction') == 'NEUTRAL':
                filter_fail = 'llm_direction_neutral'
            elif c.get('gate_reason'):
                filter_fail = 'bq_gated'
            else:
                filter_fail = 'catalyst_failed'

        c['filter_fail'] = filter_fail
        c['passed_all'] = 1 if filter_fail is None else 0

    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# SCORE CANDIDATES
# ═══════════════════════════════════════════════════════════════════════════════

def score_candidates(merged):
    qualified = [c for c in merged if c['passed_all'] == 1]
    if not qualified:
        for c in merged:
            c['gap_score'] = 0
            c['catalyst_score'] = 0
            c['retrace_score'] = 0
            c['combined_score'] = 0
            c['rank_overall'] = None
        return merged

    gaps = [c['gap_pct'] for c in qualified]
    gap_min = min(gaps)
    gap_max = max(gaps)
    gap_range = gap_max - gap_min if gap_max > gap_min else 1.0

    for c in merged:
        if c['passed_all'] != 1:
            c['gap_score'] = 0
            c['catalyst_score'] = 0
            c['retrace_score'] = 0
            c['combined_score'] = 0
            c['rank_overall'] = None
            continue

        # Gap score (0-1)
        gs = (c['gap_pct'] - gap_min) / gap_range
        c['gap_score'] = round(min(gs, 1.0), 4)

        # Catalyst score (0-1)
        src = c['signal_source']
        if src == 'gap_only':
            c['catalyst_score'] = 0.40
        elif src == 'catalyst_only':
            c['catalyst_score'] = 0.50
        else:
            ct = c.get('catalyst_tier')
            if ct == 'T1':
                c['catalyst_score'] = 1.00
            elif ct == 'T2':
                c['catalyst_score'] = 0.65
            elif ct == 'T3':
                c['catalyst_score'] = 0.40
            else:
                c['catalyst_score'] = 0.50

        # PM retrace score (0-1)
        retrace = c.get('pm_retrace_pct')
        if retrace is None:
            c['retrace_score'] = 0.50
        elif 0.20 <= retrace <= 0.50:
            c['retrace_score'] = 1.00
        elif retrace < 0.20:
            c['retrace_score'] = 0.80
        elif retrace <= 0.80:
            c['retrace_score'] = 0.60
        else:
            c['retrace_score'] = 0.0

        # Combined
        combined = (WEIGHT_GAP * c['gap_score']
                    + WEIGHT_CATALYST * c['catalyst_score']
                    + WEIGHT_RETRACE * c['retrace_score'])

        # Dual source bonus
        if src == 'both':
            combined *= 1.20

        c['combined_score'] = round(min(combined, 1.0), 4)

    # Rank qualified candidates
    qualified.sort(key=lambda x: -x['combined_score'])
    for i, c in enumerate(qualified):
        c['rank_overall'] = i + 1

    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# DB WRITE
# ═══════════════════════════════════════════════════════════════════════════════

def write_to_db(scan_date_str, merged, selected, run_meta):
    conn = _get_conn()
    now_utc = datetime.now(pytz.UTC).isoformat()
    t1, t2 = get_t1_t2_dates(scan_date_str)

    # daily_runs
    conn.execute("""INSERT OR REPLACE INTO daily_runs VALUES
        (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (scan_date_str, now_utc, run_meta['mode'], run_meta['runtime'],
         run_meta['n_polygon_total'], run_meta['n_polygon_gap'],
         run_meta.get('n_polygon_price', 0), run_meta.get('n_polygon_float', 0),
         run_meta.get('n_polygon_retrace', 0),
         run_meta['n_bq_premarket'], run_meta['n_bq_discovery'],
         run_meta['n_bq_total'],
         run_meta['n_both'], run_meta['n_a_only'], run_meta['n_b_only'],
         len(selected), run_meta.get('spy_gap_pct'), run_meta.get('spy_prev_close')))

    # candidates — ALL tickers logged
    for c in merged:
        cid = f"{scan_date_str}_{c['ticker']}"
        vals = (
            cid, scan_date_str, c['ticker'],
            c['in_polygon'], c['in_bq'], c.get('bq_source'),
            c.get('gap_pct'), c.get('gap_direction', 'up'),
            c.get('prior_close'), c.get('open_price'),
            c.get('gap_tier'),
            c.get('pm_high'), c.get('pm_last'), c.get('pm_retrace_pct'),
            c.get('float_shares'), c.get('market_cap_proxy'),
            c.get('headline'), c.get('news_time'), c.get('news_age_hrs'),
            c.get('catalyst_type'), c.get('subcategory'),
            c.get('llm_direction'), c.get('llm_confidence'),
            c.get('llm_expected_move'), c.get('llm_remaining_move'),
            c.get('llm_move_exhausted'),
            c.get('gate_reason'), c.get('catalyst_tier'),
            c.get('passed_gap'), c.get('passed_price'),
            c.get('passed_float'), c.get('passed_retrace'),
            c.get('passed_catalyst'), c.get('passed_all'),
            c.get('filter_fail'),
            c.get('gap_score'), c.get('catalyst_score'),
            c.get('retrace_score'), c.get('combined_score'),
            c.get('signal_source'), c.get('rank_overall'),
            c.get('selected', 0), c.get('selection_rank'),
            t1, t2,
            None, None, None,
            None, None, None,
            now_utc,
        )
        placeholders = ','.join(['?'] * len(vals))
        conn.execute(f"INSERT OR REPLACE INTO candidates VALUES ({placeholders})", vals)

    # trades — selected only
    for s in selected:
        tid = f"{scan_date_str}_{s['ticker']}"
        conn.execute("""INSERT OR REPLACE INTO trades
            (trade_id, run_date, ticker, selection_rank, signal_source,
             gap_pct, gap_tier, catalyst_tier, subcategory,
             combined_score, pm_retrace_pct, t1_date, t2_date)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (tid, scan_date_str, s['ticker'], s['selection_rank'],
             s['signal_source'], s['gap_pct'], s.get('gap_tier'),
             s.get('catalyst_tier'), s.get('subcategory'),
             s.get('combined_score'), s.get('pm_retrace_pct'),
             t1, t2))

    conn.commit()
    conn.close()
    log.info(f"DB write: {len(merged)} candidates, {len(selected)} trades")


# ═══════════════════════════════════════════════════════════════════════════════
# FILL OUTCOMES
# ═══════════════════════════════════════════════════════════════════════════════

def fill_outcomes(fill_date_str=None):
    if fill_date_str is None:
        fill_date_str = datetime.now(ET).strftime('%Y-%m-%d')

    log.info(f"FILL OUTCOMES — data available through {fill_date_str}")

    conn = _get_conn()

    # Fill trades table — look up open_price from candidates as fallback
    trades = conn.execute("""
        SELECT t.trade_id, t.run_date, t.ticker, t.t1_date, t.t2_date,
               COALESCE(t.entry_price, c.open_price) as ref_price,
               t.t0_close, t.t1_close, t.t2_close
        FROM trades t
        LEFT JOIN candidates c ON t.run_date = c.run_date AND t.ticker = c.ticker
        WHERE t.t0_close IS NULL OR t.t1_close IS NULL OR t.t2_close IS NULL
    """).fetchall()

    if not trades:
        log.info("  No trades need filling.")

    # Also fill candidates table (for backtest data)
    open_candidates = conn.execute("""
        SELECT candidate_id, run_date, ticker, t1_date, t2_date,
               open_price, t0_close, t1_close, t2_close
        FROM candidates
        WHERE (t0_close IS NULL OR t1_close IS NULL OR t2_close IS NULL)
          AND passed_all = 1
    """).fetchall()

    all_items = []
    for t in trades:
        all_items.append(('trades', t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8]))
    for c in open_candidates:
        all_items.append(('candidates', c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8]))

    if not all_items:
        conn.close()
        return

    log.info(f"  {len(trades)} trades + {len(open_candidates)} candidates need outcome data")

    dates_needed = set()
    for item in all_items:
        _, _, scan_date, _, t1_date, t2_date, _, _, _, _ = item
        dates_needed.add(scan_date)
        if t1_date:
            dates_needed.add(t1_date)
        if t2_date:
            dates_needed.add(t2_date)

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

    for item in all_items:
        table, item_id, scan_date, ticker, t1_date, t2_date, ref_price, t0c, t1c, t2c = item

        updates = {}
        # For trades, use entry_price; for candidates, use open_price
        if ref_price is None or ref_price <= 0:
            continue

        if t0c is None and scan_date in daily_data:
            dd = daily_data[scan_date].get(ticker, {})
            close_val = dd.get('close', 0)
            if close_val and close_val > 0:
                updates['t0_close'] = close_val
                updates['pnl_eod'] = round((ref_price - close_val) / ref_price * 100, 4)

        if t1c is None and t1_date and t1_date in daily_data:
            dd = daily_data[t1_date].get(ticker, {})
            close_val = dd.get('close', 0)
            if close_val and close_val > 0:
                updates['t1_close'] = close_val
                updates['pnl_t1'] = round((ref_price - close_val) / ref_price * 100, 4)

        if t2c is None and t2_date and t2_date in daily_data:
            dd = daily_data[t2_date].get(ticker, {})
            close_val = dd.get('close', 0)
            if close_val and close_val > 0:
                updates['t2_close'] = close_val
                updates['pnl_t2'] = round((ref_price - close_val) / ref_price * 100, 4)

        if updates:
            set_clause = ', '.join(f"{k}=?" for k in updates.keys())
            id_col = 'trade_id' if table == 'trades' else 'candidate_id'
            conn.execute(f"UPDATE {table} SET {set_clause} WHERE {id_col}=?",
                         list(updates.values()) + [item_id])
            n_filled += 1

        # Expand float cache
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
    log.info(f"  Filled {n_filled} items")

    if float_updated:
        save_float_cache(float_cache)
        log.info(f"  Float cache updated: {len(float_cache)} tickers")


# ═══════════════════════════════════════════════════════════════════════════════
# PRINT WATCHLIST
# ═══════════════════════════════════════════════════════════════════════════════

def print_watchlist(selected, merged, scan_date_str):
    t1, t2 = get_t1_t2_dates(scan_date_str)
    now_str = datetime.now(PT).strftime('%H:%M:%S')

    W = 66
    print(f"\n  {'='*W}")
    print(f"  |{'MORNING SCANNER — ' + scan_date_str:^{W-2}}|")
    print(f"  |{'':^{W-2}}|")
    print(f"  |  Run: {now_str} PST  |  Hold: T+2  |  Exit: {t2 or 'n/a':<12}       |")
    print(f"  {'='*W}")

    if not selected:
        print(f"\n  No qualifying trades found.\n")
        return

    # Selected trades table
    print(f"\n  SELECTED TRADES (SHORT, enter at local high or 9:45 fallback)")
    rows = []
    for s in selected:
        retrace = s.get('pm_retrace_pct')
        ret_str = f"{retrace:.2f}" if retrace is not None else 'n/a'
        cat = s.get('subcategory') or s.get('catalyst_type') or '—'
        if len(cat) > 20:
            cat = cat[:18] + '..'

        rows.append([
            f"#{s['selection_rank']}",
            s['ticker'],
            f"+{s['gap_pct']*100:.1f}%",
            s['signal_source'].upper()[:7],
            cat,
            ret_str,
            f"{s['combined_score']:.2f}",
        ])

    print()
    print(tabulate(rows,
                   headers=['Rank', 'Ticker', 'Gap%', 'Source', 'Catalyst', 'PM Ret', 'Score'],
                   tablefmt='simple',
                   colalign=('left', 'left', 'right', 'center', 'left', 'right', 'right')))

    # Also qualified (not selected)
    also_qualified = [c for c in merged
                      if c['passed_all'] == 1 and c.get('selected') != 1]
    also_qualified.sort(key=lambda x: -(x.get('combined_score') or 0))

    if also_qualified[:8]:
        print(f"\n  ALSO QUALIFIED (not selected)")
        rows2 = []
        for c in also_qualified[:8]:
            cat = c.get('subcategory') or c.get('catalyst_type') or '—'
            if len(cat) > 15:
                cat = cat[:13] + '..'
            rows2.append([
                c['ticker'],
                f"+{c['gap_pct']*100:.1f}%",
                c['signal_source'].upper()[:7],
                cat,
                f"{c.get('combined_score', 0):.2f}",
            ])
        print()
        print(tabulate(rows2,
                       headers=['Ticker', 'Gap%', 'Source', 'Catalyst', 'Score'],
                       tablefmt='simple',
                       colalign=('left', 'right', 'center', 'left', 'right')))

    # Entry rules
    print(f"\n  ENTRY RULES:")
    print(f"    Local high: ThinkScript alert → enter SHORT on confirmation")
    print(f"    Fallback:   9:45 AM if no local high (skip if >15% down from open)")
    print(f"    Discovery:  Enter only after ORB confirms SHORT direction")
    print(f"    Hold:       T+2 close ({t2 or 'TBD'}) — no stop loss")
    print()

    # Per-ticker detail
    for s in selected:
        retrace = s.get('pm_retrace_pct')
        cat = s.get('catalyst_type') or '—'
        headline = s.get('headline') or ''
        if len(headline) > 70:
            headline = headline[:68] + '..'

        print(f"  #{s['selection_rank']} {s['ticker']}  ({s['signal_source'].upper()})")
        prior = s.get('prior_close')
        opn = s.get('open_price')
        prior_str = f"${prior:.2f}" if prior else 'n/a'
        opn_str = f"${opn:.2f}" if opn else 'n/a'
        print(f"     Gap: +{s['gap_pct']*100:.1f}%  |  Prior: {prior_str}  |  Open: {opn_str}")
        if retrace is not None:
            if retrace < 0.2:
                note = "held (bullish PM, caution)"
            elif retrace <= 0.5:
                note = "sweet spot"
            elif retrace <= 0.8:
                note = "moderate retrace"
            else:
                note = "full retrace (excluded)"
            print(f"     PM retrace: {retrace:.2f} — {note}")
        if cat and cat != '—':
            print(f"     Catalyst: {cat}")
        if headline:
            print(f"     Headline: {headline}")
        fs = s.get('float_shares')
        if fs:
            fs_str = f"{fs/1e9:.1f}B" if fs >= 1e9 else f"{fs/1e6:.0f}M"
            print(f"     Float: {fs_str}")
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def run_report(weeks=REPORT_WEEKS):
    conn = _get_conn()
    cutoff = (datetime.now() - timedelta(weeks=weeks)).strftime('%Y-%m-%d')

    # Run stats
    runs = conn.execute("""
        SELECT COUNT(*), AVG(n_selected), AVG(runtime_seconds),
               AVG(n_both_sources), AVG(n_source_a_only), AVG(n_source_b_only)
        FROM daily_runs WHERE run_date >= ?
    """, (cutoff,)).fetchone()

    n_runs = runs[0] or 0
    avg_selected = runs[1] or 0
    avg_runtime = runs[2] or 0

    # Candidate stats
    cand_stats = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN signal_source='gap_only' THEN 1 ELSE 0 END),
            SUM(CASE WHEN signal_source='catalyst_only' THEN 1 ELSE 0 END),
            SUM(CASE WHEN signal_source='both' THEN 1 ELSE 0 END)
        FROM candidates WHERE run_date >= ?
    """, (cutoff,)).fetchone()

    # Trade performance
    trades = conn.execute("""
        SELECT ticker, run_date, signal_source, gap_tier, catalyst_tier,
               subcategory, entry_type, pnl_eod, pnl_t1, pnl_t2
        FROM trades WHERE run_date >= ?
        ORDER BY run_date
    """, (cutoff,)).fetchall()

    conn.close()

    W = 66
    print(f"\n  {'='*W}")
    print(f"  |{'MORNING SCANNER — PERFORMANCE REPORT':^{W-2}}|")
    print(f"  {'='*W}")
    print(f"\n  Period: Last {weeks} weeks (cutoff: {cutoff})")
    print(f"  Total runs: {n_runs}  |  Avg runtime: {avg_runtime:.1f}s")

    if cand_stats and cand_stats[0]:
        total = cand_stats[0]
        avg_per_day = total / max(n_runs, 1)
        print(f"\n  === SIGNAL VOLUME ===")
        print(f"  Total candidates logged:    {total} ({avg_per_day:.0f}/day avg)")
        print(f"    Gap-only:                 {cand_stats[1] or 0}")
        print(f"    Catalyst-only:            {cand_stats[2] or 0}")
        print(f"    Both sources:             {cand_stats[3] or 0}")

    pnl_t2 = [t[9] for t in trades if t[9] is not None]
    if pnl_t2:
        arr = np.array(pnl_t2)
        n = len(arr)
        avg = float(np.mean(arr))
        med = float(np.median(arr))
        std = float(np.std(arr, ddof=1)) if n > 1 else 0
        se = std / np.sqrt(n) if n > 0 else 0
        wr = float((arr > 0).mean())
        ci_lo = avg - 1.96 * se
        ci_hi = avg + 1.96 * se

        best_idx = int(np.argmax(arr))
        worst_idx = int(np.argmin(arr))
        t2_trades = [t for t in trades if t[9] is not None]
        best = t2_trades[best_idx]
        worst = t2_trades[worst_idx]

        print(f"\n  === TRADE PERFORMANCE ===")
        print(f"  Total trades with T+2:      {n}")
        print(f"  Avg PnL T+2:               {avg:+.2f}%")
        print(f"  Median PnL T+2:            {med:+.2f}%")
        print(f"  Win Rate:                   {wr*100:.0f}%")
        print(f"  CI 95%:                     [{ci_lo:+.2f}%, {ci_hi:+.2f}%]")
        print(f"  Best trade:                 {best[0]} {best[9]:+.1f}% ({best[1]})")
        print(f"  Worst trade:                {worst[0]} {worst[9]:+.1f}% ({worst[1]})")

        # By signal source
        print(f"\n  === BY SIGNAL SOURCE ===")
        for src in ['both', 'gap_only', 'catalyst_only']:
            src_pnl = [t[9] for t in trades if t[9] is not None and t[2] == src]
            if src_pnl:
                a = np.array(src_pnl)
                print(f"  {src:16s}  n={len(a):<4d}  avg={np.mean(a):+.2f}%  "
                      f"WR={float((a>0).mean())*100:.0f}%")

        # By entry type
        print(f"\n  === BY ENTRY TYPE ===")
        for etype in ['local_high', 'fallback', 'skipped']:
            ep = [t[9] for t in trades if t[9] is not None and t[6] == etype]
            if ep:
                a = np.array(ep)
                print(f"  {etype:16s}  n={len(a):<4d}  avg={np.mean(a):+.2f}%")
    else:
        print(f"\n  No T+2 outcomes available yet (trades).")

    # Candidate-level backtesting (uses open_price as SHORT entry proxy)
    conn2 = _get_conn()
    bt_rows = conn2.execute("""
        SELECT ticker, run_date, signal_source, gap_tier, catalyst_tier,
               subcategory, combined_score, gap_pct, open_price,
               pnl_eod, pnl_t1, pnl_t2, selected
        FROM candidates
        WHERE run_date >= ? AND passed_all = 1 AND pnl_t2 IS NOT NULL
        ORDER BY run_date, combined_score DESC
    """, (cutoff,)).fetchall()

    if bt_rows:
        bt_pnl = np.array([r[11] for r in bt_rows])
        sel_pnl = np.array([r[11] for r in bt_rows if r[12] == 1])
        unsel_pnl = np.array([r[11] for r in bt_rows if r[12] == 0])

        def _stats(arr, label):
            n = len(arr)
            if n == 0:
                return
            avg = float(np.mean(arr))
            med = float(np.median(arr))
            wr = float((arr > 0).mean()) * 100
            std = float(np.std(arr, ddof=1)) if n > 1 else 0
            se = std / np.sqrt(n) if n > 0 else 0
            trimmed = np.sort(arr)[max(1, n//10): n - max(1, n//10)]
            t_avg = float(np.mean(trimmed)) if len(trimmed) > 0 else avg
            print(f"  {label:24s} n={n:<4d}  avg={avg:+.2f}%  med={med:+.2f}%  "
                  f"trim={t_avg:+.2f}%  WR={wr:.0f}%  CI=[{avg-1.96*se:+.2f},{avg+1.96*se:+.2f}]")

        print(f"\n  === CANDIDATE BACKTESTING (open→T+2 short PnL) ===")
        _stats(bt_pnl, "All qualified")
        if len(sel_pnl) > 0:
            _stats(sel_pnl, "Selected (top N)")
        if len(unsel_pnl) > 0:
            _stats(unsel_pnl, "Also-qualified")

        # By signal source
        print(f"\n  --- By signal source ---")
        for src in ['both', 'gap_only', 'catalyst_only']:
            src_arr = np.array([r[11] for r in bt_rows if r[2] == src])
            if len(src_arr) > 0:
                _stats(src_arr, src)

        # By gap tier
        print(f"\n  --- By gap tier ---")
        for tier in ['T5+', 'T5', 'T4', 'T3', 'none']:
            tier_arr = np.array([r[11] for r in bt_rows if r[3] == tier])
            if len(tier_arr) > 0:
                _stats(tier_arr, tier)

        # Top 10 best/worst
        sorted_bt = sorted(bt_rows, key=lambda r: r[11])
        print(f"\n  --- Top 5 best shorts ---")
        for r in sorted_bt[-5:][::-1]:
            print(f"    {r[0]:8s} {r[1]}  gap={r[7]*100:+.1f}%  T+2={r[11]:+.2f}%  "
                  f"src={r[2]}  score={r[6]:.1f}")
        print(f"\n  --- Top 5 worst shorts ---")
        for r in sorted_bt[:5]:
            print(f"    {r[0]:8s} {r[1]}  gap={r[7]*100:+.1f}%  T+2={r[11]:+.2f}%  "
                  f"src={r[2]}  score={r[6]:.1f}")

    # Open positions
    open_trades = conn2.execute("""
        SELECT ticker, run_date, entry_price, t2_date
        FROM trades
        WHERE t2_close IS NULL AND entry_price IS NOT NULL
        ORDER BY run_date
    """).fetchall()
    conn2.close()

    if open_trades:
        print(f"\n  === OPEN POSITIONS ===")
        rows = []
        for t in open_trades:
            rows.append([t[0], t[1], f"${t[2]:.2f}", t[3] or '?'])
        print(tabulate(rows, headers=['Ticker', 'Entry', 'Price', 'T+2 Date'],
                       tablefmt='simple'))

    # Trade log
    if trades:
        print(f"\n  === TRADE LOG ===")
        rows = []
        for t in trades:
            rows.append([
                t[1], t[0], t[2] or '?',
                t[3] or '?', t[5] or '—',
                f"{t[7]:+.2f}%" if t[7] is not None else '—',
                f"{t[9]:+.2f}%" if t[9] is not None else '—',
            ])
        print()
        print(tabulate(rows,
                       headers=['Date', 'Ticker', 'Source', 'Gap Tier', 'Catalyst', 'EOD', 'T+2'],
                       tablefmt='simple'))
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT ANALYSIS — weekly deep queries
# ═══════════════════════════════════════════════════════════════════════════════

def run_report_analysis(weeks=REPORT_WEEKS):
    conn = _get_conn()
    cutoff = (datetime.now() - timedelta(weeks=weeks)).strftime('%Y-%m-%d')

    print(f"\n  {'='*66}")
    print(f"  WEEKLY ANALYSIS — cutoff: {cutoff}")
    print(f"  {'='*66}")

    # 1. Performance by catalyst_type
    print(f"\n  === BY CATALYST TYPE ===")
    rows = conn.execute("""
        SELECT
            catalyst_type,
            COUNT(*) as n_logged,
            SUM(selected) as n_traded,
            ROUND(AVG(CASE WHEN selected=1 THEN pnl_t2 END), 3) as traded_avg,
            ROUND(AVG(pnl_t2), 3) as all_avg
        FROM candidates
        WHERE run_date >= ? AND pnl_t2 IS NOT NULL
        GROUP BY catalyst_type
        ORDER BY traded_avg DESC
    """, (cutoff,)).fetchall()
    if rows:
        print(tabulate(rows,
                       headers=['Catalyst', 'Logged', 'Traded', 'Traded Avg', 'All Avg'],
                       tablefmt='simple'))

    # 2. Filter failure analysis
    print(f"\n  === FILTER FAILURE ANALYSIS ===")
    rows = conn.execute("""
        SELECT
            filter_fail,
            COUNT(*) as n_excluded,
            ROUND(AVG(pnl_t2), 3) as excluded_avg_pnl,
            ROUND(AVG(CASE WHEN pnl_t2 > 0 THEN 1.0 ELSE 0 END), 3) as wr
        FROM candidates
        WHERE run_date >= ? AND pnl_t2 IS NOT NULL AND passed_all = 0
        GROUP BY filter_fail
        ORDER BY n_excluded DESC
    """, (cutoff,)).fetchall()
    if rows:
        print(tabulate(rows,
                       headers=['Filter Fail', 'N', 'Avg PnL', 'WR'],
                       tablefmt='simple'))

    # 3. Score rank validation
    print(f"\n  === SCORE RANK VALIDATION ===")
    rows = conn.execute("""
        SELECT
            rank_overall,
            COUNT(*) as n,
            ROUND(AVG(pnl_t2), 3) as avg_pnl,
            ROUND(AVG(CASE WHEN pnl_t2 > 0 THEN 1.0 ELSE 0 END), 3) as wr
        FROM candidates
        WHERE run_date >= ? AND passed_all = 1 AND pnl_t2 IS NOT NULL
        GROUP BY rank_overall
        ORDER BY rank_overall
    """, (cutoff,)).fetchall()
    if rows:
        print(tabulate(rows,
                       headers=['Rank', 'N', 'Avg PnL', 'WR'],
                       tablefmt='simple'))

    # 4. Signal source comparison
    print(f"\n  === SIGNAL SOURCE COMPARISON ===")
    rows = conn.execute("""
        SELECT
            signal_source,
            COUNT(*) as n,
            ROUND(AVG(pnl_t2), 3) as avg_pnl,
            ROUND(AVG(CASE WHEN pnl_t2 > 0 THEN 1.0 ELSE 0 END), 3) as wr
        FROM candidates
        WHERE run_date >= ? AND selected = 1 AND pnl_t2 IS NOT NULL
        GROUP BY signal_source
    """, (cutoff,)).fetchall()
    if rows:
        print(tabulate(rows,
                       headers=['Source', 'N', 'Avg PnL', 'WR'],
                       tablefmt='simple'))

    # 5. Gap-DOWN log
    print(f"\n  === GAP-DOWN LOG (future LONG analysis) ===")
    rows = conn.execute("""
        SELECT
            catalyst_type,
            COUNT(*) as n,
            ROUND(AVG(pnl_t2), 3) as avg_pnl_if_longed
        FROM candidates
        WHERE run_date >= ? AND gap_direction = 'down' AND pnl_t2 IS NOT NULL
        GROUP BY catalyst_type
        ORDER BY avg_pnl_if_longed DESC
    """, (cutoff,)).fetchall()
    if rows:
        print(tabulate(rows,
                       headers=['Catalyst', 'N', 'Avg PnL (if longed)'],
                       tablefmt='simple'))
    else:
        print("  No gap-down outcomes yet.")

    conn.close()
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SCAN
# ═══════════════════════════════════════════════════════════════════════════════

def run_scan(scan_date_str=None, dry_run=False):
    t_start = time.time()

    if scan_date_str is None:
        scan_date_str = datetime.now(ET).strftime('%Y-%m-%d')
        mode = 'live'
    else:
        mode = 'historical'

    W = 66
    log.info(f"{'='*W}")
    log.info(f"  MORNING SCANNER — {scan_date_str}  |  Mode: {mode}  |  Dry run: {dry_run}")
    log.info(f"{'='*W}")

    if not is_trading_day(scan_date_str):
        log.info(f"  {scan_date_str} is not a trading day. Exiting.")
        return

    if mode == 'historical':
        log.warning("HISTORICAL MODE: open price as PM proxy. Approximate.")

    # Step 0: Init
    init_db()
    float_cache = load_float_cache()

    # Step 1: Fill outcomes for open trades
    if not dry_run:
        log.info(f"\n  Step 0: Fill outcomes for open trades")
        fill_outcomes(fill_date_str=scan_date_str)

    # Step 2: Prior trading day
    prior_date = get_prior_trading_date(scan_date_str)
    prior_str = prior_date.strftime('%Y-%m-%d')
    log.info(f"  Prior trading day: {prior_str}")

    # Step 3: SOURCE A — Polygon gap universe
    t_poly = time.time()
    log.info(f"\n  Source A: Polygon full universe")
    prior_data = fetch_grouped_daily(prior_str)

    if mode == 'live':
        current_data = fetch_snapshot_all()
    else:
        current_data = fetch_grouped_daily(scan_date_str)

    polygon_candidates, n_polygon_total = compute_gaps(prior_data, current_data, mode)

    # SPY gap
    spy_gap = None
    spy_prev_close = None
    if 'SPY' in prior_data:
        spy_prev_close = prior_data['SPY']['close']
        spy_cur = None
        if 'SPY' in current_data:
            spy_cur = current_data['SPY'].get('open', current_data['SPY'].get('price', 0))
        if spy_prev_close and spy_cur and spy_prev_close > 0:
            spy_gap = round((spy_cur - spy_prev_close) / spy_prev_close * 100, 3)
            log.info(f"  SPY gap: {spy_gap:+.2f}%")

    n_polygon_gap = len(polygon_candidates)
    n_polygon_price = sum(1 for c in polygon_candidates if (c.get('open_price', 0) or 0) >= PRICE_MIN)

    elapsed_poly = time.time() - t_poly
    log.info(f"  Source A: {n_polygon_gap} gaps >= {GAP_MIN_SCAN*100:.1f}% in {elapsed_poly:.1f}s")

    # Step 4: SOURCE B — BigQuery LLM signals
    t_bq = time.time()
    log.info(f"\n  Source B: BigQuery LLM signals")
    bq_premarket, bq_discovery = query_bq(scan_date_str)
    n_bq_premarket = len(bq_premarket)
    n_bq_discovery = len(bq_discovery)
    bq_tickers = set(s['ticker'] for s in bq_premarket + bq_discovery)
    n_bq_total = len(bq_tickers)
    elapsed_bq = time.time() - t_bq
    log.info(f"  Source B: {n_bq_total} unique BQ signals in {elapsed_bq:.1f}s")

    # Step 5: Merge
    t_merge = time.time()
    merged, n_both, n_a_only, n_b_only = merge_sources(
        polygon_candidates, bq_premarket, bq_discovery)

    # Step 6: PM bars
    t_pm = time.time()
    tickers_for_pm = [c['ticker'] for c in merged
                      if (c.get('gap_pct') or 0) >= GAP_MIN_SCAN
                      and (c.get('open_price') or 0) >= PRICE_MIN]
    log.info(f"\n  PM bars: fetching for {len(tickers_for_pm)} tickers")
    pm_data = fetch_pm_bars_async(tickers_for_pm, scan_date_str)
    elapsed_pm = time.time() - t_pm

    # Step 7: Filter
    merged = apply_filters(merged, pm_data, float_cache)

    n_polygon_float = sum(1 for c in merged if c.get('passed_float') == 1 and c.get('in_polygon'))
    n_polygon_retrace = sum(1 for c in merged if c.get('passed_retrace') == 1 and c.get('in_polygon'))

    # Step 8: Score
    merged = score_candidates(merged)

    # Step 9: Select top N
    qualified = [c for c in merged if c['passed_all'] == 1]
    qualified.sort(key=lambda x: -(x.get('combined_score') or 0))

    selected = qualified[:TARGET_TRADES]
    for i, s in enumerate(selected):
        s['selected'] = 1
        s['selection_rank'] = i + 1

    # Step 10: T+2 dates
    t1, t2 = get_t1_t2_dates(scan_date_str)

    # Step 11: Write to DB
    runtime = time.time() - t_start
    run_meta = {
        'mode': mode,
        'runtime': runtime,
        'n_polygon_total': n_polygon_total,
        'n_polygon_gap': n_polygon_gap,
        'n_polygon_price': n_polygon_price,
        'n_polygon_float': n_polygon_float,
        'n_polygon_retrace': n_polygon_retrace,
        'n_bq_premarket': n_bq_premarket,
        'n_bq_discovery': n_bq_discovery,
        'n_bq_total': n_bq_total,
        'n_both': n_both,
        'n_a_only': n_a_only,
        'n_b_only': n_b_only,
        'spy_gap_pct': spy_gap,
        'spy_prev_close': spy_prev_close,
    }

    if not dry_run:
        write_to_db(scan_date_str, merged, selected, run_meta)
    else:
        log.info("  DRY RUN — skipping DB write")

    # Step 12: Print watchlist
    print_watchlist(selected, merged, scan_date_str)

    # Summary
    n_qualified = len(qualified)
    log.info(f"  Funnel: {len(merged)} merged → {n_qualified} qualified → {len(selected)} selected")
    log.info(f"  Source breakdown: {n_both} both | {n_a_only} gap-only | {n_b_only} catalyst-only")
    log.info(f"  PM bars: {elapsed_pm:.1f}s  |  Total runtime: {runtime:.1f}s")

    return selected


# ═══════════════════════════════════════════════════════════════════════════════
# CLI DISPATCH
# ═══════════════════════════════════════════════════════════════════════════════

def parse_cli_args():
    """Parse CLI args when running as script. Returns None when running as notebook cell."""
    try:
        get_ipython()
        return None
    except NameError:
        pass

    parser = argparse.ArgumentParser(description='Morning Scanner — Agent 2')
    parser.add_argument('command', nargs='?', default='scan',
                        choices=['scan', 'fill', 'report', 'analysis'],
                        help='Command to run')
    parser.add_argument('--date', type=str, default=None,
                        help='Scan date (YYYY-MM-DD), default today')
    parser.add_argument('--dry-run', action='store_true',
                        help='Skip DB writes')
    parser.add_argument('--weeks', type=int, default=REPORT_WEEKS,
                        help='Report lookback weeks')
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

args = parse_cli_args()

if args is not None:
    # CLI mode
    if args.command == 'scan':
        init_db()
        run_scan(scan_date_str=args.date, dry_run=args.dry_run)
    elif args.command == 'fill':
        init_db()
        fill_outcomes(fill_date_str=args.date)
    elif args.command == 'report':
        init_db()
        run_report(weeks=args.weeks)
    elif args.command == 'analysis':
        init_db()
        run_report_analysis(weeks=args.weeks)
else:
    # Notebook mode — use config vars at top
    init_db()
    if MODE == 'scan':
        run_scan(scan_date_str=SCAN_DATE, dry_run=DRY_RUN)
    elif MODE == 'fill':
        fill_outcomes(fill_date_str=SCAN_DATE)
    elif MODE == 'report':
        run_report(weeks=REPORT_WEEKS)
    elif MODE == 'report_analysis':
        run_report_analysis(weeks=REPORT_WEEKS)
    else:
        print(f"Unknown MODE: {MODE}. Use 'scan', 'fill', 'report', or 'report_analysis'.")
