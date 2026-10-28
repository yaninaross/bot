# %% Agent 1_2 — Project X Live SHORT Scanner (H006 + DEEP4 Proximity)
#
# Strategy: SHORT gap-up stocks showing sustained_drop pattern,
#           entry at local_high within 3% of running intraday high,
#           exit at 17:00 ET (2:00 PM PT). Same-day intraday.
#
# Backtest stats (Aug-Oct 2026):
#   avg +2.25%  |  Sharpe 0.749  |  WR 93%  |  PF 14.14  |  n=28
#   Full period (Sep 2025 - Oct 2026): ALL 14 months positive
#
# Usage:
#   MODE = 'scan'                       -> morning scanner (default)
#   MODE = 'fill'                       -> fill outcomes after 2 PM PT
#   MODE = 'report'                     -> performance summary
#   SCAN_TIMESTAMP = '2026-10-08 06:45' -> time machine (PT)
#   SCAN_TIMESTAMP = None               -> live (uses current time)
#   DRY_RUN = True                      -> skip DB writes
#
# Two-phase scan:
#   Phase A (6:31 PT / 9:31 ET): Scan gap-ups, rank by gap %, build pool
#   Phase B (7:30 PT / 10:30 ET): Classify sustained_drop (bar 60),
#           find local_high entries with proximity check, fire signals
#
# Colab-cell friendly. Config vars at top, no if __name__.

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


# =============================================================================
# CONFIG
# =============================================================================

MODE = 'scan'
SCAN_TIMESTAMP = None
DRY_RUN = False
REPORT_WEEKS = 4

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs")
POLYGON_BASE = "https://api.polygon.io"

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

DB_PATH = os.path.join(BASE_DIR, 'data', 'agent1_2.db')
LOG_PATH = os.path.join(BASE_DIR, 'logs', 'agent1_2.log')

GAP_MIN = 0.02
PRICE_MIN = 5.0
FLOAT_MIN_SHARES = 5_000_000
TOP_N = 5
PRE_FILTER_N = 40
POSITION_SIZE = 100_000
EXIT_TIME_ET = '17:00'

# Entry detection
PEAK_WINDOW = 3
MIN_ENTRY_BAR = 3
MAX_PEAK_BAR = 72
FALLBACK_BAR = 75
PROX_MAX = 0.03

# Phase timing (PT)
PHASE_A_READY_PT = (6, 31)
PHASE_B_READY_PT = (7, 30)

BAR_WORKERS = 10
TICKER_REGEX = re.compile(r'^[A-Z]{1,5}$')
_EXCLUDED_SUFFIXES = ('W', 'WS', 'U', 'R')
_EXCLUDED_TICKERS = {'SPY', 'QQQ', 'IWM', 'DIA', 'VXX', 'UVXY', 'SQQQ', 'TQQQ',
                     'SPXU', 'SPXL', 'SDOW', 'UDOW', 'SDS', 'SSO', 'SH',
                     'LABU', 'LABD', 'SOXL', 'SOXS', 'FNGU', 'FNGD',
                     'NUGT', 'DUST', 'JNUG', 'JDST', 'TNA', 'TZA',
                     'UPRO', 'VIXY', 'SVXY', 'TVIX'}
_NON_CS_TYPES = {'ETF', 'ETV', 'ETN', 'ETP', 'ETS', 'ADRC', 'FUND', 'PFD'}

ET = pytz.timezone('America/New_York')
PT = pytz.timezone('America/Los_Angeles')

_NYSE_HOLIDAYS = {
    '2026-01-01', '2026-01-19', '2026-02-16', '2026-04-03', '2026-05-25',
    '2026-06-19', '2026-07-03', '2026-09-07', '2026-11-26', '2026-12-25',
    '2027-01-01', '2027-01-18', '2027-02-15', '2027-03-26', '2027-05-31',
    '2027-06-18', '2027-07-05', '2027-09-06', '2027-11-25', '2027-12-24',
    '2028-01-17', '2028-02-21', '2028-04-14', '2028-05-29',
    '2028-06-19', '2028-07-04', '2028-09-04', '2028-11-23', '2028-12-25',
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
_FLOAT_SET = {tk for tk, fs in _FLOAT.items() if fs >= FLOAT_MIN_SHARES}


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    logger = logging.getLogger('agent1_2')
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fh = logging.FileHandler(LOG_PATH)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s  %(levelname)-7s  %(message)s'))
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger

log = setup_logging()


# =============================================================================
# DATABASE
# =============================================================================

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scan_date TEXT NOT NULL,
        ticker TEXT NOT NULL,
        selection_rank INTEGER,
        gap_pct REAL,
        open_price REAL,
        prior_close REAL,
        entry_price REAL,
        entry_time_et TEXT,
        drop_from_open_pct REAL,
        proximity_pct REAL,
        pattern TEXT,
        position_size REAL,
        shares INTEGER,
        exit_price REAL,
        exit_time_et TEXT DEFAULT '17:00',
        pnl_pct REAL,
        pnl_dollar REAL,
        status TEXT DEFAULT 'OPEN',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.execute('''CREATE TABLE IF NOT EXISTS scan_pool (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scan_date TEXT NOT NULL,
        ticker TEXT NOT NULL,
        gap_pct REAL,
        open_price REAL,
        prior_close REAL,
        float_shares INTEGER,
        passed_float INTEGER,
        pattern TEXT,
        signal_fired INTEGER DEFAULT 0,
        reject_reason TEXT
    )''')
    conn.commit()
    conn.close()


def _get_conn():
    return sqlite3.connect(DB_PATH)


# =============================================================================
# CALENDAR
# =============================================================================

def is_trading_day(d):
    if isinstance(d, str):
        d = datetime.strptime(d, '%Y-%m-%d').date()
    if d.weekday() >= 5:
        return False
    return d.strftime('%Y-%m-%d') not in _NYSE_HOLIDAYS

def get_prior_trading_date(target_date):
    if isinstance(target_date, str):
        target_date = datetime.strptime(target_date, '%Y-%m-%d').date()
    d = target_date - timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d


# =============================================================================
# POLYGON API
# =============================================================================

def _polygon_get(url, params=None):
    if params is None:
        params = {}
    params['apiKey'] = POLYGON_API_KEY
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 429:
                time.sleep(2 ** (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == 2:
                log.error(f"API error: {url} — {e}")
                return {}
            time.sleep(1)
    return {}

def fetch_grouped_daily(date_str):
    url = f"{POLYGON_BASE}/v2/aggs/grouped/locale/us/market/stocks/{date_str}"
    data = _polygon_get(url, {"adjusted": "true", "include_otc": "false"})
    out = {}
    for r in data.get('results', []):
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
    log.info(f"  Grouped daily ({date_str}): {len(out)} tickers")
    return out

def fetch_snapshot_all():
    url = f"{POLYGON_BASE}/v2/snapshot/locale/us/markets/stocks/tickers"
    data = _polygon_get(url, {"include_otc": "false"})
    out = {}
    for t in data.get('tickers', []):
        tk = t.get('ticker', '')
        if not TICKER_REGEX.match(tk):
            continue
        prev_c = t.get('prevDay', {}).get('c')
        day_o = t.get('day', {}).get('o')
        if prev_c and day_o and float(day_o) > 0:
            out[tk] = {
                'open': float(day_o),
                'prev_close': float(prev_c),
                'price': float(t.get('day', {}).get('c', day_o)),
            }
    log.info(f"  Snapshot: {len(out)} tickers")
    return out


# =============================================================================
# ASYNC MINUTE BAR FETCHER
# =============================================================================

async def _fetch_bars_one(session, ticker, date_str, semaphore):
    url = f"{POLYGON_BASE}/v2/aggs/ticker/{ticker}/range/1/minute/{date_str}/{date_str}"
    params = {"adjusted": "true", "sort": "asc", "limit": "1000", "apiKey": POLYGON_API_KEY}
    async with semaphore:
        for attempt in range(3):
            try:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
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

            pm_bars, rth_bars, ah_bars = [], [], []
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

            return ticker, {'pm': pm_bars, 'rth': rth_bars, 'ah': ah_bars}
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
    n_ok = sum(1 for v in bar_data.values() if v.get('rth'))
    log.info(f"  Minute bars: {n_ok}/{len(tickers)} in {time.time()-t0:.1f}s")
    return bar_data


# =============================================================================
# ASSET TYPE CHECK (CS only)
# =============================================================================

def _is_common_stock(ticker):
    url = f"{POLYGON_BASE}/v3/reference/tickers/{ticker}"
    data = _polygon_get(url)
    results = data.get('results', {})
    if not results:
        return False
    asset_type = results.get('type', '')
    return asset_type not in _NON_CS_TYPES and asset_type != ''

_cs_cache = {}

def check_common_stock(ticker):
    if ticker in _cs_cache:
        return _cs_cache[ticker]
    result = _is_common_stock(ticker)
    _cs_cache[ticker] = result
    return result


# =============================================================================
# GAP SCAN — Phase A
# =============================================================================

def compute_gaps(prior_data, current_data, mode='historical'):
    candidates = []
    for tk in current_data:
        if tk in _EXCLUDED_TICKERS:
            continue
        if any(tk.endswith(s) for s in _EXCLUDED_SUFFIXES) and len(tk) > 2:
            continue
        if tk not in _FLOAT_SET:
            continue

        if tk not in prior_data:
            continue
        prev_c = prior_data[tk].get('close', 0) if isinstance(prior_data[tk], dict) else prior_data[tk]
        if prev_c <= 0:
            continue

        cur_open = current_data[tk].get('open', 0)
        if cur_open <= 0 or cur_open < PRICE_MIN:
            continue

        gap_pct = (cur_open - prev_c) / prev_c
        if gap_pct < GAP_MIN:
            continue
        if gap_pct > 0.75:
            continue

        candidates.append({
            'ticker': tk,
            'gap_pct': round(gap_pct, 6),
            'prior_close': round(prev_c, 4),
            'open_price': round(cur_open, 4),
            'float_shares': _FLOAT.get(tk),
        })

    candidates.sort(key=lambda x: -x['gap_pct'])
    log.info(f"  Gap scan: {len(candidates)} with gap >= {GAP_MIN*100:.0f}%, price >= ${PRICE_MIN}")
    return candidates


# =============================================================================
# PATTERN CLASSIFICATION — sustained_drop at bar 60
# =============================================================================

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


# =============================================================================
# LOCAL HIGH DETECTION + PROXIMITY FILTER
# =============================================================================

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


def get_exit_price(bar_data, exit_time='17:00'):
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


# =============================================================================
# PERSIST SCAN POOL
# =============================================================================

def persist_scan_pool(scan_date_str, candidates, bar_cache, signals):
    conn = _get_conn()
    fired_tickers = {s['ticker'] for s in signals}
    for c in candidates:
        tk = c['ticker']
        pattern = c.get('pattern', '')
        signal_fired = 1 if tk in fired_tickers else 0
        reject = c.get('reject_reason', '')
        conn.execute(
            'INSERT INTO scan_pool (scan_date, ticker, gap_pct, open_price, prior_close, '
            'float_shares, passed_float, pattern, signal_fired, reject_reason) '
            'VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?)',
            (scan_date_str, tk, c['gap_pct'], c['open_price'], c['prior_close'],
             c.get('float_shares'), pattern, signal_fired, reject)
        )
    conn.commit()
    conn.close()


def persist_signals(scan_date_str, signals):
    conn = _get_conn()
    for s in signals:
        shares = int(POSITION_SIZE / s['entry_price']) if s['entry_price'] > 0 else 0
        conn.execute(
            'INSERT INTO signals (scan_date, ticker, selection_rank, gap_pct, open_price, '
            'prior_close, entry_price, entry_time_et, drop_from_open_pct, proximity_pct, '
            'pattern, position_size, shares, status) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (scan_date_str, s['ticker'], s['rank'], s['gap_pct'], s['open_price'],
             s['prior_close'], s['entry_price'], s['entry_time'],
             s.get('drop_from_open', 0) * 100, s.get('proximity', 0) * 100,
             s.get('pattern', 'sustained_drop'), POSITION_SIZE, shares, 'OPEN')
        )
    conn.commit()
    conn.close()


# =============================================================================
# FILL OUTCOMES
# =============================================================================

def run_fill_outcomes(fill_date_str=None):
    init_db()
    conn = _get_conn()
    open_trades = conn.execute(
        "SELECT id, scan_date, ticker, entry_price FROM signals WHERE status='OPEN'"
    ).fetchall()

    if not open_trades:
        log.info("  No open trades to fill.")
        conn.close()
        return

    log.info(f"  Filling outcomes for {len(open_trades)} open trades...")

    for trade_id, scan_date, ticker, entry_price in open_trades:
        bar_data = fetch_minute_bars([ticker], scan_date)
        if ticker not in bar_data:
            log.warning(f"    {ticker}: no bar data for {scan_date}")
            continue

        exit_price = get_exit_price(bar_data[ticker])
        if exit_price is None:
            log.warning(f"    {ticker}: no exit price")
            continue

        pnl_pct = round((entry_price - exit_price) / entry_price * 100, 3)
        pnl_dollar = round(POSITION_SIZE * pnl_pct / 100, 2)

        conn.execute(
            "UPDATE signals SET exit_price=?, pnl_pct=?, pnl_dollar=?, status='FILLED' WHERE id=?",
            (round(exit_price, 4), pnl_pct, pnl_dollar, trade_id)
        )
        log.info(f"    {ticker}: entry ${entry_price:.2f} -> exit ${exit_price:.2f} = {pnl_pct:+.2f}% (${pnl_dollar:+.0f})")

    conn.commit()
    conn.close()


# =============================================================================
# REPORT
# =============================================================================

def run_report(weeks=REPORT_WEEKS):
    init_db()
    conn = _get_conn()
    cutoff = (datetime.now() - timedelta(weeks=weeks)).strftime('%Y-%m-%d')
    rows = conn.execute(
        "SELECT scan_date, ticker, gap_pct, entry_price, exit_price, pnl_pct, pnl_dollar, "
        "proximity_pct, drop_from_open_pct, status "
        "FROM signals WHERE scan_date >= ? ORDER BY scan_date, selection_rank",
        (cutoff,)
    ).fetchall()
    conn.close()

    if not rows:
        print(f"\n  No trades in last {weeks} weeks.")
        return

    filled = [r for r in rows if r[9] == 'FILLED']
    open_trades = [r for r in rows if r[9] == 'OPEN']

    print(f"\n{'='*70}")
    print(f"  AGENT 1_2 — Performance Report (last {weeks} weeks)")
    print(f"{'='*70}")

    if filled:
        pnls = [r[5] for r in filled]
        dollars = [r[6] for r in filled]
        avg = np.mean(pnls)
        std = np.std(pnls, ddof=1) if len(pnls) > 1 else 0
        sharpe = avg / std if std > 0 else 0
        wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100

        print(f"\n  Filled trades: {len(filled)}")
        print(f"  Avg PnL:       {avg:+.2f}%")
        print(f"  Sharpe:        {sharpe:.3f}")
        print(f"  Win rate:      {wr:.0f}%")
        print(f"  Total P&L:     ${sum(dollars):+,.0f}")

        print(f"\n  {'Date':<12s}  {'Ticker':<6s}  {'Gap':>6s}  {'Entry':>8s}  {'Exit':>8s}  {'PnL':>8s}  {'$PnL':>8s}  {'Prox':>5s}")
        print(f"  {'-'*12}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*5}")
        for r in filled:
            print(f"  {r[0]:<12s}  {r[1]:<6s}  {r[2]*100:>+5.1f}%  ${r[3]:>7.2f}  ${r[4]:>7.2f}  {r[5]:>+7.2f}%  ${r[6]:>+7.0f}  {r[7]:>4.1f}%")

    if open_trades:
        print(f"\n  Open trades: {len(open_trades)}")
        for r in open_trades:
            print(f"    {r[0]} {r[1]} — entry ${r[3]:.2f}, gap +{r[2]*100:.1f}%")

    print()


# =============================================================================
# PRINT SIGNAL OUTPUT — matches user's desired format exactly
# =============================================================================

def print_signals(signals, watched, scan_date_str):
    now_pt = datetime.now(PT)
    time_str = now_pt.strftime('%H:%M')
    date_str_display = scan_date_str

    n = len(signals)

    print(f"\n{'='*60}")
    print(f"SHORT AGENT — {date_str_display} — {time_str} PT")
    print(f"{'='*60}")

    if n > 0:
        stars = '*' * 51
        print(f"\n{stars}")
        print(f"  {n} SIGNAL(S) — SHORT NOW")
        print(f"{stars}")

        for i, s in enumerate(signals, 1):
            shares = int(POSITION_SIZE / s['entry_price']) if s['entry_price'] > 0 else 0
            print(f"\n  #{i}: SHORT {s['ticker']}")
            print(f"  Entry:    ${s['entry_price']:.2f} (signaled at {_et_to_pt(s['entry_time'])} PT)")
            print(f"  Gap:      +{s['gap_pct']*100:.1f}% from ${s['prior_close']:.2f}")
            print(f"  Drop:     -{abs(s['drop_from_open'])*100:.1f}% from open")
            print(f"  Prox:     {s['proximity']*100:.1f}% below running high")
            print(f"  Size:     ${POSITION_SIZE:,.0f} (~{shares:,d} shares)")
            print(f"  EXIT:     2:00 PM PT (17:00 ET)")
    else:
        print(f"\n  NO SIGNALS — no qualifying trades today")

    if watched:
        print(f"\n  Candidates watched, no signal:")
        for w in watched:
            reason = w.get('reject_reason', '')
            print(f"  {w['ticker']:<6s} gap +{w['gap_pct']*100:.1f}% | {reason}")

    print(f"\n{'='*60}")
    print(f"  REMINDER: Cover ALL shorts by 2:00 PM PT")
    print(f"{'='*60}\n")


def _et_to_pt(et_time_str):
    parts = et_time_str.split(':')
    h, m = int(parts[0]), int(parts[1])
    pt_h = h - 3
    return f"{pt_h:02d}:{m:02d}"


# =============================================================================
# MAIN SCAN — Phase A + Phase B
# =============================================================================

def run_scan(scan_date_str=None, dry_run=False):
    t_start = time.time()
    init_db()

    if scan_date_str is None:
        scan_date_str = datetime.now(ET).strftime('%Y-%m-%d')
        mode = 'live'
    else:
        mode = 'historical'

    log.info(f"\n{'='*70}")
    log.info(f"  AGENT 1_2 — Project X SHORT Scanner (H006 + DEEP4)")
    log.info(f"  Date: {scan_date_str}  |  Mode: {mode}  |  Dry run: {dry_run}")
    log.info(f"  Strategy: SHORT gap-up + sustained_drop + local_high (prox<=3%)")
    log.info(f"  Exit: {EXIT_TIME_ET} ET (2:00 PM PT) — intraday only")
    log.info(f"{'='*70}")

    if not is_trading_day(scan_date_str):
        log.info(f"  {scan_date_str} is not a trading day. Exiting.")
        return

    # Fill any open trades first
    if not dry_run:
        run_fill_outcomes()

    # ── Phase A: Gap scan ──
    prior_date = get_prior_trading_date(scan_date_str)
    prior_str = prior_date.strftime('%Y-%m-%d')
    log.info(f"\n  Phase A: Gap scan")
    log.info(f"  Prior trading day: {prior_str}")

    t1 = time.time()
    prior_data = fetch_grouped_daily(prior_str)
    log.info(f"  [1] Prior closes: {time.time()-t1:.1f}s")

    t2 = time.time()
    if mode == 'live':
        current_data = fetch_snapshot_all()
    else:
        current_data = fetch_grouped_daily(scan_date_str)
    log.info(f"  [2] Current prices: {time.time()-t2:.1f}s")

    candidates = compute_gaps(prior_data, current_data, mode)
    if not candidates:
        log.info("  No gap-up candidates found.")
        print_signals([], [], scan_date_str)
        return

    # CS verification for top candidates
    top_candidates = candidates[:PRE_FILTER_N]
    log.info(f"  Verifying CS type for top {len(top_candidates)} candidates...")
    verified = []
    for c in top_candidates:
        if check_common_stock(c['ticker']):
            verified.append(c)
        else:
            log.debug(f"    {c['ticker']}: excluded (not CS)")
    log.info(f"  CS verified: {len(verified)}/{len(top_candidates)}")
    candidates = verified

    if not candidates:
        log.info("  No CS candidates after verification.")
        print_signals([], [], scan_date_str)
        return

    # ── Phase B: Fetch minute bars, classify patterns, find entries ──
    log.info(f"\n  Phase B: Pattern classification + entry detection")

    fetch_tickers = [c['ticker'] for c in candidates]
    t3 = time.time()
    bar_cache = fetch_minute_bars(fetch_tickers, scan_date_str)
    log.info(f"  [3] Minute bars: {time.time()-t3:.1f}s")

    signals = []
    watched = []

    for c in candidates:
        tk = c['ticker']
        if tk not in bar_cache:
            c['reject_reason'] = 'no minute bars'
            continue

        rth = bar_cache[tk].get('rth', [])
        if len(rth) < 60:
            c['reject_reason'] = f'only {len(rth)} RTH bars (need 60)'
            continue

        pattern = classify_open_pattern(rth)
        c['pattern'] = pattern

        if pattern != 'sustained_drop':
            if pattern == 'drop_then_recover':
                recovery_at_60 = 0
                open_price = rth[0][1]['o']
                low_30m = min(b[1]['l'] for b in rth[:30])
                close_60m = rth[59][1]['c']
                denom = abs(open_price - low_30m)
                if denom > 0:
                    recovery_at_60 = (close_60m - low_30m) / denom
                c['reject_reason'] = f'recovery {recovery_at_60*100:.0f}%'
            else:
                c['reject_reason'] = pattern
            watched.append(c)
            continue

        entry = find_local_high_with_proximity(rth)
        if entry is None:
            c['reject_reason'] = 'no clean local high yet'
            watched.append(c)
            continue

        signals.append({
            'ticker': tk,
            'gap_pct': c['gap_pct'],
            'prior_close': c['prior_close'],
            'open_price': c['open_price'],
            'float_shares': c.get('float_shares'),
            'entry_price': entry['entry_price'],
            'entry_time': entry['entry_time'],
            'entry_type': entry['entry_type'],
            'proximity': entry['proximity'],
            'drop_from_open': entry['drop_from_open'],
            'pattern': pattern,
            'rank': 0,
        })

    signals.sort(key=lambda x: -x['gap_pct'])
    for i, s in enumerate(signals):
        s['rank'] = i + 1
    signals = signals[:TOP_N]

    # Sort watched: sustained_drop (closest to signal) first, then others by gap
    watched_sd = [w for w in watched if w.get('pattern') == 'sustained_drop']
    watched_other = [w for w in watched if w.get('pattern') != 'sustained_drop']
    watched_sorted = watched_sd + watched_other

    # Print output
    print_signals(signals, watched_sorted[:5], scan_date_str)

    # Persist
    if not dry_run and signals:
        persist_signals(scan_date_str, signals)
        persist_scan_pool(scan_date_str, candidates, bar_cache, signals)
        log.info(f"  Persisted {len(signals)} signals to DB")

    # Log summary
    n_sd = sum(1 for c in candidates if c.get('pattern') == 'sustained_drop')
    log.info(f"\n  Funnel: {len(candidates)} gap-ups -> {n_sd} sustained_drop -> {len(signals)} signals")
    log.info(f"  Runtime: {time.time()-t_start:.1f}s")

    return signals


# =============================================================================
# PARSE TIMESTAMP
# =============================================================================

def _parse_scan_timestamp(ts):
    if ts is None:
        return None
    try:
        dt = datetime.strptime(ts.strip(), '%Y-%m-%d %H:%M')
        return PT.localize(dt).strftime('%Y-%m-%d')
    except ValueError:
        return ts.strip()


# =============================================================================
# MAIN DISPATCH
# =============================================================================

log.info(f"\n{'='*70}")
log.info(f"  Agent 1_2 starting — MODE={MODE}")
log.info(f"{'='*70}")

if MODE == 'scan':
    scan_date = _parse_scan_timestamp(SCAN_TIMESTAMP)
    run_scan(scan_date_str=scan_date, dry_run=DRY_RUN)
elif MODE == 'fill':
    run_fill_outcomes()
elif MODE == 'report':
    run_report(weeks=REPORT_WEEKS)
else:
    log.error(f"  Unknown MODE: {MODE}")
