# %% Agent 1 — Advanced Experiments E1-E10
# Hold period sweep, ETF filter, liquidity, asset type, price bands,
# short interest, intraday patterns, gap freshness, sector concentration,
# combined signal score.
#
# Uses existing hardcoded _FLOAT universe as baseline.
# Single data-fetch pass, then 10 experiments run on cached data.
# Run in Colab cell. ~30-40 min for full backtest (API-bound).

import os, sys, time, re, asyncio, copy, json
from datetime import datetime, timedelta
from collections import defaultdict, Counter, OrderedDict

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
import matplotlib.colors as mcolors

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    import subprocess; subprocess.check_call([sys.executable, "-m", "pip", "install", "google-genai", "--quiet", "--break-system-packages"])
    from google import genai
    from google.genai import types as genai_types

try:
    from google.colab import auth
    try: auth.authenticate_user()
    except: pass
except: pass


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

GCP_PROJECT  = "yanina-everything"
GCP_LOCATION = "us-central1"
GEMINI_MODEL = "gemini-2.5-flash"
SECTOR_BATCH_SIZE = 30

PRE_FILTER_N = 15

BACKTEST_START = '2026-01-02'
BACKTEST_END   = '2026-09-12'

GAP_MIN_PCT = 5.0
PRICE_MIN = 10.0
FLOAT_MIN = 10_000_000
TOP_N = 5

PEAK_WINDOW = 3
MAX_PEAK_BAR = 12
FALLBACK_BAR = 15
INITIAL_CAPITAL = 100_000

ADV_LOOKBACK = 20

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "YOUR_POLYGON_API_KEY")
_BASE = "https://api.polygon.io"
_TK_RE = re.compile(r'^[A-Z]{1,5}$')
ET = pytz.timezone('America/New_York')

_EXCLUDE_TK = {'SPY','QQQ','IWM','DIA','VXX','UVXY','SQQQ','TQQQ','SPXU','SPXL',
               'SDOW','UDOW','SDS','SSO','SH','LABU','LABD','SOXL','SOXS',
               'FNGU','FNGD','NUGT','DUST','JNUG','JDST','TNA','TZA',
               'UPRO','VIXY','SVXY','TVIX'}
_EXCLUDE_SUFFIX = ('W', 'WS', 'U', 'R')

_HOLIDAYS = {
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

def _next_n_td(d, n=5):
    if isinstance(d, str): d = datetime.strptime(d, '%Y-%m-%d').date()
    out, d = [], d + timedelta(days=1)
    for _ in range(30):
        if _is_td(d):
            out.append(d)
            if len(out) >= n: return out
        d += timedelta(days=1)
    return out

def _prior_n_td(d, n=20):
    if isinstance(d, str): d = datetime.strptime(d, '%Y-%m-%d').date()
    out = []
    d = d - timedelta(days=1)
    for _ in range(60):
        if _is_td(d):
            out.append(d)
            if len(out) >= n: return list(reversed(out))
        d -= timedelta(days=1)
    return list(reversed(out))

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
# ASYNC TICKER REFERENCE (type, exchange, sector, SIC)
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
                'exchange': res.get('primary_exchange'),
                'sic_code': res.get('sic_code'),
                'sic_desc': res.get('sic_description', ''),
                'name': res.get('name', ''),
                'market': res.get('market'),
                'locale': res.get('locale'),
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
    return {tk: _ticker_ref.get(tk, {}) for tk in tickers}


# ═══════════════════════════════════════════════════════════════════════════════
# ASYNC MINUTE BAR FETCH — RTH + PM bars
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
            if not bars: return (tk, dt), None
            pm_bars, rth_bars = [], []
            for b in bars:
                ts = pd.Timestamp(b['t'], unit='ms', tz='UTC').tz_convert(ET)
                t_str = ts.strftime('%H:%M')
                rec = (t_str, {'o':float(b['o']),'h':float(b['h']),
                               'l':float(b['l']),'c':float(b['c']),'v':int(b['v'])})
                if '04:00' <= t_str < '09:30':
                    pm_bars.append(rec)
                elif '09:30' <= t_str < '16:00':
                    rth_bars.append(rec)
            return (tk, dt), {'pm': pm_bars, 'rth': rth_bars}
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
# LOCAL HIGH DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def find_local_high(bars, window=PEAK_WINDOW, max_peak_bar=MAX_PEAK_BAR,
                    fallback_bar=FALLBACK_BAR):
    if len(bars) < window * 2 + 1:
        return None
    for i in range(window, min(max_peak_bar, len(bars) - window)):
        ch = bars[i][1]['h']
        if all(bars[i-j][1]['h'] < ch and bars[i+j][1]['h'] < ch
               for j in range(1, window + 1)):
            ci = i + window
            if ci < len(bars):
                return {'entry_price':bars[ci][1]['c'],'entry_time':bars[ci][0],
                        'entry_type':'local_high','entry_bar':ci,
                        'peak_bar':i,'peak_time':bars[i][0],'peak_price':bars[i][1]['h']}
    if fallback_bar is not None and fallback_bar < len(bars):
        return {'entry_price':bars[fallback_bar][1]['c'],'entry_time':bars[fallback_bar][0],
                'entry_type':'fallback','entry_bar':fallback_bar,
                'peak_bar':None,'peak_time':None,'peak_price':None}
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS COMPUTATION (reusable across all experiments)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(pnl_list, label=''):
    if not pnl_list:
        return {'label':label,'n':0,'avg':0,'med':0,'std':0,'wr':0,
                'sharpe':0,'pf':0,'ci_lo':0,'ci_hi':0,'trimmed_mean':0}
    a = np.array(pnl_list)
    n = len(a)
    avg = float(np.mean(a))
    med = float(np.median(a))
    std = float(np.std(a, ddof=1)) if n > 1 else 0
    wr = float((a > 0).mean())
    sharpe = avg / std if std > 0 else 0
    se = std / np.sqrt(n) if n > 0 else 0
    ci_lo = avg - 1.96 * se
    ci_hi = avg + 1.96 * se
    w = a[a > 0]; l = a[a < 0]
    pf = float(w.sum() / abs(l.sum())) if len(l) > 0 and l.sum() != 0 else float('inf')
    p5, p95 = np.percentile(a, 5), np.percentile(a, 95)
    trimmed = a[(a >= p5) & (a <= p95)]
    tm = float(np.mean(trimmed)) if len(trimmed) > 0 else avg
    return {'label':label,'n':n,'avg':avg,'med':med,'std':std,'wr':wr,
            'sharpe':sharpe,'pf':pf,'ci_lo':ci_lo,'ci_hi':ci_hi,'trimmed_mean':tm}

def leave_one_out_stable(pnl_list, threshold=0.0):
    if len(pnl_list) < 10:
        return False, 0
    base_sign = np.mean(pnl_list) > threshold
    flips = 0
    for i in range(len(pnl_list)):
        subset = pnl_list[:i] + pnl_list[i+1:]
        if (np.mean(subset) > threshold) != base_sign:
            flips += 1
    return flips <= max(1, len(pnl_list) * 0.05), flips

def metrics_row(m, extra_cols=None):
    ci_flag = '✓' if m['ci_lo'] > 0 else '✗'
    row = [m['label'], m['n'], f"{m['avg']:+.2f}%", f"{m['med']:+.2f}%",
           f"{m['trimmed_mean']:+.2f}%", f"{m['wr']*100:.0f}%",
           f"{m['sharpe']:.3f}", f"{m['pf']:.2f}",
           f"[{m['ci_lo']:+.2f},{m['ci_hi']:+.2f}] {ci_flag}"]
    if extra_cols:
        row.extend(extra_cols)
    return row

def metrics_headers(extra=None):
    h = ['Label','n','Avg','Median','TrimMean','WR','Sharpe','PF','CI 95%']
    if extra:
        h.extend(extra)
    return h


# ═══════════════════════════════════════════════════════════════════════════════
# SIC CODE → SECTOR MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

def sic_to_sector(sic_code):
    if sic_code is None: return 'Unknown'
    try:
        sic = int(sic_code)
    except (ValueError, TypeError):
        return 'Unknown'
    if 100 <= sic < 1000: return 'Agriculture'
    if 1000 <= sic < 1500: return 'Mining'
    if 1500 <= sic < 1800: return 'Construction'
    if 2000 <= sic < 4000: return 'Manufacturing'
    if 4000 <= sic < 5000: return 'Transportation'
    if 5000 <= sic < 5200: return 'Wholesale'
    if 5200 <= sic < 6000: return 'Retail'
    if 6000 <= sic < 6800: return 'Finance'
    if 7000 <= sic < 9000: return 'Services'
    if 9000 <= sic < 10000: return 'Public Admin'
    return 'Other'

# known ETF types from Polygon
_ETF_TYPES = {'ETF', 'ETV', 'ETN', 'ETP'}
_EQUITY_TYPES = {'CS', 'ADRC', 'ADRP', 'ADRW'}
_LEVERAGED_KEYWORDS = {'2X','3X','ULTRA','BULL','BEAR','INVERSE','SHORT','LONG',
                        'LEVERAGED','DIREXION','PROSHARES'}

VALID_SECTORS = [
    'Technology', 'Healthcare', 'Finance', 'Energy', 'Mining',
    'Manufacturing', 'Consumer Discretionary', 'Consumer Staples',
    'Industrials', 'Utilities', 'Real Estate', 'Communications',
    'Materials', 'Transportation', 'Retail', 'Agriculture',
    'Construction', 'Aerospace & Defense', 'Wholesale',
    'Services', 'Crypto/Blockchain', 'Cannabis',
]

_llm_sector_cache = {}

def classify_sectors_llm(tickers_with_names):
    if not tickers_with_names:
        return {}
    try:
        client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
    except Exception as e:
        print(f"    Gemini init failed: {e}")
        return {}

    all_results = {}
    batches = [tickers_with_names[i:i+SECTOR_BATCH_SIZE]
               for i in range(0, len(tickers_with_names), SECTOR_BATCH_SIZE)]

    for batch_idx, batch in enumerate(batches):
        lines = [f"  {tk}: {name}" for tk, name in batch]
        prompt = (
            "Classify each stock ticker into exactly ONE sector from this list:\n"
            f"{', '.join(VALID_SECTORS)}\n\n"
            "Tickers:\n" + "\n".join(lines) + "\n\n"
            "Return ONLY a JSON object mapping ticker to sector. Example:\n"
            '{"AAPL": "Technology", "JNJ": "Healthcare"}\n'
            "Return ONLY the JSON, no explanation."
        )
        for attempt in range(3):
            try:
                resp = client.models.generate_content(
                    model=GEMINI_MODEL, contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        temperature=0.0, max_output_tokens=4096),
                )
                text = resp.text or ''
                text = re.sub(r'```(?:json)?', '', text).strip()
                match = re.search(r'\{.*\}', text, re.DOTALL)
                if match:
                    parsed = json.loads(match.group())
                    for tk, sector in parsed.items():
                        tk_upper = tk.upper().strip()
                        if sector in VALID_SECTORS:
                            all_results[tk_upper] = sector
                        else:
                            all_results[tk_upper] = sector
                    break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                else:
                    print(f"    LLM sector batch {batch_idx+1} failed: {e}")
        if (batch_idx + 1) % 5 == 0 or batch_idx == len(batches) - 1:
            print(f"    LLM sectors: {batch_idx+1}/{len(batches)} batches, {len(all_results)} labeled")
        time.sleep(0.5)

    _llm_sector_cache.update(all_results)
    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
#
#  PHASE 1: DATA COLLECTION
#
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

def collect_data():
    t_total = time.time()
    timings = {}

    print(f"\n{'='*80}")
    print(f"  AGENT 1 — EXPERIMENTS E1-E10 — DATA COLLECTION")
    print(f"  {BACKTEST_START} → {BACKTEST_END}")
    print(f"  Universe: hardcoded _FLOAT ({len(_FLOAT)} tickers)")
    print(f"  Strategy: SHORT top {TOP_N} gap-ups >= {GAP_MIN_PCT}%, local high entry")
    print(f"{'='*80}")

    # ── Trading days ──
    d = datetime.strptime(BACKTEST_START, '%Y-%m-%d').date()
    end = datetime.strptime(BACKTEST_END, '%Y-%m-%d').date()
    dates = []
    while d <= end:
        if _is_td(d): dates.append(d.strftime('%Y-%m-%d'))
        d += timedelta(days=1)
    n_dates = len(dates)

    # ── Pre-fetch ADV lookback dates (20 trading days before start) ──
    t1 = time.time()
    adv_dates = _prior_n_td(BACKTEST_START, ADV_LOOKBACK)
    adv_date_strs = [d.strftime('%Y-%m-%d') for d in adv_dates]
    print(f"\n  Phase 1a: Fetching {len(adv_date_strs)} ADV lookback dates...")
    for dt in adv_date_strs:
        fetch_grouped_daily(dt)
    timings['adv_prefetch'] = time.time() - t1

    # ── Scan all days ──
    t2 = time.time()
    print(f"\n  Phase 1b: Scanning {n_dates} days for gap-ups...")
    all_candidates = {}
    all_tickers = set()
    total_cands = 0

    for i, dt in enumerate(dates):
        prior_str = _prior_td(dt).strftime('%Y-%m-%d')
        prior = fetch_grouped_daily(prior_str)
        today = fetch_grouped_daily(dt)

        gap_min = GAP_MIN_PCT / 100.0

        # Wide pool: ALL tickers that gap up (for E10 re-ranking)
        wide_cands = []
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
            wide_cands.append({
                'ticker': tk, 'gap_pct': gap, 'prior_close': prev_c,
                'open_price': cur_p, 'float': fs,
                'tier': 'T4' if gap <= 0.08 else 'T5',
                'scan_date': dt, 'volume_today': today[tk]['v'],
                'in_float_universe': tk in _FLOAT,
            })
            all_tickers.add(tk)

        wide_cands.sort(key=lambda x: -x['gap_pct'])
        # Take top PRE_FILTER_N for E10 scoring pool
        selected = wide_cands[:PRE_FILTER_N]
        for j, s in enumerate(selected):
            s['is_top_n'] = (j < TOP_N)
        for j, s in enumerate(selected):
            s['rank'] = j + 1
            nxt = _next_n_td(dt, 5)
            for k in range(5):
                s[f't{k+1}_date'] = nxt[k].strftime('%Y-%m-%d') if k < len(nxt) else None

        all_candidates[dt] = selected
        total_cands += len(selected)

        if (i+1) % 20 == 0 or i == n_dates - 1:
            print(f"    [{i+1:>3}/{n_dates}] {dt}  |  {total_cands} total  |  {time.time()-t_total:.0f}s")

    timings['scan'] = time.time() - t2
    print(f"  → {total_cands} candidates across {n_dates} days, {len(all_tickers)} unique tickers")

    # ── Fetch ticker reference data (type, exchange, sector) ──
    t3 = time.time()
    print(f"\n  Phase 2: Fetching reference data for {len(all_tickers)} tickers...")
    tk_list = sorted(all_tickers)
    for c_start in range(0, len(tk_list), 200):
        chunk = tk_list[c_start:c_start + 200]
        fetch_ticker_refs(chunk)
        done = min(c_start + 200, len(tk_list))
        print(f"    {done}/{len(tk_list)}  |  {time.time()-t_total:.0f}s")
    timings['refs'] = time.time() - t3

    # ── LLM sector labeling for tickers without SIC codes ──
    t3b = time.time()
    needs_sector = []
    for tk in tk_list:
        ref = _ticker_ref.get(tk, {})
        sic = ref.get('sic_code')
        sector = sic_to_sector(sic)
        if sector == 'Unknown' and ref.get('name'):
            needs_sector.append((tk, ref['name']))
    if needs_sector:
        print(f"\n  Phase 2b: LLM sector labeling for {len(needs_sector)} tickers without SIC...")
        classify_sectors_llm(needs_sector)
        print(f"    → {len(_llm_sector_cache)} tickers labeled via LLM")
    timings['llm_sectors'] = time.time() - t3b

    # ── Fetch minute bars (PM + RTH) for all candidates ──
    t4 = time.time()
    all_pairs = set()
    for picks in all_candidates.values():
        for p in picks:
            all_pairs.add((p['ticker'], p['scan_date']))

    print(f"\n  Phase 3: Fetching 1-min bars (PM+RTH) for {len(all_pairs)} pairs...")
    pairs_list = list(all_pairs)
    all_bars = {}
    for c_start in range(0, len(pairs_list), 100):
        chunk = pairs_list[c_start:c_start + 100]
        result = fetch_minute_bars_batch(chunk)
        all_bars.update(result)
        done = min(c_start + 100, len(pairs_list))
        if done % 200 == 0 or done == len(pairs_list):
            print(f"    {done}/{len(pairs_list)}  |  {time.time()-t_total:.0f}s")
    timings['bars'] = time.time() - t4
    print(f"  → Got bars for {len(all_bars)}/{len(all_pairs)} pairs")

    # ── Find local high entries ──
    t5 = time.time()
    print(f"\n  Phase 4: Finding local high entries...")
    entry_cache = {}
    for key, bar_data in all_bars.items():
        rth = bar_data.get('rth', [])
        if rth and len(rth) >= PEAK_WINDOW * 2 + 1:
            entry_cache[key] = find_local_high(rth)
        else:
            entry_cache[key] = None
    timings['entry'] = time.time() - t5

    # ── Fetch outcome dates (through T+5) ──
    t6 = time.time()
    print(f"  Phase 5: Fetching outcome dates (T+0 through T+5)...")
    outcome_dates = set()
    for picks in all_candidates.values():
        for p in picks:
            outcome_dates.add(p['scan_date'])
            for k in range(1, 6):
                d = p.get(f't{k}_date')
                if d: outcome_dates.add(d)

    to_fetch = sorted(od for od in outcome_dates if od not in _gd_cache)
    for j, od in enumerate(to_fetch):
        try: fetch_grouped_daily(od)
        except Exception as e: print(f"    WARN: {od}: {e}")
        if (j+1) % 20 == 0:
            print(f"    {j+1}/{len(to_fetch)} outcome dates")
    timings['outcomes'] = time.time() - t6
    print(f"  → Done  |  {timings['outcomes']:.0f}s")

    # ── Build enriched trade list ──
    t7 = time.time()
    print(f"\n  Phase 6: Building enriched trade list...")
    trades = []

    for dt in dates:
        for c in all_candidates.get(dt, []):
            tc = copy.deepcopy(c)
            key = (tc['ticker'], tc['scan_date'])
            entry = entry_cache.get(key)
            bar_data = all_bars.get(key, {})
            rth_bars = bar_data.get('rth', []) if bar_data else []
            pm_bars = bar_data.get('pm', []) if bar_data else []

            if entry is None:
                continue

            tc['entry_price'] = entry['entry_price']
            tc['entry_time'] = entry['entry_time']
            tc['entry_type'] = entry['entry_type']
            tc['entry_bar'] = entry['entry_bar']
            tc['peak_bar'] = entry.get('peak_bar')
            tc['peak_price'] = entry.get('peak_price')
            tc['is_top_n'] = c.get('is_top_n', True)

            tk, ep = tc['ticker'], tc['entry_price']
            ref = _ticker_ref.get(tk, {})
            tc['asset_type'] = ref.get('type', 'UNK')
            tc['exchange'] = ref.get('exchange')
            tc['sic_code'] = ref.get('sic_code')
            sic_sector = sic_to_sector(ref.get('sic_code'))
            if sic_sector == 'Unknown' and tk in _llm_sector_cache:
                tc['sector'] = _llm_sector_cache[tk]
                tc['sector_source'] = 'llm'
            else:
                tc['sector'] = sic_sector
                tc['sector_source'] = 'sic' if sic_sector != 'Unknown' else 'none'
            tc['asset_name'] = ref.get('name', '')
            tc['is_etf'] = tc['asset_type'] in _ETF_TYPES
            tc['is_equity'] = tc['asset_type'] in _EQUITY_TYPES
            is_lev = any(kw in tc['asset_name'].upper() for kw in _LEVERAGED_KEYWORDS)
            tc['is_leveraged'] = is_lev

            # ── Intraday PnL at fixed bar indices from open ──
            for bar_idx, col_name in [(30, 'pnl_30m'), (60, 'pnl_60m'), (120, 'pnl_2h')]:
                if bar_idx < len(rth_bars):
                    exit_p = rth_bars[bar_idx][1]['c']
                    tc[col_name] = round((ep - exit_p) / ep * 100, 3)
                else:
                    tc[col_name] = None

            # ── Daily close PnL (EOD through T+5) ──
            for offset, col_name in [(0, 'pnl_eod'), (1, 'pnl_t1'), (2, 'pnl_t2'),
                                     (3, 'pnl_t3'), (4, 'pnl_t4'), (5, 'pnl_t5')]:
                if offset == 0:
                    date_key = tc['scan_date']
                else:
                    date_key = tc.get(f't{offset}_date', '')
                dd = _gd_cache.get(date_key, {}).get(tk, {})
                tc[col_name] = round((ep - dd['c']) / ep * 100, 3) if dd.get('c') else None

            # ── ADV (20-day average dollar volume before scan date) ──
            prior_dates = _prior_n_td(tc['scan_date'], ADV_LOOKBACK)
            vols = []
            for pd_d in prior_dates:
                pd_str = pd_d.strftime('%Y-%m-%d')
                dd = _gd_cache.get(pd_str, {}).get(tk, {})
                if dd.get('v') and dd.get('c') and dd['c'] > 0:
                    vols.append(dd['v'] * dd['c'])
            tc['adv_dollar'] = np.mean(vols) if vols else None
            tc['adv_shares'] = np.mean([v / max(tc['open_price'], 1) for v in vols]) if vols else None

            # ── PM analysis for gap freshness (E8) ──
            if pm_bars:
                pm_high = max(b[1]['h'] for b in pm_bars)
                pm_high_time = None
                for b in pm_bars:
                    if b[1]['h'] >= pm_high * 0.98:
                        pm_high_time = b[0]
                        break
                tc['pm_high'] = pm_high
                tc['pm_high_time'] = pm_high_time
                tc['pm_bar_count'] = len(pm_bars)
                if pm_high_time:
                    h, m = int(pm_high_time[:2]), int(pm_high_time[3:5])
                    mins_before_open = (9 * 60 + 30) - (h * 60 + m)
                    tc['mins_before_open'] = mins_before_open
                    if mins_before_open <= 30:
                        tc['gap_freshness'] = 'very_fresh'
                    elif mins_before_open <= 120:
                        tc['gap_freshness'] = 'fresh'
                    elif mins_before_open <= 300:
                        tc['gap_freshness'] = 'moderate'
                    else:
                        tc['gap_freshness'] = 'stale'
                else:
                    tc['mins_before_open'] = None
                    tc['gap_freshness'] = 'unknown'
                pm_last = pm_bars[-1][1]['c']
                pm_open = pm_bars[0][1]['o']
                total_move = pm_high - pm_open
                tc['pm_retrace'] = (pm_high - pm_last) / total_move if total_move > 0 else 0
            else:
                tc['pm_high'] = None
                tc['pm_high_time'] = None
                tc['pm_bar_count'] = 0
                tc['mins_before_open'] = None
                tc['gap_freshness'] = 'no_pm'
                tc['pm_retrace'] = None

            # ── Intraday pattern classification (E7) ──
            if len(rth_bars) >= 60:
                low_30m = min(b[1]['l'] for b in rth_bars[:30])
                close_30m = rth_bars[29][1]['c']
                close_60m = rth_bars[59][1]['c']
                open_price = rth_bars[0][1]['o']
                drop_30m = (low_30m - open_price) / open_price if open_price > 0 else 0
                recovery_denom = abs(open_price - low_30m)
                recovery = (close_60m - low_30m) / recovery_denom if recovery_denom > 0 else 0

                if drop_30m < -0.03 and recovery > 0.5:
                    tc['open_pattern'] = 'drop_then_recover'
                elif drop_30m < -0.03 and recovery <= 0.5:
                    tc['open_pattern'] = 'sustained_drop'
                elif drop_30m >= -0.01:
                    tc['open_pattern'] = 'flat_or_bounce'
                else:
                    tc['open_pattern'] = 'mild_drop'
            else:
                tc['open_pattern'] = 'no_data'

            trades.append(tc)

    timings['enrich'] = time.time() - t7
    elapsed = time.time() - t_total

    # ── Timing summary ──
    print(f"\n  {'─'*60}")
    print(f"  TIMING BREAKDOWN")
    print(f"  {'─'*60}")
    for step, dur in [('ADV prefetch', timings['adv_prefetch']),
                      ('Scan (grouped daily)', timings['scan']),
                      ('Ticker reference', timings['refs']),
                      ('LLM sector labeling', timings['llm_sectors']),
                      ('Minute bars (PM+RTH)', timings['bars']),
                      ('Local high entries', timings['entry']),
                      ('Outcome dates (T+5)', timings['outcomes']),
                      ('Enrichment', timings['enrich'])]:
        print(f"    {step:<30s}  {dur:>6.0f}s  ({dur/elapsed*100:>4.0f}%)")
    print(f"\n  TOTAL: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  TRADES: {len(trades)} with entry")

    return trades, dates, timings


# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
#
#  EXPERIMENTS E1-E10
#
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════


# ─── E1: Hold Period Sweep ────────────────────────────────────────────────────

def run_e1(trades):
    print(f"\n\n{'='*80}")
    print(f"  E1: HOLD PERIOD SWEEP — 30m → T+5 by gap tier")
    print(f"{'='*80}")

    hold_cols = OrderedDict([
        ('30m', 'pnl_30m'), ('60m', 'pnl_60m'), ('2h', 'pnl_2h'),
        ('EOD', 'pnl_eod'), ('T+1', 'pnl_t1'), ('T+2', 'pnl_t2'),
        ('T+3', 'pnl_t3'), ('T+4', 'pnl_t4'), ('T+5', 'pnl_t5'),
    ])

    tiers = OrderedDict([
        ('T4 (5-8%)', lambda t: t['tier'] == 'T4'),
        ('T5 (>8%)', lambda t: t['tier'] == 'T5'),
        ('All (>5%)', lambda t: True),
    ])

    results = {}
    for tier_name, tier_fn in tiers.items():
        subset = [t for t in trades if tier_fn(t)]
        results[tier_name] = {}
        print(f"\n  ── {tier_name} ({len(subset)} trades) ──")
        rows = []
        for hold_name, col in hold_cols.items():
            pnls = [t[col] for t in subset if t.get(col) is not None]
            m = compute_metrics(pnls, hold_name)
            loo, flips = leave_one_out_stable(pnls)
            results[tier_name][hold_name] = m
            rows.append(metrics_row(m, [f"{'YES' if loo else 'NO'} ({flips})"]))
        print(tabulate(rows, headers=metrics_headers(['LOO Stable']), tablefmt='simple'))

    # find optimal hold per tier
    print(f"\n  ── OPTIMAL HOLD PER TIER ──")
    for tier_name in tiers:
        best = max(results[tier_name].items(), key=lambda x: x[1]['sharpe'])
        print(f"  {tier_name:15s}  →  {best[0]:5s}  Sharpe={best[1]['sharpe']:.3f}  "
              f"Avg={best[1]['avg']:+.2f}%  n={best[1]['n']}")

    return results


# ─── E2: ETF Exclusion Filter ────────────────────────────────────────────────

def run_e2(trades):
    print(f"\n\n{'='*80}")
    print(f"  E2: ETF EXCLUSION FILTER")
    print(f"{'='*80}")

    filters = OrderedDict([
        ('no_filter', lambda t: True),
        ('exclude_etf', lambda t: not t.get('is_etf', False)),
        ('equities_only', lambda t: t.get('is_equity', False)),
        ('cs_plus_adr', lambda t: t.get('asset_type') in ('CS', 'ADRC', 'ADRP')),
    ])

    results = {}
    rows = []
    for name, fn in filters.items():
        subset = [t for t in trades if fn(t)]
        pnls = [t['pnl_t2'] for t in subset if t.get('pnl_t2') is not None]
        m = compute_metrics(pnls, name)
        results[name] = m
        n_etf = sum(1 for t in subset if t.get('is_etf'))
        rows.append(metrics_row(m, [f"{len(subset)}", f"{n_etf}"]))

    print(tabulate(rows, headers=metrics_headers(['Total','ETFs']), tablefmt='simple'))

    # ETF performance breakdown
    etf_trades = [t for t in trades if t.get('is_etf') and t.get('pnl_t2') is not None]
    eq_trades = [t for t in trades if t.get('is_equity') and t.get('pnl_t2') is not None]
    print(f"\n  ── ETF vs EQUITY HEAD-TO-HEAD ──")
    for lbl, subset in [('ETFs', etf_trades), ('Equities', eq_trades)]:
        if subset:
            pnls = [t['pnl_t2'] for t in subset]
            m = compute_metrics(pnls, lbl)
            print(f"  {lbl:12s}  n={m['n']:<4d}  avg={m['avg']:+.2f}%  WR={m['wr']*100:.0f}%  Sharpe={m['sharpe']:.3f}")

    # list ETFs that appeared
    etf_list = [t for t in trades if t.get('is_etf')]
    if etf_list:
        tk_pnl = defaultdict(list)
        for t in etf_list:
            if t.get('pnl_t2') is not None:
                tk_pnl[t['ticker']].append(t['pnl_t2'])
        print(f"\n  ── ETFs IN UNIVERSE ({len(tk_pnl)} unique) ──")
        etf_rows = []
        for tk, pnls in sorted(tk_pnl.items(), key=lambda x: -len(x[1]))[:20]:
            a = np.array(pnls)
            etf_rows.append([tk, len(a), f"{np.mean(a):+.2f}%",
                            f"{(a>0).mean()*100:.0f}%"])
        print(tabulate(etf_rows, headers=['Ticker','n','Avg T+2','WR'], tablefmt='simple'))

    return results


# ─── E3: Liquidity Filter (ADV) ──────────────────────────────────────────────

def run_e3(trades):
    print(f"\n\n{'='*80}")
    print(f"  E3: LIQUIDITY FILTER (Average Daily Dollar Volume)")
    print(f"{'='*80}")

    dollar_thresholds = OrderedDict([
        ('no_filter', 0),
        ('adv_$500K', 500_000),
        ('adv_$1M', 1_000_000),
        ('adv_$5M', 5_000_000),
        ('adv_$10M', 10_000_000),
        ('adv_$25M', 25_000_000),
        ('adv_$50M', 50_000_000),
        ('adv_$100M', 100_000_000),
    ])

    results = {}
    rows = []
    for name, thresh in dollar_thresholds.items():
        if thresh == 0:
            subset = [t for t in trades if t.get('pnl_t2') is not None]
        else:
            subset = [t for t in trades if t.get('pnl_t2') is not None
                      and t.get('adv_dollar') is not None
                      and t['adv_dollar'] >= thresh]
        pnls = [t['pnl_t2'] for t in subset]
        m = compute_metrics(pnls, name)
        results[name] = m
        active_days = len(set(t['scan_date'] for t in subset))
        tpd = m['n'] / active_days if active_days > 0 else 0
        rows.append(metrics_row(m, [f"{tpd:.1f}"]))

    print(tabulate(rows, headers=metrics_headers(['TPD']), tablefmt='simple'))

    # ADV quartile analysis
    trades_with_adv = [t for t in trades if t.get('adv_dollar') is not None
                       and t.get('pnl_t2') is not None]
    if trades_with_adv:
        advs = np.array([t['adv_dollar'] for t in trades_with_adv])
        q25, q50, q75 = np.percentile(advs, [25, 50, 75])
        print(f"\n  ADV distribution: Q25=${q25/1e6:.1f}M  Q50=${q50/1e6:.1f}M  Q75=${q75/1e6:.1f}M")

        quartiles = [
            ('Q1 (lowest)', lambda t: t['adv_dollar'] < q25),
            ('Q2', lambda t: q25 <= t['adv_dollar'] < q50),
            ('Q3', lambda t: q50 <= t['adv_dollar'] < q75),
            ('Q4 (highest)', lambda t: t['adv_dollar'] >= q75),
        ]
        print(f"\n  ── ADV QUARTILE PERFORMANCE ──")
        q_rows = []
        for ql, qfn in quartiles:
            qs = [t['pnl_t2'] for t in trades_with_adv if qfn(t)]
            qm = compute_metrics(qs, ql)
            q_rows.append(metrics_row(qm))
        print(tabulate(q_rows, headers=metrics_headers(), tablefmt='simple'))

        # correlation
        pnls = np.array([t['pnl_t2'] for t in trades_with_adv])
        log_advs = np.log10(advs + 1)
        r = np.corrcoef(log_advs, pnls)[0, 1]
        print(f"\n  Correlation log10(ADV) vs T+2 PnL: r={r:.4f}")

    return results


# ─── E4: Asset Type Stratification ───────────────────────────────────────────

def run_e4(trades):
    print(f"\n\n{'='*80}")
    print(f"  E4: ASSET TYPE STRATIFICATION")
    print(f"{'='*80}")

    type_groups = defaultdict(list)
    for t in trades:
        if t.get('pnl_t2') is None: continue
        at = t.get('asset_type', 'UNK')
        if t.get('is_leveraged'):
            at = 'ETF_LEV'
        elif t.get('is_etf'):
            at = 'ETF'
        type_groups[at].append(t)

    results = {}
    rows = []
    for at in sorted(type_groups, key=lambda x: -len(type_groups[x])):
        subset = type_groups[at]
        pnls = [t['pnl_t2'] for t in subset]
        m = compute_metrics(pnls, at)
        results[at] = m
        tickers = len(set(t['ticker'] for t in subset))
        rows.append(metrics_row(m, [f"{tickers}"]))

    print(tabulate(rows, headers=metrics_headers(['Tickers']), tablefmt='simple'))
    return results


# ─── E5: Price Bands ─────────────────────────────────────────────────────────

def run_e5(trades):
    print(f"\n\n{'='*80}")
    print(f"  E5: PRICE BAND ANALYSIS")
    print(f"{'='*80}")

    bands = OrderedDict([
        ('$10-20', (10, 20)),
        ('$20-50', (20, 50)),
        ('$50-100', (50, 100)),
        ('$100-200', (100, 200)),
        ('$200-500', (200, 500)),
        ('$500+', (500, None)),
    ])

    floors = OrderedDict([
        ('min $10', 10),
        ('min $15', 15),
        ('min $20', 20),
        ('min $30', 30),
        ('min $50', 50),
    ])

    print(f"\n  ── PRICE BANDS ──")
    results = {}
    rows = []
    for name, (lo, hi) in bands.items():
        subset = [t for t in trades if t.get('pnl_t2') is not None
                  and t['open_price'] >= lo
                  and (hi is None or t['open_price'] < hi)]
        pnls = [t['pnl_t2'] for t in subset]
        m = compute_metrics(pnls, name)
        results[name] = m
        rows.append(metrics_row(m))
    print(tabulate(rows, headers=metrics_headers(), tablefmt='simple'))

    print(f"\n  ── MINIMUM PRICE FLOORS ──")
    floor_results = {}
    rows = []
    for name, floor in floors.items():
        subset = [t for t in trades if t.get('pnl_t2') is not None
                  and t['open_price'] >= floor]
        pnls = [t['pnl_t2'] for t in subset]
        m = compute_metrics(pnls, name)
        floor_results[name] = m
        rows.append(metrics_row(m))
    print(tabulate(rows, headers=metrics_headers(), tablefmt='simple'))

    results['floors'] = floor_results
    return results


# ─── E6: Short Interest (attempted) ──────────────────────────────────────────

def run_e6(trades):
    print(f"\n\n{'='*80}")
    print(f"  E6: SHORT INTEREST FILTER")
    print(f"{'='*80}")

    print(f"\n  NOTE: Short interest data requires Polygon Business plan.")
    print(f"  Attempting to fetch for a sample ticker to check availability...")

    sample_tk = trades[0]['ticker'] if trades else 'AAPL'
    try:
        url = f"{_BASE}/v3/reference/tickers/{sample_tk}/short-interest"
        data = _pg(url)
        if data.get('results'):
            print(f"  ✓ Short interest data available! Fetching for all tickers...")
            # would proceed with full fetch here
        else:
            print(f"  ✗ No short interest data returned. Skipping E6.")
            print(f"  → Proxy: using volume-on-gap as squeeze risk indicator instead.")
    except Exception as e:
        print(f"  ✗ Short interest endpoint not available: {e}")
        print(f"  → Proxy: using volume-on-gap as squeeze risk indicator instead.")

    # Proxy analysis: gap-day volume vs ADV as squeeze indicator
    print(f"\n  ── VOLUME SPIKE AS SQUEEZE PROXY ──")
    trades_with_adv = [t for t in trades if t.get('pnl_t2') is not None
                       and t.get('adv_dollar') is not None and t['adv_dollar'] > 0]

    if trades_with_adv:
        for t in trades_with_adv:
            t['vol_ratio'] = (t['volume_today'] * t['open_price']) / t['adv_dollar']

        vol_tiers = OrderedDict([
            ('vol <1.5x', lambda t: t['vol_ratio'] < 1.5),
            ('vol 1.5-3x', lambda t: 1.5 <= t['vol_ratio'] < 3),
            ('vol 3-5x', lambda t: 3 <= t['vol_ratio'] < 5),
            ('vol 5-10x', lambda t: 5 <= t['vol_ratio'] < 10),
            ('vol >10x', lambda t: t['vol_ratio'] >= 10),
        ])

        rows = []
        results = {}
        for name, fn in vol_tiers.items():
            subset = [t for t in trades_with_adv if fn(t)]
            pnls = [t['pnl_t2'] for t in subset]
            m = compute_metrics(pnls, name)
            results[name] = m
            rows.append(metrics_row(m))
        print(tabulate(rows, headers=metrics_headers(), tablefmt='simple'))

        # correlation
        vr = np.array([t['vol_ratio'] for t in trades_with_adv])
        pnls = np.array([t['pnl_t2'] for t in trades_with_adv])
        r = np.corrcoef(np.log10(vr + 1), pnls)[0, 1]
        print(f"\n  Correlation log10(vol_ratio) vs T+2 PnL: r={r:.4f}")
    else:
        results = {}
        print(f"  No trades with ADV data.")

    return results


# ─── E7: Intraday Pattern at Open ────────────────────────────────────────────

def run_e7(trades):
    print(f"\n\n{'='*80}")
    print(f"  E7: INTRADAY PATTERN AT OPEN (first 60 min)")
    print(f"{'='*80}")

    patterns = defaultdict(list)
    for t in trades:
        if t.get('pnl_t2') is None: continue
        patterns[t.get('open_pattern', 'no_data')].append(t)

    results = {}
    rows = []
    for pat in ['drop_then_recover', 'sustained_drop', 'mild_drop', 'flat_or_bounce', 'no_data']:
        subset = patterns.get(pat, [])
        pnls = [t['pnl_t2'] for t in subset]
        m = compute_metrics(pnls, pat)
        results[pat] = m
        # also compute EOD PnL for comparison
        eod_pnls = [t['pnl_eod'] for t in subset if t.get('pnl_eod') is not None]
        eod_m = compute_metrics(eod_pnls, '')
        rows.append(metrics_row(m, [f"{eod_m['avg']:+.2f}%"]))

    print(tabulate(rows, headers=metrics_headers(['EOD Avg']), tablefmt='simple'))

    # also test: entry at 60m (after the bounce) vs local high
    print(f"\n  ── ALTERNATIVE ENTRY: close of bar 60 vs local high ──")
    dtr_trades = patterns.get('drop_then_recover', [])
    if dtr_trades:
        # local high entry (existing)
        lh_pnls = [t['pnl_t2'] for t in dtr_trades]
        # 60m entry: use pnl from bar 60 price to T+2 close
        alt_pnls = []
        for t in dtr_trades:
            if t.get('pnl_60m') is not None and t.get('pnl_t2') is not None:
                # T+2 PnL from 60m price = need to recompute from the 60m close
                # We have entry_price-based PnL. The 60m bar close relative to entry:
                # pnl_60m = (entry - bar60_close)/entry * 100
                # bar60_close = entry * (1 - pnl_60m/100)
                bar60_close = t['entry_price'] * (1 - t['pnl_60m'] / 100)
                if bar60_close > 0:
                    # T+2 close relative to bar60 entry
                    t2_close = t['entry_price'] * (1 - t['pnl_t2'] / 100)
                    alt_pnl = (bar60_close - t2_close) / bar60_close * 100
                    alt_pnls.append(alt_pnl)
        lh_m = compute_metrics(lh_pnls, 'local_high entry')
        alt_m = compute_metrics(alt_pnls, '60m bounce entry')
        print(f"  local_high:  n={lh_m['n']}  avg={lh_m['avg']:+.2f}%  Sharpe={lh_m['sharpe']:.3f}")
        print(f"  60m_bounce:  n={alt_m['n']}  avg={alt_m['avg']:+.2f}%  Sharpe={alt_m['sharpe']:.3f}")
    else:
        print(f"  No drop_then_recover trades found.")

    return results


# ─── E8: Gap Freshness ───────────────────────────────────────────────────────

def run_e8(trades):
    print(f"\n\n{'='*80}")
    print(f"  E8: GAP FRESHNESS — Time PM High Formed")
    print(f"{'='*80}")

    freshness_groups = defaultdict(list)
    for t in trades:
        if t.get('pnl_t2') is None: continue
        freshness_groups[t.get('gap_freshness', 'unknown')].append(t)

    results = {}
    rows = []
    for cat in ['very_fresh', 'fresh', 'moderate', 'stale', 'no_pm', 'unknown']:
        subset = freshness_groups.get(cat, [])
        pnls = [t['pnl_t2'] for t in subset]
        m = compute_metrics(pnls, cat)
        results[cat] = m
        avg_mins = np.mean([t['mins_before_open'] for t in subset
                           if t.get('mins_before_open') is not None]) if subset else 0
        rows.append(metrics_row(m, [f"{avg_mins:.0f}m"]))

    print(tabulate(rows, headers=metrics_headers(['Avg Mins Pre-Open']), tablefmt='simple'))

    # correlation of mins_before_open with T+2 PnL
    valid = [t for t in trades if t.get('mins_before_open') is not None
             and t.get('pnl_t2') is not None]
    if valid:
        mins = np.array([t['mins_before_open'] for t in valid])
        pnls = np.array([t['pnl_t2'] for t in valid])
        r = np.corrcoef(mins, pnls)[0, 1]
        print(f"\n  Correlation mins_before_open vs T+2 PnL: r={r:.4f}")
        print(f"  (negative = fresher gaps fade harder)")

    return results


# ─── E9: Sector Concentration ────────────────────────────────────────────────

def run_e9(trades):
    print(f"\n\n{'='*80}")
    print(f"  E9: SECTOR CONCENTRATION — Idiosyncratic vs Sector-Wide Gaps")
    print(f"{'='*80}")

    # sector source breakdown
    src_counts = Counter(t.get('sector_source', 'none') for t in trades)
    unk_count = sum(1 for t in trades if t['sector'] == 'Unknown')
    print(f"\n  Sector sources: SIC={src_counts.get('sic',0)}  "
          f"LLM={src_counts.get('llm',0)}  "
          f"None={src_counts.get('none',0)}  "
          f"Still Unknown={unk_count}")

    # group by scan_date, count sector occurrences among all candidates (not just top N)
    date_sectors = defaultdict(lambda: Counter())
    for t in trades:
        date_sectors[t['scan_date']][t['sector']] += 1

    # tag each trade with sector_count for that day
    for t in trades:
        t['sector_count'] = date_sectors[t['scan_date']][t['sector']]
        t['is_idiosyncratic'] = t['sector_count'] <= 2

    # sector performance
    sector_groups = defaultdict(list)
    for t in trades:
        if t.get('pnl_t2') is None: continue
        sector_groups[t['sector']].append(t)

    print(f"\n  ── SECTOR PERFORMANCE ──")
    rows = []
    results = {}
    for sec in sorted(sector_groups, key=lambda x: -len(sector_groups[x])):
        pnls = [t['pnl_t2'] for t in sector_groups[sec]]
        m = compute_metrics(pnls, sec)
        results[sec] = m
        rows.append(metrics_row(m))
    print(tabulate(rows, headers=metrics_headers(), tablefmt='simple'))

    # idiosyncratic vs crowded
    print(f"\n  ── IDIOSYNCRATIC vs SECTOR-WIDE ──")
    idio_trades = [t for t in trades if t.get('pnl_t2') is not None and t['is_idiosyncratic']]
    crowd_trades = [t for t in trades if t.get('pnl_t2') is not None and not t['is_idiosyncratic']]

    for lbl, subset in [('Idiosyncratic (≤2)', idio_trades), ('Sector-wide (≥3)', crowd_trades)]:
        pnls = [t['pnl_t2'] for t in subset]
        m = compute_metrics(pnls, lbl)
        loo, flips = leave_one_out_stable(pnls)
        print(f"  {lbl:25s}  n={m['n']:<4d}  avg={m['avg']:+.2f}%  med={m['med']:+.2f}%  "
              f"WR={m['wr']*100:.0f}%  Sharpe={m['sharpe']:.3f}  LOO={'YES' if loo else 'NO'}")

    return results


# ─── E10: Combined Best Signal Score ─────────────────────────────────────────

def run_e10(trades, e1_results, e2_results, e3_results):
    print(f"\n\n{'='*80}")
    print(f"  E10: COMBINED SIGNAL SCORE (re-rank from top {PRE_FILTER_N} → top {TOP_N})")
    print(f"{'='*80}")

    hold_col = 'pnl_t2'
    print(f"\n  Using {hold_col} for scoring (T+2 baseline)")

    # Use all trades (from the wider PRE_FILTER_N pool) for re-ranking
    valid = [t for t in trades if t.get(hold_col) is not None
             and t.get('adv_dollar') is not None]

    if not valid:
        print(f"  No valid trades for scoring.")
        return {}

    # count pool sizes per day
    daily_pool = defaultdict(list)
    for t in valid:
        daily_pool[t['scan_date']].append(t)
    pool_sizes = [len(v) for v in daily_pool.values()]
    avg_pool = np.mean(pool_sizes) if pool_sizes else 0
    days_with_gt5 = sum(1 for s in pool_sizes if s > TOP_N)
    print(f"  Pool: {len(valid)} trades across {len(daily_pool)} days")
    print(f"  Avg pool/day: {avg_pool:.1f}  |  Days with >{TOP_N} candidates: {days_with_gt5}")

    # normalize features
    gaps = np.array([t['gap_pct'] for t in valid])
    advs = np.log10(np.array([max(t['adv_dollar'], 1) for t in valid]))
    gap_min, gap_max = gaps.min(), gaps.max()
    adv_min, adv_max = advs.min(), advs.max()

    def norm(val, vmin, vmax):
        if vmax == vmin: return 0.5
        return (val - vmin) / (vmax - vmin)

    weight_combos = OrderedDict([
        ('pure_gap',       {'gap':1.0, 'adv':0.0, 'equity':0.0, 'fresh':0.0, 'idio':0.0, 'pattern':0.0}),
        ('gap_adv',        {'gap':0.5, 'adv':0.3, 'equity':0.1, 'fresh':0.05, 'idio':0.05, 'pattern':0.0}),
        ('balanced',       {'gap':0.30,'adv':0.15,'equity':0.15,'fresh':0.15,'idio':0.10, 'pattern':0.15}),
        ('quality',        {'gap':0.20,'adv':0.20,'equity':0.20,'fresh':0.15,'idio':0.10, 'pattern':0.15}),
        ('anti_squeeze',   {'gap':0.25,'adv':0.10,'equity':0.20,'fresh':0.10,'idio':0.10, 'pattern':0.25}),
        ('pattern_heavy',  {'gap':0.15,'adv':0.05,'equity':0.15,'fresh':0.15,'idio':0.10, 'pattern':0.40}),
    ])

    # pre-compute per-trade features
    pattern_scores = {'sustained_drop': 1.0, 'mild_drop': 0.5,
                      'flat_or_bounce': 0.4, 'drop_then_recover': 0.1, 'no_data': 0.5}
    freshness_scores = {'stale': 1.0, 'moderate': 0.7, 'fresh': 0.4,
                        'very_fresh': 0.1, 'no_pm': 0.5, 'unknown': 0.5}
    for t in valid:
        t['_f_gap'] = norm(t['gap_pct'], gap_min, gap_max)
        t['_f_adv'] = norm(np.log10(max(t['adv_dollar'], 1)), adv_min, adv_max)
        t['_f_equity'] = 1.0 if t.get('is_equity') else 0.2
        t['_f_fresh'] = freshness_scores.get(t.get('gap_freshness', 'unknown'), 0.5)
        t['_f_idio'] = 1.0 if t.get('is_idiosyncratic', True) else 0.5
        t['_f_pattern'] = pattern_scores.get(t.get('open_pattern', 'no_data'), 0.5)

    print(f"\n  ── RANKING FUNCTION COMPARISON (top {TOP_N}/day from pool of {PRE_FILTER_N}) ──")
    results = {}
    rows = []

    for combo_name, weights in weight_combos.items():
        selected_pnls = []
        for dt in sorted(daily_pool):
            day_trades = daily_pool[dt]
            for t in day_trades:
                t['_score'] = (weights['gap'] * t['_f_gap'] +
                               weights['adv'] * t['_f_adv'] +
                               weights['equity'] * t['_f_equity'] +
                               weights['fresh'] * t['_f_fresh'] +
                               weights['idio'] * t['_f_idio'] +
                               weights['pattern'] * t['_f_pattern'])
            ranked = sorted(day_trades, key=lambda x: -x['_score'])[:TOP_N]
            for t in ranked:
                selected_pnls.append(t[hold_col])

        m = compute_metrics(selected_pnls, combo_name)
        results[combo_name] = m
        rows.append(metrics_row(m))

    print(tabulate(rows, headers=metrics_headers(), tablefmt='simple'))

    best_combo = max(results.items(), key=lambda x: x[1]['sharpe'])
    gap_m = results['pure_gap']
    print(f"\n  Best scoring: {best_combo[0]}  Sharpe={best_combo[1]['sharpe']:.3f}")
    print(f"  vs pure_gap:  Sharpe={gap_m['sharpe']:.3f}")
    print(f"  Delta Sharpe: {best_combo[1]['sharpe'] - gap_m['sharpe']:+.3f}")
    print(f"  Delta Avg:    {best_combo[1]['avg'] - gap_m['avg']:+.2f}%")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
#
#  CHARTS
#
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

def plot_all(trades, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, dates):

    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        BASE_DIR = os.getcwd()

    # ── Fig 1: E1 Hold Period Heatmap ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle('E1: Hold Period Sweep — Sharpe by Tier × Hold', fontsize=13, fontweight='bold')

    holds = list(list(e1.values())[0].keys())
    tiers = list(e1.keys())

    # Sharpe heatmap
    ax = axes[0]
    sharpe_data = np.array([[e1[t][h]['sharpe'] for h in holds] for t in tiers])
    im = ax.imshow(sharpe_data, cmap='RdYlGn', aspect='auto',
                   vmin=min(0, sharpe_data.min()), vmax=max(0.5, sharpe_data.max()))
    ax.set_xticks(range(len(holds))); ax.set_xticklabels(holds, fontsize=8, rotation=45)
    ax.set_yticks(range(len(tiers))); ax.set_yticklabels(tiers, fontsize=9)
    for i in range(len(tiers)):
        for j in range(len(holds)):
            v = sharpe_data[i, j]
            ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=7,
                    fontweight='bold' if v == sharpe_data[i].max() else 'normal')
    ax.set_title('Sharpe Ratio')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Avg PnL heatmap
    ax = axes[1]
    avg_data = np.array([[e1[t][h]['avg'] for h in holds] for t in tiers])
    im2 = ax.imshow(avg_data, cmap='RdYlGn', aspect='auto',
                    vmin=min(-2, avg_data.min()), vmax=max(2, avg_data.max()))
    ax.set_xticks(range(len(holds))); ax.set_xticklabels(holds, fontsize=8, rotation=45)
    ax.set_yticks(range(len(tiers))); ax.set_yticklabels(tiers, fontsize=9)
    for i in range(len(tiers)):
        for j in range(len(holds)):
            v = avg_data[i, j]
            ax.text(j, i, f'{v:+.1f}', ha='center', va='center', fontsize=7,
                    fontweight='bold' if v == avg_data[i].max() else 'normal')
    ax.set_title('Avg PnL (%)')
    plt.colorbar(im2, ax=ax, shrink=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    p1 = os.path.join(BASE_DIR, 'e1_hold_period_heatmap.png')
    plt.savefig(p1, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {p1}")

    # ── Fig 2: E2/E3/E5 filter comparison ─────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('E2/E3/E5: Filter Impact on Sharpe', fontsize=13, fontweight='bold')

    # E2: ETF filter
    ax = axes[0]
    names = list(e2.keys()); sharpes = [e2[n]['sharpe'] for n in names]
    colors = ['#27ae60' if s > e2['no_filter']['sharpe'] else '#e74c3c' for s in sharpes]
    bars = ax.bar(range(len(names)), sharpes, color=colors, alpha=0.8)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=7, rotation=30)
    ax.axhline(e2['no_filter']['sharpe'], color='#999', ls='--', lw=1)
    ax.set_title('E2: ETF Exclusion'); ax.set_ylabel('Sharpe')
    for b, s, n in zip(bars, sharpes, names):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.003,
                f'{s:.3f}\nn={e2[n]["n"]}', ha='center', va='bottom', fontsize=7)

    # E3: Liquidity
    ax = axes[1]
    names = list(e3.keys()); sharpes = [e3[n]['sharpe'] for n in names]
    ns = [e3[n]['n'] for n in names]
    ax.bar(range(len(names)), sharpes, color='#3498db', alpha=0.8)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=6, rotation=45)
    ax.axhline(e3['no_filter']['sharpe'], color='#999', ls='--', lw=1)
    ax.set_title('E3: ADV Threshold'); ax.set_ylabel('Sharpe')
    for i, (s, n) in enumerate(zip(sharpes, ns)):
        ax.text(i, s + 0.003, f'{s:.3f}\nn={n}', ha='center', va='bottom', fontsize=6)

    # E5: Price floors
    ax = axes[2]
    if 'floors' in e5:
        fl = e5['floors']
        names = list(fl.keys()); sharpes = [fl[n]['sharpe'] for n in names]
        ax.bar(range(len(names)), sharpes, color='#e67e22', alpha=0.8)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=8, rotation=30)
        ax.set_title('E5: Min Price Floor'); ax.set_ylabel('Sharpe')
        for i, s in enumerate(zip(sharpes)):
            ax.text(i, s[0] + 0.003, f'{s[0]:.3f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    p2 = os.path.join(BASE_DIR, 'e2_e3_e5_filters.png')
    plt.savefig(p2, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {p2}")

    # ── Fig 3: E4 Asset Type + E9 Sector ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle('E4: Asset Type & E9: Sector Performance', fontsize=13, fontweight='bold')

    # E4
    ax = axes[0]
    valid_e4 = {k: v for k, v in e4.items() if v['n'] >= 5}
    names = sorted(valid_e4, key=lambda x: -valid_e4[x]['sharpe'])
    sharpes = [valid_e4[n]['sharpe'] for n in names]
    ns_e4 = [valid_e4[n]['n'] for n in names]
    bars = ax.barh(range(len(names)), sharpes, color='#2c3e50', alpha=0.8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.axvline(0, color='#333', lw=0.8)
    ax.set_title('E4: Sharpe by Asset Type'); ax.set_xlabel('Sharpe')
    ax.invert_yaxis()
    for i, (s, n) in enumerate(zip(sharpes, ns_e4)):
        ax.text(max(s, 0) + 0.005, i, f'{s:.3f} (n={n})', va='center', fontsize=7)

    # E9
    ax = axes[1]
    valid_e9 = {k: v for k, v in e9.items() if v['n'] >= 5}
    names = sorted(valid_e9, key=lambda x: -valid_e9[x]['sharpe'])[:12]
    sharpes = [valid_e9[n]['sharpe'] for n in names]
    ns_e9 = [valid_e9[n]['n'] for n in names]
    bars = ax.barh(range(len(names)), sharpes, color='#9b59b6', alpha=0.8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.axvline(0, color='#333', lw=0.8)
    ax.set_title('E9: Sharpe by Sector'); ax.set_xlabel('Sharpe')
    ax.invert_yaxis()
    for i, (s, n) in enumerate(zip(sharpes, ns_e9)):
        ax.text(max(s, 0) + 0.005, i, f'{s:.3f} (n={n})', va='center', fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    p3 = os.path.join(BASE_DIR, 'e4_e9_type_sector.png')
    plt.savefig(p3, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {p3}")

    # ── Fig 4: E7 Intraday Pattern + E8 Gap Freshness ────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle('E7: Open Pattern & E8: Gap Freshness', fontsize=13, fontweight='bold')

    # E7
    ax = axes[0]
    valid_e7 = {k: v for k, v in e7.items() if v['n'] >= 3}
    names = sorted(valid_e7, key=lambda x: -valid_e7[x]['sharpe'])
    sharpes = [valid_e7[n]['sharpe'] for n in names]
    avgs = [valid_e7[n]['avg'] for n in names]
    x = np.arange(len(names))
    bars = ax.bar(x - 0.15, sharpes, 0.3, label='Sharpe', color='#3498db', alpha=0.8)
    ax2 = ax.twinx()
    ax2.bar(x + 0.15, avgs, 0.3, label='Avg PnL%', color='#e67e22', alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7, rotation=30)
    ax.set_ylabel('Sharpe'); ax2.set_ylabel('Avg PnL (%)')
    ax.set_title('E7: T+2 PnL by Open Pattern')
    ax.legend(loc='upper left', fontsize=7); ax2.legend(loc='upper right', fontsize=7)
    for i, n in enumerate(names):
        ax.text(i - 0.15, sharpes[i] + 0.005, f'n={valid_e7[n]["n"]}',
                ha='center', va='bottom', fontsize=6)

    # E8
    ax = axes[1]
    valid_e8 = {k: v for k, v in e8.items() if v['n'] >= 3}
    names = sorted(valid_e8, key=lambda x: -valid_e8[x]['sharpe'])
    sharpes = [valid_e8[n]['sharpe'] for n in names]
    avgs = [valid_e8[n]['avg'] for n in names]
    x = np.arange(len(names))
    bars = ax.bar(x - 0.15, sharpes, 0.3, label='Sharpe', color='#2ecc71', alpha=0.8)
    ax3 = ax.twinx()
    ax3.bar(x + 0.15, avgs, 0.3, label='Avg PnL%', color='#e74c3c', alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7, rotation=30)
    ax.set_ylabel('Sharpe'); ax3.set_ylabel('Avg PnL (%)')
    ax.set_title('E8: T+2 PnL by Gap Freshness')
    ax.legend(loc='upper left', fontsize=7); ax3.legend(loc='upper right', fontsize=7)
    for i, n in enumerate(names):
        ax.text(i - 0.15, sharpes[i] + 0.005, f'n={valid_e8[n]["n"]}',
                ha='center', va='bottom', fontsize=6)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    p4 = os.path.join(BASE_DIR, 'e7_e8_pattern_freshness.png')
    plt.savefig(p4, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {p4}")

    # ── Fig 5: E10 Combined Score comparison ──────────────────────────────
    if e10:
        fig, ax = plt.subplots(figsize=(12, 5))
        names = list(e10.keys())
        sharpes = [e10[n]['sharpe'] for n in names]
        avgs = [e10[n]['avg'] for n in names]
        x = np.arange(len(names))
        colors = ['#27ae60' if s >= e10['pure_gap']['sharpe'] else '#e74c3c' for s in sharpes]
        bars = ax.bar(x, sharpes, color=colors, alpha=0.85, edgecolor='white')
        ax.axhline(e10['pure_gap']['sharpe'], color='#999', ls='--', lw=2,
                   label=f'pure_gap baseline ({e10["pure_gap"]["sharpe"]:.3f})')
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
        ax.set_ylabel('Sharpe')
        ax.set_title('E10: Combined Signal Score — Ranking Function Comparison', fontsize=12)
        ax.legend(fontsize=9)
        for b, s, n in zip(bars, sharpes, names):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                    f'{s:.3f}\navg={e10[n]["avg"]:+.2f}%\nn={e10[n]["n"]}',
                    ha='center', va='bottom', fontsize=7)

        plt.tight_layout()
        p5 = os.path.join(BASE_DIR, 'e10_combined_score.png')
        plt.savefig(p5, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  Saved: {p5}")

    # ── Fig 6: E5 Price Band detail + E6 Vol Spike ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle('E5: Price Bands & E6: Volume Spike (Squeeze Proxy)', fontsize=13, fontweight='bold')

    # E5 bands
    ax = axes[0]
    band_results = {k: v for k, v in e5.items() if k != 'floors' and v['n'] >= 3}
    names = list(band_results.keys())
    sharpes = [band_results[n]['sharpe'] for n in names]
    avgs = [band_results[n]['avg'] for n in names]
    x = np.arange(len(names))
    bars = ax.bar(x - 0.15, sharpes, 0.3, label='Sharpe', color='#3498db', alpha=0.8)
    ax4 = ax.twinx()
    ax4.bar(x + 0.15, avgs, 0.3, label='Avg PnL%', color='#e67e22', alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel('Sharpe'); ax4.set_ylabel('Avg PnL (%)')
    ax.set_title('E5: Performance by Price Band')
    ax.legend(loc='upper left', fontsize=7); ax4.legend(loc='upper right', fontsize=7)
    for i, n in enumerate(names):
        ax.text(i - 0.15, sharpes[i] + 0.005, f'n={band_results[n]["n"]}',
                ha='center', va='bottom', fontsize=6)

    # E6 vol spike
    ax = axes[1]
    if e6:
        valid_e6 = {k: v for k, v in e6.items() if v['n'] >= 3}
        names = list(valid_e6.keys())
        sharpes = [valid_e6[n]['sharpe'] for n in names]
        bars = ax.bar(range(len(names)), sharpes, color='#e74c3c', alpha=0.8)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=7, rotation=30)
        ax.set_ylabel('Sharpe')
        ax.set_title('E6: Volume Spike (Squeeze Proxy)')
        for i, (s, n) in enumerate(zip(sharpes, names)):
            ax.text(i, s + 0.005, f'{s:.3f}\nn={valid_e6[n]["n"]}',
                    ha='center', va='bottom', fontsize=7)
    else:
        ax.text(0.5, 0.5, 'No E6 data', ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    p6 = os.path.join(BASE_DIR, 'e5_e6_price_volume.png')
    plt.savefig(p6, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {p6}")


# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
#
#  SUMMARY REPORT
#
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(trades, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10):
    t2_trades = [t for t in trades if t.get('pnl_t2') is not None]
    baseline = compute_metrics([t['pnl_t2'] for t in t2_trades], 'Baseline (T+2)')

    print(f"\n\n{'#'*80}")
    print(f"  EXPERIMENT SUMMARY — KEY FINDINGS")
    print(f"{'#'*80}")

    print(f"\n  BASELINE: {baseline['n']} trades  avg={baseline['avg']:+.2f}%  "
          f"Sharpe={baseline['sharpe']:.3f}  WR={baseline['wr']*100:.0f}%")

    # E1: best hold
    print(f"\n  E1 HOLD PERIOD:")
    for tier in e1:
        best = max(e1[tier].items(), key=lambda x: x[1]['sharpe'])
        print(f"    {tier:15s}  best={best[0]:5s}  Sharpe={best[1]['sharpe']:.3f}  "
              f"avg={best[1]['avg']:+.2f}%")

    # E2: ETF impact
    print(f"\n  E2 ETF FILTER:")
    for name, m in e2.items():
        delta = m['sharpe'] - e2['no_filter']['sharpe']
        print(f"    {name:18s}  Sharpe={m['sharpe']:.3f}  Δ={delta:+.3f}  n={m['n']}")

    # E3: best ADV
    print(f"\n  E3 LIQUIDITY:")
    best_adv = max(e3.items(), key=lambda x: x[1]['sharpe'])
    print(f"    Best: {best_adv[0]}  Sharpe={best_adv[1]['sharpe']:.3f}  n={best_adv[1]['n']}")

    # E5: best price
    if 'floors' in e5:
        best_floor = max(e5['floors'].items(), key=lambda x: x[1]['sharpe'])
        print(f"\n  E5 PRICE FLOOR:")
        print(f"    Best: {best_floor[0]}  Sharpe={best_floor[1]['sharpe']:.3f}  n={best_floor[1]['n']}")

    # E7: pattern
    print(f"\n  E7 OPEN PATTERN:")
    for pat, m in sorted(e7.items(), key=lambda x: -x[1].get('sharpe', 0)):
        if m['n'] >= 3:
            print(f"    {pat:22s}  Sharpe={m['sharpe']:.3f}  avg={m['avg']:+.2f}%  n={m['n']}")

    # E8: freshness
    print(f"\n  E8 GAP FRESHNESS:")
    for cat, m in sorted(e8.items(), key=lambda x: -x[1].get('sharpe', 0)):
        if m['n'] >= 3:
            print(f"    {cat:15s}  Sharpe={m['sharpe']:.3f}  avg={m['avg']:+.2f}%  n={m['n']}")

    # E10: best score
    if e10:
        print(f"\n  E10 COMBINED SCORE:")
        for name, m in sorted(e10.items(), key=lambda x: -x[1]['sharpe']):
            delta = m['sharpe'] - e10['pure_gap']['sharpe']
            print(f"    {name:15s}  Sharpe={m['sharpe']:.3f}  Δ={delta:+.3f}  n={m['n']}")

    # actionable recommendations
    print(f"\n  {'─'*60}")
    print(f"  ACTIONABLE FILTERS FOR LIVE DEPLOYMENT:")
    print(f"  {'─'*60}")

    recommendations = []

    # E1
    all_best = max(e1['All (>5%)'].items(), key=lambda x: x[1]['sharpe'])
    if all_best[0] != 'T+2':
        recommendations.append(f"  → Change hold period from T+2 to {all_best[0]} "
                                f"(Sharpe {e1['All (>5%)']['T+2']['sharpe']:.3f} → {all_best[1]['sharpe']:.3f})")

    # E2
    best_e2 = max(e2.items(), key=lambda x: x[1]['sharpe'])
    if best_e2[0] != 'no_filter' and best_e2[1]['sharpe'] > e2['no_filter']['sharpe'] + 0.01:
        recommendations.append(f"  → Apply '{best_e2[0]}' filter "
                                f"(Sharpe +{best_e2[1]['sharpe'] - e2['no_filter']['sharpe']:.3f})")

    # E3
    if best_adv[1]['sharpe'] > e3['no_filter']['sharpe'] + 0.01:
        recommendations.append(f"  → Require {best_adv[0]} "
                                f"(Sharpe +{best_adv[1]['sharpe'] - e3['no_filter']['sharpe']:.3f})")

    if not recommendations:
        recommendations.append("  → No filter change exceeds +0.01 Sharpe improvement.")

    for r in recommendations:
        print(r)


# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

trades, dates, timings = collect_data()

# E1-E9: analyze only the top-N trades (the "deployed" set)
top_n_trades = [t for t in trades if t.get('is_top_n', True)]
n_pool = len(trades)
n_top = len(top_n_trades)
print(f"\n  Wide pool: {n_pool} trades  |  Top-{TOP_N} deployed: {n_top}")

e1_results = run_e1(top_n_trades)
e2_results = run_e2(top_n_trades)
e3_results = run_e3(top_n_trades)
e4_results = run_e4(top_n_trades)
e5_results = run_e5(top_n_trades)
e6_results = run_e6(top_n_trades)
e7_results = run_e7(top_n_trades)
e8_results = run_e8(top_n_trades)
e9_results = run_e9(top_n_trades)
# E10: re-rank from full wide pool
e10_results = run_e10(trades, e1_results, e2_results, e3_results)

plot_all(top_n_trades, e1_results, e2_results, e3_results, e4_results, e5_results,
         e6_results, e7_results, e8_results, e9_results, e10_results, dates)

print_summary(top_n_trades, e1_results, e2_results, e3_results, e4_results, e5_results,
              e6_results, e7_results, e8_results, e9_results, e10_results)
