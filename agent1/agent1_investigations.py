# %% Agent 1 — Composite Investigations I1-I10
# Runs on the E1-E10 baseline: 217 _FLOAT tickers, TOP_N=5, both entry types.
# Tests 10 filter combinations to find the best deployable strategy.
# Single data-fetch pass (~30 min), then all investigations run on cached data.
# Extended to 2025-09-15 for I10 out-of-sample validation.
# Colab-cell friendly.

import os, sys, time, re, asyncio, copy, json
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict

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

ENABLE_LLM_SECTORS = False  # set True to enable Gemini sector labeling for I9

if ENABLE_LLM_SECTORS:
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        import subprocess; subprocess.check_call([sys.executable, "-m", "pip", "install", "google-genai", "--quiet", "--break-system-packages"])
        from google import genai
        from google.genai import types as genai_types


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

GCP_PROJECT  = "yanina-everything"
GCP_LOCATION = "us-central1"
GEMINI_MODEL = "gemini-2.5-flash"
SECTOR_BATCH_SIZE = 30

BACKTEST_START = '2025-09-15'   # extended for I10 OOS validation
BACKTEST_END   = '2026-09-12'
OOS_CUTOFF     = '2026-01-01'   # before = OOS (Q4 2025), after = in-sample

GAP_MIN_PCT = 5.0
PRICE_MIN   = 10.0
FLOAT_MIN   = 10_000_000
TOP_N       = 5
ADV_LOOKBACK = 20

PEAK_WINDOW  = 3
MAX_PEAK_BAR = 12
FALLBACK_BAR = 15

INITIAL_CAPITAL = 100_000

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "YOUR_POLYGON_API_KEY")
_BASE = "https://api.polygon.io"
_TK_RE = re.compile(r'^[A-Z]{1,5}$')
ET = pytz.timezone('America/New_York')

_EXCLUDE_TK = {'SPY','QQQ','IWM','DIA','VXX','UVXY','SQQQ','TQQQ','SPXU','SPXL',
               'SDOW','UDOW','SDS','SSO','SH','LABU','LABD','SOXL','SOXS',
               'FNGU','FNGD','NUGT','DUST','JNUG','JDST','TNA','TZA',
               'UPRO','VIXY','SVXY','TVIX'}
_EXCLUDE_SUFFIX = ('W', 'WS', 'U', 'R')

_ETF_TYPES = {'ETF', 'ETV', 'ETN', 'ETP'}
_CS_TYPES = {'CS'}
_ADRC_TYPES = {'ADRC'}
_LEVERAGED_KEYWORDS = {'2X','3X','ULTRA','BULL','BEAR','INVERSE','SHORT','LONG',
                        'LEVERAGED','DIREXION','PROSHARES'}

_HOLIDAYS = {
    '2025-01-01','2025-01-20','2025-02-17','2025-04-18','2025-05-26',
    '2025-06-19','2025-07-04','2025-09-01','2025-11-27','2025-12-25',
    '2026-01-01','2026-01-19','2026-02-16','2026-04-03','2026-05-25',
    '2026-06-19','2026-07-03','2026-09-07','2026-11-26','2026-12-25',
    '2027-01-01','2027-01-18','2027-02-15','2027-03-26','2027-05-31',
    '2027-06-18','2027-07-05','2027-09-06','2027-11-25','2027-12-24',
}


# ═══════════════════════════════════════════════════════════════════════════════
# HARDCODED _FLOAT UNIVERSE (217 tickers)
# ═══════════════════════════════════════════════════════════════════════════════

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
    out, d = [], d - timedelta(days=1)
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
# ASYNC TICKER REFERENCE
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
# ASYNC MINUTE BAR FETCH
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
# ENTRY LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def find_entry(rth_bars):
    n = len(rth_bars)
    if n < PEAK_WINDOW * 2 + 1:
        return None
    for i in range(PEAK_WINDOW, min(MAX_PEAK_BAR, n - PEAK_WINDOW)):
        ch = rth_bars[i][1]['h']
        if all(rth_bars[i-j][1]['h'] < ch and rth_bars[i+j][1]['h'] < ch
               for j in range(1, PEAK_WINDOW + 1)):
            ci = i + PEAK_WINDOW
            if ci < n:
                return {'entry_price': rth_bars[ci][1]['c'],
                        'entry_time': rth_bars[ci][0],
                        'entry_type': 'local_high',
                        'entry_bar': ci, 'peak_bar': i}
    if FALLBACK_BAR is not None and FALLBACK_BAR < n:
        bar_close = rth_bars[FALLBACK_BAR][1]['c']
        return {'entry_price': bar_close,
                'entry_time': rth_bars[FALLBACK_BAR][0],
                'entry_type': 'fallback',
                'entry_bar': FALLBACK_BAR, 'peak_bar': None}
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTOR MAPPING
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

VALID_SECTORS = [
    'Technology','Healthcare','Finance','Energy','Mining','Manufacturing',
    'Consumer Discretionary','Consumer Staples','Industrials','Utilities',
    'Real Estate','Communications','Materials','Transportation','Retail',
    'Agriculture','Construction','Aerospace & Defense','Wholesale',
    'Services','Crypto/Blockchain','Cannabis',
]
_llm_sector_cache = {}

def classify_sectors_llm(tickers_with_names):
    if not tickers_with_names or not ENABLE_LLM_SECTORS:
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
            "Return ONLY a JSON object mapping ticker to sector.\n"
            '{"AAPL": "Technology", "JNJ": "Healthcare"}\n'
        )
        for attempt in range(3):
            try:
                resp = client.models.generate_content(
                    model=GEMINI_MODEL, contents=prompt,
                    config=genai_types.GenerateContentConfig(temperature=0.0, max_output_tokens=4096))
                text = resp.text or ''
                text = re.sub(r'```(?:json)?', '', text).strip()
                match = re.search(r'\{.*\}', text, re.DOTALL)
                if match:
                    parsed = json.loads(match.group())
                    for tk, sector in parsed.items():
                        all_results[tk.upper().strip()] = sector
                    break
            except Exception as e:
                if attempt < 2: time.sleep(2 * (attempt + 1))
                else: print(f"    LLM batch {batch_idx+1} failed: {e}")
        if (batch_idx + 1) % 5 == 0 or batch_idx == len(batches) - 1:
            print(f"    LLM sectors: {batch_idx+1}/{len(batches)}, {len(all_results)} labeled")
        time.sleep(0.5)
    _llm_sector_cache.update(all_results)
    return all_results


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

def loo_stable(pnl_list):
    if len(pnl_list) < 10: return True, 0
    base_avg = np.mean(pnl_list)
    base_sign = base_avg > 0
    flips = sum(1 for i in range(len(pnl_list))
                if (np.mean(pnl_list[:i] + pnl_list[i+1:]) > 0) != base_sign)
    return flips <= max(1, len(pnl_list) * 0.05), flips


# ═══════════════════════════════════════════════════════════════════════════════
# INVESTIGATION FRAMEWORK
# ═══════════════════════════════════════════════════════════════════════════════

def investigate(all_trades, filters, label, top_n=5, pnl_key='pnl_t2'):
    filtered = list(all_trades)
    funnel = [('all_trades', len(filtered))]
    for fname, fn in filters:
        filtered = [t for t in filtered if fn(t)]
        funnel.append((fname, len(filtered)))

    by_date = defaultdict(list)
    for t in filtered:
        by_date[t['scan_date']].append(t)

    selected = []
    for dt in sorted(by_date):
        day = sorted(by_date[dt], key=lambda x: -x['gap_pct'])[:top_n]
        selected.extend(day)

    pnls = [t[pnl_key] for t in selected if t.get(pnl_key) is not None]
    metrics = compute_metrics(pnls, label)
    active_days = sum(1 for d in by_date if len(by_date[d]) > 0)
    tpd = len(selected) / active_days if active_days > 0 else 0

    return {
        'label': label, 'metrics': metrics, 'trades': selected,
        'funnel': funnel, 'active_days': active_days, 'tpd': tpd,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
#
#  DATA COLLECTION (single pass)
#
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

def collect_data():
    t_total = time.time()
    timings = {}

    print(f"\n{'='*80}")
    print(f"  AGENT 1 — INVESTIGATIONS I1-I10 — DATA COLLECTION")
    print(f"  {BACKTEST_START} → {BACKTEST_END}  (OOS cutoff: {OOS_CUTOFF})")
    print(f"  Universe: _FLOAT ({len(_FLOAT)} tickers), gap>={GAP_MIN_PCT}%, px>=${PRICE_MIN}")
    print(f"  TOP_N={TOP_N}, both entry types, T+2 baseline hold")
    print(f"{'='*80}")

    d = datetime.strptime(BACKTEST_START, '%Y-%m-%d').date()
    end = datetime.strptime(BACKTEST_END, '%Y-%m-%d').date()
    dates = []
    while d <= end:
        if _is_td(d): dates.append(d.strftime('%Y-%m-%d'))
        d += timedelta(days=1)
    n_dates = len(dates)
    print(f"\n  {n_dates} trading days ({BACKTEST_START} → {BACKTEST_END})")

    # ── ADV lookback prefetch ──
    t1 = time.time()
    adv_dates = _prior_n_td(BACKTEST_START, ADV_LOOKBACK)
    adv_date_strs = [d.strftime('%Y-%m-%d') for d in adv_dates]
    print(f"\n  Phase 1a: Fetching {len(adv_date_strs)} ADV lookback dates...")
    for dt in adv_date_strs:
        fetch_grouped_daily(dt)
    timings['adv'] = time.time() - t1

    # ── Scan all days ──
    t2 = time.time()
    print(f"\n  Phase 1b: Scanning {n_dates} days for gap-ups in _FLOAT universe...")
    all_candidates = {}
    all_tickers = set()
    total_cands = 0
    gap_min = GAP_MIN_PCT / 100.0

    for i, dt in enumerate(dates):
        prior_str = _prior_td(dt).strftime('%Y-%m-%d')
        prior = fetch_grouped_daily(prior_str)
        today = fetch_grouped_daily(dt)

        cands = []
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
            if tk not in _FLOAT: continue
            if _FLOAT[tk] < FLOAT_MIN: continue
            cands.append({
                'ticker': tk, 'gap_pct': gap, 'prior_close': prev_c,
                'open_price': cur_p, 'scan_date': dt,
                'volume_today': today[tk]['v'],
                'in_float_universe': tk in _FLOAT,
            })
            all_tickers.add(tk)

        cands.sort(key=lambda x: -x['gap_pct'])
        all_candidates[dt] = cands
        total_cands += len(cands)

        if (i+1) % 30 == 0 or i == n_dates - 1:
            print(f"    [{i+1:>3}/{n_dates}] {dt}  |  {total_cands} total  |  {time.time()-t_total:.0f}s")

    timings['scan'] = time.time() - t2
    print(f"  → {total_cands} candidates, {len(all_tickers)} unique tickers")

    # ── Fetch ticker references ──
    t3 = time.time()
    print(f"\n  Phase 2: Fetching reference data for {len(all_tickers)} tickers...")
    tk_list = sorted(all_tickers)
    for c_start in range(0, len(tk_list), 200):
        chunk = tk_list[c_start:c_start + 200]
        fetch_ticker_refs(chunk)
        done = min(c_start + 200, len(tk_list))
        if done % 1000 == 0 or done == len(tk_list):
            print(f"    {done}/{len(tk_list)}  |  {time.time()-t_total:.0f}s")
    timings['refs'] = time.time() - t3

    # ── LLM sector labeling (optional) ──
    t3b = time.time()
    needs_sector = []
    for tk in tk_list:
        ref = _ticker_ref.get(tk, {})
        if sic_to_sector(ref.get('sic_code')) == 'Unknown' and ref.get('name'):
            needs_sector.append((tk, ref['name']))
    if needs_sector and ENABLE_LLM_SECTORS:
        print(f"\n  Phase 2b: LLM sector labeling for {len(needs_sector)} tickers...")
        classify_sectors_llm(needs_sector)
    timings['llm'] = time.time() - t3b

    # ── Fetch minute bars ──
    t4 = time.time()
    all_pairs = set()
    for picks in all_candidates.values():
        for p in picks:
            all_pairs.add((p['ticker'], p['scan_date']))

    print(f"\n  Phase 3: Fetching 1-min bars for {len(all_pairs)} pairs...")
    pairs_list = list(all_pairs)
    all_bars = {}
    for c_start in range(0, len(pairs_list), 100):
        chunk = pairs_list[c_start:c_start + 100]
        result = fetch_minute_bars_batch(chunk)
        all_bars.update(result)
        done = min(c_start + 100, len(pairs_list))
        if done % 500 == 0 or done == len(pairs_list):
            print(f"    {done}/{len(pairs_list)}  |  {time.time()-t_total:.0f}s")
    timings['bars'] = time.time() - t4
    print(f"  → Got bars for {len(all_bars)}/{len(all_pairs)} pairs")

    # ── Prefetch outcome dates (T+5) ──
    t5 = time.time()
    print(f"\n  Phase 4: Fetching outcome dates (T+0 through T+5)...")
    outcome_dates = set()
    for picks in all_candidates.values():
        for p in picks:
            outcome_dates.add(p['scan_date'])
            nxt = _next_n_td(p['scan_date'], 5)
            for nd in nxt:
                outcome_dates.add(nd.strftime('%Y-%m-%d'))
    to_fetch = sorted(od for od in outcome_dates if od not in _gd_cache)
    for j, od in enumerate(to_fetch):
        try: fetch_grouped_daily(od)
        except: pass
        if (j+1) % 30 == 0 or j == len(to_fetch) - 1:
            print(f"    {j+1}/{len(to_fetch)} outcome dates  |  {time.time()-t_total:.0f}s")
    timings['outcomes'] = time.time() - t5

    # ── Build enriched trade list ──
    t6 = time.time()
    print(f"\n  Phase 5: Enriching trades...")
    all_trades = []
    n_no_bars = 0
    n_no_entry = 0

    for dt in dates:
        for c in all_candidates.get(dt, []):
            tk = c['ticker']
            key = (tk, dt)
            bar_data = all_bars.get(key)
            if bar_data is None:
                n_no_bars += 1
                continue

            rth_bars = bar_data.get('rth', [])
            pm_bars = bar_data.get('pm', [])

            entry = find_entry(rth_bars)
            if entry is None:
                n_no_entry += 1
                continue

            tc = copy.deepcopy(c)
            tc['entry_price'] = entry['entry_price']
            tc['entry_time'] = entry['entry_time']
            tc['entry_type'] = entry['entry_type']
            tc['entry_bar'] = entry['entry_bar']
            tc['fallback_drop_pct'] = ((c['open_price'] - entry['entry_price']) / c['open_price']
                                       if entry['entry_type'] == 'fallback' and c['open_price'] > 0 else 0)

            ref = _ticker_ref.get(tk, {})
            tc['asset_type'] = ref.get('type', 'UNK')
            tc['asset_name'] = ref.get('name', '')
            tc['is_cs'] = tc['asset_type'] in _CS_TYPES
            tc['is_etf'] = tc['asset_type'] in _ETF_TYPES
            tc['is_adrc'] = tc['asset_type'] in _ADRC_TYPES
            tc['is_leveraged'] = any(kw in tc['asset_name'].upper() for kw in _LEVERAGED_KEYWORDS)
            tc['float_shares'] = _FLOAT.get(tk)

            sic_sector = sic_to_sector(ref.get('sic_code'))
            if sic_sector == 'Unknown' and tk in _llm_sector_cache:
                tc['sector'] = _llm_sector_cache[tk]
            else:
                tc['sector'] = sic_sector

            # ── Multi-period outcomes ──
            ep = tc['entry_price']
            nxt = _next_n_td(dt, 5)
            eod_data = _gd_cache.get(dt, {}).get(tk, {})
            tc['pnl_eod'] = round((ep - eod_data['c']) / ep * 100, 3) if eod_data.get('c') else None
            for k in range(5):
                col = f'pnl_t{k+1}'
                if k < len(nxt):
                    nd_str = nxt[k].strftime('%Y-%m-%d')
                    dd = _gd_cache.get(nd_str, {}).get(tk, {})
                    tc[col] = round((ep - dd['c']) / ep * 100, 3) if dd.get('c') else None
                else:
                    tc[col] = None

            # ── 10:30 AM price (bar 60 = 9:30 + 60 min) ──
            if len(rth_bars) > 60:
                tc['price_1030'] = rth_bars[60][1]['c']
            else:
                tc['price_1030'] = None

            # ── Volume ratio (gap day vol / 20-day ADV) ──
            prior_dates = _prior_n_td(dt, ADV_LOOKBACK)
            vols = []
            for pd_d in prior_dates:
                pd_str = pd_d.strftime('%Y-%m-%d')
                dd = _gd_cache.get(pd_str, {}).get(tk, {})
                if dd.get('v') and dd['v'] > 0:
                    vols.append(dd['v'])
            adv = np.mean(vols) if vols else None
            tc['vol_ratio'] = c['volume_today'] / adv if adv and adv > 0 else None

            # ── PM retrace ──
            if pm_bars:
                pm_high = max(b[1]['h'] for b in pm_bars)
                pm_last = pm_bars[-1][1]['c']
                denom = pm_high - c['open_price']
                tc['pm_retrace'] = (pm_high - pm_last) / denom if denom > 0 else 0.0
            else:
                tc['pm_retrace'] = None

            # ── Open pattern (E7) ──
            if len(rth_bars) >= 60:
                low_30m = min(b[1]['l'] for b in rth_bars[:30])
                close_60m = rth_bars[59][1]['c']
                open_p = rth_bars[0][1]['o']
                drop = (low_30m - open_p) / open_p if open_p > 0 else 0
                rec_d = abs(open_p - low_30m)
                recovery = (close_60m - low_30m) / rec_d if rec_d > 0 else 0
                if drop < -0.03 and recovery > 0.5:
                    tc['open_pattern'] = 'drop_then_recover'
                elif drop < -0.03 and recovery <= 0.5:
                    tc['open_pattern'] = 'sustained_drop'
                elif drop >= -0.01:
                    tc['open_pattern'] = 'flat_or_bounce'
                else:
                    tc['open_pattern'] = 'mild_drop'
            else:
                tc['open_pattern'] = 'no_data'

            # ── Gap freshness (E8) ──
            if pm_bars:
                pm_high = max(b[1]['h'] for b in pm_bars)
                for b in pm_bars:
                    if b[1]['h'] >= pm_high * 0.98:
                        h, m = int(b[0][:2]), int(b[0][3:5])
                        mins = (9 * 60 + 30) - (h * 60 + m)
                        if mins <= 30: tc['gap_freshness'] = 'very_fresh'
                        elif mins <= 120: tc['gap_freshness'] = 'fresh'
                        elif mins <= 300: tc['gap_freshness'] = 'moderate'
                        else: tc['gap_freshness'] = 'stale'
                        break
                else:
                    tc['gap_freshness'] = 'unknown'
            else:
                tc['gap_freshness'] = 'no_pm'

            tc['rth_bars'] = rth_bars

            all_trades.append(tc)

    timings['enrich'] = time.time() - t6
    elapsed = time.time() - t_total

    print(f"\n  → {len(all_trades)} enriched trades  |  {n_no_entry} no entry  |  {n_no_bars} no bars")
    print(f"\n  {'─'*50}")
    for step, dur in [('ADV prefetch', timings['adv']), ('Scan', timings['scan']),
                      ('Refs', timings['refs']), ('LLM sectors', timings['llm']),
                      ('Bars', timings['bars']), ('Outcomes', timings['outcomes']),
                      ('Enrich', timings['enrich'])]:
        print(f"    {step:<20s}  {dur:>6.0f}s  ({dur/elapsed*100:>4.0f}%)")
    print(f"  TOTAL: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    return all_trades, dates


# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
#
#  INVESTIGATIONS I1-I10
#
#  Each returns a list of result dicts for the consolidated report.
#
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

# ── I1-I3 helpers ──

def _loo_stability(pnls):
    if len(pnls) < 3:
        return {'stable': True, 'n': len(pnls), 'flips': 0, 'flip_pct': 0.0}
    full_avg = np.mean(pnls)
    flips = 0
    for i in range(len(pnls)):
        loo_avg = np.mean(pnls[:i] + pnls[i+1:])
        if (loo_avg > 0) != (full_avg > 0):
            flips += 1
    return {'stable': flips == 0, 'n': len(pnls), 'flips': flips,
            'flip_pct': round(100.0 * flips / len(pnls), 2)}

def _make_result(id_, title, inv_result, top_n=5, pnl_key='pnl_t2', extra=None, summary=''):
    pnls = [t[pnl_key] for t in inv_result['trades'] if t.get(pnl_key) is not None]
    loo = _loo_stability(pnls)
    return {
        'id': id_, 'title': title, 'metrics': inv_result['metrics'],
        'funnel': inv_result['funnel'], 'active_days': inv_result['active_days'],
        'tpd': inv_result['tpd'], 'top_n': top_n, 'pnl_key': pnl_key,
        'extra': {**(extra or {}), 'loo': loo}, 'summary': summary,
    }

# ── I4-I6 helpers ──

def _loo_sign_flips(pnls):
    if not pnls or len(pnls) < 2: return 0
    total = sum(pnls); n = len(pnls)
    base_sign = np.sign(total / n)
    return sum(1 for p in pnls if np.sign((total - p) / (n - 1)) != base_sign)

def _max_open_positions(trades, dates, date_idx, hold_days):
    open_by_date = defaultdict(int)
    for t in trades:
        sd = t['scan_date']
        if sd not in date_idx: continue
        si = date_idx[sd]
        for offset in range(hold_days + 1):
            idx = si + offset
            if idx < len(dates): open_by_date[dates[idx]] += 1
    max_open = max(open_by_date.values()) if open_by_date else 0
    return max_open, dict(open_by_date)


# ═══════════════════════════════════════════════════════════════════════════════
# I1: CS_ONLY Standalone
# ═══════════════════════════════════════════════════════════════════════════════

def run_i1(all_trades, dates):
    print("\n=== I1: CS_ONLY Standalone ===")
    results = []

    cs_tickers = [t for t, info in _ticker_ref.items() if info.get('type') == 'CS']
    non_cs_tickers = [t for t, info in _ticker_ref.items() if info.get('type') != 'CS']
    print(f"  _FLOAT universe: {len(_ticker_ref)} tickers, {len(cs_tickers)} CS, {len(non_cs_tickers)} non-CS")

    baseline = investigate(all_trades, [], 'Baseline', top_n=5, pnl_key='pnl_t2')
    print(f"  Baseline: n={baseline['metrics']['n']}, avg={baseline['metrics']['avg']:.2f}%")

    baseline_cs = [t for t in baseline['trades'] if t.get('is_cs')]
    baseline_non_cs = [t for t in baseline['trades'] if not t.get('is_cs')]
    cs_frac = len(baseline_cs) / max(len(baseline['trades']), 1)
    print(f"  Baseline trades: {len(baseline_cs)} CS ({cs_frac:.1%}), {len(baseline_non_cs)} non-CS")

    type_counts = defaultdict(int)
    for t in baseline_non_cs:
        type_counts[t.get('asset_type', 'unknown')] += 1

    cs_pnls = [t['pnl_t2'] for t in baseline_cs if t.get('pnl_t2') is not None]
    non_cs_pnls = [t['pnl_t2'] for t in baseline_non_cs if t.get('pnl_t2') is not None]
    cs_baseline_metrics = compute_metrics(cs_pnls, 'CS subset of baseline')
    non_cs_baseline_metrics = compute_metrics(non_cs_pnls, 'Non-CS subset of baseline')

    filters_cs = [('CS_ONLY', lambda t: t.get('is_cs', False))]
    inv_cs = investigate(all_trades, filters_cs, 'I1: CS_ONLY', top_n=5, pnl_key='pnl_t2')
    print(f"  I1 CS_ONLY (re-ranked): n={inv_cs['metrics']['n']}, avg={inv_cs['metrics']['avg']:.2f}%, "
          f"sharpe={inv_cs['metrics']['sharpe']:.3f}")

    results.append(_make_result(
        'I1', 'CS_ONLY (re-ranked top-5)', inv_cs,
        extra={'universe_cs': len(cs_tickers), 'universe_non_cs': len(non_cs_tickers),
               'baseline_cs_frac': round(cs_frac, 4), 'non_cs_type_breakdown': dict(type_counts),
               'cs_subset_metrics': cs_baseline_metrics, 'non_cs_subset_metrics': non_cs_baseline_metrics},
        summary=(f"CS-only universe ({len(cs_tickers)}/{len(_ticker_ref)} tickers). "
                 f"Re-ranked: avg {inv_cs['metrics']['avg']:.2f}% vs baseline {baseline['metrics']['avg']:.2f}%."),
    ))
    results.append(_make_result('I1a', 'Baseline (reference)', baseline,
        summary=f"Baseline reference: {baseline['metrics']['n']} trades, avg {baseline['metrics']['avg']:.2f}%."))
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# I2: CS + sustained_drop
# ═══════════════════════════════════════════════════════════════════════════════

def run_i2(all_trades, dates):
    print("\n=== I2: CS + sustained_drop ===")
    results = []

    all_cs = [t for t in all_trades if t.get('is_cs')]
    all_sd = [t for t in all_trades if t.get('open_pattern') == 'sustained_drop']
    all_cs_sd = [t for t in all_trades if t.get('is_cs') and t.get('open_pattern') == 'sustained_drop']
    sd_that_are_cs = len(all_cs_sd) / max(len(all_sd), 1)
    cs_that_have_sd = len(all_cs_sd) / max(len(all_cs), 1)
    print(f"  Pre-ranking pool: {len(all_trades)} total, {len(all_cs)} CS, "
          f"{len(all_sd)} sustained_drop, {len(all_cs_sd)} CS+SD")

    filters_cs_sd = [('CS_ONLY', lambda t: t.get('is_cs', False)),
                     ('sustained_drop', lambda t: t.get('open_pattern') == 'sustained_drop')]
    inv_cs_sd = investigate(all_trades, filters_cs_sd, 'I2: CS+SD', top_n=5, pnl_key='pnl_t2')
    print(f"  I2 CS+SD: n={inv_cs_sd['metrics']['n']}, avg={inv_cs_sd['metrics']['avg']:.2f}%, "
          f"sharpe={inv_cs_sd['metrics']['sharpe']:.3f}")
    results.append(_make_result('I2', 'CS + sustained_drop', inv_cs_sd,
        extra={'pool_cs': len(all_cs), 'pool_sd': len(all_sd), 'pool_cs_sd': len(all_cs_sd),
               'sd_are_cs_pct': round(sd_that_are_cs, 4), 'cs_have_sd_pct': round(cs_that_have_sd, 4)},
        summary=f"CS+sustained_drop: avg {inv_cs_sd['metrics']['avg']:.2f}%, n={inv_cs_sd['metrics']['n']}."))

    filters_sd = [('sustained_drop', lambda t: t.get('open_pattern') == 'sustained_drop')]
    inv_sd = investigate(all_trades, filters_sd, 'I2a: SD only', top_n=5, pnl_key='pnl_t2')
    results.append(_make_result('I2a', 'sustained_drop only (no CS)', inv_sd,
        summary=f"SD-only: avg {inv_sd['metrics']['avg']:.2f}%, n={inv_sd['metrics']['n']}."))

    filters_cs = [('CS_ONLY', lambda t: t.get('is_cs', False))]
    inv_cs = investigate(all_trades, filters_cs, 'I2b: CS only', top_n=5, pnl_key='pnl_t2')
    results.append(_make_result('I2b', 'CS only (reference)', inv_cs,
        summary=f"CS-only reference: avg {inv_cs['metrics']['avg']:.2f}%, n={inv_cs['metrics']['n']}."))

    # I2c: Intraday management
    print("  Computing I2c: Intraday management...")
    inv_cs_all = investigate(all_trades, filters_cs, 'CS pool for intraday', top_n=5, pnl_key='pnl_t2')
    intraday_pnls = []
    intraday_actions = defaultdict(int)
    intraday_pnl_by_action = defaultdict(list)

    for t in inv_cs_all['trades']:
        pattern = t.get('open_pattern', 'no_data')
        entry_p = t.get('entry_price')
        if entry_p is None or entry_p <= 0: continue

        if pattern == 'sustained_drop':
            pnl = t.get('pnl_t2')
            action = 'hold_t2 (sustained_drop)'
        elif pattern == 'drop_then_recover':
            p1030 = t.get('price_1030')
            if p1030 is not None and p1030 > 0:
                pnl = (entry_p - p1030) / entry_p * 100.0
                action = 'close_1030 (drop_recover)'
            else:
                pnl = t.get('pnl_t2')
                action = 'hold_t2 (drop_recover, no 1030)'
        else:
            pnl = t.get('pnl_t2')
            action = f'hold_t2 ({pattern})'

        if pnl is not None:
            intraday_pnls.append(pnl)
            intraday_actions[action] += 1
            intraday_pnl_by_action[action].append(pnl)

    intraday_metrics = compute_metrics(intraday_pnls, 'I2c: Intraday Mgmt')
    action_summary = {}
    for action, pnls in intraday_pnl_by_action.items():
        action_summary[action] = {'n': len(pnls),
                                  'avg': round(np.mean(pnls), 4) if pnls else 0,
                                  'med': round(np.median(pnls), 4) if pnls else 0}
    print(f"  I2c Intraday: n={intraday_metrics['n']}, avg={intraday_metrics['avg']:.2f}%, "
          f"sharpe={intraday_metrics['sharpe']:.3f}")

    results.append({
        'id': 'I2c', 'title': 'Intraday management: CS enter at local_high, manage by pattern',
        'metrics': intraday_metrics, 'funnel': inv_cs_all['funnel'],
        'active_days': inv_cs_all['active_days'], 'tpd': inv_cs_all['tpd'],
        'top_n': 5, 'pnl_key': 'blended',
        'extra': {'action_breakdown': action_summary, 'loo': _loo_stability(intraday_pnls)},
        'summary': f"Intraday mgmt: blended avg {intraday_metrics['avg']:.2f}%, n={intraday_metrics['n']}.",
    })
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# I3: CS + Gap Freshness
# ═══════════════════════════════════════════════════════════════════════════════

def run_i3(all_trades, dates):
    print("\n=== I3: CS + Gap Freshness ===")
    results = []

    filters_cs = [('CS_ONLY', lambda t: t.get('is_cs', False))]
    inv_cs_base = investigate(all_trades, filters_cs, 'CS baseline', top_n=5, pnl_key='pnl_t2')
    print(f"  CS baseline: n={inv_cs_base['metrics']['n']}, avg={inv_cs_base['metrics']['avg']:.2f}%")

    cs_trades = [t for t in all_trades if t.get('is_cs')]
    freshness_dist = defaultdict(int)
    for t in cs_trades:
        freshness_dist[t.get('gap_freshness', 'unknown')] += 1
    print(f"  CS freshness distribution: {dict(freshness_dist)}")

    combos = [
        ('I3a', 'CS + stale only',
         [('CS_ONLY', lambda t: t.get('is_cs', False)),
          ('stale', lambda t: t.get('gap_freshness') == 'stale')]),
        ('I3b', 'CS + stale+moderate',
         [('CS_ONLY', lambda t: t.get('is_cs', False)),
          ('stale_or_moderate', lambda t: t.get('gap_freshness') in ('stale', 'moderate'))]),
        ('I3c', 'CS + not very_fresh',
         [('CS_ONLY', lambda t: t.get('is_cs', False)),
          ('not_very_fresh', lambda t: t.get('gap_freshness') != 'very_fresh')]),
        ('I3d', 'CS + stale+moderate+fresh',
         [('CS_ONLY', lambda t: t.get('is_cs', False)),
          ('not_very_fresh_or_no_pm', lambda t: t.get('gap_freshness') in ('stale', 'moderate', 'fresh'))]),
    ]

    for id_, title, filters in combos:
        inv = investigate(all_trades, filters, title, top_n=5, pnl_key='pnl_t2')
        impractical = inv['tpd'] < 1.5
        delta_avg = inv['metrics']['avg'] - inv_cs_base['metrics']['avg']
        delta_sharpe = inv['metrics']['sharpe'] - inv_cs_base['metrics']['sharpe']
        flag = " *** IMPRACTICAL ***" if impractical else ""
        print(f"  {id_} {title}: n={inv['metrics']['n']}, avg={inv['metrics']['avg']:.2f}%, "
              f"tpd={inv['tpd']:.2f}{flag}")
        results.append(_make_result(id_, title, inv,
            extra={'impractical': impractical,
                   'vs_cs_baseline': {'delta_avg': round(delta_avg, 4), 'delta_sharpe': round(delta_sharpe, 4)},
                   'freshness_distribution': dict(freshness_dist)},
            summary=f"{title}: avg {inv['metrics']['avg']:.2f}% (delta {delta_avg:+.2f}% vs CS-only), "
                    f"tpd={inv['tpd']:.2f}{'  IMPRACTICAL' if impractical else ''}."))

    results.append(_make_result('I3e', 'CS-only baseline (reference)', inv_cs_base,
        extra={'freshness_distribution': dict(freshness_dist)},
        summary=f"CS-only reference: avg {inv_cs_base['metrics']['avg']:.2f}%, n={inv_cs_base['metrics']['n']}."))
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# I4: CS_ONLY + Price Floor
# ═══════════════════════════════════════════════════════════════════════════════

def run_i4(all_trades, dates):
    print("\n=== I4: CS_ONLY + Price Floor ===")
    results = []
    price_floors = [30, 50, 100]

    cs_base = investigate(all_trades, [('CS_ONLY', lambda t: t['is_cs'])],
                          'I4_baseline_CS', top_n=5, pnl_key='pnl_t2')
    print(f"  CS baseline: n={cs_base['metrics']['n']}, avg={cs_base['metrics']['avg']:.2f}%, "
          f"Sharpe={cs_base['metrics']['sharpe']:.3f}")

    for pmin in price_floors:
        tag = f'I4_CS_MIN${pmin}'
        res = investigate(all_trades,
            [('CS_ONLY', lambda t: t['is_cs']),
             (f'PRICE>=${pmin}', lambda t, pm=pmin: t['open_price'] >= pm)],
            tag, top_n=5, pnl_key='pnl_t2')
        m = res['metrics']

        cs_trades = [t for t in all_trades if t['is_cs']]
        excluded = [t for t in cs_trades if t['open_price'] < pmin and t['pnl_t2'] is not None]
        excl_pnls = [t['pnl_t2'] for t in excluded]
        excl_avg = np.mean(excl_pnls) if excl_pnls else float('nan')
        practical = res['tpd'] >= 1.5
        flag = "" if practical else " ** IMPRACTICAL"
        print(f"  {tag}: n={m['n']}, avg={m['avg']:.2f}%, Sharpe={m['sharpe']:.3f}, "
              f"tpd={res['tpd']:.2f}{flag}")

        pnls = [t['pnl_t2'] for t in res['trades'] if t['pnl_t2'] is not None]
        results.append({
            'id': f'I4_{pmin}', 'title': f'CS + Price >= ${pmin}',
            'metrics': m, 'funnel': res['funnel'], 'active_days': res['active_days'],
            'tpd': res['tpd'], 'top_n': 5, 'pnl_key': 'pnl_t2',
            'extra': {'excluded_n': len(excl_pnls),
                      'excluded_avg_pnl': round(excl_avg, 4) if not np.isnan(excl_avg) else None,
                      'practical': practical, 'loo_sign_flips': _loo_sign_flips(pnls)},
            'summary': f"CS + price>=${pmin}: avg {m['avg']:.2f}% (n={m['n']}), "
                       f"Sharpe {m['sharpe']:.3f}, tpd {res['tpd']:.2f}."})
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# I5: CS_ONLY + Extended Hold
# ═══════════════════════════════════════════════════════════════════════════════

def run_i5(all_trades, dates):
    print("\n=== I5: CS_ONLY + Extended Hold ===")
    results = []
    date_idx = {d: i for i, d in enumerate(dates)}

    configs = [(2,'pnl_t2',5,''), (3,'pnl_t3',5,''), (4,'pnl_t4',5,''),
               (5,'pnl_t5',5,''), (4,'pnl_t4',2,'_TOP2')]

    for hold_days, pnl_key, top_n, suffix in configs:
        tag = f'I5_CS_T+{hold_days}_N{top_n}{suffix}'
        res = investigate(all_trades, [('CS_ONLY', lambda t: t['is_cs'])],
                          tag, top_n=top_n, pnl_key=pnl_key)
        m = res['metrics']
        max_open, _ = _max_open_positions(res['trades'], dates, date_idx, hold_days)
        capital_per_pos = 100_000 / max_open if max_open > 0 else 0
        total_capital = max_open * 20_000
        print(f"  {tag}: n={m['n']}, avg={m['avg']:.2f}%, Sharpe={m['sharpe']:.3f}, "
              f"max_open={max_open}")

        pnls = [t[pnl_key] for t in res['trades'] if t.get(pnl_key) is not None]
        results.append({
            'id': f'I5_T{hold_days}_N{top_n}', 'title': f'CS T+{hold_days} hold, TOP_N={top_n}',
            'metrics': m, 'funnel': res['funnel'], 'active_days': res['active_days'],
            'tpd': res['tpd'], 'top_n': top_n, 'pnl_key': pnl_key,
            'extra': {'hold_days': hold_days, 'max_open_positions': max_open,
                      'capital_at_20k_per_pos': total_capital,
                      'capital_per_pos_at_100k': round(capital_per_pos, 2),
                      'loo_sign_flips': _loo_sign_flips(pnls)},
            'summary': f"CS T+{hold_days} N={top_n}: avg {m['avg']:.2f}%, Sharpe {m['sharpe']:.3f}. "
                       f"Max {max_open} simultaneous positions (${total_capital:,.0f} at $20K/pos)."})
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# I6: CS_ONLY + Volume Ratio
# ═══════════════════════════════════════════════════════════════════════════════

def run_i6(all_trades, dates):
    print("\n=== I6: CS_ONLY + Volume Ratio ===")
    results = []

    cs_with_vol = [t for t in all_trades
                   if t['is_cs'] and t.get('vol_ratio') is not None and t['pnl_t2'] is not None]
    if len(cs_with_vol) >= 10:
        vr = np.array([t['vol_ratio'] for t in cs_with_vol])
        pnl = np.array([t['pnl_t2'] for t in cs_with_vol])
        corr = np.corrcoef(vr, pnl)[0, 1]
    else:
        corr = float('nan')
    print(f"  CS vol_ratio vs T+2 PnL correlation: {corr:.4f} (n={len(cs_with_vol)})")

    vol_caps = [1.5, 2.0, 3.0]
    for vc in vol_caps:
        tag = f'I6_CS_VOL<{vc}x'
        res = investigate(all_trades,
            [('CS_ONLY', lambda t: t['is_cs']),
             (f'vol_ratio<{vc}x_or_None',
              lambda t, v=vc: t.get('vol_ratio') is None or t['vol_ratio'] < v)],
            tag, top_n=5, pnl_key='pnl_t2')
        m = res['metrics']
        practical = res['tpd'] >= 1.5

        none_trades = [t for t in res['trades'] if t.get('vol_ratio') is None and t['pnl_t2'] is not None]
        actual_low = [t for t in res['trades'] if t.get('vol_ratio') is not None and t['pnl_t2'] is not None]
        none_avg = np.mean([t['pnl_t2'] for t in none_trades]) if none_trades else float('nan')
        low_avg = np.mean([t['pnl_t2'] for t in actual_low]) if actual_low else float('nan')

        print(f"  {tag}: n={m['n']}, avg={m['avg']:.2f}%, Sharpe={m['sharpe']:.3f}, tpd={res['tpd']:.2f}")
        pnls = [t['pnl_t2'] for t in res['trades'] if t['pnl_t2'] is not None]
        results.append({
            'id': f'I6_VOL_LT{vc}', 'title': f'CS + vol_ratio < {vc}x (None incl)',
            'metrics': m, 'funnel': res['funnel'], 'active_days': res['active_days'],
            'tpd': res['tpd'], 'top_n': 5, 'pnl_key': 'pnl_t2',
            'extra': {'vol_cap': vc, 'practical': practical,
                      'none_n': len(none_trades), 'none_avg_pnl': round(none_avg, 4) if not np.isnan(none_avg) else None,
                      'low_vol_n': len(actual_low), 'low_vol_avg_pnl': round(low_avg, 4) if not np.isnan(low_avg) else None,
                      'vol_pnl_corr': round(corr, 4) if not np.isnan(corr) else None,
                      'loo_sign_flips': _loo_sign_flips(pnls)},
            'summary': f"CS + vol<{vc}x: avg {m['avg']:.2f}% (n={m['n']}), Sharpe {m['sharpe']:.3f}."})

    # High volume: vol_ratio >= 5.0x
    res = investigate(all_trades,
        [('CS_ONLY', lambda t: t['is_cs']),
         ('vol_ratio>=5x', lambda t: t.get('vol_ratio') is not None and t['vol_ratio'] >= 5.0)],
        'I6_CS_VOL>=5x', top_n=5, pnl_key='pnl_t2')
    m = res['metrics']
    print(f"  I6_VOL_GE5: n={m['n']}, avg={m['avg']:.2f}%, Sharpe={m['sharpe']:.3f}")
    pnls = [t['pnl_t2'] for t in res['trades'] if t['pnl_t2'] is not None]
    results.append({
        'id': 'I6_VOL_GE5', 'title': 'CS + vol_ratio >= 5.0x',
        'metrics': m, 'funnel': res['funnel'], 'active_days': res['active_days'],
        'tpd': res['tpd'], 'top_n': 5, 'pnl_key': 'pnl_t2',
        'extra': {'vol_floor': 5.0, 'practical': res['tpd'] >= 1.5,
                  'vol_pnl_corr': round(corr, 4) if not np.isnan(corr) else None,
                  'loo_sign_flips': _loo_sign_flips(pnls)},
        'summary': f"CS + vol>=5x: avg {m['avg']:.2f}% (n={m['n']}), Sharpe {m['sharpe']:.3f}."})
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# I7: CS + Stale/Moderate + $30 + T+3 (Composite)
# ═══════════════════════════════════════════════════════════════════════════════

def _select_top_n(trades, top_n=5, pnl_key='pnl_t2', score_fn=None):
    by_date = defaultdict(list)
    for t in trades:
        by_date[t['scan_date']].append(t)
    selected = []
    for dt in sorted(by_date):
        day = by_date[dt]
        if score_fn:
            day.sort(key=score_fn, reverse=True)
        else:
            day.sort(key=lambda x: x['gap_pct'], reverse=True)
        for t in day[:top_n]:
            if t.get(pnl_key) is not None:
                selected.append(t)
    return selected

def _loo_stable(pnl_list, threshold=0.0):
    if len(pnl_list) < 10: return False, 0
    base_sign = np.mean(pnl_list) > threshold
    flips = sum(1 for i in range(len(pnl_list))
                if (np.mean(pnl_list[:i] + pnl_list[i+1:]) > threshold) != base_sign)
    return flips <= max(1, len(pnl_list) * 0.05), flips

def run_i7(all_trades, dates):
    print(f"\n\n{'='*80}")
    print(f"  I7: COMPOSITE FILTER — CS + Stale/Moderate + $30 Floor + T+3")
    print(f"{'='*80}")
    TOP_N = 5; results = []

    step0 = list(all_trades); n0 = len(step0)
    step1 = [t for t in step0 if t.get('is_cs')]
    excluded1 = [t for t in step0 if not t.get('is_cs')]
    exc1_pnls = [t['pnl_t3'] for t in excluded1 if t.get('pnl_t3') is not None]
    kept1_pnls = [t['pnl_t3'] for t in step1 if t.get('pnl_t3') is not None]
    exc1_avg = np.mean(exc1_pnls) if exc1_pnls else 0
    kept1_avg = np.mean(kept1_pnls) if kept1_pnls else 0
    harmful1 = exc1_avg > kept1_avg and len(exc1_pnls) > 5
    print(f"    Step 1 — CS only: n={len(step1)}, excluded avg={exc1_avg:+.2f}%"
          f"{'  ** HARMFUL **' if harmful1 else ''}")

    step2 = [t for t in step1 if t.get('gap_freshness') in ('stale', 'moderate')]
    excluded2 = [t for t in step1 if t.get('gap_freshness') not in ('stale', 'moderate')]
    exc2_pnls = [t['pnl_t3'] for t in excluded2 if t.get('pnl_t3') is not None]
    kept2_pnls = [t['pnl_t3'] for t in step2 if t.get('pnl_t3') is not None]
    exc2_avg = np.mean(exc2_pnls) if exc2_pnls else 0
    kept2_avg = np.mean(kept2_pnls) if kept2_pnls else 0
    harmful2 = exc2_avg > kept2_avg and len(exc2_pnls) > 5
    print(f"    Step 2 — Stale|Moderate: n={len(step2)}, excluded avg={exc2_avg:+.2f}%"
          f"{'  ** HARMFUL **' if harmful2 else ''}")

    step3 = [t for t in step2 if t.get('open_price', 0) >= 30]
    excluded3 = [t for t in step2 if t.get('open_price', 0) < 30]
    exc3_pnls = [t['pnl_t3'] for t in excluded3 if t.get('pnl_t3') is not None]
    exc3_avg = np.mean(exc3_pnls) if exc3_pnls else 0
    kept3_avg = np.mean([t['pnl_t3'] for t in step3 if t.get('pnl_t3') is not None]) if step3 else 0
    harmful3 = exc3_avg > kept3_avg and len(exc3_pnls) > 5
    print(f"    Step 3 — Price>=$30: n={len(step3)}, excluded avg={exc3_avg:+.2f}%"
          f"{'  ** HARMFUL **' if harmful3 else ''}")

    selected = _select_top_n(step3, top_n=TOP_N, pnl_key='pnl_t3')
    pnls = [t['pnl_t3'] for t in selected]
    active_days = len(set(t['scan_date'] for t in selected))
    tpd = len(selected) / active_days if active_days > 0 else 0
    m = compute_metrics(pnls, 'I7a: CS+Stale/Mod+$30+T+3')
    funnel = [('all_trades', n0), ('cs_only', len(step1)), ('stale_or_moderate', len(step2)),
              ('price_gte_30', len(step3)), ('top_n_selected', len(selected))]
    harmful_flags = []
    if harmful1: harmful_flags.append('CS filter HARMFUL')
    if harmful2: harmful_flags.append('gap_freshness filter HARMFUL')
    if harmful3: harmful_flags.append('price_floor filter HARMFUL')
    loo_s, loo_f = _loo_stable(pnls) if len(pnls) > 20 else (False, 0)
    print(f"    I7a: n={m['n']}, avg={m['avg']:+.2f}%, Sharpe={m['sharpe']:.3f}")

    results.append({'id': 'I7a', 'title': 'CS + Stale/Moderate + $30 + T+3',
        'metrics': m, 'funnel': funnel, 'active_days': active_days, 'tpd': tpd,
        'top_n': TOP_N, 'pnl_key': 'pnl_t3',
        'extra': {'harmful_filters': harmful_flags, 'loo_stable': loo_s, 'loo_flips': loo_f,
                  'funnel_excluded_avgs': {'cs': exc1_avg, 'freshness': exc2_avg, 'price': exc3_avg}},
        'summary': f"Full composite: n={m['n']}, avg={m['avg']:+.2f}%, Sharpe={m['sharpe']:.3f}. "
                   f"{'LOO stable.' if loo_s else 'LOO UNSTABLE.'}"})

    # I7b: Same composite but T+2
    selected_t2 = _select_top_n(step3, top_n=TOP_N, pnl_key='pnl_t2')
    pnls_t2 = [t['pnl_t2'] for t in selected_t2]
    m_t2 = compute_metrics(pnls_t2, 'I7b: CS+Stale/Mod+$30+T+2')
    ad_t2 = len(set(t['scan_date'] for t in selected_t2))
    tpd_t2 = len(selected_t2) / ad_t2 if ad_t2 > 0 else 0
    print(f"    I7b (T+2): n={m_t2['n']}, avg={m_t2['avg']:+.2f}%, Sharpe={m_t2['sharpe']:.3f}")
    results.append({'id': 'I7b', 'title': 'CS + Stale/Moderate + $30 + T+2',
        'metrics': m_t2, 'funnel': funnel, 'active_days': ad_t2, 'tpd': tpd_t2,
        'top_n': TOP_N, 'pnl_key': 'pnl_t2', 'extra': {},
        'summary': f"Same composite T+2: n={m_t2['n']}, avg={m_t2['avg']:+.2f}%, Sharpe={m_t2['sharpe']:.3f}."})

    # I7c: CS + stale only + T+3
    step2c = [t for t in step1 if t.get('gap_freshness') == 'stale']
    selected_c = _select_top_n(step2c, top_n=TOP_N, pnl_key='pnl_t3')
    pnls_c = [t['pnl_t3'] for t in selected_c]
    m_c = compute_metrics(pnls_c, 'I7c: CS+Stale+T+3')
    ad_c = len(set(t['scan_date'] for t in selected_c))
    tpd_c = len(selected_c) / ad_c if ad_c > 0 else 0
    print(f"    I7c (stale only): n={m_c['n']}, avg={m_c['avg']:+.2f}%, Sharpe={m_c['sharpe']:.3f}")
    results.append({'id': 'I7c', 'title': 'CS + Stale only + T+3',
        'metrics': m_c, 'funnel': [('cs_only', len(step1)), ('stale_only', len(step2c)),
                                    ('top_n_selected', len(selected_c))],
        'active_days': ad_c, 'tpd': tpd_c, 'top_n': TOP_N, 'pnl_key': 'pnl_t3', 'extra': {},
        'summary': f"Stale-only: n={m_c['n']}, avg={m_c['avg']:+.2f}%, Sharpe={m_c['sharpe']:.3f}."})
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# I8: CS + Intraday Management
# ═══════════════════════════════════════════════════════════════════════════════

def run_i8(all_trades, dates):
    print(f"\n\n{'='*80}")
    print(f"  I8: INTRADAY MANAGEMENT — Close DTR at 10:30")
    print(f"{'='*80}")
    TOP_N = 5; results = []

    cs_trades = [t for t in all_trades if t.get('is_cs')]
    by_date = defaultdict(list)
    for t in cs_trades:
        by_date[t['scan_date']].append(t)

    # I8a: Close drop_then_recover at 10:30
    held_pnls, closed_pnls, closed_cf = [], [], []
    all_managed_pnls = []
    held_trades, closed_trades = [], []
    active_days = 0

    for dt in sorted(by_date):
        day = sorted(by_date[dt], key=lambda x: -x['gap_pct'])[:TOP_N]
        valid = [t for t in day if t.get('pnl_t2') is not None]
        if not valid: continue
        active_days += 1
        for t in valid:
            pattern = t.get('open_pattern', 'no_data')
            if pattern == 'drop_then_recover':
                p1030 = t.get('price_1030')
                if p1030 is not None and p1030 > 0:
                    pnl = (t['entry_price'] - p1030) / t['entry_price'] * 100
                    closed_pnls.append(pnl); all_managed_pnls.append(pnl)
                    closed_cf.append(t['pnl_t2']); closed_trades.append(t)
                else:
                    held_pnls.append(t['pnl_t2']); all_managed_pnls.append(t['pnl_t2'])
                    held_trades.append(t)
            else:
                held_pnls.append(t['pnl_t2']); all_managed_pnls.append(t['pnl_t2'])
                held_trades.append(t)

    m_managed = compute_metrics(all_managed_pnls, 'I8a: Managed')
    m_closed = compute_metrics(closed_pnls, 'closed_at_1030')
    m_cf = compute_metrics(closed_cf, 'counterfactual_T2')
    value_of_mgmt = (np.mean(closed_pnls) - np.mean(closed_cf)) if closed_pnls and closed_cf else 0
    tpd = len(all_managed_pnls) / active_days if active_days > 0 else 0
    loo_s, loo_f = _loo_stable(all_managed_pnls) if len(all_managed_pnls) > 20 else (False, 0)
    print(f"    I8a: n={m_managed['n']}, avg={m_managed['avg']:+.2f}%, Sharpe={m_managed['sharpe']:.3f}")
    print(f"    Closed: n={m_closed['n']}, avg={m_closed['avg']:+.2f}%, value={value_of_mgmt:+.2f}%/trade")

    results.append({'id': 'I8a', 'title': 'Close DTR at 10:30, hold rest T+2',
        'metrics': m_managed,
        'funnel': [('cs_trades', len(cs_trades)), ('held', len(held_trades)), ('closed_early', len(closed_trades))],
        'active_days': active_days, 'tpd': tpd, 'top_n': TOP_N, 'pnl_key': 'pnl_t2',
        'extra': {'held_metrics': compute_metrics(held_pnls, 'held'), 'closed_metrics': m_closed,
                  'counterfactual_metrics': m_cf, 'value_of_management': value_of_mgmt,
                  'loo_stable': loo_s, 'loo_flips': loo_f},
        'summary': f"Close DTR at 10:30: avg={m_managed['avg']:+.2f}%, Sharpe={m_managed['sharpe']:.3f}. "
                   f"Management value={value_of_mgmt:+.2f}%/trade."})

    # I8b: Aggressive — close DTR + mild_drop
    close_patterns = {'drop_then_recover', 'mild_drop'}
    held_b, closed_b, cf_b, all_b = [], [], [], []
    active_b = 0
    for dt in sorted(by_date):
        day = sorted(by_date[dt], key=lambda x: -x['gap_pct'])[:TOP_N]
        valid = [t for t in day if t.get('pnl_t2') is not None]
        if not valid: continue
        active_b += 1
        for t in valid:
            if t.get('open_pattern', 'no_data') in close_patterns:
                p1030 = t.get('price_1030')
                if p1030 is not None and p1030 > 0:
                    pnl = (t['entry_price'] - p1030) / t['entry_price'] * 100
                    closed_b.append(pnl); all_b.append(pnl); cf_b.append(t['pnl_t2'])
                else:
                    held_b.append(t['pnl_t2']); all_b.append(t['pnl_t2'])
            else:
                held_b.append(t['pnl_t2']); all_b.append(t['pnl_t2'])

    m_b = compute_metrics(all_b, 'I8b: Aggressive')
    value_b = (np.mean(closed_b) - np.mean(cf_b)) if closed_b and cf_b else 0
    tpd_b = len(all_b) / active_b if active_b > 0 else 0
    print(f"    I8b aggressive: n={m_b['n']}, avg={m_b['avg']:+.2f}%, value={value_b:+.2f}%/trade")

    results.append({'id': 'I8b', 'title': 'Close DTR+mild_drop at 10:30 (aggressive)',
        'metrics': m_b, 'funnel': [], 'active_days': active_b, 'tpd': tpd_b,
        'top_n': TOP_N, 'pnl_key': 'pnl_t2',
        'extra': {'value_of_management': value_b},
        'summary': f"Aggressive: avg={m_b['avg']:+.2f}%, Sharpe={m_b['sharpe']:.3f}. Value={value_b:+.2f}%/trade."})

    # Baselines
    cs_selected = _select_top_n(cs_trades, top_n=TOP_N, pnl_key='pnl_t2')
    cs_pnls = [t['pnl_t2'] for t in cs_selected]
    m_cs = compute_metrics(cs_pnls, 'CS hold-all T+2')
    ad_cs = len(set(t['scan_date'] for t in cs_selected))
    results.append({'id': 'I8c', 'title': 'Baseline: CS hold-all T+2',
        'metrics': m_cs, 'funnel': [], 'active_days': ad_cs,
        'tpd': len(cs_pnls)/ad_cs if ad_cs else 0, 'top_n': TOP_N, 'pnl_key': 'pnl_t2',
        'extra': {}, 'summary': f"CS hold-all: avg={m_cs['avg']:+.2f}%, Sharpe={m_cs['sharpe']:.3f}."})

    sd_trades = [t for t in cs_trades if t.get('open_pattern') == 'sustained_drop']
    sd_selected = _select_top_n(sd_trades, top_n=TOP_N, pnl_key='pnl_t2')
    sd_pnls = [t['pnl_t2'] for t in sd_selected]
    m_sd = compute_metrics(sd_pnls, 'CS+SD T+2')
    results.append({'id': 'I8d', 'title': 'Baseline: CS+sustained_drop T+2',
        'metrics': m_sd, 'funnel': [],
        'active_days': len(set(t['scan_date'] for t in sd_selected)),
        'tpd': len(sd_pnls)/max(1,len(set(t['scan_date'] for t in sd_selected))),
        'top_n': TOP_N, 'pnl_key': 'pnl_t2', 'extra': {},
        'summary': f"CS+SD: avg={m_sd['avg']:+.2f}%, Sharpe={m_sd['sharpe']:.3f}."})
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# I9: Sector-Weighted Ranking
# ═══════════════════════════════════════════════════════════════════════════════

SECTOR_TIERS = {
    'Healthcare': 1.3, 'Transportation': 1.3, 'Agriculture': 1.3,
    'Construction': 1.3, 'Mining': 1.3,
    'Services': 1.0, 'Finance': 1.0, 'Retail': 1.0,
    'Technology': 0.7, 'Manufacturing': 0.7, 'Crypto/Blockchain': 0.7,
}

def run_i9(all_trades, dates):
    print(f"\n\n{'='*80}")
    print(f"  I9: SECTOR-WEIGHTED RANKING")
    print(f"{'='*80}")
    TOP_N = 5; results = []

    cs_trades = [t for t in all_trades if t.get('is_cs')]
    sector_counts = defaultdict(int)
    none_count = 0
    for t in cs_trades:
        s = t.get('sector')
        if s is None or s == 'Unknown': none_count += 1
        else: sector_counts[s] += 1
    print(f"  CS trades: {len(cs_trades)}, sector=Unknown: {none_count} ({none_count/max(1,len(cs_trades))*100:.0f}%)")
    for sec, cnt in sorted(sector_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {sec:25s}  n={cnt:>4d}  weight={SECTOR_TIERS.get(sec, 1.0):.1f}")

    by_date = defaultdict(list)
    for t in cs_trades:
        by_date[t['scan_date']].append(t)

    # I9a: Sector-weighted ranking
    def sector_score(trade):
        sector = trade.get('sector')
        weight = SECTOR_TIERS.get(sector, 1.0) if sector else 1.0
        return trade['gap_pct'] * weight

    weighted_pnls, weighted_active = [], 0
    for dt in sorted(by_date):
        day = sorted(by_date[dt], key=sector_score, reverse=True)[:TOP_N]
        valid = [t for t in day if t.get('pnl_t2') is not None]
        if valid:
            weighted_active += 1
            weighted_pnls.extend([t['pnl_t2'] for t in valid])
    m_w = compute_metrics(weighted_pnls, 'I9a: Sector-weighted')
    tpd_w = len(weighted_pnls) / weighted_active if weighted_active > 0 else 0
    loo_w, flips_w = _loo_stable(weighted_pnls) if len(weighted_pnls) > 20 else (False, 0)
    print(f"  I9a sector-weighted: n={m_w['n']}, avg={m_w['avg']:+.2f}%, Sharpe={m_w['sharpe']:.3f}")

    # I9b: Pure gap (baseline)
    pure_pnls, pure_active = [], 0
    for dt in sorted(by_date):
        day = sorted(by_date[dt], key=lambda x: -x['gap_pct'])[:TOP_N]
        valid = [t for t in day if t.get('pnl_t2') is not None]
        if valid:
            pure_active += 1
            pure_pnls.extend([t['pnl_t2'] for t in valid])
    m_p = compute_metrics(pure_pnls, 'I9b: Pure gap (CS)')
    delta_sharpe = m_w['sharpe'] - m_p['sharpe']
    print(f"  I9b pure gap: n={m_p['n']}, avg={m_p['avg']:+.2f}%, Sharpe={m_p['sharpe']:.3f}")
    print(f"  Sector weighting delta Sharpe: {delta_sharpe:+.3f}")

    results.append({'id': 'I9a', 'title': 'Sector-weighted ranking (CS)',
        'metrics': m_w, 'funnel': [], 'active_days': weighted_active, 'tpd': tpd_w,
        'top_n': TOP_N, 'pnl_key': 'pnl_t2',
        'extra': {'sector_weights': SECTOR_TIERS, 'loo_stable': loo_w, 'delta_sharpe_vs_pure': delta_sharpe},
        'summary': f"Sector-weighted: avg={m_w['avg']:+.2f}%, Sharpe={m_w['sharpe']:.3f}. "
                   f"vs pure gap: delta={delta_sharpe:+.3f}."})
    results.append({'id': 'I9b', 'title': 'Pure gap ranking (CS, baseline)',
        'metrics': m_p, 'funnel': [], 'active_days': pure_active,
        'tpd': len(pure_pnls)/pure_active if pure_active else 0, 'top_n': TOP_N, 'pnl_key': 'pnl_t2',
        'extra': {}, 'summary': f"Pure gap CS: avg={m_p['avg']:+.2f}%, Sharpe={m_p['sharpe']:.3f}."})

    # I9c: Hard exclude Technology + Manufacturing
    weak_sectors = {'Technology', 'Manufacturing'}
    filtered = [t for t in cs_trades if t.get('sector') not in weak_sectors]
    excluded = [t for t in cs_trades if t.get('sector') in weak_sectors]
    exc_pnls = [t['pnl_t2'] for t in excluded if t.get('pnl_t2') is not None]
    filt_selected = _select_top_n(filtered, top_n=TOP_N, pnl_key='pnl_t2')
    filt_pnls = [t['pnl_t2'] for t in filt_selected]
    m_f = compute_metrics(filt_pnls, 'I9c: Excl Tech+Mfg')
    ad_f = len(set(t['scan_date'] for t in filt_selected))
    tpd_f = len(filt_pnls) / ad_f if ad_f > 0 else 0
    delta_f = m_f['sharpe'] - m_p['sharpe']
    print(f"  I9c excl Tech+Mfg: n={m_f['n']}, avg={m_f['avg']:+.2f}%, Sharpe={m_f['sharpe']:.3f}")
    results.append({'id': 'I9c', 'title': 'Exclude Technology + Manufacturing (CS)',
        'metrics': m_f, 'funnel': [('cs_trades', len(cs_trades)),
                                    ('after_exclude', len(filtered)), ('selected', len(filt_selected))],
        'active_days': ad_f, 'tpd': tpd_f, 'top_n': TOP_N, 'pnl_key': 'pnl_t2',
        'extra': {'excluded_sectors': list(weak_sectors), 'n_excluded': len(excluded),
                  'excluded_avg_pnl': round(np.mean(exc_pnls), 4) if exc_pnls else None,
                  'delta_sharpe_vs_pure': delta_f},
        'summary': f"Excl Tech+Mfg: avg={m_f['avg']:+.2f}%, Sharpe={m_f['sharpe']:.3f}. delta={delta_f:+.3f}."})
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# I10: Out-of-Sample Validation (Q4 2025)
# ═══════════════════════════════════════════════════════════════════════════════

def run_i10(all_trades, dates):
    print(f"\n\n{'='*80}")
    print(f"  I10: OUT-OF-SAMPLE VALIDATION (Q4 2025 vs Jan-Sep 2026)")
    print(f"{'='*80}")
    TOP_N = 5; results = []

    oos_trades = [t for t in all_trades if t['scan_date'] < OOS_CUTOFF]
    is_trades = [t for t in all_trades if t['scan_date'] >= OOS_CUTOFF]
    print(f"  OOS (before {OOS_CUTOFF}): {len(oos_trades)} trades")
    print(f"  IS  (from {OOS_CUTOFF}):    {len(is_trades)} trades")

    if not oos_trades:
        print(f"  WARNING: No OOS trades. Date range: "
              f"{min(t['scan_date'] for t in all_trades)} to "
              f"{max(t['scan_date'] for t in all_trades)}")

    def apply_i7(trades):
        return [t for t in trades if t.get('is_cs')
                and t.get('gap_freshness') in ('stale', 'moderate')
                and t.get('open_price', 0) >= 30]

    strategies = {
        'raw_baseline': {'fn': lambda tr: tr, 'pnl': 'pnl_t2', 'label': 'Raw baseline (T+2)'},
        'cs_only': {'fn': lambda tr: [t for t in tr if t.get('is_cs')], 'pnl': 'pnl_t2', 'label': 'CS_ONLY (T+2)'},
        'i7_composite': {'fn': apply_i7, 'pnl': 'pnl_t3', 'label': 'I7 Composite (T+3)'},
    }

    print(f"\n  {'Strategy':<40s}  {'Period':<5s}  {'n':>4s}  {'Avg':>8s}  {'Sharpe':>7s}  {'WR':>5s}")
    print(f"  {'─'*70}")

    for sname, s in strategies.items():
        for pname, ptrades in [('OOS', oos_trades), ('IS', is_trades)]:
            filtered = s['fn'](ptrades)
            selected = _select_top_n(filtered, top_n=TOP_N, pnl_key=s['pnl'])
            pnls = [t[s['pnl']] for t in selected]
            m = compute_metrics(pnls, f"{sname}_{pname}")
            ad = len(set(t['scan_date'] for t in selected))
            tpd = len(pnls) / ad if ad > 0 else 0
            print(f"  {s['label']:<40s}  {pname:<5s}  {m['n']:>4d}  "
                  f"{m['avg']:>+7.2f}%  {m['sharpe']:>7.3f}  {m['wr']*100:>4.0f}%")
            loo_s, loo_f = _loo_stable(pnls) if len(pnls) > 20 else (False, 0)
            regime = 'NEGATIVE OOS — regime dependency' if pname == 'OOS' and m['avg'] < 0 else ''
            results.append({'id': f'I10_{sname}_{pname}', 'title': f"{s['label']} — {pname}",
                'metrics': m, 'funnel': [], 'active_days': ad, 'tpd': tpd,
                'top_n': TOP_N, 'pnl_key': s['pnl'],
                'extra': {'period': pname, 'strategy': sname, 'loo_stable': loo_s, 'regime_note': regime},
                'summary': f"{sname} {pname}: n={m['n']}, avg={m['avg']:+.2f}%, Sharpe={m['sharpe']:.3f}. {regime}"})
        print(f"  {'─'*70}")

    # Verdict
    i7_oos = next((r for r in results if r['id'] == 'I10_i7_composite_OOS'), None)
    i7_is = next((r for r in results if r['id'] == 'I10_i7_composite_IS'), None)
    if i7_oos and i7_is and i7_oos['metrics']['n'] > 0:
        if i7_oos['metrics']['avg'] > 0 and i7_is['metrics']['avg'] > 0:
            print(f"\n    I7 composite PASSES OOS: OOS avg={i7_oos['metrics']['avg']:+.2f}%, "
                  f"IS avg={i7_is['metrics']['avg']:+.2f}%")
        elif i7_oos['metrics']['avg'] < 0:
            print(f"\n    I7 composite FAILS OOS: OOS avg={i7_oos['metrics']['avg']:+.2f}%")
    elif i7_oos and i7_oos['metrics']['n'] == 0:
        print(f"\n    No OOS trades for I7 — backtest must start before 2026-01-01")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CONSOLIDATED REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(all_results):
    print(f"\n\n{'='*120}")
    print(f"  AGENT 1 — INVESTIGATION RESULTS I1-I10")
    print(f"  {BACKTEST_START} → {BACKTEST_END}  |  OOS cutoff: {OOS_CUTOFF}")
    print(f"  Baseline: _FLOAT ({len(_FLOAT)} tickers), TOP_N={TOP_N}, T+2 hold")
    print(f"{'='*120}")

    # ── Summary comparison table ──
    print(f"\n  ── SUMMARY COMPARISON ──")
    rows = []
    for r in all_results:
        m = r['metrics']
        ci_flag = '✓' if m['ci_lo'] > 0 else '✗'
        rows.append([
            r['id'], r['title'][:45], m['n'],
            f"{r.get('tpd', 0):.1f}",
            f"{m['avg']:+.2f}%", f"{m['med']:+.2f}%",
            f"{m['wr']*100:.0f}%", f"{m['sharpe']:.3f}", f"{m['pf']:.2f}",
            f"[{m['ci_lo']:+.2f},{m['ci_hi']:+.2f}] {ci_flag}",
        ])
    print(tabulate(rows, headers=[
        'ID','Investigation','n','TPD','Avg','Median','WR','Sharpe','PF','CI 95%'
    ], tablefmt='simple'))

    # ── Detailed results per investigation ──
    for r in all_results:
        print(f"\n  {'─'*80}")
        print(f"  {r['id']}: {r['title']}")
        print(f"  {'─'*80}")
        m = r['metrics']
        print(f"    n={m['n']}  avg={m['avg']:+.2f}%  med={m['med']:+.2f}%  "
              f"WR={m['wr']*100:.0f}%  Sharpe={m['sharpe']:.3f}  PF={m['pf']:.2f}")
        print(f"    TrimMean={m['trimmed_mean']:+.2f}%  "
              f"CI=[{m['ci_lo']:+.2f},{m['ci_hi']:+.2f}]  TPD={r.get('tpd',0):.1f}")

        if r.get('funnel'):
            print(f"    Filter funnel:")
            for fname, n in r['funnel']:
                print(f"      {fname:>30s}: {n}")

        if r.get('extra'):
            print(f"    Extra:")
            for k, v in r['extra'].items():
                if isinstance(v, float):
                    print(f"      {k}: {v:.4f}")
                else:
                    print(f"      {k}: {v}")

        if r.get('summary'):
            print(f"    Summary: {r['summary']}")


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_investigations(all_results):
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        BASE_DIR = os.getcwd()

    # ── Figure 1: Sharpe comparison bar chart ──
    fig, ax = plt.subplots(figsize=(16, 8))
    names = [f"{r['id']}\n{r['title'][:25]}" for r in all_results]
    sharpes = [r['metrics']['sharpe'] for r in all_results]
    colors = ['#2ecc71' if s > 0.3 else '#3498db' if s > 0.2 else '#e74c3c' for s in sharpes]
    bars = ax.bar(range(len(names)), sharpes, color=colors, alpha=0.85, edgecolor='white')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=7, rotation=45, ha='right')
    ax.axhline(0.235, color='#333', ls='--', lw=1, label='Baseline (0.235)')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Agent 1 Investigations — Sharpe Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, sharpes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7)
    plt.tight_layout()
    p1 = os.path.join(BASE_DIR, 'agent1_investigations_sharpe.png')
    plt.savefig(p1, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n  Saved: {p1}")

    # ── Figure 2: Avg PnL vs Trade Count scatter ──
    fig, ax = plt.subplots(figsize=(12, 8))
    for r in all_results:
        m = r['metrics']
        color = '#2ecc71' if m['sharpe'] > 0.3 else '#3498db' if m['sharpe'] > 0.2 else '#e74c3c'
        ax.scatter(m['n'], m['avg'], s=max(m['sharpe'] * 200, 30), color=color, alpha=0.7,
                   edgecolor='white', linewidth=0.5)
        ax.annotate(r['id'], (m['n'], m['avg']), fontsize=7,
                    xytext=(5, 5), textcoords='offset points')
    ax.axhline(0, color='#333', lw=0.8, ls=':')
    ax.set_xlabel('Number of Trades')
    ax.set_ylabel('Avg PnL (%)')
    ax.set_title('Avg PnL vs Trade Count (size = Sharpe)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p2 = os.path.join(BASE_DIR, 'agent1_investigations_scatter.png')
    plt.savefig(p2, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {p2}")


# ═══════════════════════════════════════════════════════════════════════════════
# MARKDOWN REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def save_markdown_report(all_results):
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        BASE_DIR = os.getcwd()

    path = os.path.join(BASE_DIR, 'INVESTIGATION_RESULTS.md')
    lines = []
    lines.append(f"# Agent 1 — Investigation Results I1-I10\n")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Period:** {BACKTEST_START} → {BACKTEST_END} (OOS cutoff: {OOS_CUTOFF})")
    lines.append(f"**Baseline:** _FLOAT ({len(_FLOAT)} tickers), TOP_N={TOP_N}, T+2 hold\n")
    lines.append(f"---\n")

    # Summary table
    lines.append(f"## Summary Comparison\n")
    lines.append(f"| ID | Investigation | n | TPD | Avg PnL | Median | WR | Sharpe | PF | CI 95% |")
    lines.append(f"|---|---|---|---|---|---|---|---|---|---|")
    for r in all_results:
        m = r['metrics']
        ci_flag = '✓' if m['ci_lo'] > 0 else '✗'
        lines.append(
            f"| {r['id']} | {r['title'][:40]} | {m['n']} | {r.get('tpd',0):.1f} | "
            f"{m['avg']:+.2f}% | {m['med']:+.2f}% | {m['wr']*100:.0f}% | "
            f"{m['sharpe']:.3f} | {m['pf']:.2f} | [{m['ci_lo']:+.2f},{m['ci_hi']:+.2f}] {ci_flag} |"
        )
    lines.append("")

    # Detailed per-investigation
    for r in all_results:
        lines.append(f"\n---\n")
        lines.append(f"## {r['id']}: {r['title']}\n")
        m = r['metrics']
        lines.append(f"| Metric | Value |")
        lines.append(f"|---|---|")
        lines.append(f"| Trades | {m['n']} |")
        lines.append(f"| Avg PnL | {m['avg']:+.2f}% |")
        lines.append(f"| Median | {m['med']:+.2f}% |")
        lines.append(f"| Trimmed Mean | {m['trimmed_mean']:+.2f}% |")
        lines.append(f"| Win Rate | {m['wr']*100:.0f}% |")
        lines.append(f"| Sharpe | {m['sharpe']:.3f} |")
        lines.append(f"| Profit Factor | {m['pf']:.2f} |")
        lines.append(f"| 95% CI | [{m['ci_lo']:+.2f}, {m['ci_hi']:+.2f}] |")
        lines.append(f"| TPD | {r.get('tpd',0):.1f} |")
        lines.append("")

        if r.get('funnel'):
            lines.append(f"**Filter Funnel:**\n")
            for fname, n in r['funnel']:
                lines.append(f"- {fname}: {n}")
            lines.append("")

        if r.get('extra'):
            lines.append(f"**Extra Analysis:**\n")
            for k, v in r['extra'].items():
                if isinstance(v, float):
                    lines.append(f"- {k}: {v:.4f}")
                elif isinstance(v, dict):
                    lines.append(f"- {k}:")
                    for kk, vv in v.items():
                        lines.append(f"  - {kk}: {vv}")
                else:
                    lines.append(f"- {k}: {v}")
            lines.append("")

        if r.get('summary'):
            lines.append(f"**Summary:** {r['summary']}\n")

    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\n  Markdown report saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE / LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

import pickle

DATA_PATH = '/tmp/agent1_data.pkl'

def save_data(all_trades, dates):
    trades_clean = []
    for t in all_trades:
        tc = {k: v for k, v in t.items() if k != 'rth_bars'}
        trades_clean.append(tc)
    with open(DATA_PATH, 'wb') as f:
        pickle.dump({'trades': trades_clean, 'dates': dates,
                     'ticker_ref': dict(_ticker_ref)}, f)
    print(f"  Saved {len(trades_clean)} trades to {DATA_PATH}")

def load_data():
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    _ticker_ref.update(data['ticker_ref'])
    print(f"  Loaded {len(data['trades'])} trades, {len(data['dates'])} dates")
    return data['trades'], data['dates']


# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    all_trades, dates = collect_data()
    save_data(all_trades, dates)

    all_results = []
    for name, fn in [('I1', run_i1), ('I2', run_i2), ('I3', run_i3),
                     ('I4', run_i4), ('I5', run_i5), ('I6', run_i6),
                     ('I7', run_i7), ('I8', run_i8), ('I9', run_i9), ('I10', run_i10)]:
        print(f"\n  Running {name}...")
        results = fn(all_trades, dates)
        all_results.extend(results)

    print_report(all_results)
    plot_investigations(all_results)
    save_markdown_report(all_results)

    print(f"\n\n{'='*80}")
    print(f"  COMPLETE — {len(all_results)} investigation results")
    print(f"  Check INVESTIGATION_RESULTS.md for the full report")
    print(f"{'='*80}")
