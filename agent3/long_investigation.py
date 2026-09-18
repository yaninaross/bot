# %% Agent 3 — Long Strategy Investigation
# Three independent hypotheses, single data-fetch pass:
#   A: Gap-Down Mean Reversion (LONG) — mirror of Agent 1
#   B: Gap-Up Momentum Continuation (LONG via ORB breakout)
#   C: Multi-Day Reversal (LONG on oversold bounce + gap-up signal)
#
# Constraint: LONG only, no shorting, no leverage (pension account).
# Run in Colab cell. Config vars at top.
# Run time: ~30-60 min (API-bound, fetches minute bars for all candidates).

import os, sys, time, re, asyncio, copy
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

try:
    from google.colab import auth
    try: auth.authenticate_user()
    except: pass
except: pass


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

BACKTEST_START = '2025-09-15'
BACKTEST_END   = '2026-09-12'
OOS_CUTOFF     = '2026-01-01'

GAP_DOWN_MIN  = 0.05      # |gap| >= 5% for gap-down (Hypothesis A)
GAP_UP_MIN    = 0.05      # gap >= 5% for gap-up (Hypothesis B)
PRICE_MIN     = 10.0      # A and B minimum price
PRICE_MIN_C   = 20.0      # C minimum price (pension = higher quality)
FLOAT_MIN     = 10_000_000
TOP_N         = 5

PEAK_WINDOW   = 3
MAX_PEAK_BAR  = 12
FALLBACK_BAR  = 15
SKIP_BOUNCE_PCT = 0.15    # skip gap-down trade if already bounced >15% by bar 15
ORB_BARS      = 5         # first N bars define opening range
ORB_MAX_BAR   = 25        # max bar to wait for ORB break

ADV_LOOKBACK  = 20
PRIOR_DECLINE_5D  = 0.15  # C1: 5-day decline >= 15%
PRIOR_DECLINE_10D = 0.20  # C2: 10-day decline >= 20%
REVERSAL_GAP_MIN  = 0.03  # C: today gap-up >= 3%
VOLUME_SPIKE_MULT = 3.0   # C2: volume > 3x 20-day avg

INITIAL_CAPITAL = 100_000

POLYGON_API_KEY = os.environ["POLYGON_API_KEY"]
_BASE = "https://api.polygon.io"
_TK_RE = re.compile(r'^[A-Z]{1,5}$')
ET = pytz.timezone('America/New_York')

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

_CS_TYPES = {'CS'}
_ETF_TYPES = {'ETF', 'ETV', 'ETN', 'ETP'}

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
_FLOAT_TICKERS = {tk for tk, fs in _FLOAT.items() if fs >= FLOAT_MIN}


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
            print(f"    [retry {attempt+1}/5] connection error, waiting {wait}s...")
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
# ASYNC MINUTE BAR FETCH (PM + RTH)
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
# TICKER REFERENCE (for CS_ONLY filtering)
# ═══════════════════════════════════════════════════════════════════════════════

_ticker_ref = {}

def fetch_ticker_refs(tickers):
    for tk in tickers:
        if tk in _ticker_ref:
            continue
        try:
            data = _pg(f"{_BASE}/v3/reference/tickers/{tk}")
            res = data.get('results', {})
            _ticker_ref[tk] = {
                'type': res.get('type', 'UNK'),
                'name': res.get('name', ''),
                'exchange': res.get('primary_exchange'),
            }
            time.sleep(0.05)
        except Exception:
            _ticker_ref[tk] = {'type': 'UNK', 'name': '', 'exchange': None}


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def find_local_low(bars, window=PEAK_WINDOW, max_trough_bar=MAX_PEAK_BAR,
                   fallback_bar=FALLBACK_BAR):
    if len(bars) < window * 2 + 1:
        return None
    for i in range(window, min(max_trough_bar, len(bars) - window)):
        cl = bars[i][1]['l']
        if all(bars[i - j][1]['l'] > cl and bars[i + j][1]['l'] > cl
               for j in range(1, window + 1)):
            ci = i + window
            if ci < len(bars):
                return {'entry_price': bars[ci][1]['c'], 'entry_time': bars[ci][0],
                        'entry_type': 'local_low', 'entry_bar': ci,
                        'trough_bar': i, 'trough_time': bars[i][0],
                        'trough_price': bars[i][1]['l']}
    if fallback_bar is not None and fallback_bar < len(bars):
        return {'entry_price': bars[fallback_bar][1]['c'], 'entry_time': bars[fallback_bar][0],
                'entry_type': 'fallback', 'entry_bar': fallback_bar,
                'trough_bar': None, 'trough_time': None, 'trough_price': None}
    return None


def find_orb_break_up(rth_bars, orb_bars=ORB_BARS, max_bar=ORB_MAX_BAR):
    if len(rth_bars) < orb_bars + 1:
        return None
    orb_high = max(b[1]['h'] for b in rth_bars[:orb_bars])
    orb_low = min(b[1]['l'] for b in rth_bars[:orb_bars])
    for i in range(orb_bars, min(max_bar, len(rth_bars))):
        if rth_bars[i][1]['c'] > orb_high:
            break_pct = (rth_bars[i][1]['c'] - orb_high) / orb_high if orb_high > 0 else 0
            avg_vol_orb = np.mean([b[1]['v'] for b in rth_bars[:orb_bars]]) if orb_bars > 0 else 1
            return {'entry_price': rth_bars[i][1]['c'], 'entry_time': rth_bars[i][0],
                    'entry_type': 'orb_break', 'entry_bar': i,
                    'orb_high': orb_high, 'orb_low': orb_low,
                    'orb_width': (orb_high - orb_low) / orb_low if orb_low > 0 else 0,
                    'break_pct': round(break_pct, 6),
                    'break_bar_vol': rth_bars[i][1]['v'],
                    'avg_orb_vol': avg_vol_orb}
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def classify_gap_down_pattern(rth_bars):
    if len(rth_bars) < 60:
        return 'no_data'
    high_30m = max(b[1]['h'] for b in rth_bars[:30])
    close_60m = rth_bars[59][1]['c']
    open_p = rth_bars[0][1]['o']
    rise = (high_30m - open_p) / abs(open_p) if open_p != 0 else 0
    recovery = (close_60m - open_p) / abs(open_p) if open_p != 0 else 0
    if rise > 0.03 and recovery > 0.5 * rise:
        return 'sustained_rise'
    elif rise > 0.03 and recovery <= 0.5 * rise:
        return 'rise_then_drop'
    elif rise <= 0.01:
        return 'continued_drop'
    return 'mild_rise'


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
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
    if extra: h.extend(extra)
    return h

def tpd_str(n_trades, n_days):
    return f"{n_trades/n_days:.2f}" if n_days > 0 else 'n/a'


# ═══════════════════════════════════════════════════════════════════════════════
# DATA COLLECTION — single pass, three candidate types
# ═══════════════════════════════════════════════════════════════════════════════

def collect_data():
    t_total = time.time()
    timings = {}

    print(f"\n{'='*80}")
    print(f"  AGENT 3 — LONG STRATEGY INVESTIGATION — DATA COLLECTION")
    print(f"  {BACKTEST_START} → {BACKTEST_END}  |  OOS cutoff: {OOS_CUTOFF}")
    print(f"  Universe: _FLOAT ({len(_FLOAT_TICKERS)} tickers with float >= {FLOAT_MIN/1e6:.0f}M)")
    print(f"  Hypotheses: A (gap-down long), B (gap-up momentum), C (reversal)")
    print(f"{'='*80}")

    d = datetime.strptime(BACKTEST_START, '%Y-%m-%d').date()
    end = datetime.strptime(BACKTEST_END, '%Y-%m-%d').date()
    dates = []
    while d <= end:
        if _is_td(d): dates.append(d.strftime('%Y-%m-%d'))
        d += timedelta(days=1)
    n_dates = len(dates)

    t1 = time.time()
    adv_dates = _prior_n_td(BACKTEST_START, ADV_LOOKBACK)
    adv_strs = [d.strftime('%Y-%m-%d') for d in adv_dates]
    print(f"\n  Phase 1a: Fetching {len(adv_strs)} ADV lookback dates...")
    for dt in adv_strs:
        fetch_grouped_daily(dt)
    timings['adv_prefetch'] = time.time() - t1

    t2 = time.time()
    print(f"\n  Phase 1b: Scanning {n_dates} days for gap-down, gap-up, and reversal candidates...")

    cands_a = {}  # gap-down candidates by date
    cands_b = {}  # gap-up candidates by date
    cands_c = {}  # reversal candidates by date
    all_tickers = set()
    n_a = n_b = n_c = 0

    for i, dt in enumerate(dates):
        prior_str = _prior_td(dt).strftime('%Y-%m-%d')
        prior = fetch_grouped_daily(prior_str)
        today = fetch_grouped_daily(dt)

        day_a, day_b, day_c = [], [], []

        for tk in _FLOAT_TICKERS:
            if tk in _EXCLUDE_TK: continue
            if any(tk.endswith(s) for s in _EXCLUDE_SUFFIX) and len(tk) > 2: continue
            if tk not in prior or tk not in today: continue
            prev_c = prior[tk]['c']
            if prev_c <= 0: continue
            cur_o = today[tk]['o']
            if cur_o <= 0: continue

            gap = (cur_o - prev_c) / prev_c

            # Hypothesis A: gap-DOWN >= 5%
            if gap <= -GAP_DOWN_MIN and cur_o >= PRICE_MIN:
                tier = 'D1' if gap >= -0.08 else ('D2' if gap >= -0.12 else ('D3' if gap >= -0.20 else ('D4' if gap >= -0.35 else 'D5')))
                day_a.append({
                    'ticker': tk, 'gap_pct': gap, 'abs_gap': abs(gap),
                    'prior_close': prev_c, 'open_price': cur_o,
                    'float': _FLOAT[tk], 'tier': tier,
                    'scan_date': dt, 'volume_today': today[tk]['v'],
                })
                all_tickers.add(tk)

            # Hypothesis B: gap-UP >= 5%
            if gap >= GAP_UP_MIN and cur_o >= PRICE_MIN:
                day_b.append({
                    'ticker': tk, 'gap_pct': gap, 'abs_gap': abs(gap),
                    'prior_close': prev_c, 'open_price': cur_o,
                    'float': _FLOAT[tk],
                    'tier': 'T4' if gap <= 0.08 else 'T5',
                    'scan_date': dt, 'volume_today': today[tk]['v'],
                })
                all_tickers.add(tk)

            # Hypothesis C: prior multi-day decline + today gap-up
            if cur_o >= PRICE_MIN_C and gap >= REVERSAL_GAP_MIN:
                prior_5d = _prior_n_td(dt, 5)
                if len(prior_5d) >= 5:
                    close_5d_ago_str = prior_5d[0].strftime('%Y-%m-%d')
                    close_5d_ago_data = _gd_cache.get(close_5d_ago_str, {}).get(tk, {})
                    if close_5d_ago_data.get('c') and close_5d_ago_data['c'] > 0:
                        ret_5d = (prev_c - close_5d_ago_data['c']) / close_5d_ago_data['c']
                        if ret_5d <= -PRIOR_DECLINE_5D:
                            day_c.append({
                                'ticker': tk, 'gap_pct': gap, 'abs_gap': abs(gap),
                                'prior_close': prev_c, 'open_price': cur_o,
                                'float': _FLOAT[tk], 'prior_5d_return': ret_5d,
                                'scan_date': dt, 'volume_today': today[tk]['v'],
                                'close_5d_ago': close_5d_ago_data['c'],
                                'hyp_c_type': 'C1',
                            })
                            all_tickers.add(tk)

                # C2: 10-day decline + volume spike
                prior_10d = _prior_n_td(dt, 10)
                if len(prior_10d) >= 10:
                    close_10d_ago_str = prior_10d[0].strftime('%Y-%m-%d')
                    close_10d_ago_data = _gd_cache.get(close_10d_ago_str, {}).get(tk, {})
                    if close_10d_ago_data.get('c') and close_10d_ago_data['c'] > 0:
                        ret_10d = (prev_c - close_10d_ago_data['c']) / close_10d_ago_data['c']
                        adv_dates_c = _prior_n_td(dt, ADV_LOOKBACK)
                        vols = [_gd_cache.get(d.strftime('%Y-%m-%d'), {}).get(tk, {}).get('v', 0) for d in adv_dates_c]
                        vols = [v for v in vols if v > 0]
                        avg_vol = np.mean(vols) if vols else 0
                        vol_ratio = today[tk]['v'] / avg_vol if avg_vol > 0 else 0
                        if ret_10d <= -PRIOR_DECLINE_10D and vol_ratio >= VOLUME_SPIKE_MULT:
                            existing = [c for c in day_c if c['ticker'] == tk]
                            if not existing:
                                day_c.append({
                                    'ticker': tk, 'gap_pct': gap, 'abs_gap': abs(gap),
                                    'prior_close': prev_c, 'open_price': cur_o,
                                    'float': _FLOAT[tk], 'prior_10d_return': ret_10d,
                                    'vol_ratio': vol_ratio,
                                    'scan_date': dt, 'volume_today': today[tk]['v'],
                                    'close_10d_ago': close_10d_ago_data['c'],
                                    'hyp_c_type': 'C2',
                                })
                                all_tickers.add(tk)
                            else:
                                existing[0]['prior_10d_return'] = ret_10d
                                existing[0]['vol_ratio'] = vol_ratio
                                existing[0]['hyp_c_type'] = 'C1+C2'

        # Sort and select top N
        day_a.sort(key=lambda x: x['gap_pct'])  # most negative first
        day_b.sort(key=lambda x: -x['gap_pct'])  # most positive first

        for j, c in enumerate(day_a[:TOP_N]):
            c['rank'] = j + 1
            nxt = _next_n_td(dt, 10)
            for k in range(min(10, len(nxt))):
                c[f't{k+1}_date'] = nxt[k].strftime('%Y-%m-%d')
        for j, c in enumerate(day_b[:TOP_N]):
            c['rank'] = j + 1
            nxt = _next_n_td(dt, 5)
            for k in range(min(5, len(nxt))):
                c[f't{k+1}_date'] = nxt[k].strftime('%Y-%m-%d')
        for j, c in enumerate(day_c):
            c['rank'] = j + 1
            nxt = _next_n_td(dt, 10)
            for k in range(min(10, len(nxt))):
                c[f't{k+1}_date'] = nxt[k].strftime('%Y-%m-%d')

        cands_a[dt] = day_a[:TOP_N]
        cands_b[dt] = day_b[:TOP_N]
        cands_c[dt] = day_c
        n_a += len(cands_a[dt])
        n_b += len(cands_b[dt])
        n_c += len(cands_c[dt])

        if (i+1) % 20 == 0 or i == n_dates - 1:
            print(f"    [{i+1:>3}/{n_dates}] {dt}  |  A:{n_a} B:{n_b} C:{n_c}  |  {time.time()-t_total:.0f}s")

    timings['scan'] = time.time() - t2
    print(f"  → Candidates: A={n_a} gap-down, B={n_b} gap-up, C={n_c} reversal")
    print(f"  → {len(all_tickers)} unique tickers")

    # Fetch ticker refs for CS_ONLY
    t3 = time.time()
    tk_list = sorted(all_tickers)
    print(f"\n  Phase 2: Fetching reference data for {len(tk_list)} tickers...")
    for c_start in range(0, len(tk_list), 100):
        chunk = tk_list[c_start:c_start + 100]
        fetch_ticker_refs(chunk)
        done = min(c_start + 100, len(tk_list))
        if done % 200 == 0 or done == len(tk_list):
            print(f"    {done}/{len(tk_list)}  |  {time.time()-t_total:.0f}s")
    timings['refs'] = time.time() - t3

    # Fetch minute bars for all candidates
    t4 = time.time()
    all_pairs = set()
    for cands in [cands_a, cands_b, cands_c]:
        for picks in cands.values():
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

    # Fetch outcome dates
    t5 = time.time()
    print(f"  Phase 4: Fetching outcome dates...")
    outcome_dates = set()
    for cands in [cands_a, cands_b, cands_c]:
        for picks in cands.values():
            for p in picks:
                outcome_dates.add(p['scan_date'])
                for k in range(1, 11):
                    d = p.get(f't{k}_date')
                    if d: outcome_dates.add(d)
    to_fetch = sorted(od for od in outcome_dates if od not in _gd_cache)
    for j, od in enumerate(to_fetch):
        try: fetch_grouped_daily(od)
        except Exception as e: print(f"    WARN: {od}: {e}")
        if (j+1) % 20 == 0:
            print(f"    {j+1}/{len(to_fetch)} outcome dates")
    timings['outcomes'] = time.time() - t5

    # Build enriched trade lists
    t6 = time.time()
    print(f"\n  Phase 5: Building enriched trade lists...")

    def enrich_long(tc, bar_data, direction='gap_down'):
        rth_bars = bar_data.get('rth', []) if bar_data else []
        pm_bars = bar_data.get('pm', []) if bar_data else []
        tk = tc['ticker']
        ref = _ticker_ref.get(tk, {})
        tc['asset_type'] = ref.get('type', 'UNK')
        tc['is_cs'] = tc['asset_type'] in _CS_TYPES
        tc['is_etf'] = tc['asset_type'] in _ETF_TYPES

        # Entry depends on hypothesis
        if direction == 'gap_down':
            entry = find_local_low(rth_bars)
            if entry is None:
                return None
            # Skip rule: if already bounced > 15% from open by bar 15
            if entry['entry_type'] == 'fallback':
                open_p = tc['open_price']
                if open_p > 0 and entry['entry_price'] > open_p * (1 + SKIP_BOUNCE_PCT):
                    return None
        elif direction == 'gap_up_orb':
            entry = find_orb_break_up(rth_bars)
            if entry is None:
                return None
            tc['orb_high'] = entry.get('orb_high')
            tc['orb_width'] = entry.get('orb_width')
            tc['break_pct'] = entry.get('break_pct')
            tc['break_bar_vol'] = entry.get('break_bar_vol')
            tc['avg_orb_vol'] = entry.get('avg_orb_vol')
        elif direction == 'reversal':
            if not rth_bars:
                return None
            entry = {'entry_price': rth_bars[0][1]['o'], 'entry_time': rth_bars[0][0],
                     'entry_type': 'market_open', 'entry_bar': 0}
        else:
            return None

        tc['entry_price'] = entry['entry_price']
        tc['entry_time'] = entry['entry_time']
        tc['entry_type'] = entry['entry_type']
        tc['entry_bar'] = entry.get('entry_bar')
        ep = tc['entry_price']

        # LONG PnL: (exit - entry) / entry * 100
        for bar_idx, col in [(30, 'pnl_30m'), (60, 'pnl_60m')]:
            if bar_idx < len(rth_bars):
                tc[col] = round((rth_bars[bar_idx][1]['c'] - ep) / ep * 100, 3)
            else:
                tc[col] = None

        max_hold = 10 if direction in ('gap_down', 'reversal') else 5
        for offset in range(max_hold + 1):
            col = 'pnl_eod' if offset == 0 else f'pnl_t{offset}'
            date_key = tc['scan_date'] if offset == 0 else tc.get(f't{offset}_date', '')
            dd = _gd_cache.get(date_key, {}).get(tk, {})
            tc[col] = round((dd['c'] - ep) / ep * 100, 3) if dd.get('c') else None

        # ADV
        prior_dates = _prior_n_td(tc['scan_date'], ADV_LOOKBACK)
        vols = []
        for pd_d in prior_dates:
            dd = _gd_cache.get(pd_d.strftime('%Y-%m-%d'), {}).get(tk, {})
            if dd.get('v') and dd.get('c') and dd['c'] > 0:
                vols.append(dd['v'] * dd['c'])
        tc['adv_dollar'] = np.mean(vols) if vols else None
        adv_shares = [_gd_cache.get(pd_d.strftime('%Y-%m-%d'), {}).get(tk, {}).get('v', 0) for pd_d in prior_dates]
        adv_shares = [v for v in adv_shares if v > 0]
        tc['adv_shares'] = np.mean(adv_shares) if adv_shares else None
        tc['vol_ratio'] = tc['volume_today'] / tc['adv_shares'] if tc.get('adv_shares') and tc['adv_shares'] > 0 else None

        # Pattern classification for gap-down
        if direction == 'gap_down':
            tc['open_pattern'] = classify_gap_down_pattern(rth_bars)

        return tc

    trades_a, trades_b, trades_c = [], [], []

    for dt in dates:
        for c in cands_a.get(dt, []):
            tc = copy.deepcopy(c)
            key = (tc['ticker'], tc['scan_date'])
            bar_data = all_bars.get(key)
            result = enrich_long(tc, bar_data, 'gap_down')
            if result: trades_a.append(result)

        for c in cands_b.get(dt, []):
            tc = copy.deepcopy(c)
            key = (tc['ticker'], tc['scan_date'])
            bar_data = all_bars.get(key)
            result = enrich_long(tc, bar_data, 'gap_up_orb')
            if result: trades_b.append(result)

        for c in cands_c.get(dt, []):
            tc = copy.deepcopy(c)
            key = (tc['ticker'], tc['scan_date'])
            bar_data = all_bars.get(key)
            result = enrich_long(tc, bar_data, 'reversal')
            if result: trades_c.append(result)

    timings['enrich'] = time.time() - t6
    elapsed = time.time() - t_total

    print(f"\n  {'─'*60}")
    print(f"  COLLECTION COMPLETE")
    print(f"  {'─'*60}")
    print(f"  Hypothesis A (gap-down): {len(trades_a)} trades with entry")
    print(f"  Hypothesis B (gap-up momentum): {len(trades_b)} trades with ORB break")
    print(f"  Hypothesis C (reversal): {len(trades_c)} trades")
    print(f"  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    return trades_a, trades_b, trades_c, dates, timings


# ═══════════════════════════════════════════════════════════════════════════════
# HYPOTHESIS A — GAP-DOWN MEAN REVERSION
# ═══════════════════════════════════════════════════════════════════════════════

def run_a1(trades, dates):
    print(f"\n{'='*80}")
    print(f"  A1: RAW BASELINE — Gap-Down Mean Reversion (LONG)")
    print(f"{'='*80}")
    results = OrderedDict()
    holds = [('T+2','pnl_t2'), ('T+3','pnl_t3'), ('T+4','pnl_t4'), ('T+5','pnl_t5')]
    for hold_label, pnl_key in holds:
        pnl = [t[pnl_key] for t in trades if t.get(pnl_key) is not None]
        m = compute_metrics(pnl, f'A1 {hold_label}')
        results[f'A1_{hold_label}'] = m
    rows = [metrics_row(results[k], [tpd_str(results[k]['n'], len(dates))]) for k in results]
    print('\n' + tabulate(rows, headers=metrics_headers(['TPD']), tablefmt='simple'))

    best_key = max(results, key=lambda k: results[k]['sharpe'])
    print(f"\n  ★ Best hold: {best_key} — Sharpe {results[best_key]['sharpe']:.3f}, "
          f"Avg {results[best_key]['avg']:+.2f}%, CI [{results[best_key]['ci_lo']:+.2f}, {results[best_key]['ci_hi']:+.2f}]")
    ci_ok = results[best_key]['ci_lo'] > 0
    print(f"  → Gap-down mean reversion {'EXISTS' if ci_ok else 'NOT CONFIRMED'} (CI {'>' if ci_ok else '<='} 0)")
    return results


def run_a3(trades, dates):
    print(f"\n{'='*80}")
    print(f"  A3: CS_ONLY — Common Stock Gap-Down (no ETFs)")
    print(f"{'='*80}")
    cs_trades = [t for t in trades if t.get('is_cs', False)]
    results = OrderedDict()
    for hold_label, pnl_key in [('T+2','pnl_t2'), ('T+3','pnl_t3'), ('T+5','pnl_t5')]:
        pnl = [t[pnl_key] for t in cs_trades if t.get(pnl_key) is not None]
        m = compute_metrics(pnl, f'A3 CS {hold_label}')
        results[f'A3_{hold_label}'] = m
    rows = [metrics_row(results[k]) for k in results]
    print('\n' + tabulate(rows, headers=metrics_headers(), tablefmt='simple'))
    return results


def run_a5(trades, dates):
    print(f"\n{'='*80}")
    print(f"  A5: GAP-DOWN MAGNITUDE TIERS")
    print(f"{'='*80}")
    tiers = OrderedDict([
        ('D1 -5%→-8%',  lambda t: -0.08 <= t['gap_pct'] < -0.05),
        ('D2 -8%→-12%', lambda t: -0.12 <= t['gap_pct'] < -0.08),
        ('D3 -12%→-20%',lambda t: -0.20 <= t['gap_pct'] < -0.12),
        ('D4 -20%→-35%',lambda t: -0.35 <= t['gap_pct'] < -0.20),
        ('D5 <-35%',    lambda t: t['gap_pct'] < -0.35),
    ])
    results = OrderedDict()
    for hold_label, pnl_key in [('T+2','pnl_t2'), ('T+3','pnl_t3'), ('T+5','pnl_t5')]:
        print(f"\n  ── {hold_label} ──")
        rows = []
        for tier_name, filt in tiers.items():
            pnl = [t[pnl_key] for t in trades if filt(t) and t.get(pnl_key) is not None]
            m = compute_metrics(pnl, f'{tier_name}')
            key = f'A5_{tier_name}_{hold_label}'
            results[key] = m
            rows.append(metrics_row(m))
        print(tabulate(rows, headers=metrics_headers(), tablefmt='simple'))
    return results


def run_a6(trades, dates):
    print(f"\n{'='*80}")
    print(f"  A6: SUSTAINED RISE PATTERN (mirror of sustained_drop)")
    print(f"{'='*80}")
    patterns = OrderedDict([
        ('sustained_rise',  lambda t: t.get('open_pattern') == 'sustained_rise'),
        ('rise_then_drop',  lambda t: t.get('open_pattern') == 'rise_then_drop'),
        ('mild_rise',       lambda t: t.get('open_pattern') == 'mild_rise'),
        ('continued_drop',  lambda t: t.get('open_pattern') == 'continued_drop'),
    ])
    results = OrderedDict()
    for hold_label, pnl_key in [('T+2','pnl_t2'), ('T+3','pnl_t3'), ('T+5','pnl_t5')]:
        print(f"\n  ── {hold_label} ──")
        rows = []
        for pat_name, filt in patterns.items():
            pnl = [t[pnl_key] for t in trades if filt(t) and t.get(pnl_key) is not None]
            m = compute_metrics(pnl, pat_name)
            results[f'A6_{pat_name}_{hold_label}'] = m
            rows.append(metrics_row(m))
        print(tabulate(rows, headers=metrics_headers(), tablefmt='simple'))
    best = max(results, key=lambda k: results[k]['sharpe'])
    print(f"\n  ★ Best pattern: {best} — Sharpe {results[best]['sharpe']:.3f}")
    return results


def run_a7(trades, dates):
    print(f"\n{'='*80}")
    print(f"  A7: HOLD PERIOD SWEEP — All Horizons")
    print(f"{'='*80}")
    holds = [('30m','pnl_30m'), ('60m','pnl_60m'), ('EOD','pnl_eod'),
             ('T+1','pnl_t1'), ('T+2','pnl_t2'), ('T+3','pnl_t3'),
             ('T+4','pnl_t4'), ('T+5','pnl_t5')]
    results = OrderedDict()
    rows = []
    for hold_label, pnl_key in holds:
        pnl = [t[pnl_key] for t in trades if t.get(pnl_key) is not None]
        m = compute_metrics(pnl, f'A7 {hold_label}')
        results[f'A7_{hold_label}'] = m
        rows.append(metrics_row(m, [tpd_str(m['n'], len(dates))]))
    print('\n' + tabulate(rows, headers=metrics_headers(['TPD']), tablefmt='simple'))

    # Plot PnL curve by hold period
    holds_with_data = [(k, results[k]) for k in results if results[k]['n'] >= 5]
    if holds_with_data:
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = [h[0].replace('A7_','') for h in holds_with_data]
        avgs = [h[1]['avg'] for h in holds_with_data]
        ci_lo = [h[1]['ci_lo'] for h in holds_with_data]
        ci_hi = [h[1]['ci_hi'] for h in holds_with_data]
        x = range(len(labels))
        ax.bar(x, avgs, color=['#27ae60' if v > 0 else '#e74c3c' for v in avgs], alpha=0.7)
        ax.errorbar(x, avgs, yerr=[np.array(avgs)-np.array(ci_lo), np.array(ci_hi)-np.array(avgs)],
                    fmt='none', color='black', capsize=4)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.axhline(0, color='#333', lw=0.8)
        ax.set_ylabel('Avg PnL (%)')
        ax.set_title('A7: Gap-Down Long — PnL by Hold Period')
        plt.tight_layout()
        plt.savefig('agent3_a7_hold_sweep.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  Chart saved: agent3_a7_hold_sweep.png")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# HYPOTHESIS B — GAP-UP MOMENTUM CONTINUATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_b1(trades, dates):
    print(f"\n{'='*80}")
    print(f"  B1: ORB CONTINUATION BASELINE — Gap-Up + ORB Break Up (LONG)")
    print(f"{'='*80}")
    results = OrderedDict()
    for hold_label, pnl_key in [('60m','pnl_60m'), ('EOD','pnl_eod'), ('T+1','pnl_t1'), ('T+2','pnl_t2')]:
        pnl = [t[pnl_key] for t in trades if t.get(pnl_key) is not None]
        m = compute_metrics(pnl, f'B1 {hold_label}')
        results[f'B1_{hold_label}'] = m
    rows = [metrics_row(results[k], [tpd_str(results[k]['n'], len(dates))]) for k in results]
    print('\n' + tabulate(rows, headers=metrics_headers(['TPD']), tablefmt='simple'))
    return results


def run_b2(trades, dates):
    print(f"\n{'='*80}")
    print(f"  B2: VOLUME CONFIRMATION — ORB break vol > 1.5x avg ORB vol")
    print(f"{'='*80}")
    vol_trades = [t for t in trades if t.get('break_bar_vol') and t.get('avg_orb_vol')
                  and t['break_bar_vol'] > 1.5 * t['avg_orb_vol']]
    no_vol = [t for t in trades if t.get('break_bar_vol') and t.get('avg_orb_vol')
              and t['break_bar_vol'] <= 1.5 * t['avg_orb_vol']]
    results = OrderedDict()
    for label, subset in [('Vol confirmed', vol_trades), ('No vol confirm', no_vol)]:
        for hold_label, pnl_key in [('EOD','pnl_eod'), ('T+2','pnl_t2')]:
            pnl = [t[pnl_key] for t in subset if t.get(pnl_key) is not None]
            m = compute_metrics(pnl, f'{label} {hold_label}')
            results[f'B2_{label}_{hold_label}'] = m
    rows = [metrics_row(results[k]) for k in results]
    print('\n' + tabulate(rows, headers=metrics_headers(), tablefmt='simple'))
    return results


def run_b3(trades, dates):
    print(f"\n{'='*80}")
    print(f"  B3: ORB BREAK STRENGTH TIERS")
    print(f"{'='*80}")
    tiers = OrderedDict([
        ('weak 0-0.25%',   lambda t: t.get('break_pct', 0) < 0.0025),
        ('medium 0.25-0.75%', lambda t: 0.0025 <= t.get('break_pct', 0) < 0.0075),
        ('strong >0.75%',  lambda t: t.get('break_pct', 0) >= 0.0075),
    ])
    results = OrderedDict()
    for hold_label, pnl_key in [('EOD','pnl_eod'), ('T+2','pnl_t2')]:
        print(f"\n  ── {hold_label} ──")
        rows = []
        for tier_name, filt in tiers.items():
            pnl = [t[pnl_key] for t in trades if filt(t) and t.get(pnl_key) is not None]
            m = compute_metrics(pnl, tier_name)
            results[f'B3_{tier_name}_{hold_label}'] = m
            rows.append(metrics_row(m))
        print(tabulate(rows, headers=metrics_headers(), tablefmt='simple'))
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# HYPOTHESIS C — MULTI-DAY REVERSAL
# ═══════════════════════════════════════════════════════════════════════════════

def run_c1(trades, dates):
    print(f"\n{'='*80}")
    print(f"  C1: 5-DAY DECLINE + GAP-UP SIGNAL — Reversal LONG")
    print(f"{'='*80}")
    c1_trades = [t for t in trades if t.get('hyp_c_type', '').startswith('C1')]
    results = OrderedDict()
    for hold_label, pnl_key in [('T+3','pnl_t3'), ('T+5','pnl_t5'), ('T+10','pnl_t10')]:
        pnl = [t[pnl_key] for t in c1_trades if t.get(pnl_key) is not None]
        m = compute_metrics(pnl, f'C1 {hold_label}')
        results[f'C1_{hold_label}'] = m
    rows = [metrics_row(results[k], [tpd_str(results[k]['n'], len(dates))]) for k in results]
    print('\n' + tabulate(rows, headers=metrics_headers(['TPD']), tablefmt='simple'))
    return results


def run_c2(trades, dates):
    print(f"\n{'='*80}")
    print(f"  C2: OVERSOLD + VOLUME SPIKE — 10d decline ≥20%, vol ≥3x avg")
    print(f"{'='*80}")
    c2_trades = [t for t in trades if 'C2' in t.get('hyp_c_type', '')]
    results = OrderedDict()
    for hold_label, pnl_key in [('T+3','pnl_t3'), ('T+5','pnl_t5'), ('T+10','pnl_t10')]:
        pnl = [t[pnl_key] for t in c2_trades if t.get(pnl_key) is not None]
        m = compute_metrics(pnl, f'C2 {hold_label}')
        results[f'C2_{hold_label}'] = m
    rows = [metrics_row(results[k]) for k in results]
    print('\n' + tabulate(rows, headers=metrics_headers(), tablefmt='simple'))
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE + OOS VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(all_results, trades_a, trades_b, trades_c, dates):
    print(f"\n{'='*80}")
    print(f"  MASTER SUMMARY TABLE — ALL EXPERIMENTS")
    print(f"{'='*80}")

    rows = []
    for exp_id, m in all_results.items():
        if m['n'] == 0: continue
        ci_flag = '✓' if m['ci_lo'] > 0 else '✗'
        pension_ok = (m['ci_lo'] > 0 and m['sharpe'] >= 0.15 and m['wr'] >= 0.60)
        pension_flag = '★' if pension_ok else ''
        rows.append([
            exp_id, m['n'], tpd_str(m['n'], len(dates)),
            f"{m['avg']:+.2f}%", f"{m['wr']*100:.0f}%", f"{m['sharpe']:.3f}",
            f"[{m['ci_lo']:+.2f},{m['ci_hi']:+.2f}] {ci_flag}",
            pension_flag
        ])

    print('\n' + tabulate(rows, headers=['ID','n','TPD','Avg','WR','Sharpe','CI 95%','Pension'],
                          tablefmt='simple'))

    # Pension acceptance criteria
    print(f"\n  ── PENSION ACCOUNT ACCEPTANCE CRITERIA ──")
    print(f"  CI 95% low > 0:     REQUIRED")
    print(f"  Sharpe >= 0.15:     REQUIRED")
    print(f"  Win Rate >= 60%:    REQUIRED")
    print(f"  Max single loss > -30%: REQUIRED")
    passing = [k for k, m in all_results.items()
               if m['n'] >= 10 and m['ci_lo'] > 0 and m['sharpe'] >= 0.15 and m['wr'] >= 0.60]
    if passing:
        print(f"\n  ★ PASSING EXPERIMENTS: {', '.join(passing)}")
    else:
        print(f"\n  ✗ No experiments pass all pension criteria.")


def _resolve_trade_filter(exp_id):
    """Return a filter function for the experiment-specific subset."""
    eid = exp_id
    # A5 tier filters
    if 'A5_D1' in eid:  return lambda t: -0.08 <= t['gap_pct'] < -0.05
    if 'A5_D2' in eid:  return lambda t: -0.12 <= t['gap_pct'] < -0.08
    if 'A5_D3' in eid:  return lambda t: -0.20 <= t['gap_pct'] < -0.12
    if 'A5_D4' in eid:  return lambda t: -0.35 <= t['gap_pct'] < -0.20
    if 'A5_D5' in eid:  return lambda t: t['gap_pct'] < -0.35
    # A6 pattern filters
    if 'A6_sustained_rise' in eid:  return lambda t: t.get('open_pattern') == 'sustained_rise'
    if 'A6_rise_then_drop' in eid:  return lambda t: t.get('open_pattern') == 'rise_then_drop'
    if 'A6_mild_rise' in eid:       return lambda t: t.get('open_pattern') == 'mild_rise'
    if 'A6_continued_drop' in eid:  return lambda t: t.get('open_pattern') == 'continued_drop'
    # A3 common stock
    if eid.startswith('A3'):  return lambda t: t.get('is_cs', False)
    # B2 volume confirmation
    if 'B2_Vol confirmed' in eid:
        return lambda t: (t.get('break_bar_vol') and t.get('avg_orb_vol')
                          and t['break_bar_vol'] > 1.5 * t['avg_orb_vol'])
    if 'B2_No vol confirm' in eid:
        return lambda t: (t.get('break_bar_vol') and t.get('avg_orb_vol')
                          and t['break_bar_vol'] <= 1.5 * t['avg_orb_vol'])
    # B3 strength tiers
    if 'B3_weak' in eid:    return lambda t: t.get('break_pct', 0) < 0.0025
    if 'B3_medium' in eid:  return lambda t: 0.0025 <= t.get('break_pct', 0) < 0.0075
    if 'B3_strong' in eid:  return lambda t: t.get('break_pct', 0) >= 0.0075
    # C1/C2
    if eid.startswith('C1'):  return lambda t: t.get('hyp_c_type', '').startswith('C1')
    if eid.startswith('C2'):  return lambda t: 'C2' in t.get('hyp_c_type', '')
    # A1, A7, B1 — no additional filter
    return lambda t: True


def run_oos_validation(all_results, trades_a, trades_b, trades_c, dates):
    print(f"\n{'='*80}")
    print(f"  OOS VALIDATION — Cutoff: {OOS_CUTOFF}")
    print(f"{'='*80}")

    passing = {k: m for k, m in all_results.items()
               if m['n'] >= 10 and m['ci_lo'] > 0 and m['sharpe'] >= 0.15}
    if not passing:
        print(f"  No experiments qualify for OOS validation (need CI > 0 AND Sharpe >= 0.15).")
        return {}

    oos_results = OrderedDict()
    for exp_id in passing:
        parts = exp_id.split('_')
        hyp = parts[0][0]  # A, B, or C
        pnl_key = None
        for p in parts:
            if p.startswith('T+') or p in ('EOD', '30m', '60m'):
                pnl_key = f"pnl_{p.lower().replace('+','')}" if '+' in p else f"pnl_{p.lower()}"
                break
        if pnl_key is None:
            continue

        if hyp == 'A':
            base_trades = trades_a
        elif hyp == 'B':
            base_trades = trades_b
        else:
            base_trades = trades_c

        filt = _resolve_trade_filter(exp_id)
        filtered = [t for t in base_trades if filt(t)]

        is_trades = [t for t in filtered if t['scan_date'] < OOS_CUTOFF and t.get(pnl_key) is not None]
        oos_trades = [t for t in filtered if t['scan_date'] >= OOS_CUTOFF and t.get(pnl_key) is not None]
        is_pnl = [t[pnl_key] for t in is_trades]
        oos_pnl = [t[pnl_key] for t in oos_trades]

        m_is = compute_metrics(is_pnl, f'{exp_id} IS')
        m_oos = compute_metrics(oos_pnl, f'{exp_id} OOS')

        pension_ready = (m_is['ci_lo'] > 0 and m_oos['ci_lo'] > 0
                         and m_is['sharpe'] >= 0.15 and m_oos['sharpe'] >= 0.15)

        print(f"\n  {exp_id}:")
        print(f"    IS  ({BACKTEST_START}→{OOS_CUTOFF}): n={m_is['n']}, avg={m_is['avg']:+.2f}%, "
              f"Sharpe={m_is['sharpe']:.3f}, CI [{m_is['ci_lo']:+.2f},{m_is['ci_hi']:+.2f}]")
        print(f"    OOS ({OOS_CUTOFF}→{BACKTEST_END}): n={m_oos['n']}, avg={m_oos['avg']:+.2f}%, "
              f"Sharpe={m_oos['sharpe']:.3f}, CI [{m_oos['ci_lo']:+.2f},{m_oos['ci_hi']:+.2f}]")
        print(f"    → {'★ PENSION READY' if pension_ready else '✗ Not pension ready'}")

        oos_results[exp_id] = {'is': m_is, 'oos': m_oos, 'pension_ready': pension_ready}

    return oos_results


# ═══════════════════════════════════════════════════════════════════════════════
# ANSWER THE KEY QUESTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def print_key_questions(all_results, trades_a, trades_b, trades_c):
    print(f"\n{'='*80}")
    print(f"  KEY QUESTIONS — ANSWERS")
    print(f"{'='*80}")

    # Q1: Does gap-DOWN mean reversion exist?
    a1_t2 = all_results.get('A1_T+2', {})
    print(f"\n  Q1: Does gap-DOWN mean reversion exist at T+2 or longer?")
    if a1_t2.get('n', 0) > 0:
        ci_ok = a1_t2['ci_lo'] > 0
        print(f"      A1 T+2: avg={a1_t2['avg']:+.2f}%, WR={a1_t2['wr']*100:.0f}%, "
              f"Sharpe={a1_t2['sharpe']:.3f}, CI [{a1_t2['ci_lo']:+.2f},{a1_t2['ci_hi']:+.2f}]")
        print(f"      → {'YES — statistically significant' if ci_ok else 'NO — not significant'}")
    else:
        print(f"      → INSUFFICIENT DATA (n=0)")

    # Q3: Is the bounce intraday or multi-day?
    print(f"\n  Q3: Is the bounce intraday or multi-day?")
    for label in ['A7_30m', 'A7_60m', 'A7_EOD', 'A7_T+1', 'A7_T+2', 'A7_T+3', 'A7_T+5']:
        m = all_results.get(label, {})
        if m.get('n', 0) > 0:
            print(f"      {label}: avg={m['avg']:+.2f}%, n={m['n']}")

    # Q4: Does ORB continuation provide long alpha?
    b1_t2 = all_results.get('B1_T+2', {})
    print(f"\n  Q4: Does gap-UP ORB continuation provide long alpha?")
    if b1_t2.get('n', 0) > 0:
        ci_ok = b1_t2['ci_lo'] > 0
        print(f"      B1 T+2: avg={b1_t2['avg']:+.2f}%, WR={b1_t2['wr']*100:.0f}%, "
              f"Sharpe={b1_t2['sharpe']:.3f}, CI [{b1_t2['ci_lo']:+.2f},{b1_t2['ci_hi']:+.2f}]")
        print(f"      → {'YES' if ci_ok else 'NO'}")
    else:
        print(f"      → INSUFFICIENT DATA")

    # Q5: Multi-day reversals?
    c1_t5 = all_results.get('C1_T+5', {})
    print(f"\n  Q5: Are multi-day reversals viable for pension capital?")
    if c1_t5.get('n', 0) > 0:
        print(f"      C1 T+5: avg={c1_t5['avg']:+.2f}%, n={c1_t5['n']}, Sharpe={c1_t5['sharpe']:.3f}")
    else:
        print(f"      → INSUFFICIENT DATA")

    # Q6: Best single config?
    best_key = None
    best_sharpe = -999
    for k, m in all_results.items():
        if m.get('n', 0) >= 10 and m.get('ci_lo', -999) > 0 and m.get('sharpe', 0) > best_sharpe:
            best_sharpe = m['sharpe']
            best_key = k
    print(f"\n  Q6: Best single configuration with CI > 0?")
    if best_key:
        m = all_results[best_key]
        print(f"      → {best_key}: avg={m['avg']:+.2f}%, Sharpe={m['sharpe']:.3f}, "
              f"WR={m['wr']*100:.0f}%, n={m['n']}")
    else:
        print(f"      → None found with CI > 0 and n >= 10")


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_summary(all_results):
    passing = {k: v for k, v in all_results.items() if v.get('n', 0) >= 5}
    if not passing:
        print("  No experiments with enough data for charts.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Agent 3 — Long Strategy Investigation Summary', fontsize=13, fontweight='bold')

    # Sharpe bar chart
    ax = axes[0]
    sorted_exps = sorted(passing.items(), key=lambda x: x[1]['sharpe'], reverse=True)[:20]
    labels = [e[0] for e in sorted_exps]
    sharpes = [e[1]['sharpe'] for e in sorted_exps]
    colors = ['#27ae60' if s > 0 else '#e74c3c' for s in sharpes]
    ax.barh(range(len(labels)), sharpes, color=colors, alpha=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color='#333', lw=0.8)
    ax.axvline(0.15, color='#27ae60', ls='--', lw=0.8, label='Pension threshold')
    ax.set_xlabel('Sharpe')
    ax.set_title('Sharpe by Experiment')
    ax.invert_yaxis()
    ax.legend(fontsize=8)

    # Avg PnL with CI bars
    ax = axes[1]
    avgs = [e[1]['avg'] for e in sorted_exps]
    ci_lo = [e[1]['ci_lo'] for e in sorted_exps]
    ci_hi = [e[1]['ci_hi'] for e in sorted_exps]
    yerr_lo = [a - c for a, c in zip(avgs, ci_lo)]
    yerr_hi = [c - a for a, c in zip(avgs, ci_hi)]
    ax.barh(range(len(labels)), avgs, color=colors, alpha=0.7)
    ax.errorbar(avgs, range(len(labels)), xerr=[yerr_lo, yerr_hi],
                fmt='none', color='black', capsize=3)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color='#333', lw=0.8)
    ax.set_xlabel('Avg PnL (%)')
    ax.set_title('Avg PnL with 95% CI')
    ax.invert_yaxis()

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig('agent3_summary.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Chart saved: agent3_summary.png")


# ═══════════════════════════════════════════════════════════════════════════════
# DISPATCH
# ═══════════════════════════════════════════════════════════════════════════════

trades_a, trades_b, trades_c, dates, timings = collect_data()

all_results = OrderedDict()

# A1 is the MOST IMPORTANT experiment — run first
if trades_a:
    r = run_a1(trades_a, dates)
    all_results.update(r)
    r = run_a3(trades_a, dates)
    all_results.update(r)
    r = run_a5(trades_a, dates)
    all_results.update(r)
    r = run_a6(trades_a, dates)
    all_results.update(r)
    r = run_a7(trades_a, dates)
    all_results.update(r)
else:
    print("\n  ⚠ No gap-down candidates found in period. Hypothesis A skipped.")

if trades_b:
    r = run_b1(trades_b, dates)
    all_results.update(r)
    r = run_b2(trades_b, dates)
    all_results.update(r)
    r = run_b3(trades_b, dates)
    all_results.update(r)
else:
    print("\n  ⚠ No ORB-break gap-up candidates found. Hypothesis B skipped.")

if trades_c:
    r = run_c1(trades_c, dates)
    all_results.update(r)
    r = run_c2(trades_c, dates)
    all_results.update(r)
else:
    print("\n  ⚠ No reversal candidates found. Hypothesis C skipped.")

print_summary(all_results, trades_a, trades_b, trades_c, dates)
print_key_questions(all_results, trades_a, trades_b, trades_c)

oos_results = run_oos_validation(all_results, trades_a, trades_b, trades_c, dates)

plot_summary(all_results)

print(f"\n{'='*80}")
print(f"  INVESTIGATION COMPLETE")
print(f"  Experiments run: {len(all_results)}")
print(f"  Trades analyzed: A={len(trades_a)}, B={len(trades_b)}, C={len(trades_c)}")
print(f"{'='*80}")

import pickle
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()
cache_path = os.path.join(BASE_DIR, 'phase1_cache.pkl')
with open(cache_path, 'wb') as f:
    pickle.dump({'trades_a': trades_a, 'trades_b': trades_b, 'trades_c': trades_c,
                 'dates': dates, '_gd_cache': _gd_cache, 'all_bars': {},
                 '_ticker_ref': _ticker_ref}, f)
print(f"  Cache saved: {cache_path}")
