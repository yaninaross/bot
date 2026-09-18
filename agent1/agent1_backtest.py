# %% Agent 1 — Realistic Backtest with TOP_N Sweep
# Full 4-step trading logic:
#   1. Detect gap-up >= 5% at open
#   2. Fetch intraday 1-min bars
#   3. Find first local high (spike + reversal confirmation) → SHORT entry
#   4. Hold to T+2 close → compute PnL
# Runs N = 2,3,4,5,6 trades/day and compares performance.

import os, sys, time, re, asyncio, copy
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


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

BACKTEST_START = '2026-01-02'
BACKTEST_END   = '2026-09-12'

GAP_MIN_PCT = 5.0
PRICE_MIN = 10.0
FLOAT_MIN = 10_000_000

TOP_N_SWEEP = [2, 3, 4, 5, 6]  # compare these trade counts
MAX_TOP_N = max(TOP_N_SWEEP)    # scan with max to get all candidates

# Local high entry params (from strategy_backtest.py E17)
PEAK_WINDOW = 3
MAX_PEAK_BAR = 12
FALLBACK_BAR = 15             # set None to skip trades with no reversal

INITIAL_CAPITAL = 100_000     # equal-weight split across that day's trades

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

def _next_n_td(d, n=2):
    if isinstance(d, str): d = datetime.strptime(d, '%Y-%m-%d').date()
    out, d = [], d + timedelta(days=1)
    for _ in range(30):
        if _is_td(d):
            out.append(d)
            if len(out) >= n: return out
        d += timedelta(days=1)
    return out

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
# ASYNC MINUTE BAR FETCH
# ═══════════════════════════════════════════════════════════════════════════════

async def _fetch_bars_one(session, tk, dt, sem):
    url = f"{_BASE}/v2/aggs/ticker/{tk}/range/1/minute/{dt}/{dt}"
    params = {"adjusted":"true","sort":"asc","limit":"500","apiKey":POLYGON_API_KEY}
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
# LOCAL HIGH DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def find_local_high(bars, window=PEAK_WINDOW, max_peak_bar=MAX_PEAK_BAR,
                    fallback_bar=FALLBACK_BAR):
    if len(bars) < window * 2 + 1:
        return None
    for i in range(window, min(max_peak_bar, len(bars) - window)):
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
                    'peak_time': bars[i][0],
                    'peak_price': bars[i][1]['h'],
                }
    if fallback_bar is not None and fallback_bar < len(bars):
        return {
            'entry_price': bars[fallback_bar][1]['c'],
            'entry_time': bars[fallback_bar][0],
            'entry_type': 'fallback',
            'peak_bar': None, 'peak_time': None, 'peak_price': None,
        }
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# SCANNER — returns candidates sorted by gap, with rank
# ═══════════════════════════════════════════════════════════════════════════════

def scan_one(dt_str, top_n):
    if not _is_td(dt_str): return []
    prior_str = _prior_td(dt_str).strftime('%Y-%m-%d')
    prior = fetch_grouped_daily(prior_str)
    today = fetch_grouped_daily(dt_str)

    gap_min = GAP_MIN_PCT / 100.0
    candidates = []
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
        candidates.append({
            'ticker': tk, 'gap_pct': gap, 'prior_close': prev_c,
            'open_price': cur_p, 'float': fs,
            'tier': 'T4' if gap <= 0.08 else 'T5',
        })

    candidates.sort(key=lambda x: -x['gap_pct'])
    selected = candidates[:top_n]
    for i, s in enumerate(selected):
        s['rank'] = i + 1

    nxt = _next_n_td(dt_str, 2)
    t1 = nxt[0].strftime('%Y-%m-%d') if len(nxt) >= 1 else None
    t2 = nxt[1].strftime('%Y-%m-%d') if len(nxt) >= 2 else None
    for s in selected:
        s['scan_date'] = dt_str
        s['t1_date'] = t1
        s['t2_date'] = t2
    return selected


# ═══════════════════════════════════════════════════════════════════════════════
# EQUITY CURVE (equal-weight split each morning)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_equity(t2_trades):
    capital = INITIAL_CAPITAL
    equity = [capital]
    eq_dates = []

    daily_pnl = defaultdict(list)
    for t in t2_trades:
        daily_pnl[t['scan_date']].append(t['pnl_t2'])

    for dt in sorted(daily_pnl):
        n = len(daily_pnl[dt])
        per_trade = capital / n
        day_dollar = sum(per_trade * (r / 100) for r in daily_pnl[dt])
        capital += day_dollar
        equity.append(capital)
        eq_dates.append(dt)

    return equity, eq_dates, capital


# ═══════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE — scan once, sweep TOP_N
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest():
    t_total = time.time()

    print(f"\n{'='*80}")
    print(f"  AGENT 1 BACKTEST — TOP_N SWEEP: {TOP_N_SWEEP}")
    print(f"  {BACKTEST_START} → {BACKTEST_END}")
    print(f"  Strategy: SHORT top N gap-ups >= {GAP_MIN_PCT}%, local high entry, T+2 hold")
    print(f"  Capital: ${INITIAL_CAPITAL:,.0f}  |  equal-weight split per day")
    print(f"{'='*80}")

    # ── Step 1: Enumerate trading days ──
    d = datetime.strptime(BACKTEST_START, '%Y-%m-%d').date()
    end = datetime.strptime(BACKTEST_END, '%Y-%m-%d').date()
    dates = []
    while d <= end:
        if _is_td(d): dates.append(d.strftime('%Y-%m-%d'))
        d += timedelta(days=1)
    n_dates = len(dates)

    # ── Step 2: Scan with MAX_TOP_N to get all candidates ──
    print(f"\n  Step 1: Scanning {n_dates} days (top {MAX_TOP_N} per day)...")
    all_candidates_by_date = {}
    total_cands = 0
    for i, dt in enumerate(dates):
        picks = scan_one(dt, MAX_TOP_N)
        all_candidates_by_date[dt] = picks
        total_cands += len(picks)
        if (i+1) % 20 == 0 or i == n_dates - 1:
            print(f"    [{i+1:>3}/{n_dates}] {dt}  |  {total_cands} candidates  |  {time.time()-t_total:.0f}s")

    # ── Step 3: Fetch minute bars for ALL candidates (union across all N values) ──
    all_pairs = set()
    for picks in all_candidates_by_date.values():
        for p in picks:
            all_pairs.add((p['ticker'], p['scan_date']))

    print(f"\n  Step 2: Fetching 1-min bars for {len(all_pairs)} unique (ticker, date) pairs...")
    pairs_list = list(all_pairs)
    chunk_size = 100
    all_bars = {}
    for c_start in range(0, len(pairs_list), chunk_size):
        chunk = pairs_list[c_start:c_start + chunk_size]
        result = fetch_minute_bars_batch(chunk)
        all_bars.update(result)
        print(f"    fetched {min(c_start + chunk_size, len(pairs_list))}/{len(pairs_list)}  |  {time.time()-t_total:.0f}s")

    print(f"  → Got bars for {len(all_bars)}/{len(all_pairs)} tickers")

    # ── Step 4: Find local high entries for all candidates ──
    print(f"\n  Step 3: Finding local high entries...")
    entry_cache = {}
    for key, bars in all_bars.items():
        if bars and len(bars) >= PEAK_WINDOW * 2 + 1:
            entry_cache[key] = find_local_high(bars)
        else:
            entry_cache[key] = None

    # ── Step 5: Pre-fetch all outcome dates ──
    print(f"\n  Step 4: Fetching outcome dates...")
    outcome_dates = set()
    for picks in all_candidates_by_date.values():
        for p in picks:
            outcome_dates.add(p['scan_date'])
            if p.get('t1_date'): outcome_dates.add(p['t1_date'])
            if p.get('t2_date'): outcome_dates.add(p['t2_date'])

    to_fetch = sorted(od for od in outcome_dates if od not in _gd_cache)
    for j, od in enumerate(to_fetch):
        try: fetch_grouped_daily(od)
        except Exception as e: print(f"    WARN: {od}: {e}")
        if (j+1) % 20 == 0:
            print(f"    fetched {j+1}/{len(to_fetch)} outcome dates")
    print(f"  → Done.")

    # ── Step 6: For each TOP_N, build trade list and compute stats ──
    print(f"\n  Step 5: Evaluating TOP_N = {TOP_N_SWEEP}...")

    results = {}
    for top_n in TOP_N_SWEEP:
        trades = []
        n_lh = n_fb = n_ne = n_nb = 0

        for dt in dates:
            day_candidates = all_candidates_by_date.get(dt, [])[:top_n]
            for t in day_candidates:
                tc = copy.deepcopy(t)
                key = (tc['ticker'], tc['scan_date'])
                entry = entry_cache.get(key)

                if entry is None:
                    if key not in all_bars or not all_bars.get(key):
                        n_nb += 1
                    else:
                        n_ne += 1
                    continue

                tc['entry_price'] = entry['entry_price']
                tc['entry_time'] = entry['entry_time']
                tc['entry_type'] = entry['entry_type']
                tc['peak_time'] = entry.get('peak_time')
                tc['peak_price'] = entry.get('peak_price')

                if entry['entry_type'] == 'local_high': n_lh += 1
                else: n_fb += 1

                # fill outcomes
                tk, ep = tc['ticker'], tc['entry_price']
                t0d = _gd_cache.get(tc['scan_date'],{}).get(tk,{})
                tc['t0_close'] = t0d.get('c')
                tc['pnl_eod'] = round((ep - t0d['c'])/ep*100, 3) if t0d.get('c') else None
                t1d = _gd_cache.get(tc.get('t1_date',''),{}).get(tk,{})
                tc['t1_close'] = t1d.get('c')
                tc['pnl_t1'] = round((ep - t1d['c'])/ep*100, 3) if t1d.get('c') else None
                t2d = _gd_cache.get(tc.get('t2_date',''),{}).get(tk,{})
                tc['t2_close'] = t2d.get('c')
                tc['pnl_t2'] = round((ep - t2d['c'])/ep*100, 3) if t2d.get('c') else None

                trades.append(tc)

        t2_trades = [t for t in trades if t.get('pnl_t2') is not None]
        equity, eq_dates, final_cap = compute_equity(t2_trades)

        pnl_arr = np.array([t['pnl_t2'] for t in t2_trades]) if t2_trades else np.array([])
        n = len(pnl_arr)
        avg = float(np.mean(pnl_arr)) if n > 0 else 0
        std = float(np.std(pnl_arr, ddof=1)) if n > 1 else 0
        wr = float((pnl_arr > 0).mean()) if n > 0 else 0
        sharpe = avg / std if std > 0 else 0
        se = std / np.sqrt(n) if n > 0 else 0
        ci_lo = avg - 1.96 * se
        w = pnl_arr[pnl_arr > 0]; l = pnl_arr[pnl_arr < 0]
        pf = float(w.sum() / abs(l.sum())) if len(l) > 0 and l.sum() != 0 else float('inf')
        ret_pct = (final_cap - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

        eq_arr = np.array(equity)
        peak = np.maximum.accumulate(eq_arr)
        max_dd = float(((eq_arr - peak) / peak * 100).min())

        results[top_n] = {
            'trades': trades, 't2_trades': t2_trades,
            'equity': equity, 'eq_dates': eq_dates, 'final_cap': final_cap,
            'n': n, 'avg': avg, 'med': float(np.median(pnl_arr)) if n > 0 else 0,
            'std': std, 'wr': wr, 'sharpe': sharpe, 'pf': pf,
            'ci_lo': ci_lo, 'ret_pct': ret_pct, 'max_dd': max_dd,
            'n_lh': n_lh, 'n_fb': n_fb, 'n_ne': n_ne, 'n_nb': n_nb,
            'total_pnl': float(np.sum(pnl_arr)) if n > 0 else 0,
        }

        print(f"    N={top_n}: {n} trades, avg={avg:+.2f}%, WR={wr*100:.0f}%, "
              f"Sharpe={sharpe:.3f}, return={ret_pct:+.1f}%, maxDD={max_dd:.1f}%")

    elapsed = time.time() - t_total
    print(f"\n  Total runtime: {elapsed:.0f}s\n")

    # ── Print comparison + best N report + charts ──
    print_comparison(results, dates, elapsed)

    # detailed report for the best N
    best_n = max(results, key=lambda n: results[n]['sharpe'])
    print(f"\n\n{'#'*80}")
    print(f"  DETAILED REPORT FOR BEST N = {best_n}  (Sharpe {results[best_n]['sharpe']:.3f})")
    print(f"{'#'*80}")
    r = results[best_n]
    print_detail(r['t2_trades'], dates)
    plot_performance(results, dates)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def print_comparison(results, dates, elapsed):
    print(f"{'='*80}")
    print(f"  TOP_N COMPARISON — {dates[0]} → {dates[-1]}  |  {len(dates)} days")
    print(f"{'='*80}\n")

    rows = []
    for n in sorted(results):
        r = results[n]
        ci_flag = '✓' if r['ci_lo'] > 0 else '✗'
        rows.append([
            f"N={n}", r['n'], f"{r['avg']:+.2f}%", f"{r['med']:+.2f}%",
            f"{r['wr']*100:.0f}%", f"{r['sharpe']:.3f}", f"{r['pf']:.2f}",
            f"{r['ci_lo']:+.2f}% {ci_flag}",
            f"{r['max_dd']:.1f}%",
            f"${r['final_cap']:,.0f}", f"{r['ret_pct']:+.1f}%",
            f"{r['total_pnl']:+.1f}%",
        ])

    print(tabulate(rows, headers=[
        'TOP_N','Trades','Avg','Median','WR','Sharpe','PF','CI_lo','MaxDD',
        'Final $','Return','Total PnL%'
    ], tablefmt='simple'))

    best_sharpe = max(results, key=lambda n: results[n]['sharpe'])
    best_return = max(results, key=lambda n: results[n]['ret_pct'])
    best_wr = max(results, key=lambda n: results[n]['wr'])

    print(f"\n  Best Sharpe:  N={best_sharpe}  ({results[best_sharpe]['sharpe']:.3f})")
    print(f"  Best Return:  N={best_return}  ({results[best_return]['ret_pct']:+.1f}%)")
    print(f"  Best WR:      N={best_wr}  ({results[best_wr]['wr']*100:.0f}%)")


# ═══════════════════════════════════════════════════════════════════════════════
# DETAILED REPORT (for best N)
# ═══════════════════════════════════════════════════════════════════════════════

def print_detail(t2_trades, dates):
    if not t2_trades: return

    # Monthly breakdown
    print(f"\n  ── MONTHLY BREAKDOWN (T+2) ──")
    monthly = defaultdict(list)
    for t in t2_trades:
        monthly[t['scan_date'][:7]].append(t['pnl_t2'])
    rows = []
    for m in sorted(monthly):
        a = np.array(monthly[m])
        sh = f"{np.mean(a)/np.std(a,ddof=1):.3f}" if len(a)>1 and np.std(a,ddof=1)>0 else 'n/a'
        rows.append([m, len(a), f"{np.mean(a):+.2f}%", f"{np.median(a):+.2f}%",
                     f"{(a>0).mean()*100:.0f}%", sh, f"{np.sum(a):+.1f}%"])
    print()
    print(tabulate(rows, headers=['Month','n','Avg','Median','WR','Sharpe','Total'], tablefmt='simple'))

    # local_high vs fallback
    lh = [t['pnl_t2'] for t in t2_trades if t.get('entry_type') == 'local_high']
    fb = [t['pnl_t2'] for t in t2_trades if t.get('entry_type') == 'fallback']
    print(f"\n  ── ENTRY TYPE COMPARISON ──")
    for lbl, arr in [('local_high', lh), ('fallback', fb)]:
        if arr:
            a = np.array(arr)
            sh = f"{np.mean(a)/np.std(a,ddof=1):.3f}" if len(a)>1 and np.std(a,ddof=1)>0 else 'n/a'
            print(f"  {lbl:12s}  n={len(a):<4d}  avg={np.mean(a):+.2f}%  WR={float((a>0).mean())*100:.0f}%  Sharpe={sh}")

    # Top winners / losers
    for lbl, key in [('TOP 10 WINNERS', lambda x: -x['pnl_t2']),
                      ('TOP 10 LOSERS', lambda x: x['pnl_t2'])]:
        print(f"\n  ── {lbl} ──")
        rows = []
        for t in sorted(t2_trades, key=key)[:10]:
            rows.append([t['scan_date'], t['ticker'], t['tier'],
                         f"+{t['gap_pct']*100:.1f}%",
                         t.get('entry_type','?')[:3], t.get('entry_time','?'),
                         f"${t['entry_price']:.2f}", f"{t['pnl_t2']:+.2f}%"])
        print(tabulate(rows, headers=['Date','Ticker','Tier','Gap','Ent','Time','Entry$','T+2'],
                       tablefmt='simple'))

    # Ticker frequency
    tk_counts = defaultdict(list)
    for t in t2_trades:
        tk_counts[t['ticker']].append(t['pnl_t2'])
    print(f"\n  ── MOST TRADED TICKERS ──")
    rows = []
    for tk, pnls in sorted(tk_counts.items(), key=lambda x: -len(x[1]))[:15]:
        a = np.array(pnls)
        rows.append([tk, len(a), f"{np.mean(a):+.2f}%", f"{(a>0).mean()*100:.0f}%",
                     f"{np.sum(a):+.1f}%"])
    print(tabulate(rows, headers=['Ticker','Trades','Avg','WR','Total'], tablefmt='simple'))


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_performance(results, dates):

    colors = {2:'#e74c3c', 3:'#e67e22', 4:'#f1c40f', 5:'#27ae60', 6:'#3498db'}

    # ── Figure 1: Equity curves overlay + comparison bar charts (2×2) ──
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        f'Agent 1 — TOP_N Sweep: {TOP_N_SWEEP}\n'
        f'{dates[0]} → {dates[-1]}  |  {len(dates)} trading days',
        fontsize=13, fontweight='bold')

    # 1. Equity curves overlay
    ax = axes[0, 0]
    for n in sorted(results):
        r = results[n]
        if not r['eq_dates']: continue
        eq_d = [datetime.strptime(dates[0], '%Y-%m-%d')] + \
               [datetime.strptime(d, '%Y-%m-%d') for d in r['eq_dates']]
        ax.plot(eq_d, r['equity'], label=f"N={n} ({r['ret_pct']:+.1f}%)",
                color=colors.get(n, '#999'), linewidth=2)
    ax.axhline(INITIAL_CAPITAL, color='#999', ls='--', lw=0.8)
    ax.set_ylabel('Equity ($)')
    ax.set_title('Equity Curves')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.legend(fontsize=9)

    # 2. Sharpe comparison bar
    ax = axes[0, 1]
    ns = sorted(results)
    sharpes = [results[n]['sharpe'] for n in ns]
    bar_colors = [colors.get(n,'#999') for n in ns]
    bars = ax.bar(range(len(ns)), sharpes, color=bar_colors, alpha=0.8, edgecolor='white')
    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels([f'N={n}' for n in ns])
    ax.axhline(0, color='#333', lw=0.8)
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Sharpe by TOP_N')
    for b, s in zip(bars, sharpes):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                f'{s:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 3. Avg PnL + WR comparison
    ax = axes[1, 0]
    avgs = [results[n]['avg'] for n in ns]
    wrs = [results[n]['wr']*100 for n in ns]
    x = np.arange(len(ns))
    w = 0.35
    bars1 = ax.bar(x - w/2, avgs, w, color=[colors.get(n,'#999') for n in ns], alpha=0.8, label='Avg PnL %')
    ax2 = ax.twinx()
    ax2.plot(x, wrs, 'o-', color='#2c3e50', linewidth=2, markersize=8, label='Win Rate %')
    ax.set_xticks(x)
    ax.set_xticklabels([f'N={n}' for n in ns])
    ax.axhline(0, color='#333', lw=0.8)
    ax.set_ylabel('Avg T+2 PnL (%)')
    ax2.set_ylabel('Win Rate (%)')
    ax.set_title('Avg PnL & Win Rate')
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    for b, v in zip(bars1, avgs):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.05,
                f'{v:+.2f}', ha='center', va='bottom', fontsize=8)

    # 4. Max Drawdown + Return comparison
    ax = axes[1, 1]
    rets = [results[n]['ret_pct'] for n in ns]
    dds = [results[n]['max_dd'] for n in ns]
    ax.bar(x - w/2, rets, w, color=[colors.get(n,'#999') for n in ns], alpha=0.8, label='Return %')
    ax.bar(x + w/2, dds, w, color='#e74c3c', alpha=0.4, label='Max DD %')
    ax.set_xticks(x)
    ax.set_xticklabels([f'N={n}' for n in ns])
    ax.axhline(0, color='#333', lw=0.8)
    ax.set_ylabel('%')
    ax.set_title('Total Return vs Max Drawdown')
    ax.legend(fontsize=8)
    for i, (rt, dd) in enumerate(zip(rets, dds)):
        ax.text(i - w/2, rt + 1, f'{rt:+.0f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig('agent1_bt_sweep.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n  Chart saved: agent1_bt_sweep.png")

    # ── Figure 2: Best N detailed charts (equity + drawdown + monthly + histogram) ──
    best_n = max(results, key=lambda n: results[n]['sharpe'])
    r = results[best_n]
    t2_trades = r['t2_trades']
    if len(t2_trades) < 10:
        return

    vals = [t['pnl_t2'] for t in t2_trades]
    fig2, axes2 = plt.subplots(2, 2, figsize=(18, 12))
    a = np.array(vals)
    fig2.suptitle(
        f'Agent 1 Best: N={best_n} — {len(t2_trades)} trades  |  '
        f'Avg {np.mean(a):+.2f}%  |  WR {(a>0).mean()*100:.0f}%  |  '
        f'Sharpe {r["sharpe"]:.3f}  |  Return {r["ret_pct"]:+.1f}%',
        fontsize=12, fontweight='bold')

    # Equity
    ax = axes2[0, 0]
    eq_d = [datetime.strptime(dates[0], '%Y-%m-%d')] + \
           [datetime.strptime(d, '%Y-%m-%d') for d in r['eq_dates']]
    ax.plot(eq_d, r['equity'], color='#2c3e50', linewidth=2)
    ax.fill_between(eq_d, INITIAL_CAPITAL, r['equity'], alpha=0.15,
                    where=[e >= INITIAL_CAPITAL for e in r['equity']], color='#27ae60')
    ax.fill_between(eq_d, INITIAL_CAPITAL, r['equity'], alpha=0.15,
                    where=[e < INITIAL_CAPITAL for e in r['equity']], color='#e74c3c')
    ax.axhline(INITIAL_CAPITAL, color='#999', ls='--', lw=0.8)
    ax.set_title(f'Equity (${INITIAL_CAPITAL:,.0f} → ${r["final_cap"]:,.0f})')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))

    # Drawdown
    ax = axes2[0, 1]
    eq_arr = np.array(r['equity'])
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / peak * 100
    ax.fill_between(eq_d, dd, 0, alpha=0.4, color='#e74c3c')
    ax.plot(eq_d, dd, color='#c0392b', linewidth=1.5)
    ax.set_title(f'Drawdown (Max: {dd.min():.1f}%)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    # Monthly
    ax = axes2[1, 0]
    monthly = defaultdict(list)
    for t in t2_trades:
        monthly[t['scan_date'][:7]].append(t['pnl_t2'])
    months = sorted(monthly)
    m_avg = [np.mean(monthly[m]) for m in months]
    m_colors_list = ['#27ae60' if v > 0 else '#e74c3c' for v in m_avg]
    bp = ax.bar(range(len(months)), m_avg, color=m_colors_list, alpha=0.8)
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels([m[5:] for m in months], fontsize=9)
    ax.axhline(0, color='#333', lw=0.8)
    ax.set_title('Monthly Avg PnL')
    for i, (b, m) in enumerate(zip(bp, months)):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.05,
                f'n={len(monthly[m])}', ha='center', va='bottom', fontsize=7)

    # Histogram
    ax = axes2[1, 1]
    clip_lo, clip_hi = np.percentile(vals, 1), np.percentile(vals, 99)
    bins = np.linspace(clip_lo, clip_hi, 50)
    ax.hist(vals, bins=bins, color='#3498db', alpha=0.7, edgecolor='white')
    ax.axvline(0, color='#333', lw=1.2)
    ax.axvline(np.mean(vals), color='#e74c3c', ls='--', lw=2, label=f'Mean: {np.mean(vals):+.2f}%')
    ax.set_xlabel('T+2 PnL (%)')
    ax.set_title('PnL Distribution')
    ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig('agent1_bt_best.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Chart saved: agent1_bt_best.png")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

all_results = run_backtest()
