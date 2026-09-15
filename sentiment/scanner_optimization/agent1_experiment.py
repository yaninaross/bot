# %% Agent 1: Universe Narrowing Experiment
# Research question: What signals applied to the full B1 universe reliably identify
# top 3-5 candidates/day with avg PnL significantly above the +3.28% baseline?
# Experiments: A1 gap tiers, A2 price levels, A3 volume, A4 ranking function, A5 regime
#
# Data sources:
#   - Polygon grouped daily (all US tickers on each date) → cached
#   - Polygon minute bars (for ORB/entry computation) → reuse v3 cache
#   - BQ LLM universe (for is_llm tagging) → cached
#   - v2_daily_data.json (for volume ratios — 20-day lookback)
#   - SPY daily (for regime analysis) → cached
#
# Output: AGENT_1_NARROWING_RESULTS.md + agent1_*.png

import os, sys, json, warnings, re, time
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
import pytz
from scipy import stats as sp_stats

import matplotlib
try:
    get_ipython()
except NameError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

for _pkg, _imp in [('polygon-api-client', 'polygon'),
                    ('google-cloud-bigquery', 'google.cloud.bigquery'),
                    ('requests', 'requests')]:
    try:
        __import__(_imp)
    except ImportError:
        import subprocess; subprocess.check_call([sys.executable, "-m", "pip", "install", _pkg, "--quiet"])

from polygon import RESTClient as PolygonClient
from google.cloud import bigquery
import requests as _requests

try:
    from google.colab import auth
    try: auth.authenticate_user()
    except: pass
except: pass


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

TRADING_DATES = ['2026-09-08', '2026-09-09', '2026-09-10', '2026-09-11']
N_DATES = len(TRADING_DATES)

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "YOUR_POLYGON_API_KEY")
BQ_PREMARKET_TABLE = "YOUR_GCP_PROJECT.scoring_logs.premarket_scanner"
BQ_DISCOVERY_TABLE = "YOUR_GCP_PROJECT.scoring_logs.discovery_scanner"

API_DELAY = 0.02
TICKER_REGEX = re.compile(r'^[A-Z]{1,5}$')

ET = pytz.timezone('America/New_York')
MARKET_OPEN = "09:30"
MARKET_CLOSE = "16:00"

STARTING_CAPITAL = 100_000
FLOAT_MIN = 10_000_000

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

_W = 130

_EMBEDDED_FLOAT = {
    'AAPL':14594180000, 'AARD':21884158, 'ABM':58580923, 'ACHR':770023800, 'ACMR':64657388, 'ACV':10396028, 'ACVA':169807980, 'ADBE':397500000,
    'AEHR':32620450, 'AENT':50979630, 'AEON':49882790, 'AGCO':70031729, 'AGRZ':22573405, 'AHMA':10965000, 'ALAB':173485104, 'AMAT':793597443,
    'AMD':1632475042, 'AME':229203002, 'APA':350351510, 'APLD':291469112, 'ARBE':122649743, 'ARM':1068078760, 'ASAN':159441754, 'ASO':62018371,
    'ASTS':299789305, 'ATEC':154451076, 'ATHE':18208368, 'ATI':136170383, 'AVAV':50822615, 'AXGN':53761513, 'AXTA':214019997, 'AXTI':65573212,
    'BABA':2485623615, 'BBIO':195487474, 'BBNX':44989688, 'BE':294527346, 'BEAM':103287665, 'BHVN':151043781, 'BJDX':4468051, 'BLZE':61900000,
    'BNC':41173850, 'BNKK':8000940, 'BOW':32943005, 'BRNX':720888, 'BRZE':112631669, 'BTBT':360633503, 'CAN':749194858, 'CAT':459674889,
    'CC':150498995, 'CCB':15286327, 'CIFR':415030722, 'CLSK':256817073, 'COIN':222803032, 'COLA':4494439, 'COO':195030630, 'CORZ':321340441,
    'CPRT':925811482, 'CRDO':187951918, 'CRMT':8663493, 'CSR':16794151, 'CVX':1975771274, 'DBGI':933509, 'DELL':324873640, 'DFNS':1663806,
    'DKS':65355923, 'DNOW':180785891, 'DOV':134680225, 'DPRO':34375971, 'DSS':10042518, 'EAT':41765010, 'EDHL':5959275, 'ELOG':12832000,
    'ET':3443299601, 'ETN':388400000, 'EVRG':230567172, 'EXEL':247781692, 'FCEL':79954196, 'FCX':1436017523, 'FEIM':11177530, 'FIZZ':93615302,
    'FLNC':143163588, 'FLWS':37030262, 'GDDY':126647704, 'GENC':12338845, 'GLXG':5601515, 'GME':448691257, 'GMEX':903642, 'GOOS':46656876,
    'GRI':2186115, 'HBM':444144760, 'HCAI':7629942, 'HGTY':102070523, 'HOFT':10741650, 'HOOD':790630234, 'HPE':1327464779, 'HPQ':901790969,
    'HTLD':77332611, 'HUT':123259468, 'HXHX':8950000, 'IBTA':20197545, 'IESC':19924356, 'IMO':481577721, 'INBX':14717660, 'INDP':133242324,
    'INFQ':225357052, 'INTC':5044000000, 'INV':84612657, 'IONQ':381002314, 'IP':529569895, 'IPDN':14450323, 'IREN':394058648, 'KEEL':617573212,
    'KPTI':22681460, 'KR':612647282, 'KTOS':187721327, 'LASE':50177716, 'LEU':19233658, 'LOVE':14638418, 'LULU':105594064, 'LUNR':173231343,
    'M':263037954, 'MARA':386299297, 'MAS':197187430, 'MCFT':16279890, 'MDCX':63180610, 'META':2205128509, 'MGN':16250000, 'MOBX':16847921,
    'MRNA':399235889, 'MSTR':364585501, 'MU':1129393151, 'MVIS':23788853, 'MXL':90693191, 'NBIS':219465088, 'NEE':2085803189, 'NEGG':20973423,
    'NNE':53698146, 'NTHI':25936365, 'NTLA':140126693, 'NVDA':24100000000, 'NVS':2034819509, 'OCGN':339110401, 'OKLO':186017650, 'ONDS':571454656,
    'OPTT':270138823, 'OPTX':40279878, 'ORCL':2880471000, 'OTLK':243447554, 'PAVS':856897, 'PCLA':9613805, 'PDEX':3203334, 'PDSB':55971338,
    'PGY':72137266, 'PL':340354193, 'PLNT':75216965, 'PLTR':2300713329, 'PLUG':1397195278, 'PMTS':11530669, 'PRLD':79803485, 'QBTS':368835324,
    'QNTM':3816937, 'RDDT':146103200, 'RDW':249992609, 'REGN':101137842, 'RGTI':333768747, 'RH':18900969, 'RIOT':375258935, 'RKLB':598350482,
    'RLMD':106669846, 'ROIV':722323015, 'RTX':1347758144, 'SCCO':834331706, 'SDRL':62541443, 'SHMD':60958903, 'SHOE':27151308, 'SIG':39329783,
    'SKYX':135430994, 'SLS':201945709, 'SLXN':5755872, 'SMCI':656965384, 'SMR':410389522, 'SNDK':146419001, 'SNX':79958752, 'SNYR':15079956,
    'SOUN':411576753, 'SRPT':105627021, 'SSL':647761670, 'SSNC':234654996, 'STRL':30588518, 'STX':226644518, 'SURG':52908423, 'SWKS':150472782,
    'SWRD':211149963, 'SWVL':9964344, 'TEN':30127603, 'TNON':1032578, 'TPST':16539717, 'TRUG':1903708, 'TSLA':3949547394, 'TTAN':82746425,
    'TTD':426771565, 'TYRA':59676939, 'UBER':2042560121, 'UNFI':60519140, 'UPST':97313046, 'URI':62242489, 'VAC':34395320, 'VICR':34389014,
    'VOYG':58217592, 'VRT':384988173, 'VSME':2750784, 'VST':335635195, 'WDH':279111455, 'WULF':498968677, 'YARW':2669746, 'ZEO':41703489,
    'ZUMZ':16872215,
}


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_session_bars(bar_map):
    return [(t, b) for t, b in sorted(bar_map.items()) if MARKET_OPEN <= t < MARKET_CLOSE]


def get_pm_bars(bar_map):
    return [(t, b) for t, b in sorted(bar_map.items()) if '04:00' <= t < '09:30']


def compute_orb(bars, n_bars=5):
    if len(bars) < n_bars:
        return None, None
    return (max(bars[i][1]['h'] for i in range(n_bars)),
            min(bars[i][1]['l'] for i in range(n_bars)))


def find_local_high(bars, window=3, max_peak_bar=12, fallback_bar=15):
    for i in range(window, min(max_peak_bar, len(bars) - window)):
        ch = bars[i][1]['h']
        if all(bars[i - j][1]['h'] < ch and bars[i + j][1]['h'] < ch
               for j in range(1, window + 1)):
            ci = i + window
            if ci < len(bars):
                return {'entry_price': bars[ci][1]['c'], 'entry_bar_idx': ci,
                        'entry_time': bars[ci][0], 'entry_type': 'local_high',
                        'peak_bar': i}
    if fallback_bar < len(bars):
        return {'entry_price': bars[fallback_bar][1]['c'], 'entry_bar_idx': fallback_bar,
                'entry_time': bars[fallback_bar][0], 'entry_type': 'fallback'}
    return None


def compute_pm_retrace(bar_map):
    pm = get_pm_bars(bar_map)
    if len(pm) < 2:
        return None
    pm_open = pm[0][1]['o']
    pm_high = max(b['h'] for _, b in pm)
    pm_last = pm[-1][1]['c']
    denom = pm_high - pm_open
    if denom <= 0:
        return None
    return (pm_high - pm_last) / denom


def pm_retrace_to_score(retrace):
    if retrace is None:
        return 0.5
    if retrace > 0.8:
        return 0.0
    if 0.2 <= retrace <= 0.5:
        return 1.0
    if retrace < 0.2:
        return 0.8
    return 0.6


def normalize_val(value, mn, mx):
    if value is None:
        return 0.5
    if mx <= mn:
        return 0.5
    return max(0.0, min(1.0, (value - mn) / (mx - mn)))


def trim_mean(arr, prop=0.1):
    s = np.sort(arr)
    n = len(s)
    k = max(1, int(n * prop))
    if 2 * k >= n:
        return float(np.mean(s))
    return float(np.mean(s[k:-k]))


def full_metrics(pnl_list, n_dates=N_DATES):
    pnl = np.array([x for x in pnl_list if x is not None and np.isfinite(x)])
    n = len(pnl)
    empty = {'n': n, 'trades_per_day': round(n / n_dates, 1) if n_dates else 0}
    if n < 3:
        empty['note'] = 'insufficient'
        for k in ['avg_pnl_pct', 'median_pnl_pct', 'trimmed_mean_pct', 'std_pct',
                  'sharpe', 'win_rate', 'profit_factor', 'max_loss_pct', 'max_gain_pct',
                  'ci_95_low', 'ci_95_high', 'ci_confirmed', 'total_pnl_pct']:
            empty[k] = None
        if n > 0:
            empty['avg_pnl_pct'] = round(float(np.mean(pnl)), 3)
            empty['total_pnl_pct'] = round(float(np.sum(pnl)), 2)
        return empty

    avg = float(np.mean(pnl))
    med = float(np.median(pnl))
    std = float(np.std(pnl, ddof=1))
    se = std / np.sqrt(n)
    winners = pnl[pnl > 0]
    losers = pnl[pnl < 0]
    pf = float(winners.sum() / abs(losers.sum())) if losers.sum() != 0 else (
        float('inf') if len(winners) > 0 else 0.0)
    ci_lo = avg - 1.96 * se
    ci_hi = avg + 1.96 * se

    return {
        'n': n,
        'trades_per_day': round(n / n_dates, 1) if n_dates else 0,
        'avg_pnl_pct': round(avg, 3),
        'median_pnl_pct': round(med, 3),
        'trimmed_mean_pct': round(trim_mean(pnl, 0.1), 3),
        'std_pct': round(std, 3),
        'sharpe': round(avg / std, 3) if std > 0 else 0,
        'win_rate': round(float((pnl > 0).mean()), 3),
        'profit_factor': round(pf, 2) if pf < 100 else float('inf'),
        'max_loss_pct': round(float(np.min(pnl)), 3),
        'max_gain_pct': round(float(np.max(pnl)), 3),
        'ci_95_low': round(ci_lo, 3),
        'ci_95_high': round(ci_hi, 3),
        'ci_confirmed': bool(ci_lo > 0),
        'total_pnl_pct': round(float(np.sum(pnl)), 2),
    }


def leave_one_out_check(pnl_list):
    arr = np.array([x for x in pnl_list if x is not None and np.isfinite(x)])
    if len(arr) < 5:
        return {'stable': None, 'reason': 'too few'}
    base_avg = np.mean(arr)
    loo_avgs = [np.mean(np.delete(arr, i)) for i in range(len(arr))]
    loo_avgs = np.array(loo_avgs)
    sign_flip = (np.min(loo_avgs) < 0 < base_avg) or (np.max(loo_avgs) > 0 > base_avg)
    return {
        'stable': not sign_flip,
        'base_avg': round(float(base_avg), 3),
        'min_loo': round(float(np.min(loo_avgs)), 3),
        'max_loo': round(float(np.max(loo_avgs)), 3),
        'sign_flip': bool(sign_flip),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING — Build B1 Universe from Polygon + cached minute bars
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*_W}")
print(f"  AGENT 1: UNIVERSE NARROWING EXPERIMENT")
print(f"  Building B1_WIDE_SHORT universe (all gap-up >= 1.5%, price >= $5)")
print(f"{'='*_W}")

poly = PolygonClient(POLYGON_API_KEY)

# ── SPY daily ─────────────────────────────────────────────────────────────────

print(f"\n  [1/6] Loading SPY daily...")

spy_daily = {}
spy_cache = os.path.join(RESULTS_DIR, 'a1_spy_daily.json')
v2_spy_cache = os.path.join(RESULTS_DIR, 'v2_spy_daily.json')

if os.path.exists(spy_cache):
    with open(spy_cache) as f:
        spy_daily = json.load(f)
    print(f"    From a1 cache: {len(spy_daily)} days")
elif os.path.exists(v2_spy_cache):
    with open(v2_spy_cache) as f:
        spy_daily = json.load(f)
    print(f"    From v2 cache: {len(spy_daily)} days")
else:
    first_dt = datetime.strptime(TRADING_DATES[0], '%Y-%m-%d')
    cal_start = (first_dt - timedelta(days=30)).strftime('%Y-%m-%d')
    cal_end = TRADING_DATES[-1]
    for b in poly.get_aggs('SPY', 1, 'day', from_=cal_start, to=cal_end, limit=500):
        dt_str = pd.Timestamp(b.timestamp, unit='ms', tz='UTC').strftime('%Y-%m-%d')
        spy_daily[dt_str] = {'open': float(b.open), 'high': float(b.high),
                             'low': float(b.low), 'close': float(b.close),
                             'volume': int(b.volume)}
    with open(spy_cache, 'w') as f:
        json.dump(spy_daily, f)
    print(f"    Fetched: {len(spy_daily)} days")

trading_days = sorted(spy_daily.keys())

# Build date info with prior/T+1/T+2 mapping
date_info = {}
for dt in TRADING_DATES:
    if dt not in trading_days:
        print(f"    WARNING: {dt} not in SPY calendar")
        continue
    idx = trading_days.index(dt)
    date_info[dt] = {
        'prior': trading_days[idx - 1] if idx > 0 else None,
        't1': trading_days[idx + 1] if idx + 1 < len(trading_days) else None,
        't2': trading_days[idx + 2] if idx + 2 < len(trading_days) else None,
    }

# ── Grouped daily ────────────────────────────────────────────────────────────

print(f"\n  [2/6] Loading grouped daily data...")

grouped_daily = {}
gd_cache = os.path.join(RESULTS_DIR, 'a1_grouped_daily.json')

if os.path.exists(gd_cache):
    with open(gd_cache) as f:
        grouped_daily = json.load(f)
    print(f"    From cache: {len(grouped_daily)} dates")

all_needed_dates = set(TRADING_DATES)
for di in date_info.values():
    for k in ('prior', 't1', 't2'):
        if di.get(k):
            all_needed_dates.add(di[k])

# Add 25 calendar days before earliest trading date for 20-day volume lookback
first_dt = datetime.strptime(min(TRADING_DATES), '%Y-%m-%d')
for i in range(1, 30):
    d = (first_dt - timedelta(days=i)).strftime('%Y-%m-%d')
    if d in trading_days:
        all_needed_dates.add(d)

missing_gd = sorted(set(all_needed_dates) - set(grouped_daily.keys()))
if missing_gd:
    print(f"    Fetching {len(missing_gd)} missing dates from Polygon...")
    for dt in missing_gd:
        url = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{dt}"
        try:
            resp = _requests.get(url, params={"apiKey": POLYGON_API_KEY, "adjusted": "true"}, timeout=30)
            if resp.status_code == 200:
                results = resp.json().get('results', [])
                day_data = {}
                for r in results:
                    tk = r.get('T', '')
                    if not TICKER_REGEX.match(tk):
                        continue
                    day_data[tk] = {'o': r.get('o', 0), 'h': r.get('h', 0),
                                    'l': r.get('l', 0), 'c': r.get('c', 0),
                                    'v': int(r.get('v', 0))}
                grouped_daily[dt] = day_data
                print(f"      {dt}: {len(day_data)} tickers")
            else:
                print(f"      {dt}: HTTP {resp.status_code}")
                grouped_daily[dt] = {}
        except Exception as e:
            print(f"      {dt}: error {e}")
            grouped_daily[dt] = {}
        time.sleep(API_DELAY)
    with open(gd_cache, 'w') as f:
        json.dump(grouped_daily, f)
    print(f"    Cache saved: {os.path.getsize(gd_cache)/(1024*1024):.1f} MB")

# ── LLM Universe ─────────────────────────────────────────────────────────────

print(f"\n  [3/6] Loading LLM universe from BQ...")

llm_universe = {}
llm_cache = os.path.join(RESULTS_DIR, 'a1_llm_universe.json')

if os.path.exists(llm_cache):
    with open(llm_cache) as f:
        raw = json.load(f)
    llm_universe = {tuple(k.split('__')): v for k, v in raw.items()}
    print(f"    From cache: {len(llm_universe)} signals")
else:
    bq_client = bigquery.Client()
    _dates_sql = ", ".join(f"'{d}'" for d in TRADING_DATES)
    for table, scanner in [(BQ_PREMARKET_TABLE, 'premarket'),
                            (BQ_DISCOVERY_TABLE, 'discovery')]:
        q = f"""
        SELECT CAST(scan_date AS STRING) AS eval_date,
               ticker, direction,
               COALESCE(catalyst_type, 'unknown') AS catalyst_type,
               COALESCE(catalyst, '') AS catalyst
        FROM `{table}`
        WHERE scan_date IN ({_dates_sql})
          AND direction IN ('LONG', 'SHORT')
        ORDER BY scan_date, ticker
        """
        df_bq = bq_client.query(q).to_dataframe()
        for _, row in df_bq.iterrows():
            key = (row['ticker'], str(row['eval_date']))
            if key in llm_universe:
                continue
            cat = str(row['catalyst_type']).lower()
            llm_universe[key] = {
                'direction': row['direction'],
                'catalyst_type': cat,
                'scanner': scanner,
            }
        print(f"    {scanner}: {len(df_bq)} rows")

    cache_data = {f"{k[0]}__{k[1]}": v for k, v in llm_universe.items()}
    with open(llm_cache, 'w') as f:
        json.dump(cache_data, f)
    print(f"    Total: {len(llm_universe)} unique signals")

# ── Volume lookback ──────────────────────────────────────────────────────────

print(f"\n  [4/6] Volume lookback ready (using grouped_daily, {len(grouped_daily)} dates)")

# ── Minute bars ──────────────────────────────────────────────────────────────

print(f"\n  [5/6] Loading minute bars...")

minute_bars = {}
v3_path = os.path.join(RESULTS_DIR, 'v3_minute_bars.json')
mb_cache = os.path.join(RESULTS_DIR, 'a1_minute_bars.json')

if os.path.exists(v3_path):
    with open(v3_path) as f:
        v3_raw = json.load(f)
    if 'ticker_bars' in v3_raw:
        for k, v in v3_raw['ticker_bars'].items():
            parts = k.split('__')
            if len(parts) == 2:
                minute_bars[(parts[0], parts[1])] = v
    print(f"    v3 seed: {len(minute_bars)} ticker-dates")
    del v3_raw

if os.path.exists(mb_cache):
    with open(mb_cache) as f:
        raw_mb = json.load(f)
    for k, v in raw_mb.items():
        parts = k.split('__')
        if len(parts) == 2:
            minute_bars[(parts[0], parts[1])] = v
    print(f"    a1 cache: total {len(minute_bars)} ticker-dates")
    del raw_mb

# ── Build gapper universe ────────────────────────────────────────────────────

print(f"\n  [6/6] Building gapper universe (gap-up >= 1.5%, price >= $5)...")

gappers = []
for dt in TRADING_DATES:
    di = date_info.get(dt)
    if not di or not di.get('prior'):
        continue
    today_data = grouped_daily.get(dt, {})
    prior_data = grouped_daily.get(di['prior'], {})
    n_gap = 0

    for tk in today_data:
        if tk not in prior_data:
            continue
        prev_c = prior_data[tk]['c']
        today_o = today_data[tk]['o']
        if prev_c <= 0 or today_o <= 0:
            continue
        if today_o < 5.0:
            continue
        gap_pct = (today_o - prev_c) / prev_c * 100
        if gap_pct < 1.5:
            continue

        llm = llm_universe.get((tk, dt))

        # Volume ratio: gap_day_vol / 20-day avg (from grouped_daily)
        volume_ratio = None
        prior_vols = []
        prior_dates = sorted(d for d in grouped_daily if d < dt)[-20:]
        for pd_dt in prior_dates:
            gd_day = grouped_daily.get(pd_dt, {})
            if tk in gd_day and gd_day[tk].get('v', 0) > 0:
                prior_vols.append(gd_day[tk]['v'])
        if len(prior_vols) >= 5:
            avg_vol = np.mean(prior_vols)
            today_vol = today_data[tk]['v']
            if avg_vol > 0:
                volume_ratio = today_vol / avg_vol

        gappers.append({
            'ticker': tk, 'date': dt, 'gap_pct': gap_pct,
            'open_price': today_o, 'prev_close': prev_c,
            'today_close': today_data[tk]['c'],
            'today_high': today_data[tk]['h'],
            'today_low': today_data[tk]['l'],
            'today_volume': today_data[tk]['v'],
            'is_llm': llm is not None,
            'catalyst_type': llm['catalyst_type'] if llm else 'unknown',
            't2_date': di.get('t2'),
            'volume_ratio': volume_ratio,
            'float_shares': _EMBEDDED_FLOAT.get(tk),
        })
        n_gap += 1

    print(f"    {dt}: {n_gap} gap-up tickers >= 1.5% and >= $5")

print(f"    Total B1 universe: {len(gappers)}")

# ── Fetch missing minute bars for B1 universe ────────────────────────────────

needed_mb = set((g['ticker'], g['date']) for g in gappers)
missing_mb = sorted(needed_mb - set(minute_bars.keys()))

if missing_mb:
    print(f"\n  Fetching {len(missing_mb)} missing minute-bar sets from Polygon...")
    n_ok = 0
    t0 = time.time()
    for i, (tk, dt) in enumerate(missing_mb):
        try:
            raw = list(poly.get_aggs(ticker=tk, multiplier=1, timespan="minute",
                                     from_=dt, to=dt, limit=500))
            if raw:
                bar_map = {}
                for b in raw:
                    ts = pd.Timestamp(b.timestamp, unit='ms', tz='UTC').tz_convert(ET)
                    bar_map[ts.strftime('%H:%M')] = {
                        'o': float(b.open), 'h': float(b.high),
                        'l': float(b.low), 'c': float(b.close),
                        'v': int(b.volume)}
                minute_bars[(tk, dt)] = bar_map
                n_ok += 1
        except Exception:
            pass
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(missing_mb) - i - 1) / rate if rate > 0 else 0
            print(f"      {i+1}/{len(missing_mb)} ({n_ok} ok, {rate:.0f}/s, ETA {eta/60:.1f}m)")
        if API_DELAY > 0:
            time.sleep(API_DELAY)

    print(f"    Fetched: {n_ok}/{len(missing_mb)} ({time.time()-t0:.0f}s)")
    save_mb = {f"{k[0]}__{k[1]}": v for k, v in minute_bars.items()
               if k in needed_mb}
    with open(mb_cache, 'w') as f:
        json.dump(save_mb, f)
    print(f"    Cache saved: {os.path.getsize(mb_cache)/(1024*1024):.1f} MB")


# ── Compute entry, PnL, features for each gapper ────────────────────────────

print(f"\n  Computing entries and PnL...")

trades = []
funnel = defaultdict(int)

for g in gappers:
    tk, dt = g['ticker'], g['date']
    funnel['start'] += 1

    bar_map = minute_bars.get((tk, dt))
    if bar_map is None:
        funnel['no_bars'] += 1
        continue

    bars = get_session_bars(bar_map)
    if len(bars) < 20:
        funnel['short_session'] += 1
        continue

    entry = find_local_high(bars, window=3, max_peak_bar=12, fallback_bar=15)
    if entry is None or entry.get('entry_price', 0) <= 0:
        funnel['no_entry'] += 1
        continue

    ep = entry['entry_price']
    entry_idx = entry['entry_bar_idx']

    # T+2 PnL
    pnl_t2 = None
    t2d = g.get('t2_date')
    if t2d and t2d in grouped_daily and tk in grouped_daily.get(t2d, {}):
        exit_price = grouped_daily[t2d][tk]['c']
        pnl_t2 = (ep - exit_price) / ep * 100

    # 30m PnL
    pnl_30m = None
    if entry_idx < 29 and len(bars) > 29:
        pnl_30m = (ep - bars[29][1]['c']) / ep * 100

    # 60m PnL
    pnl_60m = None
    if entry_idx < 59 and len(bars) > 59:
        pnl_60m = (ep - bars[59][1]['c']) / ep * 100

    # PM retrace
    pm_retrace = compute_pm_retrace(bar_map)

    # ORB
    orb_high, orb_low = compute_orb(bars)
    orb_width = None
    if orb_high is not None and orb_low is not None and ep > 0:
        orb_width = (orb_high - orb_low) / ep

    # SPY context on gap day
    spy_gap_pct = None
    di = date_info.get(dt, {})
    if di.get('prior') and dt in spy_daily and di['prior'] in spy_daily:
        spy_prev_c = spy_daily[di['prior']].get('close', spy_daily[di['prior']].get('c', 0))
        spy_today_o = spy_daily[dt].get('open', spy_daily[dt].get('o', 0))
        if spy_prev_c > 0:
            spy_gap_pct = (spy_today_o - spy_prev_c) / spy_prev_c * 100

    spy_daily_ret = None
    if dt in spy_daily:
        sd = spy_daily[dt]
        so = sd.get('open', sd.get('o', 0))
        sc = sd.get('close', sd.get('c', 0))
        if so > 0:
            spy_daily_ret = (sc - so) / so * 100

    trade = {
        'ticker': tk, 'date': dt,
        'gap_pct': round(g['gap_pct'], 3),
        'open_price': g['open_price'],
        'entry_price': round(ep, 4),
        'entry_type': entry.get('entry_type', ''),
        'entry_bar_idx': entry_idx,
        'entry_time': entry.get('entry_time', ''),
        'pnl_t2': round(pnl_t2, 4) if pnl_t2 is not None else None,
        'pnl_30m': round(pnl_30m, 4) if pnl_30m is not None else None,
        'pnl_60m': round(pnl_60m, 4) if pnl_60m is not None else None,
        'pm_retrace': pm_retrace,
        'orb_width': orb_width,
        'volume_ratio': g['volume_ratio'],
        'float_shares': g['float_shares'],
        'is_llm': g['is_llm'],
        'catalyst_type': g['catalyst_type'],
        'today_volume': g['today_volume'],
        'spy_gap_pct': spy_gap_pct,
        'spy_daily_ret': spy_daily_ret,
    }
    trades.append(trade)

funnel['qualifying'] = len(trades)
n_t2 = sum(1 for t in trades if t['pnl_t2'] is not None)
print(f"    Qualifying: {len(trades)} (with T+2: {n_t2})")
print(f"    Funnel: {dict(funnel)}")

trade_df = pd.DataFrame(trades)

# Identify dates with T+2 data
dates_with_t2 = sorted(trade_df[trade_df['pnl_t2'].notna()]['date'].unique())
N_DATES_T2 = len(dates_with_t2)
print(f"    Dates with T+2: {dates_with_t2} ({N_DATES_T2} dates)")

# Baseline
baseline_pnl = trade_df['pnl_t2'].dropna().tolist()
baseline_m = full_metrics(baseline_pnl, N_DATES_T2)
print(f"\n  BASELINE (all qualifying, T+2):")
print(f"    n={baseline_m['n']}, avg={baseline_m.get('avg_pnl_pct', 0):+.3f}%, "
      f"med={baseline_m.get('median_pnl_pct', 0):+.3f}%, "
      f"WR={baseline_m.get('win_rate', 0)*100:.0f}%, "
      f"Sharpe={baseline_m.get('sharpe', 0):.3f}, "
      f"CI=[{baseline_m.get('ci_95_low', 0):+.2f}, {baseline_m.get('ci_95_high', 0):+.2f}]")

# Apply float filter = L1
l1_trades = trade_df[(trade_df['float_shares'].isna()) | (trade_df['float_shares'] >= FLOAT_MIN)]
l1_pnl = l1_trades['pnl_t2'].dropna().tolist()
l1_m = full_metrics(l1_pnl, N_DATES_T2)
print(f"\n  L1 (float >= 10M or unknown):")
print(f"    n={l1_m['n']}, avg={l1_m.get('avg_pnl_pct', 0):+.3f}%, "
      f"Sharpe={l1_m.get('sharpe', 0):.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# A1: Gap Magnitude Stratification
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*_W}")
print(f"  A1: Gap Magnitude Stratification")
print(f"{'='*_W}")

gap_tiers = {
    'tier_1_small':   (1.5, 2.0),
    'tier_2_medium':  (2.0, 3.0),
    'tier_3_large':   (3.0, 5.0),
    'tier_4_xlarge':  (5.0, 8.0),
    'tier_5_extreme': (8.0, None),
}

a1_results = {}

print(f"\n  {'Tier':>18} {'Gap Range':>12} {'n':>5} {'n_t2':>5} {'t/d':>6} "
      f"{'avg%':>8} {'med%':>8} {'WR':>6} {'Sharpe':>7} {'CI_lo':>8} {'CI_hi':>8} {'CI>0':>6}")
print(f"  {'─'*110}")

for tier_name, (lo, hi) in gap_tiers.items():
    if hi is not None:
        subset = trade_df[(trade_df['gap_pct'] >= lo) & (trade_df['gap_pct'] < hi)]
    else:
        subset = trade_df[trade_df['gap_pct'] >= lo]

    n_total = len(subset)
    pnl = subset['pnl_t2'].dropna().tolist()
    m = full_metrics(pnl, N_DATES_T2)
    loo = leave_one_out_check(pnl)
    a1_results[tier_name] = {'lo': lo, 'hi': hi, 'n_total': n_total, 'n_t2': len(pnl), **m,
                             'loo': loo}

    range_str = f"{lo}-{hi}%" if hi else f">{lo}%"
    avg_s = f"{m.get('avg_pnl_pct', 0) or 0:>+8.2f}"
    med_s = f"{m.get('median_pnl_pct', 0) or 0:>+8.2f}"
    wr_s = f"{m.get('win_rate', 0)*100 if m.get('win_rate') else 0:>5.0f}%"
    sh_s = f"{m.get('sharpe', 0) or 0:>7.3f}"
    ci_lo_s = f"{m.get('ci_95_low', 0) or 0:>+8.2f}"
    ci_hi_s = f"{m.get('ci_95_high', 0) or 0:>+8.2f}"
    ci_ok = f"{'YES' if m.get('ci_confirmed') else 'NO':>6}"
    tpd = round(n_total / N_DATES, 1)

    print(f"  {tier_name:>18} {range_str:>12} {n_total:>5} {len(pnl):>5} {tpd:>6.1f} "
          f"{avg_s} {med_s} {wr_s} {sh_s} {ci_lo_s} {ci_hi_s} {ci_ok}")

# Top percentile analysis
for pct_label, pct in [('Top 10%', 0.10), ('Top 5%', 0.05), ('Top 3%', 0.03)]:
    threshold = trade_df['gap_pct'].quantile(1 - pct)
    subset = trade_df[trade_df['gap_pct'] >= threshold]
    pnl = subset['pnl_t2'].dropna().tolist()
    m = full_metrics(pnl, N_DATES_T2)
    a1_results[f'percentile_{pct_label}'] = {
        'threshold': round(threshold, 2), 'n_total': len(subset),
        'n_t2': len(pnl), **m}
    print(f"\n  {pct_label} (gap >= {threshold:.1f}%): n={len(subset)} ({len(pnl)} T+2), "
          f"avg={m.get('avg_pnl_pct', 0) or 0:+.2f}%, Sharpe={m.get('sharpe', 0) or 0:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# A2: Price Level Stratification
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*_W}")
print(f"  A2: Price Level Stratification")
print(f"{'='*_W}")

price_thresholds = [5, 10, 15, 20, 30, 50, 100]
a2_results = {}

print(f"\n  {'Min Price':>10} {'n':>5} {'n_t2':>5} {'t/d':>6} "
      f"{'avg%':>8} {'med%':>8} {'WR':>6} {'Sharpe':>7} {'CI_lo':>8} {'CI>0':>6}")
print(f"  {'─'*80}")

for price in price_thresholds:
    subset = trade_df[trade_df['open_price'] >= price]
    pnl = subset['pnl_t2'].dropna().tolist()
    m = full_metrics(pnl, N_DATES_T2)
    a2_results[price] = {'n_total': len(subset), 'n_t2': len(pnl), **m}

    avg_s = f"{m.get('avg_pnl_pct', 0) or 0:>+8.2f}"
    med_s = f"{m.get('median_pnl_pct', 0) or 0:>+8.2f}"
    wr_s = f"{m.get('win_rate', 0)*100 if m.get('win_rate') else 0:>5.0f}%"
    sh_s = f"{m.get('sharpe', 0) or 0:>7.3f}"
    ci_lo_s = f"{m.get('ci_95_low', 0) or 0:>+8.2f}"
    ci_ok = f"{'YES' if m.get('ci_confirmed') else 'NO':>6}"
    tpd = round(len(subset) / N_DATES, 1)

    print(f"  ${price:>8} {len(subset):>5} {len(pnl):>5} {tpd:>6.1f} "
          f"{avg_s} {med_s} {wr_s} {sh_s} {ci_lo_s} {ci_ok}")

# Market cap proxy
print(f"\n  Market cap proxy (price × float shares):")
has_cap = trade_df[trade_df['float_shares'].notna()].copy()
has_cap['market_cap'] = has_cap['open_price'] * has_cap['float_shares']
cap_thresholds = [100e6, 500e6, 1e9, 5e9, 10e9]

for cap in cap_thresholds:
    subset = has_cap[has_cap['market_cap'] >= cap]
    pnl = subset['pnl_t2'].dropna().tolist()
    m = full_metrics(pnl, N_DATES_T2)
    label = f"${cap/1e9:.1f}B" if cap >= 1e9 else f"${cap/1e6:.0f}M"
    a2_results[f'mcap_{label}'] = {'n_total': len(subset), 'n_t2': len(pnl), **m}
    print(f"    >= {label}: n={len(subset)} ({len(pnl)} T+2), "
          f"avg={m.get('avg_pnl_pct', 0) or 0:+.2f}%, Sharpe={m.get('sharpe', 0) or 0:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# A3: Volume Characteristics
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*_W}")
print(f"  A3: Volume Characteristics")
print(f"{'='*_W}")

has_vr = trade_df[trade_df['volume_ratio'].notna()].copy()
print(f"\n  Trades with volume ratio: {len(has_vr)} / {len(trade_df)}")
print(f"  Volume ratio range: {has_vr['volume_ratio'].min():.2f} to {has_vr['volume_ratio'].max():.2f}")

a3_results = {}

# Volume ratio thresholds (cumulative >=)
vol_thresholds = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
print(f"\n  Volume ratio threshold sweep:")
print(f"  {'VR >=':>10} {'n':>5} {'n_t2':>5} {'t/d':>6} "
      f"{'avg%':>8} {'WR':>6} {'Sharpe':>7} {'CI_lo':>8}")
print(f"  {'─'*70}")

for vr in vol_thresholds:
    subset = has_vr[has_vr['volume_ratio'] >= vr]
    pnl = subset['pnl_t2'].dropna().tolist()
    m = full_metrics(pnl, N_DATES_T2)
    a3_results[f'vr_{vr}'] = {'threshold': vr, 'n_total': len(subset), 'n_t2': len(pnl), **m}
    tpd = round(len(subset) / N_DATES, 1)
    print(f"  {vr:>10.1f} {len(subset):>5} {len(pnl):>5} {tpd:>6.1f} "
          f"{m.get('avg_pnl_pct', 0) or 0:>+8.2f} "
          f"{m.get('win_rate', 0)*100 if m.get('win_rate') else 0:>5.0f}% "
          f"{m.get('sharpe', 0) or 0:>7.3f} "
          f"{m.get('ci_95_low', 0) or 0:>+8.2f}")

# Volume rank percentiles (top N% within day)
print(f"\n  Volume rank within day (top N%):")
for pct in [0.10, 0.05, 0.03, 0.02, 0.01]:
    selected = []
    for dt in TRADING_DATES:
        day = has_vr[has_vr['date'] == dt].sort_values('volume_ratio', ascending=False)
        n_select = max(1, int(len(day) * pct))
        selected.append(day.head(n_select))
    if selected:
        sel_df = pd.concat(selected)
        pnl = sel_df['pnl_t2'].dropna().tolist()
        m = full_metrics(pnl, N_DATES_T2)
        a3_results[f'top_{pct*100:.0f}pct'] = {'n_total': len(sel_df), 'n_t2': len(pnl), **m}
        print(f"    Top {pct*100:.0f}%: n={len(sel_df)} ({len(pnl)} T+2), "
              f"avg={m.get('avg_pnl_pct', 0) or 0:+.2f}%, "
              f"Sharpe={m.get('sharpe', 0) or 0:.3f}")

# Joint filter: large gap AND abnormal volume
print(f"\n  Joint filter: gap >= 5% AND volume ratio >= 3.0:")
joint = has_vr[(has_vr['gap_pct'] >= 5.0) & (has_vr['volume_ratio'] >= 3.0)]
pnl = joint['pnl_t2'].dropna().tolist()
m_joint = full_metrics(pnl, N_DATES_T2)
a3_results['joint_gap5_vr3'] = {'n_total': len(joint), 'n_t2': len(pnl), **m_joint}
print(f"    n={len(joint)} ({len(pnl)} T+2), avg={m_joint.get('avg_pnl_pct', 0) or 0:+.2f}%, "
      f"Sharpe={m_joint.get('sharpe', 0) or 0:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# A4: Ranking Function Optimization
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*_W}")
print(f"  A4: Ranking Function Optimization")
print(f"{'='*_W}")

# Compute percentile ranks within each day
a4_base = trade_df.copy()

def compute_day_percentile(series):
    return series.rank(pct=True, method='average')

a4_results = {}

weight_combos = {
    'gap_dominant':    (0.50, 0.30, 0.10, 0.10),
    'volume_dominant': (0.30, 0.50, 0.10, 0.10),
    'balanced':        (0.35, 0.35, 0.15, 0.15),
    'gap_vol_equal':   (0.40, 0.40, 0.10, 0.10),
    'pure_gap':        (0.60, 0.20, 0.10, 0.10),
    'pure_volume':     (0.20, 0.60, 0.10, 0.10),
    'gap_price':       (0.40, 0.20, 0.30, 0.10),
}

for combo_name, (w_gap, w_vol, w_price, w_pm) in weight_combos.items():
    for top_n in [1, 2, 3, 4, 5]:
        selected = []
        for dt in TRADING_DATES:
            day = a4_base[a4_base['date'] == dt].copy()
            if len(day) == 0:
                continue

            # Compute ranking signals
            day['gap_rank'] = compute_day_percentile(day['gap_pct'])
            day['vol_rank'] = compute_day_percentile(day['volume_ratio'].fillna(day['volume_ratio'].median() if len(day) > 0 else 0))
            day['price_rank'] = compute_day_percentile(day['open_price'])
            day['pm_score'] = day['pm_retrace'].apply(pm_retrace_to_score)

            day['agent1_score'] = (
                w_gap * day['gap_rank'] +
                w_vol * day['vol_rank'] +
                w_price * day['price_rank'] +
                w_pm * day['pm_score']
            )

            top = day.nlargest(top_n, 'agent1_score')
            selected.append(top)

        if selected:
            sel_df = pd.concat(selected)
            pnl = sel_df['pnl_t2'].dropna().tolist()
            m = full_metrics(pnl, N_DATES_T2)
            key = f"{combo_name}_top{top_n}"
            a4_results[key] = {
                'weights': {'gap': w_gap, 'vol': w_vol, 'price': w_price, 'pm': w_pm},
                'top_n': top_n,
                'n_total': len(sel_df),
                'n_t2': len(pnl),
                **m
            }

# Print best results for top-3
print(f"\n  Ranking function results (Top 3/day):")
print(f"  {'Combo':>20} {'n_t2':>5} {'avg%':>8} {'med%':>8} {'WR':>6} {'Sharpe':>7} {'CI_lo':>8}")
print(f"  {'─'*75}")

top3_results = {k: v for k, v in a4_results.items() if '_top3' in k}
for name, r in sorted(top3_results.items(), key=lambda x: -(x[1].get('sharpe') or 0)):
    avg_s = f"{r.get('avg_pnl_pct', 0) or 0:>+8.2f}"
    med_s = f"{r.get('median_pnl_pct', 0) or 0:>+8.2f}"
    wr_s = f"{r.get('win_rate', 0)*100 if r.get('win_rate') else 0:>5.0f}%"
    sh_s = f"{r.get('sharpe', 0) or 0:>7.3f}"
    ci_lo_s = f"{r.get('ci_95_low', 0) or 0:>+8.2f}"
    print(f"  {name:>20} {r['n_t2']:>5} {avg_s} {med_s} {wr_s} {sh_s} {ci_lo_s}")

# Show all top-N results for best combo
best_combo = max(top3_results.items(), key=lambda x: x[1].get('sharpe') or 0)[0]
best_prefix = best_combo.replace('_top3', '')
print(f"\n  Best combo ({best_prefix}) across top-N:")
for n in [1, 2, 3, 4, 5]:
    key = f"{best_prefix}_top{n}"
    if key in a4_results:
        r = a4_results[key]
        print(f"    Top {n}: n_t2={r['n_t2']}, avg={r.get('avg_pnl_pct', 0) or 0:+.2f}%, "
              f"Sharpe={r.get('sharpe', 0) or 0:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# A5: SPY Regime Context
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*_W}")
print(f"  A5: SPY Regime Context")
print(f"{'='*_W}")

a5_results = {}

# SPY gap day analysis
print(f"\n  Per-date SPY context:")
for dt in TRADING_DATES:
    sd = spy_daily.get(dt, {})
    di = date_info.get(dt, {})
    prior = di.get('prior')
    if prior and prior in spy_daily:
        sp = spy_daily[prior]
        spy_gap = (sd.get('open', sd.get('o', 0)) - sp.get('close', sp.get('c', 0))) / sp.get('close', sp.get('c', 1)) * 100
    else:
        spy_gap = 0
    spy_ret = 0
    so = sd.get('open', sd.get('o', 0))
    sc = sd.get('close', sd.get('c', 0))
    if so > 0:
        spy_ret = (sc - so) / so * 100
    spy_range = 0
    sh = sd.get('high', sd.get('h', 0))
    sl = sd.get('low', sd.get('l', 0))
    if so > 0:
        spy_range = (sh - sl) / so * 100

    # Count B1 trades and avg T+2
    day_trades = trade_df[trade_df['date'] == dt]
    day_t2 = day_trades['pnl_t2'].dropna()
    avg_t2 = f"{day_t2.mean():+.2f}%" if len(day_t2) > 0 else "no T+2"

    a5_results[dt] = {
        'spy_gap_pct': round(spy_gap, 3),
        'spy_intraday_ret': round(spy_ret, 3),
        'spy_range_pct': round(spy_range, 3),
        'n_trades': len(day_trades),
        'n_t2': len(day_t2),
        'avg_t2': round(float(day_t2.mean()), 3) if len(day_t2) > 0 else None,
    }

    print(f"    {dt}: SPY gap={spy_gap:+.2f}%, ret={spy_ret:+.2f}%, range={spy_range:.2f}%  "
          f"| B1: {len(day_trades)} trades, T+2 avg={avg_t2}")

# Correlation: SPY daily return vs B1 avg T+2
if len(trade_df[trade_df['spy_daily_ret'].notna() & trade_df['pnl_t2'].notna()]) >= 5:
    corr_data = trade_df[trade_df['spy_daily_ret'].notna() & trade_df['pnl_t2'].notna()]
    r, p = sp_stats.pearsonr(corr_data['spy_daily_ret'], corr_data['pnl_t2'])
    print(f"\n  SPY daily return vs B1 T+2: r={r:.3f}, p={p:.4f}")
    a5_results['correlation'] = {'r': round(r, 4), 'p': round(p, 4), 'n': len(corr_data)}

# SPY regime split: up day vs down day
print(f"\n  SPY regime filter test:")
for label, condition in [('SPY down', lambda x: x < 0), ('SPY flat/up', lambda x: x >= 0)]:
    subset = trade_df[trade_df['spy_daily_ret'].apply(condition) if trade_df['spy_daily_ret'].notna().any() else False]
    pnl = subset['pnl_t2'].dropna().tolist()
    m = full_metrics(pnl, max(1, len(subset['date'].unique())))
    a5_results[label] = m
    print(f"    {label:>15}: n={m['n']}, avg={m.get('avg_pnl_pct', 0) or 0:+.2f}%, "
          f"Sharpe={m.get('sharpe', 0) or 0:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# TOP-3 SIMULATION (Best Ranking Function)
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*_W}")
print(f"  TOP-3 SIMULATION — Best Ranking Function Applied")
print(f"{'='*_W}")

# Use the best combo from A4
best_weights = a4_results.get(f'{best_prefix}_top3', {}).get('weights', {})
w_gap = best_weights.get('gap', 0.35)
w_vol = best_weights.get('vol', 0.35)
w_price = best_weights.get('price', 0.15)
w_pm = best_weights.get('pm', 0.15)

print(f"  Using: {best_prefix} (gap={w_gap}, vol={w_vol}, price={w_price}, pm={w_pm})")

simulation_trades = []
for dt in TRADING_DATES:
    day = a4_base[a4_base['date'] == dt].copy()
    if len(day) == 0:
        print(f"\n  {dt}: 0 trades — skipped")
        continue

    day['gap_rank'] = compute_day_percentile(day['gap_pct'])
    vol_med = day['volume_ratio'].median() if day['volume_ratio'].notna().any() else 0
    day['vol_rank'] = compute_day_percentile(day['volume_ratio'].fillna(vol_med))
    day['price_rank'] = compute_day_percentile(day['open_price'])
    day['pm_score'] = day['pm_retrace'].apply(pm_retrace_to_score)
    day['agent1_score'] = (w_gap * day['gap_rank'] + w_vol * day['vol_rank'] +
                           w_price * day['price_rank'] + w_pm * day['pm_score'])

    top3 = day.nlargest(3, 'agent1_score')

    print(f"\n  {dt}: {len(day)} candidates → Top 3:")
    for rank, (_, row) in enumerate(top3.iterrows(), 1):
        t2 = f"{row['pnl_t2']:+.2f}%" if pd.notna(row['pnl_t2']) else "no T+2"
        vr = f"{row['volume_ratio']:.1f}x" if pd.notna(row['volume_ratio']) else "—"
        print(f"    #{rank}: {row['ticker']:<6} gap={row['gap_pct']:+.1f}%  "
              f"price=${row['open_price']:.2f}  VR={vr}  "
              f"score={row['agent1_score']:.3f}  T+2={t2}")
        simulation_trades.append({
            'date': dt, 'rank': rank, 'ticker': row['ticker'],
            'gap_pct': round(row['gap_pct'], 2),
            'open_price': round(row['open_price'], 2),
            'volume_ratio': round(row['volume_ratio'], 2) if pd.notna(row['volume_ratio']) else None,
            'score': round(row['agent1_score'], 4),
            'pnl_t2': round(row['pnl_t2'], 2) if pd.notna(row['pnl_t2']) else None,
            'pnl_30m': round(row['pnl_30m'], 2) if pd.notna(row['pnl_30m']) else None,
            'pnl_60m': round(row['pnl_60m'], 2) if pd.notna(row['pnl_60m']) else None,
        })

# Summary
sim_t2 = [t['pnl_t2'] for t in simulation_trades if t['pnl_t2'] is not None]
print(f"\n  SIMULATION SUMMARY ({len(simulation_trades)} trades, {len(sim_t2)} with T+2):")
if sim_t2:
    print(f"    Avg T+2: {np.mean(sim_t2):+.2f}%")
    print(f"    Med T+2: {np.median(sim_t2):+.2f}%")
    print(f"    WR: {(np.array(sim_t2) > 0).mean()*100:.0f}%")
    print(f"    Total: {np.sum(sim_t2):+.2f}%")
    # Compare to baseline
    if baseline_m.get('avg_pnl_pct'):
        improvement = np.mean(sim_t2) - baseline_m['avg_pnl_pct']
        print(f"    vs Baseline: {improvement:+.2f}% improvement")


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n  Generating charts...")

# ── Chart 1: Gap Tier Analysis ────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Agent 1 — A1: Gap Magnitude Stratification (B1 Universe, T+2)',
             fontsize=12, fontweight='bold')

tier_names = list(gap_tiers.keys())
tier_avgs = [a1_results[t].get('avg_pnl_pct', 0) or 0 for t in tier_names]
tier_ns = [a1_results[t].get('n_t2', 0) for t in tier_names]
tier_sharpes = [a1_results[t].get('sharpe', 0) or 0 for t in tier_names]

ax = axes[0]
colors = ['#27ae60' if a > 0 else '#e74c3c' for a in tier_avgs]
ax.bar(range(len(tier_names)), tier_avgs, color=colors, alpha=0.8, edgecolor='white')
ax.set_xticks(range(len(tier_names)))
ax.set_xticklabels([t.replace('tier_', 'T') for t in tier_names], fontsize=8, rotation=15)
ax.set_ylabel('Avg T+2 PnL (%)')
ax.axhline(0, color='#333', lw=0.7)
if baseline_m.get('avg_pnl_pct'):
    ax.axhline(baseline_m['avg_pnl_pct'], color='#e67e22', ls='--', lw=1.5, label=f'Baseline {baseline_m["avg_pnl_pct"]:+.2f}%')
    ax.legend(fontsize=8)
ax.set_title('Avg PnL by Gap Tier')
for i, (a, n) in enumerate(zip(tier_avgs, tier_ns)):
    ax.text(i, a + 0.1, f'n={n}', ha='center', fontsize=7)

ax = axes[1]
ax.bar(range(len(tier_names)), tier_ns, color='#3498db', alpha=0.8, edgecolor='white')
ax.set_xticks(range(len(tier_names)))
ax.set_xticklabels([t.replace('tier_', 'T') for t in tier_names], fontsize=8, rotation=15)
ax.set_ylabel('# Trades with T+2')
ax.set_title('Trade Count')

ax = axes[2]
ax.plot(range(len(tier_names)), tier_sharpes, 'o-', color='#8e44ad', markersize=8, linewidth=2)
ax.set_xticks(range(len(tier_names)))
ax.set_xticklabels([t.replace('tier_', 'T') for t in tier_names], fontsize=8, rotation=15)
ax.set_ylabel('Sharpe')
ax.axhline(0, color='#333', lw=0.7)
ax.set_title('Sharpe by Gap Tier')

plt.tight_layout(rect=[0, 0, 1, 0.92])
chart1_path = os.path.join(BASE_DIR, 'agent1_gap_tiers.png')
plt.savefig(chart1_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"    Chart 1: {chart1_path}")

# ── Chart 2: Price + Volume ───────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Agent 1 — A2: Price Levels + A3: Volume Ratios',
             fontsize=12, fontweight='bold')

# Price
ax = axes[0]
p_prices = list(a2_results.keys())[:len(price_thresholds)]
p_avgs = [a2_results[p].get('avg_pnl_pct', 0) or 0 for p in p_prices]
p_ns = [a2_results[p].get('n_t2', 0) for p in p_prices]
colors = ['#27ae60' if a > 0 else '#e74c3c' for a in p_avgs]
ax.bar(range(len(p_prices)), p_avgs, color=colors, alpha=0.8, edgecolor='white')
ax.set_xticks(range(len(p_prices)))
ax.set_xticklabels([f'${p}' for p in p_prices], fontsize=9)
ax.set_ylabel('Avg T+2 PnL (%)')
ax.axhline(0, color='#333', lw=0.7)
ax.set_title('A2: Min Price Threshold')
for i, (a, n) in enumerate(zip(p_avgs, p_ns)):
    ax.text(i, a + 0.1, f'n={n}', ha='center', fontsize=7)

# Volume
ax = axes[1]
vr_keys = [f'vr_{v}' for v in vol_thresholds if f'vr_{v}' in a3_results]
vr_avgs = [a3_results[k].get('avg_pnl_pct', 0) or 0 for k in vr_keys]
vr_ns = [a3_results[k].get('n_t2', 0) for k in vr_keys]
colors = ['#27ae60' if a > 0 else '#e74c3c' for a in vr_avgs]
ax.bar(range(len(vr_keys)), vr_avgs, color=colors, alpha=0.8, edgecolor='white')
ax.set_xticks(range(len(vr_keys)))
ax.set_xticklabels([f'{v}x' for v in vol_thresholds[:len(vr_keys)]], fontsize=9)
ax.set_ylabel('Avg T+2 PnL (%)')
ax.axhline(0, color='#333', lw=0.7)
ax.set_title('A3: Volume Ratio Threshold')
for i, (a, n) in enumerate(zip(vr_avgs, vr_ns)):
    ax.text(i, a + 0.1, f'n={n}', ha='center', fontsize=7)

plt.tight_layout(rect=[0, 0, 1, 0.92])
chart2_path = os.path.join(BASE_DIR, 'agent1_price_volume.png')
plt.savefig(chart2_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"    Chart 2: {chart2_path}")

# ── Chart 3: Ranking Function Comparison ──────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Agent 1 — A4: Ranking Function Optimization',
             fontsize=12, fontweight='bold')

# Top-3 comparison across combos
ax = axes[0]
combo_names = [k.replace('_top3', '') for k in sorted(top3_results.keys())]
combo_sharpes = [top3_results[f'{c}_top3'].get('sharpe', 0) or 0 for c in combo_names]
combo_avgs = [top3_results[f'{c}_top3'].get('avg_pnl_pct', 0) or 0 for c in combo_names]
sort_idx = np.argsort(combo_sharpes)[::-1]
ax.barh(range(len(combo_names)), [combo_sharpes[i] for i in sort_idx],
        color='#3498db', alpha=0.8)
ax.set_yticks(range(len(combo_names)))
ax.set_yticklabels([combo_names[i] for i in sort_idx], fontsize=9)
ax.set_xlabel('Sharpe (Top 3/day)')
ax.axvline(0, color='#333', lw=0.7)
ax.set_title('Sharpe by Weight Combo (Top 3/day)')

# Top-N sweep for best combo
ax = axes[1]
best_ns = []
best_avgs = []
best_sharpes = []
for n in [1, 2, 3, 4, 5]:
    key = f"{best_prefix}_top{n}"
    if key in a4_results:
        best_ns.append(n)
        best_avgs.append(a4_results[key].get('avg_pnl_pct', 0) or 0)
        best_sharpes.append(a4_results[key].get('sharpe', 0) or 0)

ax.plot(best_ns, best_avgs, 'o-', color='#27ae60', markersize=8, linewidth=2, label='Avg PnL')
ax2 = ax.twinx()
ax2.plot(best_ns, best_sharpes, 's--', color='#8e44ad', markersize=8, linewidth=2, label='Sharpe')
ax.set_xlabel('Top N per day')
ax.set_ylabel('Avg T+2 PnL (%)', color='#27ae60')
ax2.set_ylabel('Sharpe', color='#8e44ad')
ax.set_title(f'Best Combo ({best_prefix}) — Top N Sweep')
ax.axhline(0, color='#333', lw=0.5)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')

plt.tight_layout(rect=[0, 0, 1, 0.92])
chart3_path = os.path.join(BASE_DIR, 'agent1_ranking.png')
plt.savefig(chart3_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"    Chart 3: {chart3_path}")

# ── Chart 4: Top-3 Simulation Detail ─────────────────────────────────────────

fig, ax = plt.subplots(1, 1, figsize=(14, 7))
fig.suptitle(f'Agent 1 — Top-3 Simulation ({best_prefix})\n'
             f'12 trades (3/day × 4 days)',
             fontsize=12, fontweight='bold')

# Bar chart of individual trade PnLs
sim_labels = [f"{t['date'][5:]}\n{t['ticker']}" for t in simulation_trades]
sim_pnls = [t['pnl_t2'] if t['pnl_t2'] is not None else 0 for t in simulation_trades]
sim_has_data = [t['pnl_t2'] is not None for t in simulation_trades]
colors = []
for pnl, has in zip(sim_pnls, sim_has_data):
    if not has:
        colors.append('#bdc3c7')
    elif pnl > 0:
        colors.append('#27ae60')
    else:
        colors.append('#e74c3c')

bars = ax.bar(range(len(sim_labels)), sim_pnls, color=colors, alpha=0.8, edgecolor='white')
ax.set_xticks(range(len(sim_labels)))
ax.set_xticklabels(sim_labels, fontsize=8, rotation=45, ha='right')
ax.set_ylabel('T+2 PnL (%)')
ax.axhline(0, color='#333', lw=0.7)

for i, (pnl, has) in enumerate(zip(sim_pnls, sim_has_data)):
    label = f"{pnl:+.1f}%" if has else "no data"
    ax.text(i, pnl + (0.2 if pnl >= 0 else -0.4), label,
            ha='center', fontsize=7, fontweight='bold')

if sim_t2:
    ax.axhline(np.mean(sim_t2), color='#e67e22', ls='--', lw=1.5,
               label=f'Avg: {np.mean(sim_t2):+.2f}%')
    ax.legend(fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.92])
chart4_path = os.path.join(BASE_DIR, 'agent1_simulation.png')
plt.savefig(chart4_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"    Chart 4: {chart4_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS + MARKDOWN REPORT
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*_W}")
print(f"  SAVING OUTPUTS")
print(f"{'='*_W}")

results_output = {
    'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'trading_dates': TRADING_DATES,
    'baseline': baseline_m,
    'l1_baseline': l1_m,
    'a1_gap_tiers': a1_results,
    'a2_price': {str(k): v for k, v in a2_results.items()},
    'a3_volume': a3_results,
    'a4_ranking': a4_results,
    'a5_regime': a5_results,
    'simulation': simulation_trades,
    'best_ranking': best_prefix,
    'best_weights': best_weights,
    'funnel': dict(funnel),
}

results_path = os.path.join(RESULTS_DIR, 'agent1_experiment_results.json')
with open(results_path, 'w') as f:
    json.dump(results_output, f, indent=2, default=str)
print(f"  Results: {results_path}")

# Trades CSV
trade_df.to_csv(os.path.join(RESULTS_DIR, 'a1_b1_trades.csv'), index=False)
print(f"  Trades CSV: {os.path.join(RESULTS_DIR, 'a1_b1_trades.csv')}")

def fmt(v, decimals=2):
    if v is None: return '—'
    if isinstance(v, bool): return 'YES' if v else 'NO'
    if isinstance(v, float): return f'{v:+.{decimals}f}' if v != 0 else f'{v:.{decimals}f}'
    return str(v)

# ── Markdown Report ──────────────────────────────────────────────────────────

rpt = []
rpt.append(f"# Agent 1: Universe Narrowing — Experiment Results")
rpt.append(f"")
rpt.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
rpt.append(f"Dataset: B1_WIDE_SHORT universe — all gap-up >= 1.5%, price >= $5")
rpt.append(f"Dates: {', '.join(TRADING_DATES)} ({N_DATES_T2} with T+2 data)")
rpt.append(f"")

rpt.append(f"## Baseline")
rpt.append(f"")
rpt.append(f"- Total qualifying trades: {baseline_m['n']} with T+2 data")
rpt.append(f"- Avg PnL: {fmt(baseline_m.get('avg_pnl_pct'))}%")
rpt.append(f"- Median: {fmt(baseline_m.get('median_pnl_pct'))}%")
rpt.append(f"- Win Rate: {baseline_m.get('win_rate', 0)*100:.0f}%")
rpt.append(f"- Sharpe: {fmt(baseline_m.get('sharpe'), 3)}")
rpt.append(f"- CI 95%: [{fmt(baseline_m.get('ci_95_low'))}, {fmt(baseline_m.get('ci_95_high'))}]")
rpt.append(f"- CI > 0: {fmt(baseline_m.get('ci_confirmed'))}")
rpt.append(f"")

# A1
rpt.append(f"## 1. Gap Tier Analysis (A1)")
rpt.append(f"")
rpt.append(f"| Tier | Range | n | n_T+2 | t/d | Avg% | Med% | WR | Sharpe | CI_lo | CI>0 | LOO Stable |")
rpt.append(f"|------|-------|---|-------|-----|------|------|-----|--------|-------|------|------------|")
for t in tier_names:
    r = a1_results[t]
    lo, hi = r['lo'], r['hi']
    rng = f"{lo}-{hi}%" if hi else f">{lo}%"
    loo_s = fmt(r.get('loo', {}).get('stable'))
    rpt.append(f"| {t} | {rng} | {r['n_total']} | {r['n_t2']} | "
               f"{round(r['n_total']/N_DATES, 1)} | {fmt(r.get('avg_pnl_pct'))} | "
               f"{fmt(r.get('median_pnl_pct'))} | "
               f"{r.get('win_rate', 0)*100:.0f}% | {fmt(r.get('sharpe'), 3)} | "
               f"{fmt(r.get('ci_95_low'))} | {fmt(r.get('ci_confirmed'))} | {loo_s} |")
rpt.append(f"")

# A2
rpt.append(f"## 2. Price Threshold Analysis (A2)")
rpt.append(f"")
rpt.append(f"| Min Price | n | n_T+2 | t/d | Avg% | Sharpe | CI>0 |")
rpt.append(f"|-----------|---|-------|-----|------|--------|------|")
for p in price_thresholds:
    r = a2_results.get(p, {})
    rpt.append(f"| ${p} | {r.get('n_total', 0)} | {r.get('n_t2', 0)} | "
               f"{round(r.get('n_total', 0)/N_DATES, 1)} | {fmt(r.get('avg_pnl_pct'))} | "
               f"{fmt(r.get('sharpe'), 3)} | {fmt(r.get('ci_confirmed'))} |")
rpt.append(f"")

# A3
rpt.append(f"## 3. Volume Signal Analysis (A3)")
rpt.append(f"")
rpt.append(f"| Volume Ratio >= | n | n_T+2 | Avg% | Sharpe | CI_lo |")
rpt.append(f"|-----------------|---|-------|------|--------|-------|")
for vr in vol_thresholds:
    k = f'vr_{vr}'
    r = a3_results.get(k, {})
    rpt.append(f"| {vr}x | {r.get('n_total', 0)} | {r.get('n_t2', 0)} | "
               f"{fmt(r.get('avg_pnl_pct'))} | {fmt(r.get('sharpe'), 3)} | "
               f"{fmt(r.get('ci_95_low'))} |")
rpt.append(f"")

# A4
rpt.append(f"## 4. Ranking Function Optimization (A4)")
rpt.append(f"")
rpt.append(f"### Top 3/day — Weight Combos")
rpt.append(f"")
rpt.append(f"| Combo | Weights (g/v/p/pm) | n_T+2 | Avg% | Sharpe |")
rpt.append(f"|-------|--------------------|-------|------|--------|")
for name in sorted(top3_results.keys(), key=lambda x: -(top3_results[x].get('sharpe') or 0)):
    r = top3_results[name]
    w = r.get('weights', {})
    w_str = f"{w.get('gap', 0)}/{w.get('vol', 0)}/{w.get('price', 0)}/{w.get('pm', 0)}"
    rpt.append(f"| {name.replace('_top3', '')} | {w_str} | {r['n_t2']} | "
               f"{fmt(r.get('avg_pnl_pct'))} | {fmt(r.get('sharpe'), 3)} |")
rpt.append(f"")

# A5
rpt.append(f"## 5. Regime Filter Impact (A5)")
rpt.append(f"")
rpt.append(f"| Date | SPY Gap | SPY Ret | SPY Range | B1 Trades | T+2 Avg |")
rpt.append(f"|------|---------|---------|-----------|-----------|---------|")
for dt in TRADING_DATES:
    r = a5_results.get(dt, {})
    rpt.append(f"| {dt} | {fmt(r.get('spy_gap_pct'))}% | {fmt(r.get('spy_intraday_ret'))}% | "
               f"{r.get('spy_range_pct', 0):.2f}% | {r.get('n_trades', 0)} | "
               f"{fmt(r.get('avg_t2'))}% |")
rpt.append(f"")
if 'correlation' in a5_results:
    rpt.append(f"SPY daily return vs B1 T+2 correlation: r={a5_results['correlation']['r']}, "
               f"p={a5_results['correlation']['p']}")
    rpt.append(f"")

# Final Spec
rpt.append(f"## 6. FINAL AGENT 1 SPECIFICATION")
rpt.append(f"")
rpt.append(f"- **Best ranking function**: {best_prefix}")
rpt.append(f"- **Weights**: gap={w_gap}, volume={w_vol}, price={w_price}, pm_retrace={w_pm}")
rpt.append(f"- **Selection**: Top 3 per day by ranking score")
rpt.append(f"- **Expected trades/day**: 3")
if sim_t2:
    rpt.append(f"- **Expected avg PnL**: {np.mean(sim_t2):+.2f}%")
    rpt.append(f"- **Win rate**: {(np.array(sim_t2) > 0).mean()*100:.0f}%")
rpt.append(f"- **Caveat**: backtested on {N_DATES} dates ({N_DATES_T2} with T+2), n={len(sim_t2)}")
rpt.append(f"")

# Top-3 simulation
rpt.append(f"## 7. TOP-3 SIMULATION")
rpt.append(f"")
rpt.append(f"| Date | Rank | Ticker | Gap% | Price | Vol Ratio | Score | T+2 PnL |")
rpt.append(f"|------|------|--------|------|-------|-----------|-------|---------|")
for t in simulation_trades:
    vr = f"{t['volume_ratio']:.1f}x" if t['volume_ratio'] is not None else "—"
    t2 = f"{t['pnl_t2']:+.2f}%" if t['pnl_t2'] is not None else "no data"
    rpt.append(f"| {t['date']} | #{t['rank']} | {t['ticker']} | {t['gap_pct']:+.1f}% | "
               f"${t['open_price']:.2f} | {vr} | {t['score']:.3f} | {t2} |")
rpt.append(f"")
if sim_t2:
    rpt.append(f"**Simulation total**: {len(sim_t2)} trades with T+2, "
               f"avg={np.mean(sim_t2):+.2f}%, total={np.sum(sim_t2):+.2f}%")
rpt.append(f"")

report_path = os.path.join(BASE_DIR, 'AGENT_1_NARROWING_RESULTS.md')
with open(report_path, 'w') as f:
    f.write('\n'.join(rpt))
print(f"  Report: {report_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# COMPLETE
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*_W}")
print(f"  AGENT 1 EXPERIMENT COMPLETE")
print(f"{'='*_W}")
print(f"  Funnel: {dict(funnel)}")
print(f"  Total B1 trades: {len(trade_df)} ({n_t2} with T+2)")
if sim_t2:
    print(f"  Top-3 simulation: {len(sim_t2)} T+2 trades, avg={np.mean(sim_t2):+.2f}%")
print(f"\n  Output files:")
print(f"    {results_path}")
print(f"    {report_path}")
print(f"    {chart1_path}")
print(f"    {chart2_path}")
print(f"    {chart3_path}")
print(f"    {chart4_path}")
print(f"{'='*_W}")
