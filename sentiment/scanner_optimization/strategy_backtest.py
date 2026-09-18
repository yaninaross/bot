# %% Strategy Backtest — 4 Strategies × 3 Hold Periods × Feature Layers
# Strategies: BASELINE_V1, B1_WIDE_SHORT, B2_NO_ORB, B3_EOD_ENTRY (all SHORT)
# Holds: 30m (bar[29] close), 60m (bar[59] close), T+2 (2-day close)
# Layers: L0 raw, L1 float, L2 PM retrace, L3 sweet spot, L4 breaking, L5 combined
# Composite: ORB width + PM retrace + gap magnitude + offering bonus
# Portfolio PnL: finite capital, equal split across same-day trades, compounding
#
# Output: results/bt_master_table.csv, bt_results.json, strategy_backtest_*.png

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

POLYGON_API_KEY = os.environ["POLYGON_API_KEY"]
BQ_PREMARKET_TABLE = f"{GCP_PROJECT}.scoring_logs.premarket_scanner"
BQ_DISCOVERY_TABLE = f"{GCP_PROJECT}.scoring_logs.discovery_scanner"

FORCE_REFETCH = False
API_DELAY = 0.02
MAX_GAPPERS_PER_DAY = 0
TICKER_REGEX = re.compile(r'^[A-Z]{1,5}$')

ET = pytz.timezone('America/New_York')
MARKET_OPEN = "09:30"
MARKET_CLOSE = "16:00"

STARTING_CAPITAL = 100_000

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

TICKER_REF_CACHE = os.path.join(RESULTS_DIR, 'features', 'ticker_reference_cache.json')
V3_MINUTE_BARS_PATH = os.path.join(RESULTS_DIR, 'v3_minute_bars.json')

_W = 130

HOLD_KEYS = {'30m': 'pnl_30m', '60m': 'pnl_60m', 't2': 'pnl_t2'}
HOLD_LABELS = {'30m': '30m', '60m': '60m', 't2': 'T+2'}

FLOAT_MIN = 10_000_000
PM_RETRACE_MAX = 0.8
PM_SWEET_LO = 0.2
PM_SWEET_HI = 0.5

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
    'SWRD':211149963, 'SWVL':9964344, 'TEN':30127603, 'TNON':1032578, 'TPST':16539767, 'TRUG':1903708, 'TSLA':3949547394, 'TTAN':82746425,
    'TTD':426771565, 'TYRA':59676939, 'UBER':2042560121, 'UNFI':60519140, 'UPST':97313046, 'URI':62242489, 'VAC':34395320, 'VICR':34389014,
    'VOYG':58217592, 'VRT':384988173, 'VSME':2750784, 'VST':335635195, 'WDH':279111455, 'WULF':498968677, 'YARW':2669746, 'ZEO':41703489,
    'ZUMZ':16872215,
}


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY DEFINITIONS (4 strategies, all SHORT)
# ═══════════════════════════════════════════════════════════════════════════════

STRATEGIES = [
    {
        'id': 'BASELINE_V1', 'label': 'Baseline V1 (LLM+ORB)',
        'direction': 'SHORT', 'gap_sign': 'positive', 'gap_min_pct': 3.0,
        'universe': 'llm', 'min_price': 0,
        'catalyst_exclude': ['earnings', 'deal', 'macro'],
        'orb_required': True, 'orb_break_dir': 'SHORT',
        'orb_scan_bars': (5, 25),
        'entry_method': 'local_high',
        'entry_peak_window': 3, 'entry_max_peak_bar': 12, 'entry_fallback_bar': 15,
        'holds': ['30m', '60m', 't2'],
        'layers': ['L0', 'L1', 'L2', 'L3', 'L4', 'L5'],
        'color': '#2ecc71',
    },
    {
        'id': 'B2_NO_ORB', 'label': 'B2: No ORB',
        'direction': 'SHORT', 'gap_sign': 'positive', 'gap_min_pct': 3.0,
        'universe': 'llm', 'min_price': 0,
        'catalyst_exclude': ['earnings', 'deal', 'macro'],
        'orb_required': False, 'orb_break_dir': None,
        'orb_scan_bars': None,
        'entry_method': 'local_high',
        'entry_peak_window': 3, 'entry_max_peak_bar': 12, 'entry_fallback_bar': 15,
        'holds': ['30m', '60m', 't2'],
        'layers': ['L0', 'L1', 'L2', 'L3', 'L4', 'L5'],
        'color': '#3498db',
    },
    {
        'id': 'B3_EOD_ENTRY', 'label': 'B3: EOD Entry',
        'direction': 'SHORT', 'gap_sign': 'positive', 'gap_min_pct': 3.0,
        'universe': 'llm', 'min_price': 0,
        'catalyst_exclude': ['earnings', 'deal', 'macro'],
        'orb_required': True, 'orb_break_dir': 'SHORT',
        'orb_scan_bars': (5, 25),
        'entry_method': 'eod_close',
        'holds': ['t2'],
        'layers': ['L0', 'L1', 'L2', 'L3', 'L4', 'L5'],
        'color': '#f39c12',
    },
    {
        'id': 'B1_WIDE_SHORT', 'label': 'B1: Wide (All >1.5%)',
        'direction': 'SHORT', 'gap_sign': 'positive', 'gap_min_pct': 1.5,
        'universe': 'all', 'min_price': 5.0,
        'catalyst_exclude': [],
        'orb_required': False, 'orb_break_dir': None,
        'orb_scan_bars': None,
        'entry_method': 'local_high',
        'entry_peak_window': 3, 'entry_max_peak_bar': 12, 'entry_fallback_bar': 15,
        'holds': ['30m', '60m', 't2'],
        'layers': ['L0', 'L1', 'L2', 'L3', 'L5'],
        'color': '#e74c3c',
    },
]

LAYER_ORDER = ['L0', 'L1', 'L2', 'L3', 'L4', 'L5']
LAYER_NAMES = {
    'L0': 'L0_raw', 'L1': 'L1_float', 'L2': 'L2_pm_retrace',
    'L3': 'L3_sweet_spot', 'L4': 'L4_breaking', 'L5': 'L5_combined',
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


def check_orb_break(bars, orb_high, orb_low, direction, scan_start, scan_end,
                    volume_confirm=False, n_orb_bars=5):
    if orb_high is None or orb_low is None:
        return {'confirmed': False}
    avg_orb_vol = (np.mean([bars[i][1]['v'] for i in range(min(n_orb_bars, len(bars)))])
                   if volume_confirm else 0)
    for i in range(scan_start, min(scan_end, len(bars))):
        bar = bars[i][1]
        broke = (direction == 'SHORT' and bar['c'] < orb_low) or \
                (direction == 'LONG' and bar['c'] > orb_high)
        if not broke:
            continue
        if volume_confirm and bar['v'] <= 1.5 * avg_orb_vol:
            continue
        return {'confirmed': True, 'break_bar_idx': i,
                'break_bar_time': bars[i][0], 'break_bar_close': bar['c']}
    return {'confirmed': False}


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
    empty = {'n': n, 'trades_per_day': n / n_dates if n_dates else 0}
    if n < 3:
        empty['note'] = 'insufficient — do not interpret'
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


def compute_equity_curve(trades, hold_key, dates, start_cap=STARTING_CAPITAL):
    equity = start_cap
    curve = []
    for dt in sorted(dates):
        day_pnls = [t[hold_key] for t in trades
                    if t['date'] == dt and t.get(hold_key) is not None]
        n = len(day_pnls)
        daily_ret = np.mean(day_pnls) / 100 if n > 0 else 0.0
        equity *= (1 + daily_ret)
        curve.append({
            'date': dt, 'n_trades': n,
            'daily_ret_pct': round(daily_ret * 100, 3),
            'equity': round(equity, 2),
        })
    total_ret_pct = (equity / start_cap - 1) * 100
    return curve, round(total_ret_pct, 3), round(equity, 2)


def apply_layer_filter(trades, layer_id, strategy_id):
    if layer_id == 'L0':
        return trades

    out = list(trades)
    is_b1 = strategy_id == 'B1_WIDE_SHORT'

    if layer_id in ('L1', 'L5'):
        out = [t for t in out if not _fails_float_filter(t, is_b1)]

    if layer_id in ('L2', 'L3', 'L5'):
        out = [t for t in out if not _fails_pm_retrace_filter(t, is_b1)]

    if layer_id == 'L3':
        out = [t for t in out
               if t.get('pm_retrace_pct') is not None
               and PM_SWEET_LO <= t['pm_retrace_pct'] <= PM_SWEET_HI]

    if layer_id in ('L4', 'L5') and not is_b1:
        out = [t for t in out if t.get('catalyst_type', '') != 'breaking']

    return out


def _fails_float_filter(t, is_b1):
    fs = t.get('float_shares')
    if fs is None:
        return False
    return fs < FLOAT_MIN


def _fails_pm_retrace_filter(t, is_b1):
    if is_b1 and not t.get('is_llm', False):
        return False
    pr = t.get('pm_retrace_pct')
    if pr is None:
        return False
    return pr > PM_RETRACE_MAX


# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════════

poly = PolygonClient(POLYGON_API_KEY)
bq_client = bigquery.Client()

print(f"\n{'='*_W}")
print(f"  STRATEGY BACKTEST — 4 Strategies × 3 Holds × Feature Layers")
print(f"  Dates: {', '.join(TRADING_DATES)}")
print(f"{'='*_W}")

# ── 1. Trading Calendar ────────────────────────────────────────────────────────

print(f"\n  [1/6] Building trading calendar...")

first_dt = datetime.strptime(TRADING_DATES[0], '%Y-%m-%d')
last_dt = datetime.strptime(TRADING_DATES[-1], '%Y-%m-%d')
cal_start = (first_dt - timedelta(days=10)).strftime('%Y-%m-%d')
cal_end = (last_dt + timedelta(days=10)).strftime('%Y-%m-%d')

spy_daily = {}
cache_spy = os.path.join(RESULTS_DIR, 'bt_spy_daily.json')
if os.path.exists(cache_spy) and not FORCE_REFETCH:
    with open(cache_spy) as f:
        spy_daily = json.load(f)
    print(f"    Loaded SPY daily from cache ({len(spy_daily)} bars)")
else:
    for b in poly.get_aggs('SPY', 1, 'day', from_=cal_start, to=cal_end, limit=500):
        dt_str = pd.Timestamp(b.timestamp, unit='ms', tz='UTC').strftime('%Y-%m-%d')
        spy_daily[dt_str] = {'o': float(b.open), 'h': float(b.high), 'l': float(b.low),
                             'c': float(b.close), 'v': int(b.volume)}
    with open(cache_spy, 'w') as f:
        json.dump(spy_daily, f)
    print(f"    Fetched SPY daily: {len(spy_daily)} bars")

trading_days = sorted(spy_daily.keys())

date_info = {}
for dt in TRADING_DATES:
    if dt not in trading_days:
        print(f"    WARNING: {dt} not in calendar — skipping")
        continue
    idx = trading_days.index(dt)
    prior = trading_days[idx - 1] if idx > 0 else None
    t1 = trading_days[idx + 1] if idx + 1 < len(trading_days) else None
    t2 = trading_days[idx + 2] if idx + 2 < len(trading_days) else None
    date_info[dt] = {'prior': prior, 't1': t1, 't2': t2}
    t2_flag = '' if t2 else ' [NO T+2]'
    print(f"    {dt}: prior={prior}, T+2={t2}{t2_flag}")

all_needed_dates = set(TRADING_DATES)
for di in date_info.values():
    for k in ('prior', 't1', 't2'):
        if di.get(k):
            all_needed_dates.add(di[k])
all_needed_dates = sorted(all_needed_dates)

# ── 2. Grouped Daily ──────────────────────────────────────────────────────────

print(f"\n  [2/6] Fetching grouped daily data...")

grouped_daily = {}
cache_gd = os.path.join(RESULTS_DIR, 'bt_grouped_daily.json')
if os.path.exists(cache_gd) and not FORCE_REFETCH:
    with open(cache_gd) as f:
        grouped_daily = json.load(f)
    print(f"    Loaded from cache ({len(grouped_daily)} dates)")

missing_dates = [d for d in all_needed_dates if d not in grouped_daily]
if missing_dates:
    print(f"    Fetching {len(missing_dates)} missing dates...")
    for dt in missing_dates:
        url = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{dt}"
        try:
            resp = _requests.get(url, params={"apiKey": POLYGON_API_KEY, "adjusted": "true"},
                                 timeout=30)
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
                grouped_daily[dt] = {}
        except Exception as e:
            print(f"      {dt}: error {e}")
            grouped_daily[dt] = {}
        time.sleep(API_DELAY)
    with open(cache_gd, 'w') as f:
        json.dump(grouped_daily, f)

# ── 3. BQ LLM Universe ────────────────────────────────────────────────────────

print(f"\n  [3/6] Fetching LLM-flagged universe from BQ...")

llm_universe = {}
cache_llm = os.path.join(RESULTS_DIR, 'bt_llm_universe.json')
if os.path.exists(cache_llm) and not FORCE_REFETCH:
    with open(cache_llm) as f:
        raw = json.load(f)
    llm_universe = {tuple(k.split('__')): v for k, v in raw.items()}
    print(f"    Loaded from cache ({len(llm_universe)} signals)")
else:
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
            cat_text = str(row['catalyst']).lower()
            llm_universe[key] = {
                'direction': row['direction'],
                'catalyst_type': cat,
                'is_earnings': 'earnings' in cat or any(
                    w in cat_text for w in ['earnings', 'eps ', 'revenue miss', 'revenue beat']),
                'is_deal': 'deal' in cat or any(
                    w in cat_text for w in ['acqui', 'merger', 'buyout', 'takeover']),
                'is_macro': 'macro' in cat,
                'is_breaking': 'breaking' in cat,
                'scanner': scanner,
            }
        print(f"    {scanner}: {len(df_bq)} rows")

    cache_data = {f"{k[0]}__{k[1]}": v for k, v in llm_universe.items()}
    with open(cache_llm, 'w') as f:
        json.dump(cache_data, f)
    print(f"    Total LLM-flagged: {len(llm_universe)} unique (ticker, date)")

# ── 4. Build Gapper Universe ──────────────────────────────────────────────────

print(f"\n  [4/6] Building gapper universe...")

MIN_GAP_FOR_BARS = 1.5
gappers = []

for dt in TRADING_DATES:
    di = date_info.get(dt)
    if not di or not di['prior']:
        continue
    today_data = grouped_daily.get(dt, {})
    prior_data = grouped_daily.get(di['prior'], {})
    n_up, n_down = 0, 0

    for tk in today_data:
        if tk not in prior_data:
            continue
        prev_c = prior_data[tk]['c']
        today_o = today_data[tk]['o']
        if prev_c <= 0 or today_o <= 0:
            continue
        gap_pct = (today_o - prev_c) / prev_c * 100
        if abs(gap_pct) < MIN_GAP_FOR_BARS:
            continue

        llm = llm_universe.get((tk, dt))
        gappers.append({
            'ticker': tk, 'date': dt, 'gap_pct': gap_pct,
            'open_price': today_o, 'prev_close': prev_c,
            'today_close': today_data[tk]['c'],
            'today_high': today_data[tk]['h'],
            'today_low': today_data[tk]['l'],
            'is_llm': llm is not None,
            'catalyst_type': llm['catalyst_type'] if llm else 'unknown',
            'is_earnings': llm['is_earnings'] if llm else False,
            'is_deal': llm['is_deal'] if llm else False,
            'is_macro': llm['is_macro'] if llm else False,
            'is_breaking': llm.get('is_breaking', False) if llm else False,
            't2_date': di.get('t2'),
        })
        if gap_pct > 0:
            n_up += 1
        else:
            n_down += 1

    print(f"    {dt}: {n_up} gap-up, {n_down} gap-down (|gap| >= {MIN_GAP_FOR_BARS}%)")

gappers.sort(key=lambda x: -abs(x['gap_pct']))

if MAX_GAPPERS_PER_DAY > 0:
    limited = []
    day_counts = defaultdict(int)
    for g in gappers:
        if day_counts[g['date']] < MAX_GAPPERS_PER_DAY:
            limited.append(g)
            day_counts[g['date']] += 1
    print(f"    Capped: {len(gappers)} -> {len(limited)}")
    gappers = limited

print(f"    Total gappers: {len(gappers)}")

# ── 5. Minute Bars ─────────────────────────────────────────────────────────────

print(f"\n  [5/6] Loading/fetching minute bars...")

minute_bars = {}
cache_mb = os.path.join(RESULTS_DIR, 'bt_minute_bars.json')

if os.path.exists(V3_MINUTE_BARS_PATH):
    with open(V3_MINUTE_BARS_PATH) as f:
        v3_raw = json.load(f)
    for k, v in v3_raw.items():
        parts = k.split('__')
        if len(parts) == 2:
            minute_bars[(parts[0], parts[1])] = v
    print(f"    Pre-seeded from v3_minute_bars: {len(minute_bars)} ticker-dates")
    del v3_raw

if os.path.exists(cache_mb) and not FORCE_REFETCH:
    with open(cache_mb) as f:
        raw_mb = json.load(f)
    for k, v in raw_mb.items():
        parts = k.split('__')
        if len(parts) == 2:
            minute_bars[(parts[0], parts[1])] = v
    print(f"    Loaded bt cache: total {len(minute_bars)} ticker-dates")
    del raw_mb

needed_pairs = set((g['ticker'], g['date']) for g in gappers)
missing_pairs = sorted(needed_pairs - set(minute_bars.keys()))

if missing_pairs:
    print(f"    Fetching {len(missing_pairs)} missing ticker-dates...")
    n_ok = 0
    t0 = time.time()
    for i, (tk, dt) in enumerate(missing_pairs):
        try:
            raw_bars = list(poly.get_aggs(ticker=tk, multiplier=1, timespan="minute",
                                          from_=dt, to=dt, limit=500))
            if raw_bars:
                bar_map = {}
                for b in raw_bars:
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
            eta = (len(missing_pairs) - i - 1) / rate if rate > 0 else 0
            print(f"      {i+1}/{len(missing_pairs)} ({n_ok} ok, {rate:.0f}/s, ETA {eta/60:.1f}m)")
        if API_DELAY > 0:
            time.sleep(API_DELAY)

    print(f"    Fetched: {n_ok}/{len(missing_pairs)} ({time.time()-t0:.0f}s)")
    raw_mb_save = {f"{k[0]}__{k[1]}": v for k, v in minute_bars.items()}
    with open(cache_mb, 'w') as f:
        json.dump(raw_mb_save, f)
    del raw_mb_save
    print(f"    Cache saved: {os.path.getsize(cache_mb) / (1024*1024):.1f} MB")

# ── 6. Feature Data ────────────────────────────────────────────────────────────

print(f"\n  [6/6] Loading feature data...")

ticker_ref = {}
if os.path.exists(TICKER_REF_CACHE):
    with open(TICKER_REF_CACHE) as f:
        ticker_ref = json.load(f)
    print(f"    Ticker reference: {len(ticker_ref)} tickers from file (float proxy)")
else:
    ticker_ref = {tk: {'shares_outstanding': so} for tk, so in _EMBEDDED_FLOAT.items()}
    print(f"    Ticker reference: {len(ticker_ref)} tickers from embedded data (float proxy)")

print(f"    Feature data loaded\n")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN STRATEGIES — Compute trades with multi-hold PnL + features
# ═══════════════════════════════════════════════════════════════════════════════

print(f"{'='*_W}")
print(f"  RUNNING STRATEGIES")
print(f"{'='*_W}")

all_strategy_trades = {}
all_funnels = {}

for strat in STRATEGIES:
    sid = strat['id']
    direction = strat['direction']
    trades = []
    funnel = defaultdict(int)
    funnel['start'] = len(gappers)

    for g in gappers:
        tk, dt = g['ticker'], g['date']

        # ── Universe filter ──
        if strat['universe'] == 'llm' and not g['is_llm']:
            funnel['excl_universe'] += 1
            continue

        # ── Gap direction + threshold ──
        if strat['gap_sign'] == 'positive' and g['gap_pct'] < strat['gap_min_pct']:
            funnel['excl_gap'] += 1
            continue
        if strat['gap_sign'] == 'negative' and g['gap_pct'] > -strat['gap_min_pct']:
            funnel['excl_gap'] += 1
            continue

        # ── Price filter ──
        if strat.get('min_price', 0) > 0 and g['open_price'] < strat['min_price']:
            funnel['excl_price'] += 1
            continue

        # ── Catalyst filter ──
        cat = g['catalyst_type']
        if strat.get('catalyst_exclude') and cat != 'unknown':
            skip = False
            for exc_cat in strat['catalyst_exclude']:
                if exc_cat == 'earnings' and g['is_earnings']:
                    skip = True; break
                if exc_cat == 'deal' and g['is_deal']:
                    skip = True; break
                if exc_cat == 'macro' and g['is_macro']:
                    skip = True; break
                if exc_cat in cat:
                    skip = True; break
            if skip:
                funnel['excl_catalyst'] += 1
                continue
        funnel['after_catalyst'] += 1

        # ── Minute bars ──
        bar_map = minute_bars.get((tk, dt))
        if bar_map is None:
            funnel['excl_no_bars'] += 1
            continue
        bars = get_session_bars(bar_map)
        if len(bars) < 20:
            funnel['excl_short_session'] += 1
            continue

        # ── ORB check ──
        orb_high, orb_low = compute_orb(bars)
        if strat['orb_required']:
            s_start, s_end = strat['orb_scan_bars']
            orb_result = check_orb_break(bars, orb_high, orb_low,
                                         strat['orb_break_dir'], s_start, s_end)
            if not orb_result['confirmed']:
                funnel['excl_orb'] += 1
                continue
        funnel['after_orb'] += 1

        # ── Entry ──
        entry = None
        if strat['entry_method'] == 'local_high':
            entry = find_local_high(bars, strat.get('entry_peak_window', 3),
                                    strat.get('entry_max_peak_bar', 12),
                                    strat.get('entry_fallback_bar', 15))
        elif strat['entry_method'] == 'eod_close':
            entry = {'entry_price': g['today_close'],
                     'entry_bar_idx': len(bars) - 1,
                     'entry_time': '16:00', 'entry_type': 'eod_close'}

        if entry is None or entry.get('entry_price', 0) <= 0:
            funnel['excl_no_entry'] += 1
            continue

        ep = entry['entry_price']
        entry_idx = entry['entry_bar_idx']

        # ── Multi-hold PnL ──
        pnl_30m = None
        pnl_60m = None
        pnl_t2 = None
        exit_30m_price = None
        exit_60m_price = None
        exit_t2_price = None

        if '30m' in strat['holds'] and entry_idx < 29 and len(bars) > 29:
            exit_30m_price = bars[29][1]['c']
            pnl_30m = (ep - exit_30m_price) / ep * 100

        if '60m' in strat['holds'] and entry_idx < 59 and len(bars) > 59:
            exit_60m_price = bars[59][1]['c']
            pnl_60m = (ep - exit_60m_price) / ep * 100

        if 't2' in strat['holds']:
            t2d = g.get('t2_date')
            if t2d and t2d in grouped_daily and tk in grouped_daily.get(t2d, {}):
                exit_t2_price = grouped_daily[t2d][tk]['c']
                pnl_t2 = (ep - exit_t2_price) / ep * 100

        has_any_pnl = any(x is not None for x in [pnl_30m, pnl_60m, pnl_t2])
        if not has_any_pnl:
            funnel['excl_no_exit'] += 1
            continue

        # ── Feature values ──
        ref = ticker_ref.get(tk, {})
        float_shares = ref.get('shares_outstanding')
        pm_retrace_pct = compute_pm_retrace(bar_map)

        orb_width_raw = None
        if orb_high is not None and orb_low is not None and ep > 0:
            orb_width_raw = (orb_high - orb_low) / ep

        trade = {
            'strategy': sid, 'ticker': tk, 'date': dt,
            'direction': direction, 'gap_pct': round(g['gap_pct'], 3),
            'catalyst_type': g['catalyst_type'], 'is_llm': g['is_llm'],
            'is_breaking': g.get('is_breaking', False),
            'entry_type': entry.get('entry_type', ''),
            'entry_time': entry.get('entry_time', ''),
            'entry_bar_idx': entry_idx,
            'entry_price': round(ep, 4),
            'pnl_30m': round(pnl_30m, 4) if pnl_30m is not None else None,
            'pnl_60m': round(pnl_60m, 4) if pnl_60m is not None else None,
            'pnl_t2': round(pnl_t2, 4) if pnl_t2 is not None else None,
            'exit_30m_price': exit_30m_price,
            'exit_60m_price': exit_60m_price,
            'exit_t2_price': exit_t2_price,
            'float_shares': float_shares,
            'pm_retrace_pct': pm_retrace_pct,
            'orb_width_raw': orb_width_raw,
            'orb_high': orb_high,
            'orb_low': orb_low,
            'composite_score': None,
        }
        trades.append(trade)

    funnel['qualifying'] = len(trades)
    all_strategy_trades[sid] = trades
    all_funnels[sid] = dict(funnel)

    n_30m = sum(1 for t in trades if t['pnl_30m'] is not None)
    n_60m = sum(1 for t in trades if t['pnl_60m'] is not None)
    n_t2 = sum(1 for t in trades if t['pnl_t2'] is not None)
    avg_t2 = np.mean([t['pnl_t2'] for t in trades if t['pnl_t2'] is not None]) if n_t2 > 0 else 0
    print(f"  {sid:<20} raw: {len(trades):>5} trades  "
          f"(30m:{n_30m}, 60m:{n_60m}, T+2:{n_t2}, T+2 avg={avg_t2:+.2f}%)")


# ═══════════════════════════════════════════════════════════════════════════════
# COMPOSITE SCORES — normalize within day per strategy
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n  Computing composite scores...")

for strat in STRATEGIES:
    sid = strat['id']
    trades = all_strategy_trades[sid]
    needs_orb = strat['orb_required']

    for dt in TRADING_DATES:
        day_trades = [t for t in trades if t['date'] == dt]
        if not day_trades:
            continue

        orb_vals = [t['orb_width_raw'] for t in day_trades if t['orb_width_raw'] is not None]
        gap_vals = [abs(t['gap_pct']) for t in day_trades]
        orb_mn = min(orb_vals) if orb_vals else 0
        orb_mx = max(orb_vals) if orb_vals else 0
        gap_mn = min(gap_vals) if gap_vals else 0
        gap_mx = max(gap_vals) if gap_vals else 0

        for t in day_trades:
            if needs_orb:
                orb_score = normalize_val(t['orb_width_raw'], orb_mn, orb_mx)
            else:
                orb_score = 0.5

            pm_score = pm_retrace_to_score(t['pm_retrace_pct'])
            gap_score = normalize_val(abs(t['gap_pct']), gap_mn, gap_mx)
            offering_score = 1.0 if t.get('catalyst_type') == 'offering' else 0.0

            t['composite_score'] = round(
                0.35 * orb_score + 0.30 * pm_score + 0.20 * gap_score + 0.15 * offering_score, 4)

    scored = sum(1 for t in trades if t['composite_score'] is not None)
    print(f"    {sid}: {scored}/{len(trades)} trades scored")


# ═══════════════════════════════════════════════════════════════════════════════
# APPLY LAYERS + COMPUTE MASTER TABLE
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*_W}")
print(f"  MASTER TABLE — Strategy × Layer × Hold")
print(f"{'='*_W}")

master_rows = []
equity_data = {}
filter_funnels = {}

for strat in STRATEGIES:
    sid = strat['id']
    trades = all_strategy_trades[sid]

    for layer_id in LAYER_ORDER:
        if layer_id not in strat['layers']:
            continue

        filtered = apply_layer_filter(trades, layer_id, sid)

        for hold in strat['holds']:
            hold_key = HOLD_KEYS[hold]
            pnl_list = [t[hold_key] for t in filtered if t.get(hold_key) is not None]
            m = full_metrics(pnl_list)

            curve, total_compound, final_equity = compute_equity_curve(
                filtered, hold_key, TRADING_DATES)

            row = {
                'strategy': sid, 'layer': LAYER_NAMES[layer_id],
                'hold': HOLD_LABELS[hold],
                **m,
                'total_compound_pct': total_compound,
                'final_equity': final_equity,
            }
            master_rows.append(row)

            eq_key = (sid, layer_id, hold)
            equity_data[eq_key] = curve

        # Filter funnel for this strategy × layer
        n_after_layer = len(filtered)
        n_with = {h: sum(1 for t in filtered if t.get(HOLD_KEYS[h]) is not None)
                  for h in strat['holds']}
        filter_funnels[(sid, layer_id)] = {
            'raw_qualifying': len(trades),
            'after_layer_filter': n_after_layer,
            **{f'with_{h}': n_with[h] for h in strat['holds']},
            'trades_per_day': {h: round(n_with[h] / N_DATES, 1) for h in strat['holds']},
        }


# ── Print Master Table ─────────────────────────────────────────────────────────

header = (f"  {'STRATEGY':<16} {'LAYER':<18} {'HOLD':>5} {'n':>5} {'t/d':>5} "
          f"{'avg%':>8} {'med%':>8} {'WR':>6} {'Sharpe':>7} {'CI_lo':>8} {'CI_hi':>8} "
          f"{'CI_OK':>6} {'compd%':>8} {'equity$':>10}")
print(f"\n{header}")
print(f"  {'─'*(_W-4)}")

for row in master_rows:
    n = row['n']
    if n < 3:
        avg_s = f"{'—':>8}"
        med_s = f"{'—':>8}"
        wr_s = f"{'—':>6}"
        sh_s = f"{'—':>7}"
        ci_lo_s = f"{'—':>8}"
        ci_hi_s = f"{'—':>8}"
        ci_ok_s = f"{'—':>6}"
    else:
        avg_s = f"{row['avg_pnl_pct']:>+8.2f}"
        med_s = f"{row['median_pnl_pct']:>+8.2f}"
        wr_s = f"{row['win_rate']*100:>5.0f}%"
        sh_s = f"{row['sharpe']:>7.3f}"
        ci_lo_s = f"{row['ci_95_low']:>+8.2f}"
        ci_hi_s = f"{row['ci_95_high']:>+8.2f}"
        ci_ok_s = f"{'YES' if row['ci_confirmed'] else 'NO':>6}"

    is_candidate = (n >= 3 and row.get('ci_confirmed') and
                    row.get('trades_per_day', 0) >= 1.0)
    marker = ' ★' if is_candidate else ''

    print(f"  {row['strategy']:<16} {row['layer']:<18} {row['hold']:>5} "
          f"{n:>5} {row['trades_per_day']:>5} "
          f"{avg_s} {med_s} {wr_s} {sh_s} {ci_lo_s} {ci_hi_s} {ci_ok_s} "
          f"{row['total_compound_pct']:>+8.2f} {row['final_equity']:>10,.0f}{marker}")


# ═══════════════════════════════════════════════════════════════════════════════
# FILTER FUNNELS
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*_W}")
print(f"  FILTER FUNNELS")
print(f"{'='*_W}")

for strat in STRATEGIES:
    sid = strat['id']
    base_funnel = all_funnels[sid]
    print(f"\n  {sid}:")
    print(f"    Raw gappers:         {base_funnel['start']:>6}")
    if base_funnel.get('excl_universe'):
        print(f"    After universe:      {base_funnel['start'] - base_funnel.get('excl_universe', 0):>6}  "
              f"(-{base_funnel.get('excl_universe', 0)} not LLM)")
    if base_funnel.get('excl_gap'):
        remaining = (base_funnel['start'] - base_funnel.get('excl_universe', 0)
                     - base_funnel.get('excl_gap', 0))
        print(f"    After gap filter:    {remaining:>6}  "
              f"(-{base_funnel.get('excl_gap', 0)} gap too small)")
    if base_funnel.get('excl_price'):
        print(f"    After price filter:  (-{base_funnel.get('excl_price', 0)} < $5)")
    if base_funnel.get('excl_catalyst'):
        print(f"    After catalyst:      (-{base_funnel.get('excl_catalyst', 0)} excluded)")
    if base_funnel.get('excl_orb'):
        print(f"    After ORB:           (-{base_funnel.get('excl_orb', 0)} no ORB break)")
    if base_funnel.get('excl_no_bars'):
        print(f"    After bars check:    (-{base_funnel.get('excl_no_bars', 0)} no minute bars)")
    print(f"    Raw qualifying:      {base_funnel['qualifying']:>6}  "
          f"({base_funnel['qualifying']/N_DATES:.1f}/day)")

    for layer_id in LAYER_ORDER:
        if layer_id not in strat['layers'] or layer_id == 'L0':
            continue
        ff = filter_funnels.get((sid, layer_id))
        if ff:
            tpd_str = ", ".join(f"{h}:{ff['trades_per_day'][h]}" for h in strat['holds'])
            print(f"    {LAYER_NAMES[layer_id]}:  {ff['after_layer_filter']:>4} trades  ({tpd_str})")


# ═══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO PnL WITH FINITE CAPITAL
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*_W}")
print(f"  PORTFOLIO PnL — Finite Capital (${STARTING_CAPITAL:,.0f}), Equal Split per Day")
print(f"{'='*_W}")

print(f"\n  {'STRATEGY':<16} {'LAYER':<18} {'HOLD':>5} "
      f"{'Day 1':>10} {'Day 2':>10} {'Day 3':>10} {'Day 4':>10} "
      f"{'Final$':>12} {'Return%':>10}")
print(f"  {'─'*(_W-4)}")

for strat in STRATEGIES:
    sid = strat['id']
    for layer_id in ['L0', 'L5']:
        if layer_id not in strat['layers']:
            continue
        for hold in strat['holds']:
            eq_key = (sid, layer_id, hold)
            curve = equity_data.get(eq_key, [])
            if not curve:
                continue
            day_strs = []
            for pt in curve:
                if pt['n_trades'] > 0:
                    day_strs.append(f"{pt['daily_ret_pct']:>+7.2f}%")
                else:
                    day_strs.append(f"{'—':>8}")
            while len(day_strs) < 4:
                day_strs.append(f"{'—':>8}")

            final = curve[-1]['equity'] if curve else STARTING_CAPITAL
            ret = (final / STARTING_CAPITAL - 1) * 100
            print(f"  {sid:<16} {LAYER_NAMES[layer_id]:<18} {HOLD_LABELS[hold]:>5} "
                  f"{day_strs[0]:>10} {day_strs[1]:>10} {day_strs[2]:>10} {day_strs[3]:>10} "
                  f"{final:>12,.0f} {ret:>+9.2f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# COMPOSITE QUARTILE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*_W}")
print(f"  COMPOSITE SCORE QUARTILE ANALYSIS (at L5)")
print(f"{'='*_W}")

composite_results = {}

for strat in STRATEGIES:
    sid = strat['id']
    if 'L5' not in strat['layers']:
        continue

    trades = all_strategy_trades[sid]
    filtered = apply_layer_filter(trades, 'L5', sid)

    for hold in strat['holds']:
        hold_key = HOLD_KEYS[hold]
        valid = [t for t in filtered
                 if t.get(hold_key) is not None and t.get('composite_score') is not None]
        if len(valid) < 8:
            continue

        scores = np.array([t['composite_score'] for t in valid])
        pnls = np.array([t[hold_key] for t in valid])
        q25, q50, q75 = np.percentile(scores, [25, 50, 75])

        quartiles = {'Q1': [], 'Q2': [], 'Q3': [], 'Q4': []}
        for t in valid:
            s = t['composite_score']
            if s <= q25:
                quartiles['Q1'].append(t[hold_key])
            elif s <= q50:
                quartiles['Q2'].append(t[hold_key])
            elif s <= q75:
                quartiles['Q3'].append(t[hold_key])
            else:
                quartiles['Q4'].append(t[hold_key])

        print(f"\n  {sid} / {HOLD_LABELS[hold]} (n={len(valid)}):")
        print(f"    {'Quartile':>10} {'n':>5} {'avg%':>8} {'WR':>6} {'Sharpe':>7}")

        q_results = {}
        for qname in ['Q1', 'Q2', 'Q3', 'Q4']:
            pnl_q = np.array(quartiles[qname])
            n_q = len(pnl_q)
            if n_q == 0:
                continue
            avg_q = float(np.mean(pnl_q))
            wr_q = float((pnl_q > 0).mean())
            std_q = float(np.std(pnl_q, ddof=1)) if n_q > 1 else 0
            sharpe_q = avg_q / std_q if std_q > 0 else 0
            q_results[qname] = {'n': n_q, 'avg': avg_q, 'wr': wr_q, 'sharpe': sharpe_q}
            print(f"    {qname:>10} {n_q:>5} {avg_q:>+8.2f} {wr_q*100:>5.0f}% {sharpe_q:>7.3f}")

        if 'Q1' in q_results and 'Q4' in q_results:
            spread = q_results['Q4']['avg'] - q_results['Q1']['avg']
            ratio = (q_results['Q4']['avg'] / q_results['Q1']['avg']
                     if q_results['Q1']['avg'] != 0 else float('inf'))
            validated = q_results['Q4']['avg'] > 2 * q_results['Q1']['avg'] if q_results['Q1']['avg'] > 0 else False
            print(f"    → Q4-Q1 spread: {spread:+.2f}%  "
                  f"Q4/Q1 ratio: {ratio:.2f}x  "
                  f"Composite validated: {'YES' if validated else 'NO'}")
            composite_results[(sid, hold)] = {
                'quartiles': q_results, 'spread': spread,
                'ratio': ratio, 'validated': validated}


# ═══════════════════════════════════════════════════════════════════════════════
# KEY QUESTIONS
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*_W}")
print(f"  KEY QUESTIONS")
print(f"{'='*_W}")

print(f"\n  Q1 — Trade frequency after filtering:")
for strat in STRATEGIES:
    sid = strat['id']
    if 'L5' not in strat['layers']:
        continue
    for hold in strat['holds']:
        row_match = [r for r in master_rows
                     if r['strategy'] == sid and r['layer'] == LAYER_NAMES['L5']
                     and r['hold'] == HOLD_LABELS[hold]]
        if row_match:
            r = row_match[0]
            flag = '✓ >= 4' if r['trades_per_day'] >= 4 else (
                '~ >= 1' if r['trades_per_day'] >= 1 else '✗ < 1')
            print(f"    {sid:<16} {HOLD_LABELS[hold]:>5}: {r['trades_per_day']:>5.1f} t/d  "
                  f"avg={r['avg_pnl_pct'] if r['avg_pnl_pct'] is not None else 0:>+.2f}%  [{flag}]")

print(f"\n  Q2 — Does 30m show confirmed edge?")
for strat in STRATEGIES:
    if '30m' not in strat['holds']:
        continue
    for layer_id in ['L0', 'L2', 'L5']:
        if layer_id not in strat['layers']:
            continue
        row_match = [r for r in master_rows
                     if r['strategy'] == strat['id'] and r['layer'] == LAYER_NAMES[layer_id]
                     and r['hold'] == '30m']
        if row_match:
            r = row_match[0]
            ci_ok = 'CI>0 ✓' if r.get('ci_confirmed') else 'CI<0 ✗'
            avg_v = r['avg_pnl_pct'] if r['avg_pnl_pct'] is not None else 0
            print(f"    {strat['id']:<16} {LAYER_NAMES[layer_id]:<18} 30m: "
                  f"avg={avg_v:>+.2f}%  {ci_ok}")

print(f"\n  Q3 — B1_WIDE_SHORT after filtering:")
for layer_id in LAYER_ORDER:
    if layer_id not in STRATEGIES[3]['layers']:
        continue
    for hold in STRATEGIES[3]['holds']:
        row_match = [r for r in master_rows
                     if r['strategy'] == 'B1_WIDE_SHORT' and r['layer'] == LAYER_NAMES[layer_id]
                     and r['hold'] == HOLD_LABELS[hold]]
        if row_match:
            r = row_match[0]
            sh = r['sharpe'] if r['sharpe'] is not None else 0
            avg_v = r['avg_pnl_pct'] if r['avg_pnl_pct'] is not None else 0
            print(f"    {LAYER_NAMES[layer_id]:<18} {HOLD_LABELS[hold]:>5}: "
                  f"n={r['n']:>5}  t/d={r['trades_per_day']:>5.1f}  "
                  f"avg={avg_v:>+.2f}%  Sharpe={sh:>.3f}")

print(f"\n  Q4 — B3_EOD_ENTRY Sharpe stability:")
for layer_id in LAYER_ORDER:
    if layer_id not in STRATEGIES[2]['layers']:
        continue
    row_match = [r for r in master_rows
                 if r['strategy'] == 'B3_EOD_ENTRY' and r['layer'] == LAYER_NAMES[layer_id]
                 and r['hold'] == 'T+2']
    if row_match:
        r = row_match[0]
        sh = r['sharpe'] if r['sharpe'] is not None else 0
        avg_v = r['avg_pnl_pct'] if r['avg_pnl_pct'] is not None else 0
        print(f"    {LAYER_NAMES[layer_id]:<18} T+2: n={r['n']:>3}  "
              f"avg={avg_v:>+.2f}%  Sharpe={sh:>.3f}")

print(f"\n  Q5 — Composite score validation:")
for key, cr in composite_results.items():
    sid, hold = key
    print(f"    {sid:<16} {HOLD_LABELS[hold]:>5}: "
          f"Q4-Q1={cr['spread']:>+.2f}%  ratio={cr['ratio']:.2f}x  "
          f"validated={'YES ✓' if cr['validated'] else 'NO ✗'}")


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n  Generating charts...")

strategy_ids = [s['id'] for s in STRATEGIES]
strategy_colors = {s['id']: s['color'] for s in STRATEGIES}

# ── Chart 1: Master Comparison — L0 vs L5 per strategy ─────────────────────

fig, axes = plt.subplots(3, 3, figsize=(22, 16))
fig.suptitle('Strategy Backtest — L0 (raw) vs L5 (combined filters)\n'
             f'{N_DATES} dates ({TRADING_DATES[0]} to {TRADING_DATES[-1]})',
             fontsize=14, fontweight='bold', y=0.98)

hold_list = ['30m', '60m', 't2']
metric_list = [('avg_pnl_pct', 'Avg PnL (%)'), ('sharpe', 'Sharpe'), ('win_rate', 'Win Rate')]

for row_i, hold in enumerate(hold_list):
    for col_i, (metric_key, metric_label) in enumerate(metric_list):
        ax = axes[row_i, col_i]
        x_labels = []
        l0_vals = []
        l5_vals = []

        for strat in STRATEGIES:
            sid = strat['id']
            if hold not in strat['holds']:
                continue
            x_labels.append(sid.replace('BASELINE_', 'BL_').replace('_SHORT', ''))

            for layer_id, val_list in [('L0', l0_vals), ('L5', l5_vals)]:
                if layer_id not in strat['layers']:
                    val_list.append(0)
                    continue
                row_match = [r for r in master_rows
                             if r['strategy'] == sid and r['layer'] == LAYER_NAMES[layer_id]
                             and r['hold'] == HOLD_LABELS[hold]]
                if row_match and row_match[0].get(metric_key) is not None:
                    v = row_match[0][metric_key]
                    if metric_key == 'win_rate':
                        v *= 100
                    val_list.append(v)
                else:
                    val_list.append(0)

        if not x_labels:
            ax.set_visible(False)
            continue

        x = np.arange(len(x_labels))
        w = 0.35
        ax.bar(x - w/2, l0_vals, w, label='L0 raw', color='#3498db', alpha=0.75)
        ax.bar(x + w/2, l5_vals, w, label='L5 combined', color='#27ae60', alpha=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=8, rotation=15)
        ax.set_ylabel(metric_label)
        ax.set_title(f'{HOLD_LABELS[hold]} — {metric_label}')
        ax.axhline(0, color='#333', lw=0.5)
        if row_i == 0 and col_i == 0:
            ax.legend(fontsize=8)

        for i, (v0, v5) in enumerate(zip(l0_vals, l5_vals)):
            ax.text(i - w/2, v0, f'{v0:.1f}', ha='center', va='bottom', fontsize=6)
            ax.text(i + w/2, v5, f'{v5:.1f}', ha='center', va='bottom', fontsize=6)

plt.tight_layout(rect=[0, 0, 1, 0.95])
chart1_path = os.path.join(BASE_DIR, 'strategy_backtest_comparison.png')
plt.savefig(chart1_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"    Chart 1: {chart1_path}")

# ── Chart 2: Equity Curves ────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle(f'Equity Curves — ${STARTING_CAPITAL:,.0f} Start, Equal Weight per Day\n'
             f'L0 (dotted) vs L5 (solid)',
             fontsize=14, fontweight='bold', y=0.98)

hold_styles = {'30m': ('o', '-'), '60m': ('s', '-'), 't2': ('^', '-')}
hold_colors_eq = {'30m': '#3498db', '60m': '#e67e22', 't2': '#27ae60'}

for ax_i, strat in enumerate(STRATEGIES):
    ax = axes[ax_i // 2, ax_i % 2]
    sid = strat['id']

    for hold in strat['holds']:
        for layer_id, ls in [('L0', ':'), ('L5', '-')]:
            if layer_id not in strat['layers']:
                continue
            eq_key = (sid, layer_id, hold)
            curve = equity_data.get(eq_key, [])
            if not curve:
                continue

            dates_plot = [pt['date'] for pt in curve]
            equities = [pt['equity'] for pt in curve]
            marker, _ = hold_styles[hold]
            color = hold_colors_eq[hold]
            label = f"{HOLD_LABELS[hold]} {LAYER_NAMES[layer_id]}"
            ax.plot(range(len(dates_plot)), equities, marker=marker, linestyle=ls,
                    color=color, markersize=6, linewidth=2, label=label, alpha=0.85)

    ax.axhline(STARTING_CAPITAL, color='#999', ls='--', lw=1, alpha=0.5)
    ax.set_xticks(range(N_DATES))
    ax.set_xticklabels([d[5:] for d in TRADING_DATES], fontsize=9)
    ax.set_ylabel('Portfolio Value ($)')
    ax.set_title(f'{sid}', fontweight='bold')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.94])
chart2_path = os.path.join(BASE_DIR, 'strategy_backtest_equity.png')
plt.savefig(chart2_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"    Chart 2: {chart2_path}")

# ── Chart 3: Composite Quartile Analysis ──────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Composite Score Quartile Analysis (L5 Combined)',
             fontsize=14, fontweight='bold', y=0.98)

for ax_i, strat in enumerate(STRATEGIES):
    ax = axes[ax_i // 2, ax_i % 2]
    sid = strat['id']
    best_hold = strat['holds'][-1]

    cr = composite_results.get((sid, best_hold))
    if cr and cr.get('quartiles'):
        q_names = ['Q1', 'Q2', 'Q3', 'Q4']
        q_avgs = [cr['quartiles'].get(q, {}).get('avg', 0) for q in q_names]
        q_ns = [cr['quartiles'].get(q, {}).get('n', 0) for q in q_names]
        colors = ['#e74c3c' if v < 0 else '#27ae60' for v in q_avgs]
        bars = ax.bar(q_names, q_avgs, color=colors, alpha=0.8, edgecolor='white')
        for i, (v, n) in enumerate(zip(q_avgs, q_ns)):
            ax.text(i, v + (0.1 if v >= 0 else -0.3), f'n={n}\n{v:+.1f}%',
                    ha='center', fontsize=8, fontweight='bold')
        ax.axhline(0, color='#333', lw=0.7)
        validated = cr.get('validated', False)
        ax.set_title(f'{sid} / {HOLD_LABELS[best_hold]}\n'
                     f'Q4-Q1={cr["spread"]:+.1f}%  '
                     f'{"✓ VALIDATED" if validated else "✗ not validated"}',
                     fontweight='bold', fontsize=10)
    else:
        ax.text(0.5, 0.5, f'{sid}\nInsufficient data\nfor quartile analysis',
                ha='center', va='center', fontsize=11, transform=ax.transAxes)
        ax.set_title(sid, fontweight='bold')

    ax.set_ylabel('Avg PnL (%)')
    ax.set_xlabel('Composite Score Quartile')

plt.tight_layout(rect=[0, 0, 1, 0.94])
chart3_path = os.path.join(BASE_DIR, 'strategy_backtest_composite.png')
plt.savefig(chart3_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"    Chart 3: {chart3_path}")

# ── Chart 4: Per-Day Trade Count + PnL Time Series ────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Daily Trade Count and Average PnL — L0 vs L5',
             fontsize=14, fontweight='bold', y=0.98)

for ax_i, strat in enumerate(STRATEGIES):
    ax = axes[ax_i // 2, ax_i % 2]
    sid = strat['id']
    best_hold = strat['holds'][-1]
    hold_key = HOLD_KEYS[best_hold]

    for layer_id, style, alpha in [('L0', '--', 0.6), ('L5', '-', 1.0)]:
        if layer_id not in strat['layers']:
            continue
        filtered = apply_layer_filter(all_strategy_trades[sid], layer_id, sid)
        daily_n = []
        daily_avg = []
        for dt in TRADING_DATES:
            day_pnls = [t[hold_key] for t in filtered
                        if t['date'] == dt and t.get(hold_key) is not None]
            daily_n.append(len(day_pnls))
            daily_avg.append(np.mean(day_pnls) if day_pnls else 0)

        x = np.arange(N_DATES)
        color = strat['color']
        ax.bar(x + (0.2 if layer_id == 'L5' else -0.2), daily_avg, 0.35,
               label=f'{LAYER_NAMES[layer_id]} avg PnL', color=color, alpha=alpha,
               edgecolor='white')

        ax2 = ax.twinx()
        ax2.plot(x + (0.2 if layer_id == 'L5' else -0.2), daily_n,
                 'D', color='#333', markersize=5, alpha=alpha*0.7)
        ax2.set_ylabel('# Trades', fontsize=8, color='#666')
        ax2.tick_params(axis='y', labelcolor='#666', labelsize=7)

    ax.set_xticks(range(N_DATES))
    ax.set_xticklabels([d[5:] for d in TRADING_DATES], fontsize=9)
    ax.set_ylabel(f'Avg PnL % ({HOLD_LABELS[best_hold]})')
    ax.set_title(f'{sid}', fontweight='bold')
    ax.axhline(0, color='#333', lw=0.7)
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.2)

plt.tight_layout(rect=[0, 0, 1, 0.94])
chart4_path = os.path.join(BASE_DIR, 'strategy_backtest_timeseries.png')
plt.savefig(chart4_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"    Chart 4: {chart4_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*_W}")
print(f"  SAVING OUTPUTS")
print(f"{'='*_W}")

master_df = pd.DataFrame(master_rows)
master_path = os.path.join(RESULTS_DIR, 'bt_master_table.csv')
master_df.to_csv(master_path, index=False)
print(f"    Master table: {master_path} ({len(master_rows)} rows)")

for strat in STRATEGIES:
    sid = strat['id']
    trades = all_strategy_trades[sid]
    if trades:
        df = pd.DataFrame(trades)
        path = os.path.join(RESULTS_DIR, f'bt_trades_{sid}.csv')
        df.to_csv(path, index=False)
        print(f"    Trades: {os.path.basename(path):40s} ({len(trades)} trades)")

results_output = {
    'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'trading_dates': TRADING_DATES, 'n_dates': N_DATES,
    'starting_capital': STARTING_CAPITAL,
    'strategies': [s['id'] for s in STRATEGIES],
    'layers': LAYER_NAMES,
    'hold_periods': HOLD_LABELS,
    'master_table': master_rows,
    'filter_funnels': {f"{k[0]}__{k[1]}": v for k, v in filter_funnels.items()},
    'composite_results': {
        f"{k[0]}__{k[1]}": {
            'spread': v['spread'], 'ratio': v['ratio'], 'validated': v['validated'],
            'quartiles': {q: dict(qv) for q, qv in v['quartiles'].items()},
        }
        for k, v in composite_results.items()
    },
    'equity_curves': {
        f"{k[0]}__{k[1]}__{k[2]}": v
        for k, v in equity_data.items()
    },
}

results_path = os.path.join(RESULTS_DIR, 'bt_results.json')
with open(results_path, 'w') as f:
    json.dump(results_output, f, indent=2, default=str)
print(f"    Results: {results_path}")


# ── Summary ──────────────────────────────────────────────────────────────────

print(f"\n{'='*_W}")
print(f"  COMPLETE")
print(f"{'='*_W}")

total_trades = sum(len(v) for v in all_strategy_trades.values())
print(f"  {len(STRATEGIES)} strategies × {len(LAYER_ORDER)} layers × 3 holds = {len(master_rows)} cells")
print(f"  Total raw trades: {total_trades}")

candidates = [r for r in master_rows
               if r.get('ci_confirmed') and r.get('trades_per_day', 0) >= 1.0 and r['n'] >= 3]
if candidates:
    print(f"\n  ★ LIVE TRADING CANDIDATES (CI>0 AND t/d >= 1.0):")
    for r in candidates:
        print(f"    {r['strategy']:<16} {r['layer']:<18} {r['hold']:>5}  "
              f"avg={r['avg_pnl_pct']:>+.2f}%  Sharpe={r['sharpe']:.3f}  "
              f"t/d={r['trades_per_day']:.1f}  compound={r['total_compound_pct']:+.2f}%")
else:
    print(f"\n  No live trading candidates (CI>0 AND t/d >= 1.0)")

print(f"\n  Output files:")
print(f"    {master_path}")
print(f"    {results_path}")
print(f"    {chart1_path}")
print(f"    {chart2_path}")
print(f"    {chart3_path}")
print(f"    {chart4_path}")
print(f"{'='*_W}")
