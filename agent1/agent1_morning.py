# %% Agent 1 — Morning Gap-Fade Scanner
# OG baseline: hardcoded _FLOAT dict, unknown tickers pass, gap >= 5%, price >= $10, N=5
# Run pre-market (~6:00 AM PST / 9:00 AM ET). Outputs top 5 SHORT picks and logs to CSV.
# Set SCAN_DATE = None for live (today), or 'YYYY-MM-DD' for historical replay.

import os, sys, time, re, csv
from datetime import date, datetime, timedelta

import pytz
import requests

try:
    from tabulate import tabulate
except ImportError:
    import subprocess; subprocess.check_call([sys.executable, "-m", "pip", "install", "tabulate", "--quiet", "--break-system-packages"])
    from tabulate import tabulate

try:
    from google.colab import auth
    try: auth.authenticate_user()
    except: pass
except: pass


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

SCAN_DATE = None              # None = today (live), 'YYYY-MM-DD' = historical replay

GAP_MIN_PCT = 5.0             # minimum gap %
PRICE_MIN = 10.0              # minimum stock price $
FLOAT_MIN = 10_000_000        # minimum float shares
TOP_N = 5                     # select top N per day

LOG_FILE = 'agent1_picks.csv' # append picks here each day

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs")
_BASE = "https://api.polygon.io"
_TK_RE = re.compile(r'^[A-Z]{1,5}$')

ET = pytz.timezone('America/New_York')
PT = pytz.timezone('America/Los_Angeles')

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

def _is_trading_day(d):
    if isinstance(d, str):
        d = datetime.strptime(d, '%Y-%m-%d').date()
    return d.weekday() < 5 and d.strftime('%Y-%m-%d') not in _HOLIDAYS

def _prior_td(d):
    if isinstance(d, str):
        d = datetime.strptime(d, '%Y-%m-%d').date()
    d = d - timedelta(days=1)
    for _ in range(10):
        if _is_trading_day(d):
            return d
        d -= timedelta(days=1)
    raise ValueError(f"No trading day found before {d}")

def _next_n_td(d, n=2):
    if isinstance(d, str):
        d = datetime.strptime(d, '%Y-%m-%d').date()
    out = []
    d = d + timedelta(days=1)
    for _ in range(30):
        if _is_trading_day(d):
            out.append(d)
            if len(out) >= n:
                return out
        d += timedelta(days=1)
    return out

def _pg(url, params=None):
    p = dict(params or {})
    p['apiKey'] = POLYGON_API_KEY
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


# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCH
# ═══════════════════════════════════════════════════════════════════════════════

_gd_cache = {}

def fetch_grouped_daily(dt_str):
    if dt_str in _gd_cache:
        return _gd_cache[dt_str]
    data = _pg(f"{_BASE}/v2/aggs/grouped/locale/us/market/stocks/{dt_str}",
               {"adjusted": "true", "include_otc": "false"})
    out = {}
    for r in data.get('results', []):
        tk = r.get('T', '')
        if not _TK_RE.match(tk):
            continue
        out[tk] = {'o': float(r.get('o',0)), 'h': float(r.get('h',0)),
                   'l': float(r.get('l',0)), 'c': float(r.get('c',0)),
                   'v': int(r.get('v',0))}
    _gd_cache[dt_str] = out
    return out

def fetch_snapshot():
    data = _pg(f"{_BASE}/v2/snapshot/locale/us/markets/stocks/tickers",
               {"include_otc": "false"})
    out = {}
    for t in data.get('tickers', []):
        tk = t.get('ticker', '')
        if not _TK_RE.match(tk):
            continue
        prev_c = (t.get('prevDay') or {}).get('c')
        pm_p = (t.get('min') or {}).get('c')
        if not pm_p or pm_p <= 0:
            pm_p = prev_c
        if pm_p and pm_p > 0:
            out[tk] = {'price': float(pm_p), 'prev_c': float(prev_c) if prev_c else None}
    return out



# ═══════════════════════════════════════════════════════════════════════════════
# SCANNER — OG baseline (exact agent1_backtest.py logic)
# ═══════════════════════════════════════════════════════════════════════════════

def scan(dt_str=None):
    t0 = time.time()

    is_live = dt_str is None
    if is_live:
        dt_str = datetime.now(ET).strftime('%Y-%m-%d')

    if not _is_trading_day(dt_str):
        print(f"  {dt_str} is not a trading day — skipping")
        return [], dt_str

    prior_str = _prior_td(dt_str).strftime('%Y-%m-%d')
    prior = fetch_grouped_daily(prior_str)

    if is_live:
        snap = fetch_snapshot()
    else:
        snap = None
        today = fetch_grouped_daily(dt_str)

    gap_min = GAP_MIN_PCT / 100.0
    candidates = []
    for tk in (snap or today):
        if tk in _EXCLUDE_TK:
            continue
        if any(tk.endswith(s) for s in _EXCLUDE_SUFFIX) and len(tk) > 2:
            continue
        if tk not in prior:
            continue

        prev_c = prior[tk]['c']
        if prev_c <= 0:
            continue

        if is_live:
            cur_p = snap[tk]['price']
        else:
            cur_p = today[tk]['o']
        if cur_p <= 0 or cur_p < PRICE_MIN:
            continue

        gap = (cur_p - prev_c) / prev_c
        if gap < gap_min:
            continue

        fs = _FLOAT.get(tk)
        if fs is not None and fs < FLOAT_MIN:
            continue

        candidates.append({
            'ticker': tk, 'gap_pct': gap, 'prior_close': prev_c,
            'open_price': cur_p, 'float': fs,
        })

    candidates.sort(key=lambda x: -x['gap_pct'])
    selected = candidates[:TOP_N]
    for i, s in enumerate(selected):
        s['rank'] = i + 1

    nxt = _next_n_td(dt_str, 2)
    t1_str = nxt[0].strftime('%Y-%m-%d') if len(nxt) >= 1 else None
    t2_str = nxt[1].strftime('%Y-%m-%d') if len(nxt) >= 2 else None
    for s in selected:
        s['scan_date'] = dt_str
        s['t1_date'] = t1_str
        s['t2_date'] = t2_str

    elapsed = time.time() - t0
    now_et = datetime.now(ET).strftime('%H:%M ET')

    print(f"\n{'='*70}")
    print(f"  AGENT 1 — OG BASELINE SCAN")
    print(f"  Date: {dt_str}  |  {now_et}  |  {elapsed:.1f}s")
    print(f"  Filter: gap >= {GAP_MIN_PCT}%, price >= ${PRICE_MIN:.0f}, float >= {FLOAT_MIN/1e6:.0f}M (known only)")
    print(f"  {len(candidates)} candidates → top {TOP_N} by gap%")
    print(f"  T+2 exit: {t2_str or 'n/a'}")
    print(f"{'='*70}")

    if selected:
        rows = []
        for s in selected:
            fs = s.get('float')
            fs_str = f"{fs/1e6:.0f}M" if fs and fs < 1e9 else (f"{fs/1e9:.1f}B" if fs else 'unk')
            rows.append([
                f"#{s['rank']}", s['ticker'], f"+{s['gap_pct']*100:.1f}%",
                f"${s['prior_close']:.2f}", f"${s['open_price']:.2f}", fs_str
            ])
        print()
        print(tabulate(rows, headers=['Rank','Ticker','Gap%','Prev Close','Open','Float'],
                       tablefmt='simple', colalign=('left','left','right','right','right','right')))

        print(f"\n  ACTION: SHORT these {len(selected)} at local high, exit T+2 close ({t2_str})")
        print(f"  ENTRY: Watch 1-min chart, enter SHORT when bar[i] high > W={PEAK_WINDOW} bars")
        print(f"         before AND after (first 12 bars). Fallback entry at bar 15 (~09:45).")
    else:
        print(f"\n  No qualifying trades today.")

    return selected, dt_str


# ═══════════════════════════════════════════════════════════════════════════════
# LOG PICKS TO CSV
# ═══════════════════════════════════════════════════════════════════════════════

def log_picks(picks, log_file=LOG_FILE):
    if not picks:
        return
    file_exists = os.path.exists(log_file)
    fields = ['scan_date','rank','ticker','gap_pct','prior_close','open_price','float','t1_date','t2_date']
    with open(log_file, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        if not file_exists:
            w.writeheader()
        for p in picks:
            row = dict(p)
            row['gap_pct'] = round(p['gap_pct'] * 100, 2)
            w.writerow(row)
    print(f"\n  Logged {len(picks)} picks to {log_file}")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

picks, scan_date = scan(SCAN_DATE)
log_picks(picks)
