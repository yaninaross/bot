# %% Agent 1 — V8.2 Unified Live Scanner: ALL SHORT + LONG Candidates (Single-Cell Colab Ready)
# ==============================================================================================
# 100% Standalone Single-Cell Script (Google Colab / Vertex AI / Local CLI).
# Zero local file dependencies, zero pip installs, zero auth popups.
# Combines Official V8.2-S (SHORT, Choice 1: Max Profit) + Official V8.2-L (LONG, C1 + Intraday A/B).
#
# 12.5M Split-Proof Unleveraged Cash ($50K Short + $50K Long):
#   $100,000 -> $361,151 (+261% real profit) | Max DD: -13.9% | Sharpe: 3.32 | 12/13 green months
#
# HOW TO USE IN GOOGLE COLAB:
#   Paste this entire file into ONE cell and click Run!
#   - 06:30:15 PST (Wed-Fri): Catches V8.2-L Engine C1 Open Longs + previews V8.2-S Short watchlist.
#   - 06:31:15 PST (Mon-Fri): Locks top 3 ETB + Catalyst-Clean V8.2-S Short watchlist.
#   - 06:33-06:44 PST       : Re-run every 1-2 min to confirm V8.2-S Shorts (Bar0 Flush / Local High).
#   - 06:44:10 PST (Mon-Fri): Final morning run — confirms V8.2-L Intraday T&Q Longs (Engine A/B) too!
#
# CLI EXAMPLES:
#   python3 agent1/agent1_v82_unified.py                        # Live Today (SHORT + LONG)
#   python3 agent1/agent1_v82_unified.py --date=2026-11-18      # Historical full-day replay
#   python3 agent1/agent1_v82_unified.py --htb=SYM1 --open=AEHR # Manual HTB & open-long overrides
# ==============================================================================================

import os
import sys
import re
import time
import urllib.request
import warnings
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytz
import requests

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# USER CONTROLS (Edit directly in Colab or pass CLI flags)
# ═══════════════════════════════════════════════════════════════════════════════
SCAN_DATE            = None     # None = Live Today (PST), or 'YYYY-MM-DD' / 'YYYY-MM-DD HH:MM' (PST)
PICKY_SHORT_MODE     = False    # False = Choice 1 (Max Profit, Official V8.2); True (--picky) = Choice 2
LONG_MODE            = "hybrid" # "hybrid" (C1 Swing + Intraday A/B), "c1", or "intraday"
FORCE_INTRADAY_LONG  = False    # False = auto-runs Intraday Long A/B at >=06:43 PST (or historical) for speed
MANUAL_HTB_TODAY     = ""       # Optional comma-separated HTB symbols to block from shorting, e.g. "XYZ,ABC"
MANUAL_OPEN_LONGS    = ""       # Optional comma-separated open C1 longs to protect from dup buy, e.g. "AEHR"
POLYGON_API_KEY      = os.environ.get("POLYGON_API_KEY", "2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs")

# Strategy Constants (V8.2 Locked — 12.5M Split-Proof Ablation Winner)
STRATEGY_VERSION             = "V8.2"
SHORT_GAP_MIN_PCT            = 3.0
SHORT_PRICE_MIN              = 10.0
SHORT_PRICE_MAX              = 100.0
FLOAT_MIN                    = 10_000_000
PRE_FILTER_N                 = 30
GAP_CONSUMED_MAX             = 0.50
SCAN_TIME_ET                 = "09:31"
EXIT_TIME_INST_ET            = "11:30"   # 08:30 PST — <15% gap shorts & $50+ local_high
EXIT_TIME_LH_ET              = "12:30"   # 09:30 PST — <$50 local_high >=15% gap
EXIT_TIME_ET                 = "14:00"   # 11:00 PST — bar0_flush >=15% gap (Hard Flat)
MIN_RTH_BARS                 = 2
MIN_OPEN_VOL                 = 5_000
PEAK_WINDOW                  = 3
MAX_PEAK_BAR                 = 12
MAX_PEAK_TIME_ET             = "09:41"
MAX_ENTRY_TIME_ET            = "09:44"
MIN_PULLBACK_PCT             = 0.5
MAX_MISSING_BARS_MOD_GAP     = 1
MAX_OPEN_RUNUP_MOD_GAP       = 0.07
TIER1_MOD_GAP_MAX_PRICE      = 25.0
TIER1_MOD_GAP_MAX_PRIOR_DVOL = 25_000_000
SHORT_HARD_STOP_PCT          = 0.10      # -10.0% hard stop (NO trailing profit lock, NO _B2 re-shorts)
TQ_15_30_MIN_MSYN            = 0.35      # 15-30% gap unlock requires Peak->Entry MSYN >= +0.35
TQ_15_30_MAX_BLK_OFI         = -0.15     # 15-30% gap unlock requires Peak->Entry Block_OFI <= -0.15
MIN_ETB_SHARES               = 5_000
MAX_BORROW_FEE_PCT           = 30.0
BLOCK_DUPLICATE_OPEN_C1      = True      # V8.2-L Guard: never re-buy a C1 stock while its T+3 trade is open

ET = pytz.timezone("America/New_York")
PT = pytz.timezone("America/Los_Angeles")
_BASE = "https://api.polygon.io"
_TK_RE = re.compile(r"^[A-Z]{1,5}$")

_EXCLUDE_TK = {
    "SPY", "QQQ", "IWM", "DIA", "VXX", "UVXY", "SQQQ", "TQQQ", "SPXU", "SPXL",
    "SDOW", "UDOW", "SDS", "SSO", "SH", "LABU", "LABD", "SOXL", "SOXS",
    "FNGU", "FNGD", "NUGT", "DUST", "JNUG", "JDST", "TNA", "TZA",
    "UPRO", "VIXY", "SVXY", "TVIX",
}
_EXCLUDE_SUFFIX = ("W", "WS", "U", "R")
_SPAC_KEYWORDS  = ("ACQUISITION", "BLANK CHECK", "SPAC")
_ETF_BAD_WORDS  = (
    "ETF", "ETN", "TRUST", "FUND", "2X", "3X", "1.5X", "ULTRA", "PROSHARES",
    "DIREXION", "GRANITESHARES", "DEFIANCE", "T-REX", "YIELDMAX", "INVERSE",
    "BEAR", "BULL", "SHARES",
)
_HOLIDAYS = {
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
    "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
    "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
}

# Verified Float Registry (194 liquid >=10M float stocks for C1 LONG + 23 known <10M exclusions for SHORT)
_FLOAT = {
    "AAPL":14594180000,"AARD":21884158,"ABM":58580923,"ACHR":770023800,"ACMR":64657388,
    "ACV":10396028,"ACVA":169807980,"ADBE":397500000,"AEHR":32620450,"AENT":50979630,
    "AEON":49882790,"AGCO":70031729,"AGRZ":22573405,"AHMA":10965000,"ALAB":173485104,
    "AMAT":793597443,"AMD":1632475042,"AME":229203002,"APA":350351510,"APLD":291469112,
    "ARBE":122649743,"ARM":1068078760,"ASAN":159441754,"ASO":62018371,"ASTS":299789305,
    "ATEC":154451076,"ATHE":18208368,"ATI":136170383,"AVAV":50822615,"AXGN":53761513,
    "AXTA":214019997,"AXTI":65573212,"BABA":2485623615,"BBIO":195487474,"BBNX":44989688,
    "BE":294527346,"BEAM":103287665,"BHVN":151043781,"BJDX":4468051,"BLZE":61900000,
    "BNC":41173850,"BNKK":8000940,"BOW":32943005,"BRNX":720888,"BRZE":112631669,
    "BTBT":360633503,"CAN":749194858,"CAT":459674889,"CC":150498995,"CCB":15286327,
    "CIFR":415030722,"CLSK":256817073,"COIN":222803032,"COLA":4494439,"COO":195030630,
    "CORZ":321340441,"CPRT":925811482,"CRDO":187951918,"CRMT":8663493,"CSR":16794151,
    "CVX":1975771274,"DBGI":933509,"DELL":324873640,"DFNS":1663806,"DKS":65355923,
    "DNOW":180785891,"DOV":134680225,"DPRO":34375971,"DSS":10042518,"EAT":41765010,
    "EDHL":5959275,"ELOG":12832000,"ET":3443299601,"ETN":388400000,"EVRG":230567172,
    "EXEL":247781692,"FCEL":79954196,"FCX":1436017523,"FEIM":11177530,"FIZZ":93615302,
    "FLNC":143163588,"FLWS":37030262,"GDDY":126647704,"GENC":12338845,"GLXG":5601515,
    "GME":448691257,"GMEX":903642,"GOOS":46656876,"GRI":2186115,"HBM":444144760,
    "HCAI":7629942,"HGTY":102070523,"HOFT":10741650,"HOOD":790630234,"HPE":1327464779,
    "HPQ":901790969,"HTLD":77332611,"HUT":123259468,"HXHX":8950000,"IBTA":20197545,
    "IESC":19924356,"IMO":481577721,"INBX":14717660,"INDP":133242324,"INFQ":225357052,
    "INTC":5044000000,"INV":84612657,"IONQ":381002314,"IP":529569895,"IPDN":14450323,
    "IREN":394058648,"KEEL":617573212,"KPTI":22681460,"KR":612647282,"KTOS":187721327,
    "LASE":50177716,"LEU":19233658,"LOVE":14638418,"LULU":105594064,"LUNR":173231343,
    "M":263037954,"MARA":386299297,"MAS":197187430,"MCFT":16279890,"MDCX":63180610,
    "META":2205128509,"MGN":16250000,"MOBX":16847921,"MRNA":399235889,"MSTR":364585501,
    "MU":1129393151,"MVIS":23788853,"MXL":90693191,"NBIS":219465088,"NEE":2085803189,
    "NEGG":20973423,"NNE":53698146,"NTHI":25936365,"NTLA":140126693,"NVDA":24100000000,
    "NVS":2034819509,"OCGN":339110401,"OKLO":186017650,"ONDS":571454656,"OPTT":270138823,
    "OPTX":40279878,"ORCL":2880471000,"OTLK":243447554,"PAVS":856897,"PCLA":9613805,
    "PDEX":3203334,"PDSB":55971338,"PGY":72137266,"PL":340354193,"PLNT":75216965,
    "PLTR":2300713329,"PLUG":1397195278,"PMTS":11530669,"PRLD":79803485,"QBTS":368835324,
    "QNTM":3816937,"RDDT":146103200,"RDW":249992609,"REGN":101137842,"RGTI":333768747,
    "RH":18900969,"RIOT":375258935,"RKLB":598350482,"RLMD":106669846,"ROIV":722323015,
    "RTX":1347758144,"SCCO":834331706,"SDRL":62541443,"SHMD":60958903,"SHOE":27151308,
    "SIG":39329783,"SKYX":135430994,"SLS":201945709,"SLXN":5755872,"SMCI":656965384,
    "SMR":410389522,"SNDK":146419001,"SNX":79958752,"SNYR":15079956,"SOUN":411576753,
    "SRPT":105627021,"SSL":647761670,"SSNC":234654996,"STRL":30588518,"STX":226644518,
    "SURG":52908423,"SWKS":150472782,"SWRD":211149963,"SWVL":9964344,"TEN":30127603,
    "TNON":1032578,"TPST":16539717,"TRUG":1903708,"TSLA":3949547394,"TTAN":82746425,
    "TTD":426771565,"TYRA":59676939,"UBER":2042560121,"UNFI":60519140,"UPST":97313046,
    "URI":62242489,"VAC":34395320,"VICR":34389014,"VOYG":58217592,"VRT":384988173,
    "VSME":2750784,"VST":335635195,"WDH":279111455,"WULF":498968677,"YARW":2669746,
    "ZEO":41703489,"ZUMZ":16872215,
}
FLOAT_10M = {k for k, v in _FLOAT.items() if v >= FLOAT_MIN}


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED DATA CLIENT & CACHES (Zero Duplicate API Calls Between Short & Long)
# ═══════════════════════════════════════════════════════════════════════════════
_HTTP = requests.Session()
_HTTP.mount("https://", requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32))
_REF_CACHE  = {}
_BARS_CACHE = {}
_NEWS_CACHE = {}
_TQ_CACHE   = {}
_IBKR_CACHE = None


def _is_td(d):
    if isinstance(d, str):
        d = datetime.strptime(d, "%Y-%m-%d").date()
    return d.weekday() < 5 and d.strftime("%Y-%m-%d") not in _HOLIDAYS


def _prev_trading_days(dt_str, n_days=8):
    d = datetime.strptime(dt_str, "%Y-%m-%d").date()
    out = []
    while len(out) < n_days:
        d -= timedelta(days=1)
        s = d.strftime("%Y-%m-%d")
        if d.weekday() < 5 and s not in _HOLIDAYS:
            out.append(s)
    return out


def _next_trading_day(dt_str, n_steps=3):
    d = datetime.strptime(dt_str, "%Y-%m-%d").date()
    steps = 0
    while steps < n_steps:
        d += timedelta(days=1)
        s = d.strftime("%Y-%m-%d")
        if d.weekday() < 5 and s not in _HOLIDAYS:
            steps += 1
    return d.strftime("%Y-%m-%d")


def _et_to_pst(hhmm_et):
    if not hhmm_et or ":" not in str(hhmm_et):
        return str(hhmm_et)
    try:
        h, m = map(int, str(hhmm_et).split(":")[:2])
        return f"{(h - 3) % 24:02d}:{m:02d}"
    except Exception:
        return str(hhmm_et)


def _pg(url, params=None, timeout=18):
    p = dict(params or {})
    p["apiKey"] = POLYGON_API_KEY
    for attempt in range(4):
        try:
            r = _HTTP.get(url, params=p, timeout=timeout)
            if r.status_code == 429:
                time.sleep(2 ** (attempt + 1))
                continue
            if r.status_code == 200:
                return r.json()
            return {}
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return {}


def fetch_grouped_daily(dt_str):
    data = _pg(f"{_BASE}/v2/aggs/grouped/locale/us/market/stocks/{dt_str}",
               {"adjusted": "true", "include_otc": "false"}, timeout=25)
    out = {}
    for row in data.get("results", []):
        tk = row.get("T", "")
        if not _TK_RE.match(tk):
            continue
        out[tk] = {
            "o": float(row.get("o", 0)), "h": float(row.get("h", 0)),
            "l": float(row.get("l", 0)), "c": float(row.get("c", 0)),
            "v": int(row.get("v", 0)), "price": float(row.get("o", 0)),
            "has_official_open": True,
        }
    return out


def fetch_snapshot_all():
    data = _pg(f"{_BASE}/v2/snapshot/locale/us/markets/stocks/tickers",
               {"include_otc": "false"}, timeout=25)
    out = {}
    for t in data.get("tickers", []):
        tk = t.get("ticker", "")
        if not _TK_RE.match(tk):
            continue
        day_d  = t.get("day", {}) or {}
        min_d  = t.get("min", {}) or {}
        prev_d = t.get("prevDay", {}) or {}
        day_o  = float(day_d.get("o") or 0.0)
        pm_c   = float(min_d.get("c") or 0.0)
        prev_c = float(prev_d.get("c") or 0.0)
        open_p = day_o if day_o > 0 else (pm_c if pm_c > 0 else prev_c)
        cur_c  = float(day_d.get("c") or pm_c or open_p)
        if open_p > 0:
            out[tk] = {
                "o": open_p,
                "h": float(day_d.get("h") or open_p),
                "l": float(day_d.get("l") or open_p),
                "c": cur_c if cur_c > 0 else open_p,
                "v": int(day_d.get("v") or 0),
                "price": pm_c if pm_c > 0 else open_p,
                "prev_close": prev_c if prev_c > 0 else None,
                "has_official_open": day_o > 0,
            }
    return out


def fetch_ticker_ref(tk):
    if tk in _REF_CACHE:
        return _REF_CACHE[tk]
    data = _pg(f"{_BASE}/v3/reference/tickers/{tk}", timeout=10)
    res = data.get("results", {}) or {}
    val = {"type": (res.get("type") or "UNK").upper(), "name": res.get("name") or ""}
    _REF_CACHE[tk] = val
    return val


def fetch_ticker_refs_parallel(tickers):
    missing = [tk for tk in tickers if tk not in _REF_CACHE]
    if missing:
        with ThreadPoolExecutor(max_workers=16) as ex:
            list(ex.map(fetch_ticker_ref, missing))
    return {tk: _REF_CACHE.get(tk, {"type": "UNK", "name": ""}) for tk in tickers}


def fetch_minute_bars_one(tk, dt_str, max_time_et=None):
    key = (tk, dt_str, max_time_et)
    if key in _BARS_CACHE:
        return _BARS_CACHE[key]
    data = _pg(f"{_BASE}/v2/aggs/ticker/{tk}/range/1/minute/{dt_str}/{dt_str}",
               {"adjusted": "true", "sort": "asc", "limit": "5000"}, timeout=15)
    bars = data.get("results", [])
    if not bars:
        _BARS_CACHE[key] = None
        return None
    pm, rth, ah = [], [], []
    for b in bars:
        ts = pd.Timestamp(b["t"], unit="ms", tz="UTC").tz_convert(ET)
        t_str = ts.strftime("%H:%M")
        if max_time_et is not None and t_str > max_time_et:
            continue
        bd = {"o": float(b["o"]), "h": float(b["h"]), "l": float(b["l"]), "c": float(b["c"]), "v": int(b["v"])}
        if "04:00" <= t_str < "09:30":
            pm.append((t_str, bd))
        elif "09:30" <= t_str < "16:00":
            rth.append((t_str, bd))
        elif "16:00" <= t_str <= "20:00":
            ah.append((t_str, bd))
    out = {"pm": pm, "rth": rth, "ah": ah}
    _BARS_CACHE[key] = out
    return out


def fetch_minute_bars_parallel(tickers, dt_str, max_time_et=None):
    if not tickers:
        return {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        res = list(ex.map(lambda tk: (tk, fetch_minute_bars_one(tk, dt_str, max_time_et)), tickers))
    return {tk: bd for tk, bd in res if bd is not None}


def fetch_preopen_news_flags(tk, dt_str):
    """Single cached Polygon News call strictly <= 09:30 ET (3-day lookback) used by BOTH Short & Long."""
    key = (tk, dt_str)
    if key in _NEWS_CACHE:
        return _NEWS_CACHE[key]
    t0 = pd.Timestamp(f"{dt_str} 09:30:00", tz="America/New_York")
    pub_lte = t0.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    pub_gte = (t0 - pd.Timedelta(days=3)).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    data = _pg(f"{_BASE}/v2/reference/news",
               {"ticker": tk, "published_utc.lte": pub_lte, "published_utc.gte": pub_gte, "limit": 10}, timeout=10)
    txt = " | ".join(((x.get("title") or "") + " " + (x.get("description") or "")) for x in data.get("results", [])).upper()

    # Short Catalyst Flags (V8.2-S exact)
    is_dil_short = any(w in txt for w in (
        "PUBLIC OFFERING", "DIRECT OFFERING", "CONVERTIBLE NOTE", "WARRANT", "DILUTION", "SHELF REGISTRATION"
    ))
    is_ma = (not is_dil_short) and any(w in txt for w in (
        "TO BE ACQUIRED", "DEFINITIVE AGREEMENT TO ACQUIRE", "BUYOUT", "TAKEOVER",
        "STRATEGIC STAKE", "DIVESTITURE", "SELL EUROPE BUSINESS", "GOING PRIVATE", "MERGER AGREEMENT"
    ))
    is_upg = (not is_dil_short) and any(w in txt for w in (
        "UPGRADES TO BUY", "UPGRADED TO BUY", "UPGRADES TO OUTPERFORM", "UPGRADED TO OUTPERFORM",
        "RAISES PRICE TARGET", "INITIATES WITH BUY", "OVERWEIGHT"
    ))
    short_lbl = "M&A/Buyout" if is_ma else ("Upgrade" if is_upg else ("Dilution" if is_dil_short else "Clean"))

    # Long Catalyst Flags (V8.2-L exact)
    is_bear_long = any(w in txt for w in (
        "PUBLIC OFFERING", "DIRECT OFFERING", "PRIVATE PLACEMENT", "WARRANT", "DILUTION",
        "SHELF REGISTRATION", "MISSES ESTIMATES", "CUTS GUIDANCE", "LOWERS GUIDANCE"
    ))
    is_bull_long = (not is_bear_long) and any(w in txt for w in (
        "BEATS ESTIMATES", "RAISES GUIDANCE", "UPGRADE", "PRICE TARGET", "ACQUISITION", "BUYOUT",
        "FDA APPROVAL", "PHASE 2", "PHASE 3", "CONTRACT", "AWARD", "PARTNERSHIP"
    ))
    long_lbl = "BEARISH_DILUTION_OR_MISS" if is_bear_long else ("BULLISH_CATALYST" if is_bull_long else "NEUTRAL_OR_TECH")

    out = {
        "is_ma": is_ma, "is_upg": is_upg, "is_dil": is_dil_short, "short_lbl": short_lbl,
        "is_bear_long": is_bear_long, "is_bull_long": is_bull_long, "long_lbl": long_lbl,
    }
    _NEWS_CACHE[key] = out
    return out


def fetch_tq_metrics(tk, dt_str, start_et="09:30", cutoff_et="09:44"):
    """Unified Polygon v3 Trades & Quotes Lee-Ready Order Flow metrics (cached per window)."""
    key = (tk, dt_str, start_et, cutoff_et)
    if key in _TQ_CACHE:
        return _TQ_CACHE[key]
    ts_start = int(pd.Timestamp(f"{dt_str} {start_et}:00", tz="America/New_York").timestamp() * 1e9)
    ts_end   = int(pd.Timestamp(f"{dt_str} {cutoff_et}:59", tz="America/New_York").timestamp() * 1e9)
    p = {"timestamp.gte": ts_start, "timestamp.lte": ts_end, "limit": 50000, "sort": "timestamp", "order": "asc"}
    trades = _pg(f"{_BASE}/v3/trades/{tk}", p, timeout=12).get("results", [])
    quotes = _pg(f"{_BASE}/v3/quotes/{tk}", p, timeout=12).get("results", [])

    if len(trades) < 10:
        res = {"lr_ofi": 0.0, "lr_buy_ratio": 0.5, "block_ofi": 0.0, "odd_ofi": 0.0,
               "ask_step_dn": 0.0, "msyn": 0.0, "p_long": 0.0, "p_cont": 0.0, "n_trades": len(trades)}
        _TQ_CACHE[key] = res
        return res

    t_ts = np.array([int(x.get("sip_timestamp") or x.get("participant_timestamp") or 0) for x in trades], dtype=np.int64)
    t_px = np.array([float(x.get("price", 0.0)) for x in trades], dtype=float)
    t_sz = np.array([float(x.get("size", 0.0)) for x in trades], dtype=float)
    signs = np.sign(np.diff(t_px, prepend=t_px[0]))
    ask_step_dn = 0.0

    if len(quotes) >= 5:
        q_ts  = np.array([int(x.get("sip_timestamp") or x.get("participant_timestamp") or 0) for x in quotes], dtype=np.int64)
        q_bid = np.array([float(x.get("bid_price", 0.0)) for x in quotes], dtype=float)
        q_ask = np.array([float(x.get("ask_price", 0.0)) for x in quotes], dtype=float)
        vq = (q_bid > 0) & (q_ask > 0) & (q_ask >= q_bid)
        if np.any(vq):
            q_ts, q_bid, q_ask = q_ts[vq], q_bid[vq], q_ask[vq]
            q_mid = 0.5 * (q_bid + q_ask)
            mid_at_t = q_mid[np.clip(np.searchsorted(q_ts, t_ts, side="right") - 1, 0, len(q_mid) - 1)]
            signs = np.where(t_px > mid_at_t, 1.0, np.where(t_px < mid_at_t, -1.0, 0.0))
            ad = np.diff(q_ask)
            ask_step_dn = float((np.sum(ad < -1e-4) - np.sum(ad > 1e-4)) / max(len(q_ask), 1))

    for i in range(1, len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i - 1] if signs[i - 1] != 0 else 1.0

    tot_vol    = max(float(np.sum(t_sz)), 1.0)
    signed_vol = t_sz * signs
    lr_ofi     = float(np.sum(signed_vol) / tot_vol)
    buy_ratio  = float(np.sum(t_sz[signs > 0]) / tot_vol)
    odd_m      = t_sz < 100
    blk_m      = (t_sz >= 1000) | ((t_sz * t_px) >= 20000.0)
    odd_ofi    = float(np.sum(signed_vol[odd_m]) / max(float(np.sum(t_sz[odd_m])), 1.0)) if np.any(odd_m) else 0.0
    blk_ofi    = float(np.sum(signed_vol[blk_m]) / max(float(np.sum(t_sz[blk_m])), 1.0)) if np.any(blk_m) else 0.0

    msyn   = float(np.clip(-1.25 * lr_ofi - 0.85 * blk_ofi + 0.65 * odd_ofi + 1.10 * ask_step_dn, -2.5, 2.5))
    p_cont = float(np.clip(-1.10 * lr_ofi - 0.90 * blk_ofi + 0.80 * ask_step_dn, -2.5, 2.5))
    p_long = float(np.clip( 1.30 * lr_ofi + 0.95 * blk_ofi - 0.50 * odd_ofi - 0.85 * ask_step_dn + 0.25, -2.5, 2.5))

    res = {
        "lr_ofi": lr_ofi, "lr_buy_ratio": buy_ratio, "block_ofi": blk_ofi, "odd_ofi": odd_ofi,
        "ask_step_dn": ask_step_dn, "msyn": msyn, "p_long": p_long, "p_cont": p_cont, "n_trades": len(trades),
    }
    _TQ_CACHE[key] = res
    return res


def _load_ibkr_short_feed():
    global _IBKR_CACHE
    if _IBKR_CACHE is not None:
        return _IBKR_CACHE
    imap = {}
    try:
        raw = urllib.request.urlopen("ftp://shortstock:@ftp2.interactivebrokers.com/usa.txt", timeout=12).read().decode("utf-8", errors="ignore")
        for ln in raw.splitlines():
            if not ln or ln.startswith("#"):
                continue
            p = ln.split("|")
            if len(p) < 8:
                continue
            sym = p[0].strip().upper()
            fee = float(p[6].strip()) if p[6].strip() and p[6].strip() != "NA" else 99.0
            avs = p[7].strip().replace(">", "").replace("<", "")
            av  = int(float(avs)) if avs and avs != "NA" else 0
            imap[sym] = {"fee_rate": fee, "avail_shares": av}
    except Exception:
        pass
    _IBKR_CACHE = imap
    return imap


def evaluate_short_locate(ticker, open_price, prior_dvol, gap_pct_dec, htb_blacklist):
    btk = str(ticker).split("_")[0].strip().upper()
    if htb_blacklist and btk in htb_blacklist:
        return {"is_etb": False, "badge": "HTB-MANUAL", "reason": "Manual HTB blacklist"}
    imap = _load_ibkr_short_feed()
    info = imap.get(btk)
    if info is not None:
        av, fee = int(info["avail_shares"]), float(info["fee_rate"])
        av_s = f"{av//1000}K" if av < 1_000_000 else f"{av/1_000_000:.1f}M"
        if av < MIN_ETB_SHARES or fee > MAX_BORROW_FEE_PCT:
            return {"is_etb": False, "badge": f"HTB({fee:.0f}%/{av_s})", "reason": "IBKR fee/avail fail"}
        if gap_pct_dec >= 0.35 and (prior_dvol or 0) < 10e6 and av < 20000:
            return {"is_etb": False, "badge": f"HTB-SQZ({av_s})", "reason": "Parabolic low-avail"}
        return {"is_etb": True, "badge": f"ETB({fee:.1f}%/{av_s})", "reason": "IBKR FTP verified"}
    if (prior_dvol or 0) >= 20e6 and (open_price or 0) >= 15.0:
        return {"is_etb": True, "badge": "ETB-LIQ", "reason": "Structural $20M/$15 fallback"}
    return {"is_etb": False, "badge": "HTB-UNLISTED", "reason": "Missing from IBKR & <$20M ADV"}


# ═══════════════════════════════════════════════════════════════════════════════
# SHORT ENGINE HELPERS (V8.2-S Exact)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_price_at_T(bar_data, T):
    if bar_data is None:
        return None
    best = None
    for t_str, bd in (bar_data.get("pm") or []) + (bar_data.get("rth") or []):
        if t_str <= T:
            best = bd["c"]
        else:
            break
    return best


def _is_tier1_short(c):
    gap = c.get("gap_pct", 0.0)
    if gap >= 0.30:
        return True
    if gap < 0.15:
        op = c.get("open_price", 0.0)
        dv = c.get("prior_dvol")
        if op <= TIER1_MOD_GAP_MAX_PRICE and (dv is None or dv < TIER1_MOD_GAP_MAX_PRIOR_DVOL):
            return True
    return False


def _find_short_entry(rth_bars, open_price, gap_pct, ref_price):
    if not rth_bars or len(rth_bars) < 4:
        return None
    # 1. Bar 0 Opening Flush (09:30 red-bar peak confirmed at 09:33 ET bar 3)
    b0 = rth_bars[0][1]
    h0 = b0["h"]
    if b0["c"] <= b0["o"] and all(rth_bars[j][1]["h"] < h0 for j in range(1, 4)):
        ci, ep, et = 3, rth_bars[3][1]["c"], rth_bars[3][0]
        pb_pct = (h0 - ep) / h0 * 100.0 if h0 > 0 else 0.0
        drop_open = (open_price - ep) / open_price * 100.0 if open_price > 0 else 0.0
        gap_rng = (open_price - ref_price) if (open_price and ref_price and open_price > ref_price) else 0.0
        gap_ret = (ep - ref_price) / gap_rng if gap_rng > 0 else 0.0
        if pb_pct >= MIN_PULLBACK_PCT and drop_open >= 1.0 and gap_ret >= 0.35:
            mod_ok = True
            if gap_pct < 0.15:
                clock_idx = (int(et[:2]) - 9) * 60 + (int(et[3:5]) - 30)
                if (clock_idx - ci) > MAX_MISSING_BARS_MOD_GAP:
                    mod_ok = False
                if open_price > 0 and (ep - open_price) / open_price >= MAX_OPEN_RUNUP_MOD_GAP:
                    mod_ok = False
            if mod_ok:
                return {
                    "entry_price": ep, "entry_time": et, "entry_bar_idx": ci,
                    "entry_type": "bar0_flush", "peak_bar": 0, "peak_time": rth_bars[0][0],
                    "peak_high": h0, "pullback_pct": round(pb_pct, 2),
                }

    # 2. 3-Bar Local High (peak bars 3-11, confirmed peak+3 <= 09:44 ET)
    if len(rth_bars) < PEAK_WINDOW * 2 + 1:
        return None
    for i in range(PEAK_WINDOW, min(MAX_PEAK_BAR, len(rth_bars) - PEAK_WINDOW)):
        if rth_bars[i][0] > MAX_PEAK_TIME_ET:
            break
        ch = rth_bars[i][1]["h"]
        if all(rth_bars[i - j][1]["h"] < ch and rth_bars[i + j][1]["h"] < ch for j in range(1, PEAK_WINDOW + 1)):
            ci = i + PEAK_WINDOW
            if ci < len(rth_bars) and rth_bars[ci][0] <= MAX_ENTRY_TIME_ET:
                ep, et = rth_bars[ci][1]["c"], rth_bars[ci][0]
                pb_pct = (ch - ep) / ch * 100.0 if ch > 0 else 0.0
                if pb_pct < MIN_PULLBACK_PCT:
                    continue
                if gap_pct < 0.15:
                    clock_idx = (int(et[:2]) - 9) * 60 + (int(et[3:5]) - 30)
                    if (clock_idx - ci) > MAX_MISSING_BARS_MOD_GAP:
                        return None
                    if open_price > 0 and (ep - open_price) / open_price >= MAX_OPEN_RUNUP_MOD_GAP:
                        return None
                return {
                    "entry_price": ep, "entry_time": et, "entry_bar_idx": ci,
                    "entry_type": "local_high", "peak_bar": i, "peak_time": rth_bars[i][0],
                    "peak_high": ch, "pullback_pct": round(pb_pct, 2),
                }
            return None
    return None


def _get_short_exit_et(open_price, gap_pct, entry_type):
    if entry_type == "local_high":
        if open_price >= 50.0 or gap_pct < 0.15:
            return EXIT_TIME_INST_ET
        return EXIT_TIME_LH_ET
    if entry_type == "bar0_flush":
        return EXIT_TIME_INST_ET if gap_pct < 0.15 else EXIT_TIME_ET
    return EXIT_TIME_INST_ET if (open_price >= 50.0 or gap_pct < 0.15) else EXIT_TIME_LH_ET


def _short_telemetry(bar_data, entry, exit_et):
    rth = (bar_data or {}).get("rth", [])
    if not entry or not rth or entry["entry_price"] <= 0:
        return None
    ep, et, sp = entry["entry_price"], entry["entry_time"], entry["stop_price"]
    post = [(t, b) for t, b in rth if t > et]
    if not post:
        return {"cur_price": ep, "cur_time": et, "pnl_pct": 0.0, "exited": False, "stopped_out": False, "mfe_pct": 0.0}
    stopped, stop_t, stop_px = False, None, None
    for t, b in post:
        if t > exit_et:
            break
        if b["h"] >= sp:
            stopped, stop_t, stop_px = True, t, round(max(sp, b["o"]), 2)
            break
    if stopped:
        cur_t, cur_p, exited = stop_t, stop_px, True
        sub = [(t, b) for t, b in rth if et < t <= stop_t]
    else:
        b_to_ex = [(t, b) for t, b in rth if et <= t <= exit_et]
        exited = (rth[-1][0] >= exit_et) and bool(b_to_ex)
        cur_t, cur_b = (b_to_ex[-1] if exited else post[-1])
        cur_p = cur_b["c"]
        sub = [(t, b) for t, b in rth if et < t <= exit_et]
    low_p = min((b["l"] for _, b in sub), default=ep)
    return {
        "cur_price": cur_p, "cur_time": cur_t,
        "pnl_pct": round((ep - cur_p) / ep * 100.0, 2),
        "exited": exited, "stopped_out": stopped,
        "mfe_pct": round(max(0.0, (ep - low_p) / ep * 100.0), 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED V8.2 PIPELINE (SHORT + LONG IN ONE FAST PASS)
# ═══════════════════════════════════════════════════════════════════════════════

def run_v82_unified_scanner(
    scan_date_arg=None,
    picky_short=False,
    long_mode="hybrid",
    force_intraday_long=False,
    manual_htb="",
    manual_open_longs="",
):
    t0 = time.time()
    now_pt = datetime.now(PT)
    now_et = now_pt.astimezone(ET)

    # Parse date / simulated PST cutoff timestamp
    sim_cutoff_et = None
    if scan_date_arg:
        parts = str(scan_date_arg).strip().split(" ", 1)
        scan_date = parts[0]
        if len(parts) > 1 and parts[1].strip():
            mode = "historical"
            try:
                dt_pt = PT.localize(datetime.strptime(f"{scan_date} {parts[1].strip()}", "%Y-%m-%d %H:%M"))
                sim_cutoff_et = dt_pt.astimezone(ET).strftime("%H:%M")
            except ValueError:
                sim_cutoff_et = None
        else:
            mode = "live" if scan_date == now_pt.strftime("%Y-%m-%d") else "historical"
    else:
        scan_date = now_et.strftime("%Y-%m-%d")
        mode = "live"

    if not _is_td(scan_date):
        print(f"\n  {scan_date} is not a US market trading day. Nothing to scan.")
        return

    dow = datetime.strptime(scan_date, "%Y-%m-%d").weekday()
    dow_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][dow]
    now_et_hhmm = sim_cutoff_et if sim_cutoff_et is not None else now_et.strftime("%H:%M")
    now_pst_hhmm = _et_to_pst(now_et_hhmm)

    if mode == "historical" and sim_cutoff_et is None:
        scan_phase = "HISTORICAL REPLAY (Full Day)"
    elif now_et_hhmm < "09:30":
        scan_phase = "PRE-MARKET (< 06:30 PST — Preview Only)"
    elif now_et_hhmm < "09:33":
        scan_phase = "MARKET OPEN (06:30-06:32 PST — C1 Longs LIVE | Short Watchlist Locking)"
    elif now_et_hhmm < "09:45":
        scan_phase = "MORNING ENTRY WINDOW (06:33-06:44 PST — Short Triggers + 06:44 Intraday Longs)"
    elif now_et_hhmm <= "14:00":
        scan_phase = "HOLD / EXIT WINDOW (06:45-11:00 PST — Manage Stops & Scheduled Exits)"
    else:
        scan_phase = "POST-WINDOW (> 11:00 PST — Shorts Flat | Manage Longs to 13:00 PST Close)"

    is_pre_market  = (mode == "live" and now_et_hhmm < "09:30") or (sim_cutoff_et is not None and sim_cutoff_et < "09:30")
    is_pre_short   = (mode == "live" and now_et_hhmm < "09:33") or (sim_cutoff_et is not None and sim_cutoff_et < "09:33")
    is_post_short  = (mode == "historical" and sim_cutoff_et is None) or (now_et_hhmm >= "09:45")
    run_intraday_l = (long_mode in ("hybrid", "intraday")) and (
        force_intraday_long or (mode == "historical" and (sim_cutoff_et is None or sim_cutoff_et >= "09:43")) or (mode == "live" and now_et_hhmm >= "09:43")
    )

    htb_set = {s.strip().upper() for s in (manual_htb + "," + os.environ.get("HTB_BLACKLIST_TODAY", "")).split(",") if s.strip()}
    open_l_set = {s.strip().upper() for s in (manual_open_longs + "," + os.environ.get("OPEN_LONG_POSITIONS", "")).split(",") if s.strip()}

    short_ver = "V8.2-S-Picky" if picky_short else "V8.2-S"
    print("\n" + "=" * 106)
    print(f"  AGENT 1 — OFFICIAL {STRATEGY_VERSION} UNIFIED SCANNER (SHORT + LONG)  |  {scan_date} ({dow_name})")
    print(f"  Clock: {now_pt.strftime('%H:%M:%S')} PST ({now_et.strftime('%H:%M:%S')} ET)  |  Mode: {mode.upper()}  |  Short: {short_ver}  |  Long: V8.2-L ({long_mode.upper()})")
    print(f"  Phase: {scan_phase}")
    print("=" * 106)

    # ── STEP 1: SHARED MARKET DATA FETCH (Today + Prior 11 Trading Days in Parallel) ──
    prev_days = _prev_trading_days(scan_date, n_days=11)
    d_m1, d_m3, d_m5 = prev_days[0], prev_days[2], prev_days[4]
    print(f"  [1] Loading market snapshot ({scan_date}) + prior 11 trading days ({prev_days[-1]} -> {d_m1})...")

    with ThreadPoolExecutor(max_workers=12) as ex:
        fut_today = ex.submit(fetch_snapshot_all if mode == "live" else fetch_grouped_daily, *([] if mode == "live" else [scan_date]))
        fut_hist  = {d: ex.submit(fetch_grouped_daily, d) for d in prev_days}
        gd_today  = fut_today.result()
        gd_hist   = {d: f.result() for d, f in fut_hist.items()}

    if mode == "live" and not gd_today:
        gd_today = fetch_grouped_daily(scan_date)

    gd_m1, gd_m3, gd_m5 = gd_hist[d_m1], gd_hist[d_m3], gd_hist[d_m5]
    spy_str = ""
    if "SPY" in gd_m1 and "SPY" in gd_today and gd_m1["SPY"]["c"] > 0:
        spy_gap = (gd_today["SPY"]["o"] - gd_m1["SPY"]["c"]) / gd_m1["SPY"]["c"] * 100.0
        spy_str = f"  |  SPY Gap: {spy_gap:+.2f}%"
    print(f"      Loaded {len(gd_today):,} live tickers & {len(gd_m1):,} prior-day closes{spy_str}")

    # ═══════════════════════════════════════════════════════════════════════════
    # PART A — V8.2-L LONG ENGINE (CARRY GUARD + C1 OPEN SWING + INTRADAY A/B)
    # ═══════════════════════════════════════════════════════════════════════════
    active_open_c1 = {}
    for mtk in sorted(open_l_set):
        cur_p = (gd_today.get(mtk) or gd_m1.get(mtk) or {}).get("c", 0.0)
        active_open_c1[mtk] = {
            "entry_date": "MANUAL", "entry_price": cur_p, "stop_price": round(cur_p * 0.88, 2),
            "exit_date": "MANUAL", "cur_price": cur_p, "pnl_pct": 0.0,
        }

    if long_mode in ("hybrid", "c1") and BLOCK_DUPLICATE_OPEN_C1:
        sim_open = {}  # tk -> exit_date (tracks causal T+3 window across T-5..T-1)
        for k in (5, 4, 3, 2, 1, 0):
            dk = prev_days[k]
            # Remove positions whose scheduled T+3 close was strictly BEFORE dk
            sim_open = {tk: inf for tk, inf in sim_open.items() if inf["exit_date"] >= dk}
            if datetime.strptime(dk, "%Y-%m-%d").weekday() not in (2, 3, 4):
                continue
            gd_k, gd_k1, gd_k3, gd_k5 = gd_hist.get(dk, {}), gd_hist.get(prev_days[k + 1], {}), gd_hist.get(prev_days[k + 3], {}), gd_hist.get(prev_days[k + 5], {})
            if not (gd_k and gd_k1 and gd_k5):
                continue
            hcands = []
            for tk in sorted(FLOAT_10M):
                if tk in sim_open:
                    continue
                bk, bk1, bk5, bk3 = gd_k.get(tk), gd_k1.get(tk), gd_k5.get(tk), gd_k3.get(tk)
                if not (bk and bk1 and bk5):
                    continue
                ok_p, ck1, ck5 = bk["o"], bk1["c"], bk5["c"]
                ck3 = bk3["c"] if bk3 else ck1
                if not (20.0 <= ok_p <= 100.0) or not (20.0 <= ck1 <= 120.0) or ck5 <= 0:
                    continue
                g_k  = (ok_p - ck1) / ck1
                r5_k = (ck1 - ck5) / ck5
                r2_k = (ck1 - ck3) / ck3 if ck3 > 0 else 0.0
                if 0.03 <= g_k <= 0.50 and r5_k <= -0.15 and r2_k <= 0.05:
                    if fetch_preopen_news_flags(tk, dk)["is_bear_long"]:
                        continue
                    hcands.append((abs(r5_k), tk, ok_p))
            hcands.sort(reverse=True)
            for _, tk, ok_p in hcands[:3]:
                stop_p = round(ok_p * 0.88, 2)
                ex_dt_k = _next_trading_day(dk, 3)
                stopped = False
                for chk_i in range(k, -1, -1):
                    chk_d = prev_days[chk_i]
                    if chk_d > ex_dt_k:
                        break
                    bc = gd_hist.get(chk_d, {}).get(tk)
                    if bc and ((chk_i < k and bc["o"] <= stop_p) or bc["l"] <= stop_p):
                        stopped = True
                        break
                if not stopped:
                    cur_p = (gd_today.get(tk) or gd_m1.get(tk) or {}).get("c", ok_p)
                    sim_open[tk] = {
                        "entry_date": dk, "entry_price": ok_p, "stop_price": stop_p,
                        "exit_date": ex_dt_k, "cur_price": cur_p,
                        "pnl_pct": (cur_p - ok_p) / ok_p * 100.0 if ok_p > 0 else 0.0,
                    }
        for tk, inf in sim_open.items():
            if inf["exit_date"] >= scan_date and tk not in active_open_c1:
                active_open_c1[tk] = inf

    # Engine C1 Today (Wed/Thu/Fri)
    c1_picks, c1_dup_skipped = [], []
    if long_mode in ("hybrid", "c1") and dow in (2, 3, 4):
        c1_cands = []
        for tk in sorted(FLOAT_10M):
            b_t, b_m1, b_m3, b_m5 = gd_today.get(tk), gd_m1.get(tk), gd_m3.get(tk), gd_m5.get(tk)
            if not (b_t and b_m1 and b_m5):
                continue
            o_t, c_m1, c_m5 = b_t["o"], b_m1["c"], b_m5["c"]
            c_m3 = b_m3["c"] if b_m3 else c_m1
            if not (20.0 <= o_t <= 100.0) or not (20.0 <= c_m1 <= 120.0) or c_m5 <= 0:
                continue
            gap    = (o_t - c_m1) / c_m1
            ret_5d = (c_m1 - c_m5) / c_m5
            ret_2d = (c_m1 - c_m3) / c_m3 if c_m3 > 0 else 0.0
            if 0.03 <= gap <= 0.50 and ret_5d <= -0.15 and ret_2d <= 0.05:
                if BLOCK_DUPLICATE_OPEN_C1 and tk in active_open_c1:
                    inf = active_open_c1[tk]
                    c1_dup_skipped.append(f"{tk}(open {inf['entry_date']}->{inf['exit_date']})")
                    continue
                nf = fetch_preopen_news_flags(tk, scan_date)
                if nf["is_bear_long"]:
                    continue
                ex_dt = _next_trading_day(scan_date, 3)
                c1_cands.append({
                    "ticker": tk, "engine": "C1_CRASH_REV_T3" if dow in (2, 3) else "C1_FRI_CAT_T3",
                    "open_price": o_t, "entry_price": o_t,
                    "entry_time_pst": "06:30 PST Open" + (" (PREVIEW)" if is_pre_market else ""),
                    "stop_price": round(o_t * 0.88, 2), "stop_pct": -12.0,
                    "exit_target": f"T+3 Close ({ex_dt} 13:00 PST)",
                    "gap_pct": gap * 100.0, "ret_5d_pct": ret_5d * 100.0,
                    "catalyst": nf["long_lbl"], "score": abs(ret_5d) * 100.0,
                    "cur_price": b_t["c"], "live_pnl_pct": (b_t["c"] - o_t) / o_t * 100.0,
                    "status_str": "PRE-MARKET PREVIEW (Confirm 06:30:15 PST)" if is_pre_market else "ACTIVE SWING",
                    "is_preview": is_pre_market,
                })
        c1_cands.sort(key=lambda x: -x["score"])
        c1_picks = c1_cands[:3]

    # Intraday LONG Engines B & A (06:44 PST)
    intraday_long_picks, long_evaluated, morn_short_cnt = [], [], 0
    if run_intraday_l:
        print(f"  [2] Scanning V8.2-L Intraday T&Q LONGs (Engine B 10-30% Cont. + Engine A $25+/$25M+ Inst.)...")
        lg_cands = []
        for tk, b_t in gd_today.items():
            if tk in _EXCLUDE_TK or not _TK_RE.match(tk) or tk.endswith(_EXCLUDE_SUFFIX):
                continue
            b_m1 = gd_m1.get(tk)
            if not b_m1 or b_m1["c"] <= 0:
                continue
            o_t = b_t["o"]
            if not (10.0 <= o_t <= 100.0):
                continue
            gap = (o_t - b_m1["c"]) / b_m1["c"]
            dvol = b_m1["c"] * b_m1["v"]
            if 0.03 <= gap <= 0.30 and dvol >= 15_000_000:
                lg_cands.append({
                    "ticker": tk, "open_price": o_t, "prior_close": b_m1["c"],
                    "gap_pct": gap * 100.0, "prior_dvol": dvol,
                })
        lg_cands.sort(key=lambda x: -x["gap_pct"])
        need_ref = [c["ticker"] for c in lg_cands[:150] if c["ticker"] not in _FLOAT]
        fetch_ticker_refs_parallel(need_ref)
        verified_lg = []
        for c in lg_cands[:150]:
            tk = c["ticker"]
            if tk in _FLOAT:
                verified_lg.append(c)
            else:
                rf = _REF_CACHE.get(tk, {})
                if rf.get("type") in ("CS", "ADRC") and not any(w in (rf.get("name") or "").upper() for w in _ETF_BAD_WORDS):
                    verified_lg.append(c)
        top_lg = verified_lg[:25]

        def _eval_long(c):
            tk = c["ticker"]
            bd = fetch_minute_bars_one(tk, scan_date, max_time_et=sim_cutoff_et)
            rth = (bd or {}).get("rth", [])
            morn = [(t, b) for t, b in rth if t <= "09:44"]
            if len(morn) < 4:
                return None
            ent_t, ent_b = morn[-1]
            ent_px, b0 = ent_b["c"], morn[0][1]
            tq = fetch_tq_metrics(tk, scan_date, start_et="09:30", cutoff_et=ent_t)
            nf = fetch_preopen_news_flags(tk, scan_date)
            c.update({
                "entry_time_et": ent_t, "entry_price": ent_px,
                "is_short_setup": (b0["c"] < b0["o"]) and ((b0["o"] - ent_px) / b0["o"] >= 0.01),
                "tq": tq, "catalyst": nf["long_lbl"], "is_bear_cat": nf["is_bear_long"], "rth": rth,
            })
            return c

        with ThreadPoolExecutor(max_workers=10) as ex:
            long_evaluated = [x for x in ex.map(_eval_long, top_lg) if x is not None]

        morn_short_cnt = sum(1 for x in long_evaluated if x["is_short_setup"] and not (15.0 <= x["gap_pct"] < 30.0))
        is_lg_preview = (mode == "live" and now_et_hhmm < "09:44")
        for c in long_evaluated:
            tq, gap, opx, dvol = c["tq"], c["gap_pct"], c["open_price"], c["prior_dvol"]
            msyn, p_cont, p_long, ofi = tq["msyn"], tq["p_cont"], tq["p_long"], tq["lr_ofi"]
            if c["is_bear_cat"]:
                c["gate_status"] = "REJECT: Bearish/Dilution Catalyst"
                continue
            is_b = (
                10.0 <= gap <= 30.0 and dvol >= 15e6 and msyn <= -0.20 and p_cont <= -0.20
                and (morn_short_cnt <= 1 or (opx >= 25.0 and dvol >= 25e6 and msyn <= -0.55 and p_cont <= -0.55 and ofi >= 0.20))
            )
            is_a = (
                (not is_b) and morn_short_cnt <= 1 and opx >= 25.0 and dvol >= 25e6
                and 3.0 <= gap <= 25.0 and p_long >= 0.30 and msyn <= -0.10 and ofi > 0.05
            )
            if is_b or is_a:
                eng = "TQ_CONT_10_30_B" if is_b else "TQ_INST_ACCUM_A"
                c["gate_status"] = f"PASS -> {eng}"
                sc = (120.0 - msyn - p_cont + 0.5 * ofi) if is_b else (100.0 + p_long - 0.5 * msyn + 0.5 * ofi)
                ep, sp = c["entry_price"], round(c["entry_price"] * 0.945, 2)
                cur_px = c["rth"][-1][1]["c"]
                st_str = "PREVIEW (Confirm 06:44 PST)" if is_lg_preview else "ACTIVE (Hold -> 13:00 PST)"
                for t_s, b in [(t, x) for t, x in c["rth"] if t > c["entry_time_et"]]:
                    if b["l"] <= sp:
                        cur_px, st_str = sp, f"STOPPED (-5.5%) @ {_et_to_pst(t_s)} PST"
                        break
                    if t_s == "10:35" and (b["c"] - ep) / ep <= -0.015:
                        cur_px, st_str = b["c"], f"TIME-STOP @ 07:35 PST (${cur_px:.2f})"
                        break
                intraday_long_picks.append({
                    "ticker": c["ticker"], "engine": eng, "open_price": opx, "entry_price": ep,
                    "entry_time_pst": "06:44 PST" + (" (PREVIEW)" if is_lg_preview else ""),
                    "stop_price": sp, "stop_pct": -5.5,
                    "exit_target": "07:35 PST Check -> 13:00 PST Close",
                    "gap_pct": gap, "prior_dvol_m": dvol / 1e6,
                    "lr_ofi": ofi, "block_ofi": tq["block_ofi"], "msyn": msyn, "p_long": p_long,
                    "catalyst": c["catalyst"], "score": sc, "cur_price": cur_px,
                    "status_str": st_str, "live_pnl_pct": (cur_px - ep) / ep * 100.0,
                    "is_preview": is_lg_preview,
                })
            else:
                if 10.0 <= gap <= 30.0:
                    c["gate_status"] = f"SKIP EngB: Weak T&Q (MSYN={msyn:+.2f})"
                elif morn_short_cnt > 1:
                    c["gate_status"] = f"SKIP EngA: High Short Breadth ({morn_short_cnt}>1)"
                elif opx < 25.0 or dvol < 25e6:
                    c["gate_status"] = "SKIP EngA: <$25 or <$25M ADV"
                else:
                    c["gate_status"] = f"SKIP EngA: T&Q (P_LONG={p_long:+.2f}, OFI={ofi:+.2f})"
        intraday_long_picks.sort(key=lambda x: -x["score"])

    long_slots, seen_l = [], set()
    for p in c1_picks + intraday_long_picks:
        if p["ticker"] not in seen_l and len(long_slots) < 3:
            seen_l.add(p["ticker"])
            long_slots.append(p)
    long_slot_pct = round(min(1.0 / len(long_slots), 0.50) * 100.0, 1) if long_slots else 0.0

    # ═══════════════════════════════════════════════════════════════════════════
    # PART B — V8.2-S SHORT ENGINE (ETB + CATALYST VETO + 15-30% T&Q + -10% STOP)
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"  [3] Scanning {short_ver} SHORT gap-faders (Gap >= {SHORT_GAP_MIN_PCT}%, $10-$100, ETB + Catalyst-Clean)...")
    gap_min = SHORT_GAP_MIN_PCT / 100.0
    raw_short_cands = []
    for tk, cur_d in gd_today.items():
        if tk in _EXCLUDE_TK or (any(tk.endswith(s) for s in _EXCLUDE_SUFFIX) and len(tk) > 2):
            continue
        pm1 = gd_m1.get(tk)
        if not pm1 or pm1["c"] <= 0:
            continue
        cur_p = cur_d.get("o", cur_d.get("price", 0.0))
        if cur_p < SHORT_PRICE_MIN:
            continue
        gap = (cur_p - pm1["c"]) / pm1["c"]
        if gap < gap_min:
            continue
        fs = _FLOAT.get(tk)
        if fs is not None and fs < FLOAT_MIN:
            continue
        raw_short_cands.append({
            "ticker": tk, "gap_pct": gap, "raw_open_gap": gap,
            "prior_close": pm1["c"], "prior_dvol": pm1["c"] * pm1["v"],
            "open_price": cur_p, "float_shares": fs,
        })
    raw_short_cands.sort(key=lambda x: -x["gap_pct"])

    # Select top PRE_FILTER_N (30) Common Stocks (CS, no SPACs)
    cs_cands = []
    for i in range(0, len(raw_short_cands), 40):
        batch = raw_short_cands[i:i + 40]
        refs = fetch_ticker_refs_parallel([c["ticker"] for c in batch])
        for c in batch:
            rf = refs.get(c["ticker"], {})
            at, nm = rf.get("type", "UNK"), rf.get("name", "")
            c["asset_type"], c["name"] = at, nm
            if at != "CS" or any(w in nm.upper() for w in _SPAC_KEYWORDS):
                continue
            cs_cands.append(c)
            if len(cs_cands) >= PRE_FILTER_N:
                break
        if len(cs_cands) >= PRE_FILTER_N:
            break

    top_short_tks = [c["ticker"] for c in cs_cands]
    prior_bars = fetch_minute_bars_parallel(top_short_tks, d_m1)
    scan_bars  = fetch_minute_bars_parallel(top_short_tks, scan_date, max_time_et=sim_cutoff_et)

    qualified_s, n_stale, n_eroded = [], 0, 0
    for c in cs_cands:
        tk = c["ticker"]
        sb, pb = scan_bars.get(tk), prior_bars.get(tk)
        if sb and sb.get("rth"):
            c["open_price"] = sb["rth"][0][1]["o"]
            if c["prior_close"] > 0:
                c["raw_open_gap"] = (c["open_price"] - c["prior_close"]) / c["prior_close"]
        if pb and pb.get("ah"):
            ah_ref = pb["ah"][-1][1]["c"]
            c["ah_ref_price"] = ah_ref
            fresh_gap = (c["open_price"] - ah_ref) / ah_ref
            c["gap_pct"] = fresh_gap
            if fresh_gap < gap_min:
                n_stale += 1
                continue
        if sb:
            c["bar_data"] = sb
            price_T = _get_price_at_T(sb, SCAN_TIME_ET)
            if price_T is not None and not is_pre_market:
                ref_p = c.get("ah_ref_price", c["prior_close"])
                gap_T = (price_T - ref_p) / ref_p
                if gap_T < gap_min or price_T < SHORT_PRICE_MIN:
                    continue
                c["gap_pct"], c["scan_price"] = gap_T, price_T
            rth = sb.get("rth", [])
            if rth:
                daily_open = rth[0][1]["o"]
                ref_p = c.get("ah_ref_price", c["prior_close"])
                scan_rth = [b for b in rth if b[0] <= SCAN_TIME_ET]
                gap_rng = daily_open - ref_p
                if scan_rth and gap_rng > 0:
                    gap_cons = (daily_open - min(b[1]["l"] for b in scan_rth)) / gap_rng
                    c["gap_consumed"] = gap_cons
                    if gap_cons > GAP_CONSUMED_MAX:
                        n_eroded += 1
                        continue
        qualified_s.append(c)
    qualified_s.sort(key=lambda x: -x["gap_pct"])
    cs_pool = qualified_s

    # Step 5b: RTH Liquidity Filter (runs once RTH has >=2 bars; skipped only pre-06:32 PST)
    n_illiq = 0
    if not is_pre_short:
        liq_pool = []
        for c in cs_pool:
            bd = c.get("bar_data") or scan_bars.get(c["ticker"])
            if bd:
                rth = bd.get("rth", [])
                scan_rth = [(t, b) for t, b in rth if t <= SCAN_TIME_ET]
                bars_v = sum(1 for _, b in scan_rth if b["v"] > 0)
                b0_v = rth[0][1]["v"] if rth else 0
                if bars_v < MIN_RTH_BARS or b0_v < MIN_OPEN_VOL:
                    n_illiq += 1
                    continue
            liq_pool.append(c)
        cs_pool = liq_pool

    # Step 5c: Live IBKR ETB Locate Filter (ALWAYS runs — even in pre-market!)
    etb_pool, htb_excluded = [], []
    for c in cs_pool:
        loc = evaluate_short_locate(c["ticker"], c["open_price"], c.get("prior_dvol", 0), c["gap_pct"], htb_set)
        c["locate_badge"] = loc["badge"]
        if not loc["is_etb"]:
            htb_excluded.append(f"{c['ticker']}({loc['badge']})")
            continue
        etb_pool.append(c)
    cs_pool = etb_pool

    # Step 5d: Pre-Open News / Catalyst Veto (<= 09:30 ET, ALWAYS runs — even in pre-market!)
    cat_pool, cat_excluded = [], []
    if cs_pool:
        with ThreadPoolExecutor(max_workers=12) as ex:
            nflags = list(ex.map(lambda c: fetch_preopen_news_flags(c["ticker"], scan_date), cs_pool))
        for c, nf in zip(cs_pool, nflags):
            c["catalyst_badge"] = nf["short_lbl"]
            if nf["is_ma"] or (c["gap_pct"] < 0.15 and nf["is_upg"]):
                cat_excluded.append(f"{c['ticker']}({nf['short_lbl']})")
                continue
            cat_pool.append(c)
        cs_pool = cat_pool

    # Priority Tier Ranking (Tier 1: >=30% Parabolic & <15% Retail <=$25/<$25M before Tier 2)
    cs_pool = sorted(cs_pool, key=lambda c: (1 if _is_tier1_short(c) else 0, c["gap_pct"]), reverse=True)

    # Top 3 V8.2-S Watchlist & Trigger Confirmation
    short_picks = []
    for c in cs_pool:
        if not (SHORT_PRICE_MIN <= c["open_price"] <= SHORT_PRICE_MAX):
            continue
        tk = c["ticker"]
        bd = c.get("bar_data") or scan_bars.get(tk)
        rth = (bd or {}).get("rth", [])
        ref_p = c.get("ah_ref_price", c["prior_close"])
        entry = _find_short_entry(rth, c["open_price"], c["gap_pct"], ref_p) if len(rth) >= 4 else None
        tq_rej = None
        gap_v, raw_g = c["gap_pct"], c.get("raw_open_gap", c["gap_pct"])
        is_15_30 = (0.15 <= gap_v < 0.30) or (0.15 <= raw_g < 0.30)
        if entry is not None:
            tq_start = entry["peak_time"] if entry["entry_type"] == "local_high" else "09:30"
            need_tq = is_15_30 or (picky_short and gap_v < 0.15)
            if need_tq:
                tqi = fetch_tq_metrics(tk, scan_date, start_et=tq_start, cutoff_et=entry["entry_time"])
                entry["tq_msyn"], entry["tq_blk_ofi"], entry["tq_lr_ofi"] = tqi["msyn"], tqi["block_ofi"], tqi["lr_ofi"]
                if is_15_30 and not (tqi["msyn"] >= TQ_15_30_MIN_MSYN and tqi["block_ofi"] <= TQ_15_30_MAX_BLK_OFI):
                    tq_rej = f"15-30% T&Q Gate [{_et_to_pst(tq_start)}->{_et_to_pst(entry['entry_time'])} PST] (MSYN={tqi['msyn']:+.2f}, BlkOFI={tqi['block_ofi']:+.2f})"
                    entry = None
                elif picky_short and gap_v < 0.15:
                    if entry["entry_type"] == "local_high" and tqi["msyn"] < 0.0:
                        tq_rej = f"Picky <15% Bullish Pullback (MSYN={tqi['msyn']:+.2f})"
                        entry = None
                    elif entry["entry_type"] == "bar0_flush" and tqi["msyn"] < -0.50:
                        tq_rej = f"Picky <15% Bullish Absorption (MSYN={tqi['msyn']:+.2f})"
                        entry = None
        if entry is not None:
            entry["stop_price"] = round(entry["entry_price"] * (1.0 + SHORT_HARD_STOP_PCT), 2)
        ex_et = _get_short_exit_et(c["open_price"], gap_v, entry["entry_type"] if entry else None)
        short_picks.append({
            "ticker": tk, "gap_pct": gap_v * 100.0, "open_price": c["open_price"],
            "prior_close": c["prior_close"], "prior_dvol": c.get("prior_dvol") or 0.0,
            "tier": "T1" if _is_tier1_short(c) else "T2",
            "locate_badge": c.get("locate_badge", "ETB"),
            "catalyst_badge": c.get("catalyst_badge", "Clean"),
            "is_15_30": is_15_30,
            "entry": entry, "tq_reject": tq_rej,
            "exit_et": ex_et, "exit_pst": _et_to_pst(ex_et),
            "telemetry": _short_telemetry(bd, entry, ex_et) if entry else None,
        })
        if len(short_picks) >= 3:
            break

    confirmed_shorts = [p for p in short_picks if p["entry"] is not None]
    pending_shorts   = [p for p in short_picks if p["entry"] is None]
    short_slot_pct   = round(min(1.0 / len(confirmed_shorts), 0.50) * 100.0, 1) if confirmed_shorts else 0.0
    runtime = time.time() - t0

    # Check for any rare cross-side ticker overlap
    overlap_tks = ({p["ticker"] for p in confirmed_shorts} | {p["ticker"] for p in short_picks}) & ({p["ticker"] for p in long_slots} | set(active_open_c1.keys()))

    # ═══════════════════════════════════════════════════════════════════════════
    # EXECUTIVE UNIFIED DASHBOARD — ALL SHORT & LONG ACTIONS ON ONE SCREEN
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"      Short Funnel: {len(cs_cands)} CS -> -{n_stale} stale, -{n_eroded} eroded, -{n_illiq} illiq, -{len(htb_excluded)} HTB, -{len(cat_excluded)} catalyst -> {len(short_picks)}/3 Top Watchlist  ({runtime:.1f}s)")
    if htb_excluded:
        print(f"      HTB Blocked : {', '.join(htb_excluded[:8])}")
    if cat_excluded:
        print(f"      News Vetoed : {', '.join(cat_excluded[:8])}")
    if overlap_tks:
        print(f"\n  [!] CROSS-SIDE OVERLAP DETECTED: {', '.join(sorted(overlap_tks))} — review before executing both sides!")

    board_clk = "EOD" if (mode == "historical" and sim_cutoff_et is None) else f"{now_pst_hhmm} PST"
    print("\n" + "█" * 106)
    print(f"  UNIFIED V8.2 PORTFOLIO ACTION BOARD — {scan_date} ({dow_name}) @ {board_clk}")
    print(f"  Allocation: 50% SHORT Sleeve ({short_ver}, Max 3)  +  50% LONG Sleeve (V8.2-L, Max 3)")
    print("█" * 106)

    # ── 1. CARRYING OVER OPEN LONG SWINGS ──
    print("\n  [1] OPEN LONG POSITIONS CARRYING INTO TODAY (V8.2-L T+3 Swings — Duplicate Re-Buy Blocked):")
    if not active_open_c1:
        print("      • None carrying over.")
    else:
        for tk, inf in sorted(active_open_c1.items()):
            ex_tag = ">>> SELL AT TODAY'S 13:00 PST CLOSE! <<<" if inf["exit_date"] == scan_date else f"Hold -> Sell {inf['exit_date']} 13:00 PST Close"
            print(f"      • HOLD LONG {tk:<6} | Opened {inf['entry_date']} @ ${inf['entry_price']:.2f} | GTC Stop ${inf['stop_price']:.2f} (-12%) | Mark ${inf['cur_price']:.2f} ({inf['pnl_pct']:+.2f}%) | {ex_tag}")

    # ── 2. TODAY'S LONG SIGNALS ──
    print(f"\n  [2] TODAY'S `LONG` CANDIDATES & ORDERS (V8.2-L  |  {len(long_slots)}/3 Slots  |  Sleeve Size: {long_slot_pct:.1f}% each):")
    if long_slots:
        for i, p in enumerate(long_slots, 1):
            action = "PREVIEW BUY" if p.get("is_preview") else "BUY LONG   "
            sig = f"5dCrash={p['ret_5d_pct']:+.1f}%" if "ret_5d_pct" in p else f"OFI={p['lr_ofi']:+.2f}, MSYN={p['msyn']:+.2f}, P_LONG={p['p_long']:+.2f}"
            print(
                f"      #{i} {action} {p['ticker']:<6} [{p['engine']:<15}] | Size={long_slot_pct:.1f}% of LONG | "
                f"Gap=+{p['gap_pct']:.1f}% | Entry=${p['entry_price']:.2f} @ {p['entry_time_pst']}"
            )
            print(
                f"         Stop=${p['stop_price']:.2f} ({p['stop_pct']:.1f}%) | Exit={p['exit_target']} | "
                f"{sig} | Cat={p['catalyst']} | {p['status_str']} (Mark ${p['cur_price']:.2f}, {p['live_pnl_pct']:+.2f}%)"
            )
    else:
        c1_note = "C1 Standby (trades Wed/Thu/Fri only)" if dow not in (2, 3, 4) else "0 new C1 crash-reversal picks"
        dup_note = f" | Dup-Guard blocked: {', '.join(c1_dup_skipped)}" if c1_dup_skipped else ""
        int_note = "Intraday A/B Standby until 06:44 PST" if (long_mode in ("hybrid", "intraday") and not run_intraday_l) else f"0 Intraday A/B triggers ({len(long_evaluated)} gappers checked, ShortBreadth={morn_short_cnt})"
        print(f"      • No new LONG orders right now. ({c1_note}{dup_note} | {int_note})")
    if long_slots and long_mode in ("hybrid", "intraday") and not run_intraday_l:
        print(f"      • Note: Intraday LONGs (Engine A/B) auto-activate at 06:43–06:44 PST — re-run cell at 06:44:10 PST.")
    if c1_dup_skipped and long_slots:
        print(f"      • Duplicate-Guard Blocked: {', '.join(c1_dup_skipped)}")

    # ── 3. TODAY'S SHORT SIGNALS & WATCHLIST ──
    print(f"\n  [3] TODAY'S `SHORT` CANDIDATES & ORDERS ({short_ver}  |  {len(confirmed_shorts)}/3 Confirmed  |  Sleeve Size: {short_slot_pct:.1f}% each):")
    if not short_picks:
        print("      • Zero stocks passed Short ETB + Catalyst + Gap ($10-$100) filters today.")
    else:
        for i, p in enumerate(short_picks, 1):
            e = p["entry"]
            tel = p["telemetry"]
            g15_tag = " [15-30% T&Q]" if p["is_15_30"] else ""
            if e is not None:
                ent_pst = _et_to_pst(e["entry_time"])
                tq_s = f" | T&Q MSYN={e['tq_msyn']:+.2f}, BlkOFI={e['tq_blk_ofi']:+.2f}" if "tq_msyn" in e else ""
                if tel and tel["stopped_out"]:
                    tag = "STOPPED    "
                    st_s = f"STOPPED OUT @ {_et_to_pst(tel['cur_time'])} PST (${tel['cur_price']:.2f}, {tel['pnl_pct']:+.2f}%) — DO NOT ENTER LATE"
                elif tel and tel["exited"]:
                    tag = "COVERED    "
                    st_s = f"COVERED @ {_et_to_pst(tel['cur_time'])} PST (${tel['cur_price']:.2f}, {tel['pnl_pct']:+.2f}% | MFE +{tel['mfe_pct']:.1f}%)"
                else:
                    tag = "SHORT NOW  "
                    st_s = f"LIVE @ {_et_to_pst(tel['cur_time'])} PST (${tel['cur_price']:.2f}, {tel['pnl_pct']:+.2f}% | MFE +{tel['mfe_pct']:.1f}%)" if tel else "ACTIVE"
                print(
                    f"      #{i} {tag} {p['ticker']:<6} [{p['tier']}{g15_tag}] | Size={short_slot_pct:.1f}% of SHORT | "
                    f"Gap=+{p['gap_pct']:.1f}% | Entry=${e['entry_price']:.2f} @ {ent_pst} PST ({e['entry_type']})"
                )
                print(
                    f"         Stop=${e['stop_price']:.2f} (-10%, NO Profit Lock) | Cover={p['exit_pst']} PST | "
                    f"{p['locate_badge']} | Cat={p['catalyst_badge']}{tq_s} | {st_s}"
                )
            else:
                if is_post_short:
                    tag = "SKIPPED    "
                    why = p["tq_reject"] if p["tq_reject"] else "No Bar0 Flush or 3-Bar Local High by 06:44 PST"
                else:
                    tag = "WATCHLIST  "
                    why = (
                        "Wait for 06:33 PST Bar0 Flush or <=06:44 PST 3-Bar Local High"
                        + (" + Bearish T&Q (MSYN>=+0.35, BlkOFI<=-0.15)" if p["is_15_30"] else "")
                    )
                print(
                    f"      #{i} {tag} {p['ticker']:<6} [{p['tier']}{g15_tag}] | Gap=+{p['gap_pct']:.1f}% | Open=${p['open_price']:.2f} | "
                    f"{p['locate_badge']} | Cat={p['catalyst_badge']} | Target Exit={p['exit_pst']} PST"
                )
                print(f"         -> {why}")

    # ── 4. OPTIONAL COMPACT INTRADAY LONG DIAGNOSTICS (Top 8 when evaluated) ──
    if run_intraday_l and long_evaluated:
        print("\n  [4] INTRADAY LONG T&Q DIAGNOSTIC (Top 8 Liquid Gappers | Short Breadth=" + str(morn_short_cnt) + "):")
        print(f"      {'Ticker':<6} | {'Open $':>7} | {'06:44 $':>7} | {'Gap %':>6} | {'Prior$Vol':>9} | {'LR_OFI':>6} | {'MSYN':>6} | {'P_LONG':>6} | Gate Status")
        print("      " + "-" * 96)
        for c in long_evaluated[:8]:
            tq = c["tq"]
            print(
                f"      {c['ticker']:<6} | ${c['open_price']:>6.2f} | ${c['entry_price']:>6.2f} | "
                f"{c['gap_pct']:>+5.1f}% | ${c['prior_dvol']/1e6:>7.1f}M | "
                f"{tq['lr_ofi']:>+6.2f} | {tq['msyn']:>+6.2f} | {tq['p_long']:>+6.2f} | {c.get('gate_status', '')}"
            )

    # ── 5. UNIFIED SINGLE-CELL DAILY CLICK SCHEDULE (PST) ──
    print("\n" + "-" * 106)
    print("  DAILY SINGLE-CELL RUN SCHEDULE (PST) — Click Run on This ONE Cell:")
    print("    • 06:30:15 PST (Wed-Fri) : Check & execute V8.2-L Engine C1 Open LONGs (-12% GTC stop, hold T+3).")
    print("    • 06:31:15 PST (Mon-Fri) : Lock Top 3 ETB + Catalyst-Clean V8.2-S SHORT watchlist.")
    print("    • 06:33-06:44 PST        : Re-run every 1-2 min to catch V8.2-S SHORT triggers (-10% stop, NO _B2 re-shorts).")
    print("    • 06:44:10 PST (Mon-Fri) : Final morning run — catches V8.2-L Intraday T&Q LONGs (Engine A/B, -5.5% stop).")
    print("    • Exits (No re-run)      : Intraday Long check 07:35 PST (-1.5%) | Shorts cover 08:30 / 09:30 / 11:00 PST | Longs 13:00 PST.")
    print("=" * 106 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT (Safely handles BOTH Google Colab single-cell & Local CLI flags)
# ═══════════════════════════════════════════════════════════════════════════════
_cli_date        = SCAN_DATE
_cli_picky       = PICKY_SHORT_MODE
_cli_long_mode   = LONG_MODE
_cli_force_int   = FORCE_INTRADAY_LONG
_cli_htb         = MANUAL_HTB_TODAY
_cli_open_longs  = MANUAL_OPEN_LONGS

for _arg in sys.argv[1:]:
    if _arg.startswith("--date="):
        _cli_date = _arg.split("=", 1)[1].strip()
    elif _arg == "--picky":
        _cli_picky = True
    elif _arg.startswith("--long-mode="):
        _cli_long_mode = _arg.split("=", 1)[1].strip().lower()
    elif _arg == "--force-intraday-long":
        _cli_force_int = True
    elif _arg.startswith("--htb="):
        _cli_htb = (_cli_htb + "," + _arg.split("=", 1)[1].strip()).strip(",")
    elif _arg.startswith("--open="):
        _cli_open_longs = (_cli_open_longs + "," + _arg.split("=", 1)[1].strip()).strip(",")

run_v82_unified_scanner(
    scan_date_arg=_cli_date,
    picky_short=_cli_picky,
    long_mode=_cli_long_mode,
    force_intraday_long=_cli_force_int,
    manual_htb=_cli_htb,
    manual_open_longs=_cli_open_longs,
)
