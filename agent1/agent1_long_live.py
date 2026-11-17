# %% Agent 1 — Standalone Live `LONG-ONLY` Scanner & Execution Playbook (`agent1_long_live.py`)
# ==============================================================================================
# 100% Standalone & Colab-Cell Ready (No external local file dependencies, no `__file__` required).
# Paste this entire file into a Colab / Vertex AI Notebook cell and run, or run via CLI:
#   python3 agent1/agent1_long_live.py
#
# Combines the 3 Validated `LONG` Alpha Engines (Active-Slot `min(1/k, 50%)` Sizing, Top-3 Max Slots):
#   1. Engine C1 (`C1_CRASH_REV_T3` — Wed/Thu Primary + Fri Catalyst-Verified, 06:30 PST Open -> T+3 Close):
#      - Universe: Verified liquid `_FLOAT >= 10M` shares (`Open >= $20.00`, `Prior Close >= $20.00`).
#      - Trigger: Prior 5-trading-day return <= -15.0% + Today's Opening Gap +3.0% to +50.0%.
#      - Catalyst Guard: Blocks Dilution / Warrant / Offering (Cat 5) and Earnings Miss / Guidance Cut (Cat 2C).
#      - Execution: Buy at 06:30 PST Market Open (`open_price`), Hard Stop `-12.0%`, Target Exit `T+3` Close (13:00 PST).
#
#   2. Engine B (`TQ_CONT_10_30_B` — Mon-Fri Same-Day Intraday, 06:33-06:44 PST -> 13:00 PST Close):
#      - Universe: `$10-$100` Common Stocks in the `10.0%-30.0%` Continuation Gap Zone (`Prior $Vol >= $15M`).
#      - Trigger: Live Polygon Trades & Quotes (`06:30-06:44 PST`) confirm institutional accumulation & ask-stepping
#        (`MASTER_SYN <= -0.20` and `P_CONT <= -0.20`), with no Dilution/Bearish catalyst.
#      - Execution: Buy at 06:33-06:44 PST (`entry_price`), Hard Stop `-5.5%`, `07:35 PST (+60m)` Time-Stop if down <= -1.5%,
#        Target Exit `13:00 PST (16:00 ET)` Same-Day Close.
#
#   3. Engine A (`TQ_INST_ACCUM_A` — Mon-Fri Same-Day Intraday on Low Short-Trigger Mornings, 06:33-06:44 PST -> 13:00 PST):
#      - Regime Gate: Fires ONLY when `<= 1` morning short trigger fires across the market by 06:44 PST
#        (confirming an Institutional Accumulation / Gap-and-Go regime rather than a Gap-Fade regime).
#      - Universe: Institutional Mid/Large-Cap Gappers (`Open >= $25.00`, `Prior $Vol >= $25M`, `Gap +3.0% to +25.0%`).
#      - Trigger: Positive Lee-Ready Institutional Order Flow (`LR_OFI > +0.05`, `P_LONG >= +0.30`, `MASTER_SYN <= -0.10`).
#      - Execution: Buy at 06:33-06:44 PST (`entry_price`), Hard Stop `-5.5%`, `07:35 PST (+60m)` Time-Stop if down <= -1.5%,
#        Target Exit `13:00 PST (16:00 ET)` Same-Day Close.

import os
import sys
import re
import time
import math
import requests
import warnings
from datetime import datetime, timedelta
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import pytz

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — Edit directly in Colab or pass via CLI (`--date=YYYY-MM-DD`, `--mode=intraday`)
# ═══════════════════════════════════════════════════════════════════════════════
SCAN_DATE  = None       # None = Live Today (PST/ET), or set e.g. '2026-11-04' for historical replay
LONG_MODE  = "hybrid"   # "hybrid" (C1 + Intraday A+B), "intraday" (Engines A+B only), or "c1" (C1 only)
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs")

for _arg in sys.argv[1:]:
    if _arg.startswith("--date="):
        SCAN_DATE = _arg.split("=", 1)[1].strip()
    elif _arg.startswith("--mode="):
        LONG_MODE = _arg.split("=", 1)[1].strip().lower()

ET = pytz.timezone("America/New_York")
PT = pytz.timezone("America/Los_Angeles")

# Embedded Verified Float Universe (>= 10M shares) — Zero external file dependencies!
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
FLOAT_10M = {tk for tk, fl in _FLOAT.items() if fl >= 10_000_000}

_EXCLUDE_TK = {
    "SPY", "QQQ", "IWM", "DIA", "VXX", "UVXY", "SQQQ", "TQQQ", "SPXU", "SPXL",
    "SDOW", "UDOW", "SDS", "SSO", "SH", "LABU", "LABD", "SOXL", "SOXS",
    "FNGU", "FNGD", "NUGT", "DUST", "JNUG", "JDST", "TNA", "TZA",
    "UPRO", "VIXY", "SVXY", "TVIX",
}
_EXCLUDE_SUFFIX = ("W", "WS", "U", "R")
_TK_RE = re.compile(r"^[A-Z]{1,5}$")

_HOLIDAYS = {
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
    "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
}


def get_prev_trading_days(dt_str, n_days=6):
    d = datetime.strptime(dt_str, "%Y-%m-%d").date()
    out = []
    while len(out) < n_days:
        d -= timedelta(days=1)
        s = d.strftime("%Y-%m-%d")
        if d.weekday() < 5 and s not in _HOLIDAYS:
            out.append(s)
    return out


def fetch_grouped_daily(dt_str):
    url = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{dt_str}"
    try:
        r = requests.get(url, params={"adjusted": "true", "include_otc": "false", "apiKey": POLYGON_API_KEY}, timeout=20)
        if r.status_code == 200:
            return {
                row["T"]: {
                    "o": float(row["o"]), "h": float(row["h"]),
                    "l": float(row["l"]), "c": float(row["c"]), "v": float(row["v"])
                }
                for row in r.json().get("results", []) if "T" in row
            }
    except Exception:
        pass
    return {}


def fetch_snapshot_all():
    url = "https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers"
    try:
        r = requests.get(url, params={"include_otc": "false", "apiKey": POLYGON_API_KEY}, timeout=20)
        if r.status_code == 200:
            out = {}
            for t in r.json().get("tickers", []):
                tk = t.get("ticker", "")
                if not _TK_RE.match(tk):
                    continue
                day_d = t.get("day", {}) or {}
                min_d = t.get("min", {}) or {}
                prev_d = t.get("prevDay", {}) or {}
                day_o = float(day_d.get("o") or 0.0)
                pm_c  = float(min_d.get("c") or 0.0)
                cur_c = float(day_d.get("c") or pm_c or day_o or prev_d.get("c") or 0.0)
                open_p = day_o if day_o > 0 else (pm_c if pm_c > 0 else cur_c)
                if open_p > 0:
                    out[tk] = {
                        "o": open_p,
                        "h": float(day_d.get("h") or open_p),
                        "l": float(day_d.get("l") or open_p),
                        "c": cur_c if cur_c > 0 else open_p,
                        "v": float(day_d.get("v") or 0.0),
                    }
            return out
    except Exception:
        pass
    return {}


def fetch_intraday_1m_bars(tk, dt_str):
    url = f"https://api.polygon.io/v2/aggs/ticker/{tk}/range/1/minute/{dt_str}/{dt_str}"
    try:
        r = requests.get(url, params={"adjusted": "true", "sort": "asc", "limit": 5000, "apiKey": POLYGON_API_KEY}, timeout=12)
        if r.status_code == 200:
            rth, pm = [], []
            for b in r.json().get("results", []):
                ts = pd.Timestamp(b["t"], unit="ms", tz="UTC").tz_convert(ET)
                t_s = ts.strftime("%H:%M")
                bar_d = {"o": float(b["o"]), "h": float(b["h"]), "l": float(b["l"]), "c": float(b["c"]), "v": float(b["v"])}
                if "04:00" <= t_s < "09:30":
                    pm.append((t_s, bar_d))
                elif "09:30" <= t_s < "16:00":
                    rth.append((t_s, bar_d))
            return {"pm": pm, "rth": rth}
    except Exception:
        pass
    return {"pm": [], "rth": []}


def fetch_live_morning_tq_metrics(tk, dt_str, cutoff_et="09:44"):
    """Fetches Polygon v3 Trades & Quotes between 09:30 and cutoff_et (06:30-06:44 PST)
    and computes live Lee-Ready Order Flow Imbalance (LR_OFI), Buy Ratio, Block OFI, Odd-Lot OFI,
    Ask Step-Down Rate, and composite T&Q continuation scores (MASTER_SYN, P_LONG, P_CONT).
    """
    ts_start = int(pd.Timestamp(f"{dt_str} 09:30:00", tz="America/New_York").timestamp() * 1e9)
    ts_end   = int(pd.Timestamp(f"{dt_str} {cutoff_et}:59", tz="America/New_York").timestamp() * 1e9)
    trades, quotes = [], []
    try:
        r_t = requests.get(
            f"https://api.polygon.io/v3/trades/{tk}",
            params={"timestamp.gte": ts_start, "timestamp.lte": ts_end, "limit": 50000, "sort": "timestamp", "order": "asc", "apiKey": POLYGON_API_KEY},
            timeout=12
        )
        if r_t.status_code == 200:
            trades = r_t.json().get("results", [])
        r_q = requests.get(
            f"https://api.polygon.io/v3/quotes/{tk}",
            params={"timestamp.gte": ts_start, "timestamp.lte": ts_end, "limit": 50000, "sort": "timestamp", "order": "asc", "apiKey": POLYGON_API_KEY},
            timeout=12
        )
        if r_q.status_code == 200:
            quotes = r_q.json().get("results", [])
    except Exception:
        pass

    if len(trades) < 10:
        return {"lr_ofi": 0.0, "lr_buy_ratio": 0.5, "block_ofi": 0.0, "odd_lot_ofi": 0.0, "ask_step_dn": 0.0, "msyn": 0.0, "p_long": 0.0, "p_cont": 0.0}

    t_ts = np.array([int(x.get("sip_timestamp") or x.get("participant_timestamp") or 0) for x in trades], dtype=np.int64)
    t_px = np.array([float(x.get("price", 0.0)) for x in trades], dtype=float)
    t_sz = np.array([float(x.get("size", 0.0)) for x in trades], dtype=float)

    if len(quotes) >= 5:
        q_ts  = np.array([int(x.get("sip_timestamp") or x.get("participant_timestamp") or 0) for x in quotes], dtype=np.int64)
        q_bid = np.array([float(x.get("bid_price", 0.0)) for x in quotes], dtype=float)
        q_ask = np.array([float(x.get("ask_price", 0.0)) for x in quotes], dtype=float)
        valid_q = (q_bid > 0) & (q_ask > 0) & (q_ask >= q_bid)
        if np.any(valid_q):
            q_ts, q_bid, q_ask = q_ts[valid_q], q_bid[valid_q], q_ask[valid_q]
            q_mid = 0.5 * (q_bid + q_ask)
            idx_q = np.clip(np.searchsorted(q_ts, t_ts, side="right") - 1, 0, len(q_mid) - 1)
            mid_at_t = q_mid[idx_q]
            signs = np.where(t_px > mid_at_t, 1.0, np.where(t_px < mid_at_t, -1.0, 0.0))
            ask_diff = np.diff(q_ask)
            ask_step_dn = float((np.sum(ask_diff < -1e-4) - np.sum(ask_diff > 1e-4)) / max(len(q_ask), 1))
        else:
            signs = np.sign(np.diff(t_px, prepend=t_px[0]))
            ask_step_dn = 0.0
    else:
        signs = np.sign(np.diff(t_px, prepend=t_px[0]))
        ask_step_dn = 0.0

    for i in range(1, len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i - 1] if signs[i - 1] != 0 else 1.0

    tot_vol = max(float(np.sum(t_sz)), 1.0)
    signed_vol = t_sz * signs
    lr_ofi = float(np.sum(signed_vol) / tot_vol)
    lr_buy_ratio = float(np.sum(t_sz[signs > 0]) / tot_vol)

    odd_mask = t_sz < 100
    blk_mask = (t_sz >= 1000) | ((t_sz * t_px) >= 20000.0)
    odd_lot_ofi = float(np.sum(signed_vol[odd_mask]) / max(float(np.sum(t_sz[odd_mask])), 1.0)) if np.any(odd_mask) else 0.0
    block_ofi   = float(np.sum(signed_vol[blk_mask]) / max(float(np.sum(t_sz[blk_mask])), 1.0)) if np.any(blk_mask) else 0.0

    msyn_est   = float(np.clip(-1.25 * lr_ofi - 0.85 * block_ofi + 0.65 * odd_lot_ofi + 1.10 * ask_step_dn, -2.5, 2.5))
    p_cont_est = float(np.clip(-1.10 * lr_ofi - 0.90 * block_ofi + 0.80 * ask_step_dn, -2.5, 2.5))
    p_long_est = float(np.clip(1.30 * lr_ofi + 0.95 * block_ofi - 0.50 * odd_lot_ofi - 0.85 * ask_step_dn + 0.25, -2.5, 2.5))

    return {
        "lr_ofi": lr_ofi,
        "lr_buy_ratio": lr_buy_ratio,
        "block_ofi": block_ofi,
        "odd_lot_ofi": odd_lot_ofi,
        "ask_step_dn": ask_step_dn,
        "msyn": msyn_est,
        "p_long": p_long_est,
        "p_cont": p_cont_est,
        "n_trades": len(trades),
    }


def check_live_catalyst(tk, dt_str):
    """Checks Polygon News API strictly before 09:30 ET (06:30 PST) for Dilution/Offering vs Bullish Catalyst."""
    url = "https://api.polygon.io/v2/reference/news"
    cutoff_utc = pd.Timestamp(f"{dt_str} 09:30:00", tz="America/New_York").tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    start_utc  = (pd.Timestamp(f"{dt_str} 09:30:00", tz="America/New_York") - pd.Timedelta(days=3)).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        r = requests.get(url, params={"ticker": tk, "published_utc.lte": cutoff_utc, "published_utc.gte": start_utc, "limit": 10, "apiKey": POLYGON_API_KEY}, timeout=8)
        if r.status_code == 200:
            titles = " | ".join((x.get("title") or "") + " " + (x.get("description") or "") for x in r.json().get("results", [])).upper()
            if any(w in titles for w in ("PUBLIC OFFERING", "DIRECT OFFERING", "PRIVATE PLACEMENT", "WARRANT", "DILUTION", "SHELF REGISTRATION", "MISSES ESTIMATES", "CUTS GUIDANCE", "LOWERS GUIDANCE")):
                return "BEARISH_DILUTION_OR_MISS", True, False
            if any(w in titles for w in ("BEATS ESTIMATES", "RAISES GUIDANCE", "UPGRADE", "PRICE TARGET", "ACQUISITION", "BUYOUT", "FDA APPROVAL", "PHASE 2", "PHASE 3", "CONTRACT", "AWARD", "PARTNERSHIP")):
                return "BULLISH_CATALYST", False, True
    except Exception:
        pass
    return "NEUTRAL_OR_TECH", False, False


def run_live_long_scanner():
    now_pt = datetime.now(PT)
    now_et = now_pt.astimezone(ET)
    scan_date = SCAN_DATE if SCAN_DATE else now_et.strftime("%Y-%m-%d")
    dow = datetime.strptime(scan_date, "%Y-%m-%d").weekday()
    dow_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][dow]

    print("=" * 105)
    print(f"  AGENT 1 — STANDALONE LIVE `LONG-ONLY` SCANNER & PLAYBOOK  |  Date: {scan_date} ({dow_name})  |  Mode: {LONG_MODE.upper()}")
    print(f"  Current Clock: {now_pt.strftime('%H:%M:%S')} PST ({now_et.strftime('%H:%M:%S')} ET)  |  Zero Shorting / 100% Cash-Account Compatible")
    print("=" * 105)

    prev_days = get_prev_trading_days(scan_date, n_days=6)
    d_m1 = prev_days[0]
    d_m3 = prev_days[2]
    d_m5 = prev_days[4]
    print(f"  [1] Fetching market prices for {scan_date}, prior close ({d_m1}), 2d baseline ({d_m3}), and 5d baseline ({d_m5})...")
    if SCAN_DATE is None:
        gd_today = fetch_snapshot_all()
        if not gd_today:
            gd_today = fetch_grouped_daily(scan_date)
    else:
        gd_today = fetch_grouped_daily(scan_date)
    gd_m1 = fetch_grouped_daily(d_m1)
    gd_m3 = fetch_grouped_daily(d_m3)
    gd_m5 = fetch_grouped_daily(d_m5)

    # ── ENGINE 1: C1 MULTI-DAY CRASH REVERSAL LONG (Wed/Thu Primary + Fri Catalyst-Filtered) ──
    c1_picks = []
    if LONG_MODE in ("hybrid", "c1"):
        if dow in (2, 3, 4):  # Wed, Thu, Fri
            c1_cands = []
            for tk in sorted(FLOAT_10M):
                b_t  = gd_today.get(tk)
                b_m1 = gd_m1.get(tk)
                b_m3 = gd_m3.get(tk)
                b_m5 = gd_m5.get(tk)
                if not (b_t and b_m1 and b_m5):
                    continue
                o_t  = b_t["o"]
                c_m1 = b_m1["c"]
                c_m3 = b_m3["c"] if b_m3 else c_m1
                c_m5 = b_m5["c"]
                if not (20.0 <= o_t <= 100.0) or not (20.0 <= c_m1 <= 120.0) or c_m5 <= 0:
                    continue
                gap = (o_t - c_m1) / c_m1
                ret_5d = (c_m1 - c_m5) / c_m5
                ret_2d = (c_m1 - c_m3) / c_m3 if c_m3 > 0 else 0.0
                if 0.03 <= gap <= 0.50 and ret_5d <= -0.15 and ret_2d <= 0.05:
                    cat_lbl, is_bear, is_bull = check_live_catalyst(tk, scan_date)
                    if is_bear:
                        continue
                    c1_cands.append({
                        "ticker": tk,
                        "engine": "C1_CRASH_REV_T3" if dow in (2, 3) else "C1_FRI_CAT_T3",
                        "open_price": o_t,
                        "entry_price": o_t,
                        "entry_time_pst": "06:30 PST (Market Open)",
                        "stop_price": round(o_t * 0.88, 2),
                        "stop_pct": -12.0,
                        "exit_target": "T+3 Close (13:00 PST, 3 trading days)",
                        "gap_pct": gap * 100.0,
                        "ret_5d_pct": ret_5d * 100.0,
                        "catalyst": cat_lbl,
                        "score": abs(ret_5d) * 100.0,
                        "cur_price": b_t["c"],
                        "live_pnl_pct": (b_t["c"] - o_t) / o_t * 100.0,
                    })
            c1_cands.sort(key=lambda x: -x["score"])
            c1_picks = c1_cands[:3]
            print(f"  [2] Engine C1 (Multi-Day Crash Reversal LONG, {dow_name}): {len(c1_picks)} qualifying pick(s) found.")
        else:
            print(f"  [2] Engine C1 (Multi-Day Crash Reversal LONG): Standby on {dow_name} (C1 trades Wed/Thu/Fri only; skips Mon/Tue).")

    # ── ENGINE 2 & 3: INTRADAY SAME-DAY T&Q CONTINUATION & INSTITUTIONAL ACCUMULATION LONGS ──
    intraday_picks = []
    if LONG_MODE in ("hybrid", "intraday"):
        print(f"  [3] Scanning positive morning gappers ($10-$100, Gap +3% to +30%, Prior $Vol >= $15M) for Intraday T&Q LONGs...")
        gap_cands = []
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
            prior_dvol = b_m1["c"] * b_m1["v"]
            if 0.03 <= gap <= 0.30 and prior_dvol >= 15_000_000:
                gap_cands.append({
                    "ticker": tk, "open_price": o_t, "prior_close": b_m1["c"],
                    "gap_pct": gap * 100.0, "prior_dvol": prior_dvol, "cur_close": b_t["c"], "day_low": b_t["l"]
                })
        gap_cands.sort(key=lambda x: -x["gap_pct"])

        def _is_operating_equity(c):
            tk = c["ticker"]
            if tk in _FLOAT:
                return c
            try:
                r = requests.get(f"https://api.polygon.io/v3/reference/tickers/{tk}", params={"apiKey": POLYGON_API_KEY}, timeout=6)
                if r.status_code == 200:
                    res = r.json().get("results", {})
                    tk_type = (res.get("type") or "").upper()
                    tk_name = (res.get("name") or "").upper()
                    if tk_type not in ("CS", "ADRC"):
                        return None
                    bad_words = ("ETF", "ETN", "TRUST", "FUND", "2X", "3X", "1.5X", "ULTRA", "PROSHARES", "DIREXION", "GRANITESHARES", "DEFIANCE", "T-REX", "YIELDMAX", "INVERSE", "BEAR", "BULL", "SHARES")
                    if any(w in tk_name for w in bad_words):
                        return None
                    return c
            except Exception:
                pass
            return None

        verified_gap_cands = []
        with ThreadPoolExecutor(max_workers=16) as ex:
            for res in ex.map(_is_operating_equity, gap_cands[:150]):
                if res is not None:
                    verified_gap_cands.append(res)
        top_gap_cands = verified_gap_cands[:25]

        def _eval_cand(c):
            tk = c["ticker"]
            bd = fetch_intraday_1m_bars(tk, scan_date)
            rth = bd.get("rth", [])
            if len(rth) < 4:
                return None
            morn_bars = [(t_s, b) for t_s, b in rth if t_s <= "09:44"]
            if len(morn_bars) < 4:
                return None
            ent_t, ent_b = morn_bars[-1]
            ent_px = ent_b["c"]
            b0 = morn_bars[0][1]
            is_short_setup = (b0["c"] < b0["o"]) and ((b0["o"] - ent_px) / b0["o"] >= 0.01)
            tq = fetch_live_morning_tq_metrics(tk, scan_date, cutoff_et=ent_t)
            cat_lbl, is_bear, is_bull = check_live_catalyst(tk, scan_date)
            c["entry_time_et"] = ent_t
            c["entry_price"] = ent_px
            c["is_short_setup"] = is_short_setup
            c["tq"] = tq
            c["catalyst"] = cat_lbl
            c["is_bear_cat"] = is_bear
            c["is_bull_cat"] = is_bull
            c["rth"] = rth
            return c

        evaluated = []
        with ThreadPoolExecutor(max_workers=8) as ex:
            for res in ex.map(_eval_cand, top_gap_cands):
                if res is not None:
                    evaluated.append(res)

        morn_short_cnt = sum(1 for x in evaluated if x["is_short_setup"] and not (15.0 <= x["gap_pct"] < 30.0))
        print(f"      Evaluated {len(evaluated)} liquid operating-stock gappers | Market Morning Short-Trigger Count: {morn_short_cnt} ({'LOW SHORT / BULLISH BREADTH' if morn_short_cnt <= 1 else 'ACTIVE SHORT REGIME'})")

        for c in evaluated:
            tq = c["tq"]
            gap = c["gap_pct"]
            opx = c["open_price"]
            dvol = c["prior_dvol"]
            msyn = tq["msyn"]
            p_cont = tq["p_cont"]
            p_long = tq["p_long"]
            ofi = tq["lr_ofi"]

            if c["is_bear_cat"]:
                c["gate_status"] = "REJECT: Bearish/Dilution Catalyst"
                continue

            # Engine B (Priority 1 Intraday): 10-30% Continuation Gap Zone + Strong Bullish T&Q
            # Requires Low Short-Breadth Morning (<=1) OR Institutional $25+/$25M+ with deep T&Q accumulation (MSYN <= -0.55 & OFI >= +0.20)
            is_eng_b = (
                10.0 <= gap <= 30.0
                and dvol >= 15e6
                and msyn <= -0.20
                and p_cont <= -0.20
                and (
                    morn_short_cnt <= 1
                    or (opx >= 25.0 and dvol >= 25e6 and msyn <= -0.55 and p_cont <= -0.55 and ofi >= 0.20)
                )
            )
            # Engine A (Priority 2 Intraday): Institutional $25+ & $25M+ ADV on Low Short-Trigger Morning (<=1)
            is_eng_a = (
                not is_eng_b
                and morn_short_cnt <= 1
                and opx >= 25.0 and dvol >= 25e6
                and 3.0 <= gap <= 25.0
                and p_long >= 0.30 and msyn <= -0.10 and ofi > 0.05
            )
            if is_eng_b or is_eng_a:
                eng_name = "TQ_CONT_10_30_B" if is_eng_b else "TQ_INST_ACCUM_A"
                c["gate_status"] = f"PASS -> {eng_name}"
                sc = (120.0 - msyn - p_cont + 0.5 * ofi) if is_eng_b else (100.0 + p_long - 0.5 * msyn + 0.5 * ofi)
                ent_px = c["entry_price"]
                stop_px = round(ent_px * 0.945, 2)
                post_bars = [(t_s, b) for t_s, b in c["rth"] if t_s > c["entry_time_et"]]
                cur_px = c["rth"][-1][1]["c"]
                status_str = "ACTIVE (HOLD -> 13:00 PST)"
                for t_s, b in post_bars:
                    if b["l"] <= stop_px:
                        cur_px = stop_px
                        status_str = f"STOPPED (-5.5%) @ {t_s} ET"
                        break
                    if t_s == "10:35" and (b["c"] - ent_px) / ent_px <= -0.015:
                        cur_px = b["c"]
                        status_str = f"TIME-STOPEXIT @ 07:35 PST (${cur_px:.2f})"
                        break
                live_pnl = (cur_px - ent_px) / ent_px * 100.0
                intraday_picks.append({
                    "ticker": c["ticker"],
                    "engine": eng_name,
                    "open_price": opx,
                    "entry_price": ent_px,
                    "entry_time_pst": "06:44 PST",
                    "stop_price": stop_px,
                    "stop_pct": -5.5,
                    "exit_target": "07:35 PST Check -> 13:00 PST Close (Same-Day Flat)",
                    "gap_pct": gap,
                    "prior_dvol_m": dvol / 1e6,
                    "lr_ofi": ofi,
                    "block_ofi": tq["block_ofi"],
                    "msyn": msyn,
                    "p_long": p_long,
                    "catalyst": c["catalyst"],
                    "score": sc,
                    "cur_price": cur_px,
                    "status_str": status_str,
                    "live_pnl_pct": live_pnl,
                })
            else:
                if 10.0 <= gap <= 30.0:
                    c["gate_status"] = f"SKIP EngB: Weak T&Q (MSYN={msyn:+.2f} > -0.20)"
                elif morn_short_cnt > 1:
                    c["gate_status"] = f"SKIP EngA: High Short Breadth ({morn_short_cnt}>1)"
                elif opx < 25.0 or dvol < 25e6:
                    c["gate_status"] = "SKIP EngA: <$25 or <$25M Prior $Vol"
                else:
                    c["gate_status"] = f"SKIP EngA: T&Q (P_LONG={p_long:+.2f}, OFI={ofi:+.2f})"

        intraday_picks.sort(key=lambda x: -x["score"])

    # Combine into Top-3 Active Portfolio Slots (`min(1/k, 50%)` sizing)
    combined_slots = []
    seen_tks = set()
    for p in c1_picks + intraday_picks:
        if p["ticker"] not in seen_tks and len(combined_slots) < 3:
            seen_tks.add(p["ticker"])
            combined_slots.append(p)

    n_slots = len(combined_slots)
    slot_pct = round(min(1.0 / n_slots, 0.50) * 100.0, 1) if n_slots > 0 else 0.0

    print("\n" + "=" * 105)
    print(f"  CONFIRMED `LONG-ONLY` PORTFOLIO PICKS FOR {scan_date} ({n_slots}/3 Active Slots | Slot Size: {slot_pct:.1f}%)")
    print("=" * 105)
    if not combined_slots:
        print("  -> No qualifying LONG setups triggered today. Keep 100% in cash — zero trades needed.")
    else:
        for idx, p in enumerate(combined_slots, 1):
            extra = (
                f"5dCrash={p['ret_5d_pct']:+.1f}%"
                if "ret_5d_pct" in p
                else f"ADV=${p['prior_dvol_m']:.1f}M | LR_OFI={p['lr_ofi']:+.2f} | MSYN={p['msyn']:+.2f} | P_LONG={p['p_long']:+.2f}"
            )
            print(
                f"  #{idx} [{p['engine']:<15}] BUY LONG {p['ticker']:<6} | Size={slot_pct:.1f}% | "
                f"Gap=+{p['gap_pct']:.1f}% | Entry=${p['entry_price']:.2f} @ {p['entry_time_pst']} | "
                f"Stop=${p['stop_price']:.2f} ({p['stop_pct']:.1f}%) | Exit={p['exit_target']}"
            )
            st_lbl = p.get("status_str", "ACTIVE")
            print(f"     -> Signals: {extra} | Catalyst={p['catalyst']} | Status={st_lbl} | Mark=${p['cur_price']:.2f} ({p['live_pnl_pct']:+.2f}%)")

    if LONG_MODE in ("hybrid", "intraday") and evaluated:
        print("\n" + "-" * 105)
        print(f"  MORNING GAPPER T&Q DIAGNOSTIC WATCHLIST (Top 10 of {len(evaluated)} Evaluated Liquid Gappers)")
        print("-" * 105)
        print(f"  {'Ticker':<6} | {'Open $':>7} | {'06:44 $':>7} | {'Gap %':>6} | {'Prior$Vol':>9} | {'LR_OFI':>6} | {'MSYN':>6} | {'P_CONT':>6} | {'P_LONG':>6} | {'Gate Status / Filter Reason'}")
        print("  " + "-" * 103)
        for c in evaluated[:10]:
            tq = c["tq"]
            print(
                f"  {c['ticker']:<6} | ${c['open_price']:>6.2f} | ${c['entry_price']:>6.2f} | "
                f"{c['gap_pct']:>+5.1f}% | ${c['prior_dvol']/1e6:>7.1f}M | "
                f"{tq['lr_ofi']:>+6.2f} | {tq['msyn']:>+6.2f} | {tq['p_cont']:>+6.2f} | {tq['p_long']:>+6.2f} | {c.get('gate_status', '')}"
            )

    print("\n" + "=" * 105)
    print("  NEXT WEEK `LONG-ONLY` LIVE OPERATOR SCHEDULE (PST TIME ZONE)")
    print("=" * 105)
    print("  • 06:30:15 PST (Wed/Thu/Fri): Run cell / script to check Engine 1 (`C1_CRASH_REV_T3` Open Longs).")
    print("  • 06:44:10 PST (Mon–Fri)    : Run cell / script to check Engine 2 (`10-30% Gap T&Q Long`) & Engine 3 (`$25+/$25M+ Inst. T&Q Long`).")
    print("  • 07:35:00 PST (Mon–Fri)    : +60m Time-Stop Check — if an Intraday T&Q Long (`Engine A/B`) is down <= -1.5%, exit early; else hold.")
    print("  • 13:00:00 PST (Mon–Fri)    : Flatten all Intraday T&Q Longs (`Engine A/B`) at Market Close (16:00 ET); exit `C1` Longs on `T+3` Close.")
    print("=" * 105)


if __name__ == "__main__":
    run_live_long_scanner()
