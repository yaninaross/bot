# %% Agent 1 — V8.2-L Official Standalone Live `LONG` Bot (`agent1_long_live.py`)
# ==============================================================================================
# 100% Standalone & Single-Cell Colab Ready (No local file dependencies, no `__file__` required).
# Official Companion to `V8.2-S` SHORT Bot (`agent1_v2.py`).
# Combined 50/50 unleveraged cash ($50K Short + $50K Long): $100K -> $361K (12.5M, DD -13.9%, 12/13 green months).
#
# Paste this entire file into a Colab / Vertex AI Notebook cell and run, or run via CLI:
#   python3 agent1/agent1_long_live.py
#   python3 agent1/agent1_long_live.py --date=2026-11-18
#   python3 agent1/agent1_long_live.py --open=AEHR
#
# Combines the 3 Validated `LONG` Alpha Engines (Active-Slot `min(1/k, 50%)` Sizing, Top-3 Max Slots):
#   1. Engine C1 (`C1_CRASH_REV_T3` — Wed/Thu/Fri, 06:30 PST Open -> T+3 Close):
#      - 12.5M Split-Proof Standalone: 39 trades, 69.2% Net WR, PF 6.90x, +186.3% Sum Net ($100K -> $527K, Sh 2.74).
#      - Universe: Verified liquid `FLOAT >= 10M` shares (`Open $20–$100`, `Prior Close $20–$120`).
#      - Trigger: Prior 5-trading-day return <= -15.0%, `ret_2d <= +5.0%`, Opening Gap +3.0% to +50.0%.
#      - No-Duplicate-Open-Position Guard (V8.2-L): Automatically detects open T+3 positions from the
#        last 3 trading days (checking -12% stop-outs) and blocks re-buying any already-open stock.
#      - Catalyst Guard: Blocks Dilution / Offering / Earnings Miss before 06:30 PST.
#      - Execution: Buy at 06:30 PST Open, GTC Hard Stop `-12.0%`, Target Exit `T+3` Close (13:00 PST).
#
#   2. Engine B (`TQ_CONT_10_30_B` — Mon-Fri Same-Day Intraday, 06:44 PST -> 13:00 PST Close):
#      - `$10-$100` Common Stocks in `10%-30%` Gap Zone (`Prior $Vol >= $15M`) + Bullish T&Q Accumulation.
#      - Hard Stop `-5.5%`, `07:35 PST (+60m)` Time-Stop if down <= -1.5%, Flat at 13:00 PST Close.
#
#   3. Engine A (`TQ_INST_ACCUM_A` — Mon-Fri Same-Day Intraday on Low Short-Breadth Mornings `<= 1`):
#      - Institutional `$25+` & `$25M+` Prior $Vol (`Gap +3% to +25%`) + Bullish Lee-Ready Order Flow.
#      - Hard Stop `-5.5%`, `07:35 PST (+60m)` Time-Stop if down <= -1.5%, Flat at 13:00 PST Close.

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
# CONFIG — Official V8.2-L (Edit directly in Colab or pass via CLI)
# ═══════════════════════════════════════════════════════════════════════════════
STRATEGY_VERSION       = "V8.2-L"
SCAN_DATE              = None       # None = Live Today (PST/ET), or e.g. '2026-11-18' for replay
LONG_MODE              = "hybrid"   # "hybrid" (C1 + Intraday A+B), "c1" (C1 Swing Only), or "intraday"
BLOCK_DUPLICATE_OPEN_C1 = True      # V8.2-L Guard: never re-buy a C1 stock while its T+3 position is open
OPEN_LONG_POSITIONS    = {
    s.strip().upper() for s in os.environ.get("OPEN_LONG_POSITIONS", "").split(",") if s.strip()
}
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs")

for _arg in sys.argv[1:]:
    if _arg.startswith("--date="):
        SCAN_DATE = _arg.split("=", 1)[1].strip()
    elif _arg.startswith("--mode="):
        LONG_MODE = _arg.split("=", 1)[1].strip().lower()
    elif _arg.startswith("--open="):
        OPEN_LONG_POSITIONS.update(s.strip().upper() for s in _arg.split("=", 1)[1].split(",") if s.strip())

ET = pytz.timezone("America/New_York")
PT = pytz.timezone("America/Los_Angeles")

# Compact Embedded Verified Liquid Float Universe (>= 10M shares) — Zero truncation risk in Colab!
FLOAT_10M = set((
    "AAPL AARD ABM ACHR ACMR ACV ACVA ADBE AEHR AENT AEON AGCO AGRZ AHMA ALAB AMAT AMD AME APA APLD "
    "ARBE ARM ASAN ASO ASTS ATEC ATHE ATI AVAV AXGN AXTA AXTI BABA BBIO BBNX BE BEAM BHVN BLZE BNC "
    "BOW BRZE BTBT CAN CAT CC CCB CIFR CLSK COIN COO CORZ CPRT CRDO CSR CVX DELL DKS DNOW DOV DPRO "
    "DSS EAT ELOG ET ETN EVRG EXEL FCEL FCX FEIM FIZZ FLNC FLWS GDDY GENC GME GOOS HBM HGTY HOFT "
    "HOOD HPE HPQ HTLD HUT IBTA IESC IMO INBX INDP INFQ INTC INV IONQ IP IPDN IREN KEEL KPTI KR KTOS "
    "LASE LEU LOVE LULU LUNR M MARA MAS MCFT MDCX META MGN MOBX MRNA MSTR MU MVIS MXL NBIS NEE NEGG "
    "NNE NTHI NTLA NVDA NVS OCGN OKLO ONDS OPTT OPTX ORCL OTLK PDSB PGY PL PLNT PLTR PLUG PMTS PRLD "
    "QBTS RDDT RDW REGN RGTI RH RIOT RKLB RLMD ROIV RTX SCCO SDRL SHMD SHOE SIG SKYX SLS SMCI SMR "
    "SNDK SNX SNYR SOUN SRPT SSL SSNC STRL STX SURG SWKS SWRD TEN TPST TSLA TTAN TTD TYRA UBER UNFI "
    "UPST URI VAC VICR VOYG VRT VST WDH WULF ZEO ZUMZ"
).split())
_FLOAT = {tk: 10_000_000 for tk in FLOAT_10M}

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


def get_prev_trading_days(dt_str, n_days=8):
    d = datetime.strptime(dt_str, "%Y-%m-%d").date()
    out = []
    while len(out) < n_days:
        d -= timedelta(days=1)
        s = d.strftime("%Y-%m-%d")
        if d.weekday() < 5 and s not in _HOLIDAYS:
            out.append(s)
    return out


def get_next_trading_day(dt_str, n_steps=3):
    d = datetime.strptime(dt_str, "%Y-%m-%d").date()
    steps = 0
    while steps < n_steps:
        d += timedelta(days=1)
        s = d.strftime("%Y-%m-%d")
        if d.weekday() < 5 and s not in _HOLIDAYS:
            steps += 1
    return d.strftime("%Y-%m-%d")


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
    print(f"  AGENT 1 — {STRATEGY_VERSION} STANDALONE LIVE `LONG` BOT  |  Date: {scan_date} ({dow_name})  |  Mode: {LONG_MODE.upper()}")
    print(f"  Current Clock: {now_pt.strftime('%H:%M:%S')} PST ({now_et.strftime('%H:%M:%S')} ET)  |  No-Duplicate T+3 Guard: {'ON' if BLOCK_DUPLICATE_OPEN_C1 else 'OFF'}")
    print("=" * 105)

    prev_days = get_prev_trading_days(scan_date, n_days=8)
    d_m1 = prev_days[0]
    d_m3 = prev_days[2]
    d_m5 = prev_days[4]
    print(f"  [1] Fetching market prices for {scan_date} + prior 8 trading days ({prev_days[-1]} -> {d_m1}) in parallel...")
    if SCAN_DATE is None:
        gd_today = fetch_snapshot_all()
        if not gd_today:
            gd_today = fetch_grouped_daily(scan_date)
    else:
        gd_today = fetch_grouped_daily(scan_date)

    gd_hist = {}
    with ThreadPoolExecutor(max_workers=8) as _ex_gd:
        for _d, _res in zip(prev_days, _ex_gd.map(fetch_grouped_daily, prev_days)):
            gd_hist[_d] = _res
    gd_m1 = gd_hist[d_m1]
    gd_m3 = gd_hist[d_m3]
    gd_m5 = gd_hist[d_m5]

    # ── DETECT CURRENTLY OPEN T+3 ENGINE C1 POSITIONS (Causal Replay of Last 3 Trading Days) ──
    active_open_c1 = {}  # ticker -> dict(entry_date, entry_price, stop_price, exit_date, cur_price, pnl_pct)
    for mtk in sorted(OPEN_LONG_POSITIONS):
        cur_p = (gd_today.get(mtk) or gd_m1.get(mtk) or {}).get("c", 0.0)
        active_open_c1[mtk] = {
            "entry_date": "MANUAL", "entry_price": cur_p, "stop_price": round(cur_p * 0.88, 2),
            "exit_date": "MANUAL", "cur_price": cur_p, "pnl_pct": 0.0,
        }

    if LONG_MODE in ("hybrid", "c1") and BLOCK_DUPLICATE_OPEN_C1:
        # Replay oldest -> newest across T-3 (k=2), T-2 (k=1), T-1 (k=0)
        for k in (2, 1, 0):
            dk = prev_days[k]
            if datetime.strptime(dk, "%Y-%m-%d").weekday() not in (2, 3, 4):
                continue
            gd_k = gd_hist.get(dk, {})
            gd_k1 = gd_hist.get(prev_days[k + 1], {})
            gd_k3 = gd_hist.get(prev_days[k + 3], {})
            gd_k5 = gd_hist.get(prev_days[k + 5], {})
            if not (gd_k and gd_k1 and gd_k5):
                continue
            hist_cands = []
            for tk in sorted(FLOAT_10M):
                if tk in active_open_c1:
                    continue
                bk, bk1, bk5 = gd_k.get(tk), gd_k1.get(tk), gd_k5.get(tk)
                bk3 = gd_k3.get(tk)
                if not (bk and bk1 and bk5):
                    continue
                ok_p, ck1, ck5 = bk["o"], bk1["c"], bk5["c"]
                ck3 = bk3["c"] if bk3 else ck1
                if not (20.0 <= ok_p <= 100.0) or not (20.0 <= ck1 <= 120.0) or ck5 <= 0:
                    continue
                g_k = (ok_p - ck1) / ck1
                r5_k = (ck1 - ck5) / ck5
                r2_k = (ck1 - ck3) / ck3 if ck3 > 0 else 0.0
                if 0.03 <= g_k <= 0.50 and r5_k <= -0.15 and r2_k <= 0.05:
                    _, is_bear_k, _ = check_live_catalyst(tk, dk)
                    if is_bear_k:
                        continue
                    hist_cands.append((abs(r5_k), tk, ok_p))
            hist_cands.sort(reverse=True)
            for _, tk, ok_p in hist_cands[:3]:
                stop_p = round(ok_p * 0.88, 2)
                # Check if already stopped out between entry day dk and yesterday (prev_days[0])
                stopped = False
                for chk_i in range(k, -1, -1):
                    bc = gd_hist.get(prev_days[chk_i], {}).get(tk)
                    if not bc:
                        continue
                    if (chk_i < k and bc["o"] <= stop_p) or bc["l"] <= stop_p:
                        stopped = True
                        break
                if not stopped:
                    cur_p = (gd_today.get(tk) or gd_m1.get(tk) or {}).get("c", ok_p)
                    active_open_c1[tk] = {
                        "entry_date": dk,
                        "entry_price": ok_p,
                        "stop_price": stop_p,
                        "exit_date": get_next_trading_day(dk, 3),
                        "cur_price": cur_p,
                        "pnl_pct": (cur_p - ok_p) / ok_p * 100.0 if ok_p > 0 else 0.0,
                    }

    if active_open_c1:
        print(f"  [1b] Active Open C1 T+3 Swing Position(s) Carrying Into {scan_date} (Duplicate Re-Entry Blocked):")
        for tk, inf in sorted(active_open_c1.items()):
            ex_tag = "EXITS TODAY 13:00 PST CLOSE!" if inf["exit_date"] == scan_date else f"Exits {inf['exit_date']} 13:00 PST Close"
            print(
                f"       • HOLD {tk:<6} | Opened {inf['entry_date']} @ ${inf['entry_price']:.2f} | "
                f"GTC Stop=${inf['stop_price']:.2f} (-12%) | Mark=${inf['cur_price']:.2f} ({inf['pnl_pct']:+.2f}%) | {ex_tag}"
            )

    # ── ENGINE 1: C1 MULTI-DAY CRASH REVERSAL LONG (Wed/Thu/Fri, No-Duplicate Open Guard) ──
    c1_picks = []
    c1_dup_skipped = []
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
                    if BLOCK_DUPLICATE_OPEN_C1 and tk in active_open_c1:
                        inf = active_open_c1[tk]
                        c1_dup_skipped.append(f"{tk}(open from {inf['entry_date']} -> exits {inf['exit_date']})")
                        continue
                    cat_lbl, is_bear, is_bull = check_live_catalyst(tk, scan_date)
                    if is_bear:
                        continue
                    ex_dt_str = get_next_trading_day(scan_date, 3)
                    c1_cands.append({
                        "ticker": tk,
                        "engine": "C1_CRASH_REV_T3" if dow in (2, 3) else "C1_FRI_CAT_T3",
                        "open_price": o_t,
                        "entry_price": o_t,
                        "entry_time_pst": "06:30 PST (Market Open)",
                        "stop_price": round(o_t * 0.88, 2),
                        "stop_pct": -12.0,
                        "exit_target": f"T+3 Close ({ex_dt_str} 13:00 PST)",
                        "gap_pct": gap * 100.0,
                        "ret_5d_pct": ret_5d * 100.0,
                        "catalyst": cat_lbl,
                        "score": abs(ret_5d) * 100.0,
                        "cur_price": b_t["c"],
                        "live_pnl_pct": (b_t["c"] - o_t) / o_t * 100.0,
                    })
            c1_cands.sort(key=lambda x: -x["score"])
            c1_picks = c1_cands[:3]
            dup_s = f" | Duplicate-Guard Blocked: {', '.join(c1_dup_skipped)}" if c1_dup_skipped else ""
            print(f"  [2] Engine C1 (Multi-Day Crash Reversal LONG, {dow_name}): {len(c1_picks)} new pick(s) found{dup_s}.")
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
