"""
Live Broker Short-Locate Pre-Check & Automatic ETB (Easy-to-Borrow) Fallback Module
===================================================================================
Provides a 3-layer pre-trade short-locate & borrow-fee check for `agent1_v2.py` and
the T&Q Synergy engines:

1. Layer 1 (IBKR Public Short-Stock Feed — Zero API Key Needed):
   Downloads and caches `ftp://shortstock:@ftp2.interactivebrokers.com/usa.txt`
   (refreshed every 15 minutes during pre-market/RTH, ~19,600+ US equities),
   returning exact `avail_shares` and annualized `fee_rate` (%).

2. Layer 2 (Optional Alpaca Markets Assets API `/v2/assets/{symbol}`):
   If `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` are set in environment variables,
   queries Alpaca's real-time `shortable` and `easy_to_borrow` flags.

3. Layer 3 (Manual Broker Override & Point-in-Time Liquidity Fallback):
   - Allows passing `--htb=SYM1,SYM2` on the CLI or setting `HTB_BLACKLIST_TODAY`
     if your specific broker (Schwab/TOS, Fidelity, IBKR, CenterPoint, Cobra)
     rejects a locate at 06:31 PST.
   - If the FTP/API is unreachable, falls back to structural ETB rules
     (`prior_dvol >= $20M` and `open_price >= $15`).
"""

import os
import time
import pickle
import urllib.request
import requests

_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".ibkr_usa_short_cache.pkl")
_MEM_IBKR_MAP = None
_MEM_IBKR_TS  = 0.0

DEFAULT_MIN_AVAIL_SHARES = int(os.environ.get("MIN_ETB_SHARES", "5000"))
DEFAULT_MAX_BORROW_FEE   = float(os.environ.get("MAX_BORROW_FEE_PCT", "30.0"))


def load_ibkr_short_table(max_age_sec=900):
    """Fetches or loads cached IBKR USA short availability table (15-min TTL)."""
    global _MEM_IBKR_MAP, _MEM_IBKR_TS
    now = time.time()
    if _MEM_IBKR_MAP is not None and (now - _MEM_IBKR_TS) < max_age_sec:
        return _MEM_IBKR_MAP

    if os.path.exists(_CACHE_PATH):
        try:
            mtime = os.path.getmtime(_CACHE_PATH)
            if (now - mtime) < max_age_sec:
                with open(_CACHE_PATH, "rb") as f:
                    _MEM_IBKR_MAP = pickle.load(f)
                    _MEM_IBKR_TS = mtime
                    return _MEM_IBKR_MAP
        except Exception:
            pass

    try:
        req = urllib.request.urlopen("ftp://shortstock:@ftp2.interactivebrokers.com/usa.txt", timeout=12)
        raw_txt = req.read().decode("utf-8", errors="ignore")
        ibkr_map = {}
        for line in raw_txt.splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) < 8:
                continue
            sym = parts[0].strip().upper()
            fee_s = parts[6].strip()
            avail_s = parts[7].strip().replace(">", "").replace("<", "")
            try:
                fee_rate = float(fee_s) if fee_s and fee_s != "NA" else 99.0
            except Exception:
                fee_rate = 99.0
            try:
                avail_sh = int(float(avail_s)) if avail_s and avail_s != "NA" else 0
            except Exception:
                avail_sh = 0
            ibkr_map[sym] = {
                "fee_rate": fee_rate,
                "avail_shares": avail_sh,
                "name": parts[2].strip(),
            }
        if ibkr_map:
            _MEM_IBKR_MAP = ibkr_map
            _MEM_IBKR_TS = now
            try:
                with open(_CACHE_PATH, "wb") as f:
                    pickle.dump(ibkr_map, f)
            except Exception:
                pass
            return _MEM_IBKR_MAP
    except Exception:
        pass

    # Fallback to stale disk cache if network/FTP timed out
    if os.path.exists(_CACHE_PATH):
        try:
            with open(_CACHE_PATH, "rb") as f:
                _MEM_IBKR_MAP = pickle.load(f)
                return _MEM_IBKR_MAP
        except Exception:
            pass
    return {}


def check_alpaca_etb(symbol):
    """Optional check against Alpaca /v2/assets/{symbol} if ALPACA_API_KEY is configured."""
    api_key = os.environ.get("ALPACA_API_KEY", "").strip()
    sec_key = os.environ.get("ALPACA_SECRET_KEY", "").strip()
    if not api_key or not sec_key:
        return None
    try:
        url = f"https://paper-api.alpaca.markets/v2/assets/{symbol.upper()}"
        resp = requests.get(
            url,
            headers={"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": sec_key},
            timeout=3,
        )
        if resp.status_code == 200:
            data = resp.json()
            return bool(data.get("shortable", False) and data.get("easy_to_borrow", False))
    except Exception:
        pass
    return None


def evaluate_short_locate(
    ticker,
    open_price=20.0,
    prior_dvol=15e6,
    gap_pct_dec=0.10,
    manual_htb_blacklist=None,
    min_avail_shares=DEFAULT_MIN_AVAIL_SHARES,
    max_borrow_fee=DEFAULT_MAX_BORROW_FEE,
):
    """
    Evaluates whether `ticker` passes the Short-Locate Pre-Check.
    Returns a dict:
      {
        "is_etb": bool,
        "tier": "ETB_PRIME" | "ETB_MOD" | "HTB_REJECT",
        "avail_shares": int,
        "fee_rate_pct": float,
        "daily_borrow_bps": float,
        "badge": str,
        "reason": str,
      }
    """
    btk = str(ticker).split("_")[0].strip().upper()
    if manual_htb_blacklist and btk in {s.strip().upper() for s in manual_htb_blacklist}:
        return {
            "is_etb": False,
            "tier": "HTB_REJECT",
            "avail_shares": 0,
            "fee_rate_pct": 99.9,
            "daily_borrow_bps": 40.0,
            "badge": "HTB-MANUAL",
            "reason": "Manual broker HTB blacklist override",
        }

    alp_etb = check_alpaca_etb(btk)
    if alp_etb is False:
        return {
            "is_etb": False,
            "tier": "HTB_REJECT",
            "avail_shares": 0,
            "fee_rate_pct": 50.0,
            "daily_borrow_bps": 20.0,
            "badge": "HTB-ALPACA",
            "reason": "Alpaca /v2/assets marked easy_to_borrow=False",
        }

    ibkr_map = load_ibkr_short_table()
    info = ibkr_map.get(btk)

    if info is not None:
        avail = int(info["avail_shares"])
        fee = float(info["fee_rate"])
        daily_bps = round((fee / 252.0) * 100.0, 1)
        av_str = f"{avail // 1000}K" if avail < 1_000_000 else f"{avail / 1_000_000:.1f}M"

        if avail < min_avail_shares or fee > max_borrow_fee:
            return {
                "is_etb": False,
                "tier": "HTB_REJECT",
                "avail_shares": avail,
                "fee_rate_pct": fee,
                "daily_borrow_bps": daily_bps,
                "badge": f"HTB({fee:.0f}%/{av_str})",
                "reason": f"IBKR fee {fee:.1f}% > {max_borrow_fee:.0f}% or avail {avail:,} < {min_avail_shares:,}",
            }

        # Extra caution on >=35% parabolic gaps with low normal dollar volume (<$10M/day): require >=20K shares
        if gap_pct_dec >= 0.35 and (prior_dvol or 0) < 10e6 and avail < 20000:
            return {
                "is_etb": False,
                "tier": "HTB_REJECT",
                "avail_shares": avail,
                "fee_rate_pct": fee,
                "daily_borrow_bps": daily_bps,
                "badge": f"HTB-SQUEEZE({av_str})",
                "reason": f"Parabolic >=35% gap with <$10M dvol and only {avail:,} IBKR shares",
            }

        tier = "ETB_PRIME" if (avail >= 25000 and fee <= 5.0) else "ETB_MOD"
        return {
            "is_etb": True,
            "tier": tier,
            "avail_shares": avail,
            "fee_rate_pct": fee,
            "daily_borrow_bps": daily_bps,
            "badge": f"ETB({fee:.1f}%/{av_str})",
            "reason": "Verified in IBKR live shortable feed",
        }

    # Structural fallback if ticker is not in IBKR feed (or historical ticker):
    if (prior_dvol or 0) >= 20e6 and (open_price or 0) >= 15.0:
        return {
            "is_etb": True,
            "tier": "ETB_STRUCTURAL",
            "avail_shares": 50000,
            "fee_rate_pct": 1.5,
            "daily_borrow_bps": 0.6,
            "badge": "ETB-LIQ",
            "reason": "Institutional liquidity fallback ($Vol >= $20M, Price >= $15)",
        }

    return {
        "is_etb": False,
        "tier": "HTB_REJECT",
        "avail_shares": 0,
        "fee_rate_pct": 75.0,
        "daily_borrow_bps": 30.0,
        "badge": "HTB-UNLISTED",
        "reason": "Missing from IBKR shortable feed and < $20M daily dollar volume",
    }
