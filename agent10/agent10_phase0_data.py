# %% Agent 10 — Phase 0: Data Acquisition
# Fetches all data needed for RXRX cyclical mean reversion investigation.
#
# Outputs saved to agent10/data/ as parquet files:
#   rxrx_daily.parquet      — RXRX daily OHLCV since IPO (Apr 2021)
#   rxrx_30min.parquet      — RXRX 30-min bars, last 2+ years
#   context_daily.parquet   — ARKG, XBI, IWM, SPY, VIX daily
#   btc_daily.parquet       — BTC daily (X:BTCUSD)
#   rxrx_events.parquet     — earnings, corporate events
#   ark_trades_rxrx.parquet — ARK daily RXRX holdings changes
#
# Run as a Colab cell. Config vars at top.

import os, sys, time, json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ── Config ─────────────────────────────────────────────────────────
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs")
DATA_DIR = Path(__file__).parent / "data" if "__file__" in dir() else Path("agent10/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

RXRX_TICKER = "RXRX"
CONTEXT_TICKERS = ["ARKG", "XBI", "IWM", "SPY"]
VIX_TICKER = "VIX"  # Polygon uses I:VIX for indices
BTC_TICKER = "X:BTCUSD"

IPO_DATE = "2021-04-16"
INTRADAY_START = "2024-01-01"
END_DATE = date.today().isoformat()

HOLDOUT_START = "2026-04-01"

RATE_LIMIT_SLEEP = 12.5  # seconds between paginated calls (free tier: 5/min)

# ── Helpers ────────────────────────────────────────────────────────
def polygon_aggs(ticker, multiplier, timespan, from_date, to_date, limit=50000):
    """Fetch aggregates from Polygon REST API with pagination."""
    all_results = []
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
    params = {"adjusted": "true", "sort": "asc", "limit": limit, "apiKey": POLYGON_API_KEY}

    while url:
        resp = requests.get(url, params=params)
        if resp.status_code == 429:
            print(f"  Rate limited, sleeping 60s...")
            time.sleep(60)
            continue
        data = resp.json()
        if data.get("resultsCount", 0) > 0:
            all_results.extend(data["results"])
        url = data.get("next_url")
        if url:
            params = {"apiKey": POLYGON_API_KEY}
            time.sleep(RATE_LIMIT_SLEEP)

    if not all_results:
        print(f"  WARNING: No data for {ticker} {timespan} {from_date} to {to_date}")
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    if timespan == "day":
        df["date"] = df["date"].dt.date
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "vw": "vwap", "n": "num_trades"})
    df = df.drop(columns=["t"], errors="ignore")
    return df


def fetch_daily(ticker, from_date=IPO_DATE, to_date=END_DATE):
    """Fetch daily OHLCV."""
    print(f"Fetching daily: {ticker} from {from_date} to {to_date}")
    df = polygon_aggs(ticker, 1, "day", from_date, to_date)
    if not df.empty:
        df["ticker"] = ticker
        print(f"  Got {len(df)} bars")
    return df


def fetch_intraday(ticker, multiplier=30, timespan="minute", from_date=INTRADAY_START, to_date=END_DATE):
    """Fetch intraday bars in yearly chunks to respect API limits."""
    print(f"Fetching {multiplier}-min: {ticker} from {from_date} to {to_date}")
    all_dfs = []

    start = pd.Timestamp(from_date)
    end = pd.Timestamp(to_date)
    chunk_start = start

    while chunk_start < end:
        chunk_end = min(chunk_start + pd.DateOffset(months=3), end)
        cs = chunk_start.strftime("%Y-%m-%d")
        ce = chunk_end.strftime("%Y-%m-%d")
        print(f"  Chunk: {cs} to {ce}")
        df = polygon_aggs(ticker, multiplier, timespan, cs, ce)
        if not df.empty:
            all_dfs.append(df)
        chunk_start = chunk_end + pd.DateOffset(days=1)
        time.sleep(RATE_LIMIT_SLEEP)

    if all_dfs:
        result = pd.concat(all_dfs, ignore_index=True)
        result["ticker"] = ticker
        print(f"  Total: {len(result)} bars")
        return result
    return pd.DataFrame()


# ── 1. RXRX Daily ─────────────────────────────────────────────────
# %%
print("=" * 60)
print("STEP 1: RXRX Daily OHLCV")
print("=" * 60)

rxrx_daily = fetch_daily(RXRX_TICKER)
if not rxrx_daily.empty:
    rxrx_daily.to_parquet(DATA_DIR / "rxrx_daily.parquet", index=False)
    print(f"Saved rxrx_daily.parquet: {len(rxrx_daily)} rows, "
          f"{rxrx_daily['date'].min()} to {rxrx_daily['date'].max()}")
    print(f"Price range: ${rxrx_daily['close'].min():.2f} - ${rxrx_daily['close'].max():.2f}")

# ── 2. RXRX 30-min Intraday ───────────────────────────────────────
# %%
print("\n" + "=" * 60)
print("STEP 2: RXRX 30-min Intraday Bars")
print("=" * 60)

rxrx_30min = fetch_intraday(RXRX_TICKER, multiplier=30, timespan="minute")
if not rxrx_30min.empty:
    rxrx_30min.to_parquet(DATA_DIR / "rxrx_30min.parquet", index=False)
    print(f"Saved rxrx_30min.parquet: {len(rxrx_30min)} rows")

# ── 3. Context Series (Daily) ─────────────────────────────────────
# %%
print("\n" + "=" * 60)
print("STEP 3: Context Series Daily")
print("=" * 60)

context_dfs = []
for ticker in CONTEXT_TICKERS:
    df = fetch_daily(ticker)
    if not df.empty:
        context_dfs.append(df)
    time.sleep(RATE_LIMIT_SLEEP)

if context_dfs:
    context_daily = pd.concat(context_dfs, ignore_index=True)
    context_daily.to_parquet(DATA_DIR / "context_daily.parquet", index=False)
    print(f"Saved context_daily.parquet: {len(context_daily)} rows")

# ── 4. VIX ─────────────────────────────────────────────────────────
# %%
print("\n" + "=" * 60)
print("STEP 4: VIX Index")
print("=" * 60)

# Polygon uses I:VIX for VIX index
vix_daily = fetch_daily("I:VIX")
if vix_daily.empty:
    print("  Trying VIXY ETF as VIX proxy...")
    vix_daily = fetch_daily("VIXY")
if not vix_daily.empty:
    vix_daily["ticker"] = "VIX"
    vix_daily.to_parquet(DATA_DIR / "vix_daily.parquet", index=False)
    print(f"Saved vix_daily.parquet: {len(vix_daily)} rows")

# ── 5. BTC ─────────────────────────────────────────────────────────
# %%
print("\n" + "=" * 60)
print("STEP 5: BTC (Risk Appetite Proxy)")
print("=" * 60)

btc_daily = fetch_daily(BTC_TICKER)
if not btc_daily.empty:
    btc_daily.to_parquet(DATA_DIR / "btc_daily.parquet", index=False)
    print(f"Saved btc_daily.parquet: {len(btc_daily)} rows")

# ── 6. RXRX Corporate Events / Earnings ───────────────────────────
# %%
print("\n" + "=" * 60)
print("STEP 6: RXRX Events & Earnings")
print("=" * 60)

events = []

# Fetch ticker details for context
resp = requests.get(
    f"https://api.polygon.io/v3/reference/tickers/{RXRX_TICKER}",
    params={"apiKey": POLYGON_API_KEY}
)
if resp.status_code == 200:
    details = resp.json().get("results", {})
    print(f"  {RXRX_TICKER}: {details.get('name', 'N/A')}, "
          f"Market cap: {details.get('market_cap', 'N/A')}")

# Fetch dividends/splits/events
time.sleep(RATE_LIMIT_SLEEP)
resp = requests.get(
    f"https://api.polygon.io/v3/reference/splits",
    params={"ticker": RXRX_TICKER, "apiKey": POLYGON_API_KEY}
)
if resp.status_code == 200:
    splits = resp.json().get("results", [])
    for s in splits:
        events.append({"date": s.get("execution_date"), "type": "split",
                        "detail": f"Split {s.get('split_from')}:{s.get('split_to')}"})

# Known major events (from brief — hardcoded since not all available via API)
known_events = [
    {"date": "2023-07-18", "type": "investment", "detail": "NVIDIA strategic investment announced"},
    {"date": "2024-11-18", "type": "merger", "detail": "Exscientia merger agreement announced"},
    {"date": "2024-01-08", "type": "pipeline", "detail": "Q4 2023 pipeline update"},
]
events.extend(known_events)

# Attempt to get earnings dates from Polygon (stock financials)
time.sleep(RATE_LIMIT_SLEEP)
resp = requests.get(
    f"https://api.polygon.io/v3/reference/tickers/{RXRX_TICKER}/events",
    params={"apiKey": POLYGON_API_KEY}
)
if resp.status_code == 200:
    event_data = resp.json().get("results", {}).get("events", [])
    for e in event_data:
        events.append({"date": e.get("date"), "type": e.get("type", "corporate"),
                        "detail": json.dumps(e)})

if events:
    events_df = pd.DataFrame(events)
    events_df.to_parquet(DATA_DIR / "rxrx_events.parquet", index=False)
    print(f"Saved rxrx_events.parquet: {len(events_df)} events")
else:
    print("  No events fetched — check API access")

# ── 7. ARK Trades (RXRX) ──────────────────────────────────────────
# %%
print("\n" + "=" * 60)
print("STEP 7: ARK Trade Files (RXRX)")
print("=" * 60)

# ARK publishes daily trade CSVs at ark-funds.com
# We'll try to fetch the trade history
ark_url = "https://arkfunds.io/api/v2/stock/trades"
try:
    resp = requests.get(ark_url, params={"symbol": "RXRX"}, timeout=30)
    if resp.status_code == 200:
        ark_data = resp.json()
        if "stock" in ark_data:
            ark_df = pd.DataFrame(ark_data["stock"])
            ark_df.to_parquet(DATA_DIR / "ark_trades_rxrx.parquet", index=False)
            print(f"Saved ark_trades_rxrx.parquet: {len(ark_df)} trades")
        else:
            print("  No ARK trade data in response")
    else:
        print(f"  ARK trades API returned {resp.status_code}")
        print("  You can manually download from: https://cathiesark.com/ark-funds-combined/trades/rxrx")
except Exception as e:
    print(f"  ARK trades fetch failed: {e}")
    print("  Manual alternative: download from cathiesark.com or ark-funds.com")

# ── 8. Short Interest (if available) ──────────────────────────────
# %%
print("\n" + "=" * 60)
print("STEP 8: Short Interest Data")
print("=" * 60)

# Polygon short interest endpoint (may require paid plan)
time.sleep(RATE_LIMIT_SLEEP)
resp = requests.get(
    f"https://api.polygon.io/v3/reference/tickers/{RXRX_TICKER}/short-interest",
    params={"apiKey": POLYGON_API_KEY}
)
if resp.status_code == 200 and resp.json().get("results"):
    si_df = pd.DataFrame(resp.json()["results"])
    si_df.to_parquet(DATA_DIR / "rxrx_short_interest.parquet", index=False)
    print(f"Saved rxrx_short_interest.parquet: {len(si_df)} rows")
else:
    print(f"  Short interest not available (status {resp.status_code})")
    print("  Will use FINRA data or estimate from volume patterns")
    print("  Manual source: https://www.nasdaqtrader.com/trader.aspx?id=ShortInterest")

# ── 9. Summary ─────────────────────────────────────────────────────
# %%
print("\n" + "=" * 60)
print("PHASE 0 DATA SUMMARY")
print("=" * 60)

files = list(DATA_DIR.glob("*.parquet"))
for f in sorted(files):
    df = pd.read_parquet(f)
    print(f"  {f.name:35s} {len(df):>8,} rows")

print(f"\nHoldout locked: {HOLDOUT_START} onward — DO NOT use in analysis")
print(f"Development window: {IPO_DATE} to {HOLDOUT_START}")
print(f"\nTotal files: {len(files)}")
print("\nPhase 0 complete. Proceed to Phase 1: Cycle Characterization.")
