"""
Agent 10 — Fetch daily OHLCV for biotech peer basket from Polygon API.
Saves individual + combined parquet files to agent10/data/.
"""

import requests
import pandas as pd
import time
import os

# --- Config ---
API_KEY = "2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs"
TICKERS = ["SDGR", "ABCL", "ABSI", "BEAM", "CRSP", "NTLA", "TXG", "PACB", "TWST"]
START_DATE = "2021-04-16"
END_DATE = "2026-09-26"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SLEEP_BETWEEN = 12.5  # free tier rate limit

os.makedirs(OUTPUT_DIR, exist_ok=True)

all_frames = []

for i, ticker in enumerate(TICKERS):
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
        f"{START_DATE}/{END_DATE}?adjusted=true&sort=asc&limit=50000&apiKey={API_KEY}"
    )
    print(f"[{i+1}/{len(TICKERS)}] Fetching {ticker}...")
    resp = requests.get(url, timeout=30)
    data = resp.json()

    if data.get("resultsCount", 0) == 0 or "results" not in data:
        print(f"  WARNING: No results for {ticker} — status={data.get('status')}")
        if i < len(TICKERS) - 1:
            print(f"  Sleeping {SLEEP_BETWEEN}s...")
            time.sleep(SLEEP_BETWEEN)
        continue

    rows = data["results"]
    df = pd.DataFrame(rows)

    # Rename columns: t->date, o->open, h->high, l->low, c->close, v->volume, vw->vwap
    df = df.rename(columns={"t": "date", "o": "open", "h": "high", "l": "low",
                            "c": "close", "v": "volume", "vw": "vwap"})

    # Convert epoch ms to date
    df["date"] = pd.to_datetime(df["date"], unit="ms").dt.date

    # Add ticker column
    df["ticker"] = ticker

    # Keep only the columns we need
    df = df[["date", "open", "high", "low", "close", "volume", "vwap", "ticker"]]

    # Save individual parquet
    out_path = os.path.join(OUTPUT_DIR, f"peer_{ticker}.parquet")
    df.to_parquet(out_path, index=False)

    # Summary
    print(f"  {ticker}: {len(df)} bars, {df['date'].min()} to {df['date'].max()}, "
          f"price ${df['close'].min():.2f}-${df['close'].max():.2f}")

    all_frames.append(df)

    # Rate limit sleep (skip after last ticker)
    if i < len(TICKERS) - 1:
        print(f"  Sleeping {SLEEP_BETWEEN}s...")
        time.sleep(SLEEP_BETWEEN)

# Combined parquet
if all_frames:
    combined = pd.concat(all_frames, ignore_index=True)
    combined_path = os.path.join(OUTPUT_DIR, "peers_daily.parquet")
    combined.to_parquet(combined_path, index=False)
    print(f"\nCombined: {len(combined)} total bars across {combined['ticker'].nunique()} tickers")
    print(f"Saved to {combined_path}")
else:
    print("\nNo data fetched.")

print("\nDone.")
