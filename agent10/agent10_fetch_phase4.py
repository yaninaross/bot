import os, time, json, requests
import pandas as pd
from pathlib import Path

KEY = os.environ.get("POLYGON_API_KEY", "2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs")
OUT = Path("agent10/data")
OUT.mkdir(parents=True, exist_ok=True)

# Hedge ETFs we don't already have
HEDGE_ETFS = ["IBB", "ARKK", "ICLN", "TAN", "WGMI", "XLV", "XLE", "XLY"]

# New biotech candidates
NEW_BIO = ["EDIT", "DNA", "TEM", "ADPT", "GH", "SANA", "ARCT", "RGNX"]

# Additional biotech
ADD_BIO = ["FATE", "VERV", "RARE", "DCPH", "NUVB", "ACAD", "TGTX"]

# Out-of-sector
OUT_SECTOR = ["MARA", "RIOT", "CLSK", "PLUG", "RUN", "RIVN", "LCID"]

ALL = HEDGE_ETFS + NEW_BIO + ADD_BIO + OUT_SECTOR
print(f"Fetching {len(ALL)} tickers...")

results = {}
for i, tk in enumerate(ALL):
    out_path = OUT / f"p4_{tk}.parquet"
    if out_path.exists():
        print(f"  {tk}: already exists, skipping")
        continue
    if i > 0:
        time.sleep(12.5)
    url = (f"https://api.polygon.io/v2/aggs/ticker/{tk}/range/1/day/"
           f"2021-04-16/2026-09-26?adjusted=true&sort=asc&limit=50000&apiKey={KEY}")
    try:
        r = requests.get(url, timeout=30)
        data = r.json()
        if data.get("resultsCount", 0) == 0:
            print(f"  {tk}: NO DATA")
            continue
        rows = data["results"]
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["t"], unit="ms").dt.normalize()
        df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume","vw":"vwap"})
        df["ticker"] = tk
        df = df[["date","open","high","low","close","volume","vwap","ticker"]]
        df.to_parquet(out_path, index=False)
        results[tk] = len(df)
        print(f"  {tk}: {len(df)} bars, {df['date'].min().date()} to {df['date'].max().date()}, "
              f"${df['close'].min():.2f}-${df['close'].max():.2f}")
    except Exception as e:
        print(f"  {tk}: ERROR - {e}")

print(f"\nDone. Fetched {len(results)} tickers successfully.")
for tk, n in results.items():
    print(f"  {tk}: {n} bars")
