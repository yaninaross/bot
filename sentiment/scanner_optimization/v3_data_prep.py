# %% V3 Data Prep — Cache minute bars for V3 experiments
# Fetches raw minute bars from Polygon and saves to JSON for reuse by experiment agents.

import os, sys, json
import pandas as pd
import pytz

try:
    from polygon import RESTClient as PolygonClient
except ImportError:
    import subprocess; subprocess.check_call([sys.executable, "-m", "pip", "install", "polygon-api-client", "--quiet"])
    from polygon import RESTClient as PolygonClient

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "YOUR_POLYGON_API_KEY")
ET = pytz.timezone('America/New_York')

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

poly = PolygonClient(POLYGON_API_KEY)

df = pd.read_csv(os.path.join(RESULTS_DIR, 'v2_features_full.csv'))
pairs = df[['ticker', 'eval_date']].drop_duplicates().values.tolist()
dates = sorted(set(d for _, d in pairs))

_W = 80
print(f"\n{'='*_W}")
print(f"  V3 DATA PREP — Minute Bar Cache")
print(f"  {len(pairs)} ticker-date pairs across {len(dates)} dates")
print(f"{'='*_W}")

# ── Fetch minute bars for all ticker-date pairs ─────────────────────────────
print(f"\n  Fetching minute bars...")
minute_bars = {}
n_ok = 0
for i, (tk, dt) in enumerate(pairs):
    try:
        bars = list(poly.get_aggs(ticker=tk, multiplier=1, timespan="minute",
                                   from_=dt, to=dt, limit=500))
        if bars:
            bar_map = {}
            for b in bars:
                ts = pd.Timestamp(b.timestamp, unit='ms', tz='UTC').tz_convert(ET)
                bar_map[ts.strftime('%H:%M')] = {
                    'o': float(b.open), 'h': float(b.high),
                    'l': float(b.low), 'c': float(b.close),
                    'v': int(b.volume)}
            minute_bars[f"{tk}__{dt}"] = bar_map
            n_ok += 1
    except Exception as e:
        print(f"    Error {tk} {dt}: {e}")
    if (i + 1) % 50 == 0:
        print(f"    {i+1}/{len(pairs)} ({n_ok} ok)")

print(f"  Got minute bars for {n_ok}/{len(pairs)} pairs")

# ── Fetch SPY minute bars for each date ──────────────────────────────────────
print(f"\n  Fetching SPY minute bars for {len(dates)} dates...")
spy_minute = {}
for dt in dates:
    try:
        bars = list(poly.get_aggs(ticker='SPY', multiplier=1, timespan="minute",
                                   from_=dt, to=dt, limit=500))
        if bars:
            bar_map = {}
            for b in bars:
                ts = pd.Timestamp(b.timestamp, unit='ms', tz='UTC').tz_convert(ET)
                bar_map[ts.strftime('%H:%M')] = {
                    'o': float(b.open), 'h': float(b.high),
                    'l': float(b.low), 'c': float(b.close),
                    'v': int(b.volume)}
            spy_minute[dt] = bar_map
    except Exception as e:
        print(f"    Error SPY {dt}: {e}")

print(f"  Got SPY minute bars for {len(spy_minute)}/{len(dates)} dates")

# ── Save ─────────────────────────────────────────────────────────────────────
out = {'ticker_bars': minute_bars, 'spy_bars': spy_minute}
out_path = os.path.join(RESULTS_DIR, 'v3_minute_bars.json')
with open(out_path, 'w') as f:
    json.dump(out, f)

sz = os.path.getsize(out_path) / 1024
print(f"\n  Saved to {out_path}")
print(f"  Size: {sz:.0f} KB")
print(f"  Tickers: {n_ok}, SPY dates: {len(spy_minute)}")
print(f"{'='*_W}")
