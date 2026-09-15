# %% Refresh daily bars to include Sep 12+ for T+2 computation

import os, sys, json
import pandas as pd
from datetime import datetime, timedelta

try:
    from polygon import RESTClient as PolygonClient
except ImportError:
    import subprocess; subprocess.check_call([sys.executable, "-m", "pip", "install", "polygon-api-client", "--quiet"])
    from polygon import RESTClient as PolygonClient

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "YOUR_POLYGON_API_KEY")

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

poly = PolygonClient(POLYGON_API_KEY)

df = pd.read_csv(os.path.join(RESULTS_DIR, 'v2_features_full.csv'))
tickers = sorted(df['ticker'].unique().tolist())
tickers_plus = list(set(tickers + ['SPY']))

with open(os.path.join(RESULTS_DIR, 'v2_daily_data.json')) as f:
    old_daily = json.load(f)

from_date = '2026-08-01'
to_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

print(f"Refreshing daily bars for {len(tickers_plus)} tickers")
print(f"Range: {from_date} to {to_date}")

daily_data = {}
spy_daily = {}
n_ok = 0

for i, tk in enumerate(tickers_plus):
    try:
        bars = list(poly.get_aggs(ticker=tk, multiplier=1, timespan="day",
                                   from_=from_date, to=to_date, limit=60))
        if bars:
            result = {}
            for b in bars:
                dt = pd.Timestamp(b.timestamp, unit='ms', tz='UTC').strftime('%Y-%m-%d')
                result[dt] = {'open': float(b.open), 'high': float(b.high),
                              'low': float(b.low), 'close': float(b.close),
                              'volume': int(b.volume)}
            if tk == 'SPY':
                spy_daily = result
            else:
                daily_data[tk] = result
            n_ok += 1
    except Exception as e:
        print(f"  Error {tk}: {e}")
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(tickers_plus)} ({n_ok} ok)")

print(f"Got daily bars for {n_ok}/{len(tickers_plus)} tickers")

sample_tk = tickers[0]
dates = sorted(daily_data.get(sample_tk, {}).keys())
print(f"\nSample ({sample_tk}): {len(dates)} bars, {dates[0]} to {dates[-1]}")

spy_dates = sorted(spy_daily.keys())
print(f"SPY: {len(spy_dates)} bars, {spy_dates[0]} to {spy_dates[-1]}")

out_path = os.path.join(RESULTS_DIR, 'v2_daily_data.json')
with open(out_path, 'w') as f:
    json.dump(daily_data, f)
print(f"\nSaved daily data to {out_path} ({os.path.getsize(out_path)/1024:.0f} KB)")

spy_path = os.path.join(RESULTS_DIR, 'v2_spy_daily.json')
with open(spy_path, 'w') as f:
    json.dump(spy_daily, f)
print(f"Saved SPY daily to {spy_path}")
