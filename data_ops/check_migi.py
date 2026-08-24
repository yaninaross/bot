"""Quick check: pull latest MIGI OHLCV from Polygon to see what's happening."""
import pandas as pd
from polygon import RESTClient

poly = RESTClient("2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs")

end = pd.Timestamp.now(tz='UTC')
start = end - pd.Timedelta(days=14)

print(f"Fetching MIGI 1-min bars: {start.date()} → {end.date()}")

bars = []
for a in poly.list_aggs("MIGI", 1, "minute", start, end, limit=50000):
    bars.append({
        'timestamp': pd.to_datetime(a.timestamp, unit='ms', utc=True),
        'Open': a.open, 'High': a.high, 'Low': a.low,
        'Close': a.close, 'Volume': a.volume,
    })

if not bars:
    print("NO BARS RETURNED — ticker may be halted, delisted, or symbol changed")
else:
    df = pd.DataFrame(bars).set_index('timestamp').sort_index()
    print(f"\nTotal bars: {len(df)}")
    print(f"First bar:  {df.index[0]}")
    print(f"Last bar:   {df.index[-1]}")
    print(f"\nLast 10 bars:")
    print(df.tail(10).to_string())

    dates = df.index.normalize().unique()
    print(f"\nTrading days with data: {len(dates)}")
    for d in dates[-7:]:
        day_bars = df[df.index.normalize() == d]
        print(f"  {d.date()}: {len(day_bars)} bars, vol={day_bars['Volume'].sum():,.0f}")
