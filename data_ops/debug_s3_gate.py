"""
Debug HUT data — compare OHLCV, master_inference, and live Polygon for today.
"""
import pandas as pd
import pytz
from polygon import RESTClient

PST = pytz.timezone('America/Los_Angeles')
BUCKET = os.environ.get("GCS_BUCKET", "")
POLYGON_API_KEY = os.environ["POLYGON_API_KEY"]

today_utc = pd.Timestamp.now(tz='UTC').normalize()
today_str = today_utc.strftime('%Y-%m-%d')
print(f"Today: {today_str}\n")

# ── 1. OHLCV parquet on GCS ──
print("=" * 70)
print("1. OHLCV (GCS parquet)")
print("=" * 70)
ohlcv_uri = f"gs://{BUCKET}/Data/Inference/HUT_1m_full_historical_data.parquet"
df_ohlcv = pd.read_parquet(ohlcv_uri)
if df_ohlcv.index.tz is None:
    df_ohlcv.index = df_ohlcv.index.tz_localize('UTC')
print(f"Total rows: {len(df_ohlcv):,}")
print(f"Last bar:   {df_ohlcv.index[-1]}")

ohlcv_today = df_ohlcv[df_ohlcv.index >= today_utc]
print(f"Today rows: {len(ohlcv_today)}")
if len(ohlcv_today) > 0:
    ohlcv_today_pst = ohlcv_today.copy()
    ohlcv_today_pst['min_pst'] = ohlcv_today_pst.index.tz_convert(PST).hour * 60 + ohlcv_today_pst.index.tz_convert(PST).minute
    window = ohlcv_today_pst[(ohlcv_today_pst['min_pst'] >= 360) & (ohlcv_today_pst['min_pst'] <= 390)]
    if len(window) > 0:
        print(f"  Bars 360-390:")
        for _, row in window.iterrows():
            ts = row.name.tz_convert(PST)
            print(f"    min={row['min_pst']}  {ts.strftime('%H:%M')}  V={row['Volume']:.0f}  C={row['Close']:.2f}")
    else:
        print("  No bars in 360-390 window")
else:
    last5 = df_ohlcv.tail(5)
    print(f"  Last 5 bars:")
    for ts, row in last5.iterrows():
        print(f"    {ts}  V={row['Volume']:.0f}  C={row['Close']:.2f}")

# ── 2. Master inference on GCS ──
print("\n" + "=" * 70)
print("2. MASTER INFERENCE (GCS parquet)")
print("=" * 70)
master_uri = f"gs://{BUCKET}/Data/Inference_v2/HUT_master_inference.parquet"
df_master = pd.read_parquet(master_uri)
df_master = df_master.reset_index()
dt_cols = [c for c in df_master.columns if pd.api.types.is_datetime64_any_dtype(df_master[c])]
if dt_cols:
    df_master['Datetime'] = pd.to_datetime(df_master[dt_cols[0]], utc=True)
else:
    df_master['Datetime'] = pd.to_datetime(df_master[df_master.columns[0]], unit='ms', utc=True)
df_master = df_master.set_index('Datetime').sort_index()
print(f"Total rows: {len(df_master):,}")
print(f"Last bar:   {df_master.index[-1]}")

master_today = df_master[df_master.index >= today_utc]
print(f"Today rows: {len(master_today)}")
if len(master_today) > 0:
    master_today_c = master_today.copy()
    master_today_c['min_pst'] = master_today_c.index.tz_convert(PST).hour * 60 + master_today_c.index.tz_convert(PST).minute
    window = master_today_c[(master_today_c['min_pst'] >= 360) & (master_today_c['min_pst'] <= 390)]
    if len(window) > 0:
        vol_col = 'Volume' if 'Volume' in window.columns else 'volume'
        close_col = 'Close' if 'Close' in window.columns else 'close'
        print(f"  Bars 360-390:")
        for _, row in window.iterrows():
            ts = row.name.tz_convert(PST)
            print(f"    min={row['min_pst']}  {ts.strftime('%H:%M')}  V={row[vol_col]:.0f}  C={row[close_col]:.2f}")
else:
    last5 = df_master.tail(5)
    print(f"  Last 5 bars:")
    for ts, row in last5.iterrows():
        vol_col = 'Volume' if 'Volume' in last5.columns else 'volume'
        close_col = 'Close' if 'Close' in last5.columns else 'close'
        print(f"    {ts}  V={row[vol_col]:.0f}  C={row[close_col]:.2f}")

# ── 3. Live Polygon API ──
print("\n" + "=" * 70)
print("3. POLYGON API (live)")
print("=" * 70)
poly = RESTClient(POLYGON_API_KEY)
aggs = []
try:
    for a in poly.list_aggs('HUT', 1, 'minute', today_str, today_str, limit=50000):
        aggs.append({'timestamp': pd.to_datetime(a.timestamp, unit='ms', utc=True),
                     'Open': a.open, 'High': a.high, 'Low': a.low,
                     'Close': a.close, 'Volume': a.volume})
except Exception as e:
    print(f"  API error: {e}")

if aggs:
    df_poly = pd.DataFrame(aggs).set_index('timestamp').sort_index()
    df_poly['min_pst'] = df_poly.index.tz_convert(PST).hour * 60 + df_poly.index.tz_convert(PST).minute
    print(f"Total bars from Polygon today: {len(df_poly)}")
    print(f"First: {df_poly.index[0]}  Last: {df_poly.index[-1]}")
    window = df_poly[(df_poly['min_pst'] >= 360) & (df_poly['min_pst'] <= 390)]
    print(f"\n  Bars 360-390:")
    if len(window) > 0:
        for _, row in window.iterrows():
            ts = row.name.tz_convert(PST)
            print(f"    min={row['min_pst']}  {ts.strftime('%H:%M')}  V={row['Volume']:.0f}  C={row['Close']:.2f}")
    else:
        print("    (none)")
    print(f"\n  Bar at minute 375 specifically:")
    at375 = df_poly[df_poly['min_pst'] == 375]
    if len(at375) > 0:
        for _, row in at375.iterrows():
            print(f"    V={row['Volume']:.0f}  C={row['Close']:.2f}")
    else:
        print("    NOT PRESENT in Polygon data")
else:
    print("  No bars returned from Polygon for today")

# ── 4. Gap analysis ──
print("\n" + "=" * 70)
print("4. GAP ANALYSIS")
print("=" * 70)
ohlcv_last = df_ohlcv.index[-1]
master_last = df_master.index[-1]
print(f"OHLCV last bar:   {ohlcv_last}  ({ohlcv_last.tz_convert(PST).strftime('%Y-%m-%d %H:%M PST')})")
print(f"Master last bar:  {master_last}  ({master_last.tz_convert(PST).strftime('%Y-%m-%d %H:%M PST')})")
if aggs:
    print(f"Polygon last bar: {df_poly.index[-1]}  ({df_poly.index[-1].tz_convert(PST).strftime('%Y-%m-%d %H:%M PST')})")
    gap_ohlcv = df_poly.index[-1] - ohlcv_last
    gap_master = df_poly.index[-1] - master_last
    print(f"\nOHLCV gap:  {gap_ohlcv}")
    print(f"Master gap: {gap_master}")
    if gap_ohlcv > pd.Timedelta(hours=1):
        print("  → OHLCV pipeline has NOT run today — data is stale")
    if gap_master > gap_ohlcv:
        print("  → Stitch has NOT run after last OHLCV update")
