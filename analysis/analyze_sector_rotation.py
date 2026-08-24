"""
analyze_sector_rotation.py — Sector Rotation Analysis (2-year)
==============================================================
Pure market-structure analysis: did capital rotate out of crypto mining
into other sectors during June 2026?

Downloads daily bars from Polygon for 16 sector ETFs + SPY over 2 years.
No models, no trade sims, no GCS scores, no BQ.

Tables:
  A: Absolute returns by quarter (2-year overview)
  B: Relative strength vs SPY by quarter (rotation trends)
  C: Monthly zoom Apr/May/Jun 2026 (absolute + RS)
  D: RS momentum (10d, 20d rolling rate-of-change)
  E: Cross-sector correlation matrix vs WGMI
  F: Rotation scorecard (ranked June RS shift)

Colab-ready (no if __name__).
"""

import time, gc, ctypes, sys, subprocess, warnings, pickle, os
from collections import defaultdict
from datetime import timedelta
import numpy as np, pandas as pd

warnings.filterwarnings('ignore')

def _release_memory():
    gc.collect()
    try: ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception: pass

try:
    from polygon import RESTClient
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
        "polygon-api-client", "--quiet", "--break-system-packages"])
    from polygon import RESTClient

try:
    import matplotlib
    try:
        from google.colab import auth
        _IN_COLAB = True
    except ImportError:
        matplotlib.use('Agg')
        _IN_COLAB = False
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

if _IN_COLAB:
    try:
        auth.authenticate_user()
    except Exception:
        pass


# ==============================================================================
# CONFIG
# ==============================================================================
POLYGON_API_KEY = "2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs"
CACHE_FILE = '/tmp/sector_etf_daily.pkl'

DATA_START = '2024-07-01'
DATA_END   = '2026-07-17'

SECTOR_ETFS = {
    'WGMI': 'Crypto Mining',
    'XLK':  'Technology',
    'XLF':  'Financials',
    'XLV':  'Healthcare',
    'XLE':  'Energy',
    'XLI':  'Industrials',
    'XLU':  'Utilities',
    'XLC':  'Communications',
    'XLY':  'Consumer Disc.',
    'XLP':  'Consumer Staples',
    'XLRE': 'Real Estate',
    'XLB':  'Materials',
    'SMH':  'Semiconductors',
    'ITA':  'Defense/Aero',
    'ARKK': 'Spec/Innovation',
}
BENCHMARK = 'SPY'

QUARTERS = [
    ('Q3-24', '2024-07-01', '2024-09-30'),
    ('Q4-24', '2024-10-01', '2024-12-31'),
    ('Q1-25', '2025-01-01', '2025-03-31'),
    ('Q2-25', '2025-04-01', '2025-06-30'),
    ('Q3-25', '2025-07-01', '2025-09-30'),
    ('Q4-25', '2025-10-01', '2025-12-31'),
    ('Q1-26', '2026-01-01', '2026-03-31'),
    ('Q2-26', '2026-04-01', '2026-06-30'),
    ('Jul26', '2026-07-01', '2026-07-16'),
]

MONTHS_ZOOM = [
    ('Apr', '2026-04-01', '2026-04-30'),
    ('May', '2026-05-01', '2026-05-31'),
    ('Jun', '2026-06-01', '2026-06-30'),
    ('Jul', '2026-07-01', '2026-07-16'),
]

W = 100
poly_client = RESTClient(POLYGON_API_KEY)


# ==============================================================================
# DATA LOADING
# ==============================================================================
def fetch_daily_bars(ticker, start, end, sleep_secs=2):
    all_bars = []
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    try:
        for a in poly_client.list_aggs(ticker, 1, 'day', start_dt, end_dt, limit=50000):
            all_bars.append({
                'Date': pd.to_datetime(a.timestamp, unit='ms').normalize(),
                'Open': a.open, 'High': a.high, 'Low': a.low,
                'Close': a.close, 'Volume': a.volume,
            })
    except Exception as e:
        print(f"    {ticker}: fetch failed ({e}), retrying...")
        time.sleep(10)
        try:
            for a in poly_client.list_aggs(ticker, 1, 'day', start_dt, end_dt, limit=50000):
                all_bars.append({
                    'Date': pd.to_datetime(a.timestamp, unit='ms').normalize(),
                    'Open': a.open, 'High': a.high, 'Low': a.low,
                    'Close': a.close, 'Volume': a.volume,
                })
        except Exception as e2:
            print(f"    {ticker}: FAILED — {e2}")
            return None
    if not all_bars:
        return None
    df = pd.DataFrame(all_bars).set_index('Date').sort_index()
    df = df[~df.index.duplicated(keep='last')]
    time.sleep(sleep_secs)
    return df


def period_return(series, start, end):
    chunk = series[(series.index >= start) & (series.index <= end)].dropna()
    if len(chunk) < 2:
        return np.nan
    return (chunk.iloc[-1] / chunk.iloc[0]) - 1


print("=" * W)
print(f"  SECTOR ROTATION ANALYSIS — 2-Year Study")
print(f"  {DATA_START} → {DATA_END}")
print("=" * W)

# ── Try cache first ──
closes = None
if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, 'rb') as f:
            cache = pickle.load(f)
        if cache.get('start') == DATA_START and cache.get('end') == DATA_END:
            closes = cache['closes']
            print(f"\n  Cache loaded: {len(closes)} days, {len(closes.columns)} ETFs")
    except Exception:
        pass

# ── Download from Polygon if needed ──
if closes is None:
    all_symbols = list(SECTOR_ETFS.keys()) + [BENCHMARK]
    print(f"\n  Downloading daily bars from Polygon ({len(all_symbols)} symbols)...")
    frames = {}
    for symbol in all_symbols:
        df = fetch_daily_bars(symbol, DATA_START, DATA_END)
        if df is not None:
            frames[symbol] = df['Close']
            print(f"    {symbol}: {len(df)} daily bars")
        else:
            print(f"    {symbol}: NO DATA")

    closes = pd.DataFrame(frames).sort_index().ffill()
    print(f"\n  Total: {len(closes)} trading days, {len(closes.columns)} symbols")

    with open(CACHE_FILE, 'wb') as f:
        pickle.dump({'closes': closes, 'start': DATA_START, 'end': DATA_END}, f)
    print(f"  Cached to {CACHE_FILE}")

available_etfs = [s for s in SECTOR_ETFS if s in closes.columns]
if BENCHMARK not in closes.columns:
    raise SystemExit("SPY data not available — cannot compute relative strength")

spy = closes[BENCHMARK]
daily_rets = closes.pct_change().dropna()

# ── Pre-compute RS ratios ──
rs_ratios = {}
for etf in available_etfs:
    rs_ratios[etf] = closes[etf] / spy

rs_10d = {etf: rs_ratios[etf].pct_change(10) for etf in available_etfs}
rs_20d = {etf: rs_ratios[etf].pct_change(20) for etf in available_etfs}


# ==============================================================================
# TABLE A: ABSOLUTE RETURNS BY QUARTER
# ==============================================================================
print(f"\n{'═' * W}")
print(f"  TABLE A: ABSOLUTE QUARTERLY RETURNS (%)")
print(f"{'═' * W}")

hdr = f"  {'ETF':<6} {'Sector':<16}"
for ql, _, _ in QUARTERS:
    hdr += f" {ql:>6}"
print(f"\n{hdr}")
print(f"  {'─'*6} {'─'*16}", end="")
for _ in QUARTERS:
    print(f" {'─'*6}", end="")
print()

for etf in available_etfs:
    label = SECTOR_ETFS[etf][:16]
    row = f"  {etf:<6} {label:<16}"
    for _, qs, qe in QUARTERS:
        r = period_return(closes[etf], qs, qe)
        row += f" {r:>+5.0%}" if not np.isnan(r) else f" {'—':>6}"
    print(row)

print(f"  {'─'*6} {'─'*16}", end="")
for _ in QUARTERS:
    print(f" {'─'*6}", end="")
print()
row = f"  {'SPY':<6} {'Benchmark':<16}"
for _, qs, qe in QUARTERS:
    r = period_return(spy, qs, qe)
    row += f" {r:>+5.0%}" if not np.isnan(r) else f" {'—':>6}"
print(row)


# ==============================================================================
# TABLE B: RELATIVE STRENGTH VS SPY BY QUARTER
# ==============================================================================
print(f"\n{'═' * W}")
print(f"  TABLE B: RELATIVE STRENGTH vs SPY — QUARTERLY (ETF/SPY ratio change)")
print(f"{'═' * W}")

hdr = f"  {'ETF':<6} {'Sector':<16}"
for ql, _, _ in QUARTERS:
    hdr += f" {ql:>6}"
print(f"\n{hdr}")
print(f"  {'─'*6} {'─'*16}", end="")
for _ in QUARTERS:
    print(f" {'─'*6}", end="")
print()

for etf in available_etfs:
    label = SECTOR_ETFS[etf][:16]
    row = f"  {etf:<6} {label:<16}"
    for _, qs, qe in QUARTERS:
        r = period_return(rs_ratios[etf], qs, qe)
        row += f" {r:>+5.1%}" if not np.isnan(r) else f" {'—':>6}"
    print(row)


# ==============================================================================
# TABLE C: MONTHLY ZOOM (Apr/May/Jun 2026)
# ==============================================================================
print(f"\n{'═' * W}")
print(f"  TABLE C: MONTHLY ZOOM — Apr/May/Jun 2026 (Absolute + RS)")
print(f"{'═' * W}")

print(f"\n  {'ETF':<6} {'Sector':<16}", end="")
for ml, _, _ in MONTHS_ZOOM:
    print(f"  {ml+' Abs':>8} {ml+' RS':>7}", end="")
print(f"  {'Jun-Apr':>8}")
print(f"  {'─'*6} {'─'*16}", end="")
for _ in MONTHS_ZOOM:
    print(f"  {'─'*8} {'─'*7}", end="")
print(f"  {'─'*8}")

for etf in available_etfs:
    label = SECTOR_ETFS[etf][:16]
    abs_rets = {}
    rs_rets = {}
    for ml, ms, me in MONTHS_ZOOM:
        abs_rets[ml] = period_return(closes[etf], ms, me)
        rs_rets[ml] = period_return(rs_ratios[etf], ms, me)

    jun_vs_apr_rs = (rs_rets.get('Jun', 0) or 0) - (rs_rets.get('Apr', 0) or 0)

    row = f"  {etf:<6} {label:<16}"
    for ml, _, _ in MONTHS_ZOOM:
        a = abs_rets.get(ml)
        r = rs_rets.get(ml)
        row += f"  {a:>+7.1%}" if (a is not None and not np.isnan(a)) else f"  {'—':>8}"
        row += f" {r:>+6.1%}" if (r is not None and not np.isnan(r)) else f" {'—':>7}"
    row += f"  {jun_vs_apr_rs:>+7.1%}"
    print(row)

print(f"  {'─'*6} {'─'*16}", end="")
for _ in MONTHS_ZOOM:
    print(f"  {'─'*8} {'─'*7}", end="")
print(f"  {'─'*8}")
row = f"  {'SPY':<6} {'Benchmark':<16}"
for ml, ms, me in MONTHS_ZOOM:
    a = period_return(spy, ms, me)
    row += f"  {a:>+7.1%}" if not np.isnan(a) else f"  {'—':>8}"
    row += f" {'—':>7}"
row += f"  {'':>8}"
print(row)


# ==============================================================================
# TABLE D: RS MOMENTUM (avg 10d, 20d RS RoC by month)
# ==============================================================================
print(f"\n{'═' * W}")
print(f"  TABLE D: RS MOMENTUM — Average 10d & 20d RS Rate-of-Change")
print(f"{'═' * W}")

print(f"\n  {'ETF':<6} {'Sector':<16}", end="")
for ml, _, _ in MONTHS_ZOOM:
    print(f"  {ml+' 10d':>8} {ml+' 20d':>8}", end="")
print(f"  {'Signal':>16}")
print(f"  {'─'*6} {'─'*16}", end="")
for _ in MONTHS_ZOOM:
    print(f"  {'─'*8} {'─'*8}", end="")
print(f"  {'─'*16}")

for etf in available_etfs:
    label = SECTOR_ETFS[etf][:16]
    row = f"  {etf:<6} {label:<16}"

    jun_10d_avg = np.nan
    jun_20d_avg = np.nan
    for ml, ms, me in MONTHS_ZOOM:
        r10 = rs_10d[etf][(rs_10d[etf].index >= ms) & (rs_10d[etf].index <= me)]
        r20 = rs_20d[etf][(rs_20d[etf].index >= ms) & (rs_20d[etf].index <= me)]
        avg10 = r10.mean() if len(r10) > 0 else np.nan
        avg20 = r20.mean() if len(r20) > 0 else np.nan
        row += f"  {avg10:>+7.2%}" if not np.isnan(avg10) else f"  {'—':>8}"
        row += f"  {avg20:>+7.2%}" if not np.isnan(avg20) else f"  {'—':>8}"
        if ml == 'Jun':
            jun_10d_avg = avg10
            jun_20d_avg = avg20

    if not np.isnan(jun_10d_avg) and not np.isnan(jun_20d_avg):
        if jun_10d_avg > 0.01 and jun_20d_avg > 0.01:
            sig = "Strong inflow"
        elif jun_10d_avg < -0.01 and jun_20d_avg < -0.01:
            sig = "Strong outflow"
        elif jun_10d_avg > 0 and jun_20d_avg > 0:
            sig = "Mild inflow"
        elif jun_10d_avg < 0 and jun_20d_avg < 0:
            sig = "Mild outflow"
        else:
            sig = "Mixed"
    else:
        sig = "—"
    row += f"  {sig:>16}"
    print(row)


# ==============================================================================
# TABLE E: CROSS-SECTOR CORRELATION vs WGMI
# ==============================================================================
print(f"\n{'═' * W}")
print(f"  TABLE E: DAILY RETURN CORRELATIONS vs WGMI")
print(f"{'═' * W}")

if 'WGMI' in daily_rets.columns:
    apr_may_rets = daily_rets[(daily_rets.index >= '2026-04-01') & (daily_rets.index <= '2026-05-31')]
    jun_rets = daily_rets[(daily_rets.index >= '2026-06-01') & (daily_rets.index <= '2026-06-30')]
    jul_rets = daily_rets[(daily_rets.index >= '2026-07-01') & (daily_rets.index <= '2026-07-16')]
    h2_24_rets = daily_rets[(daily_rets.index >= '2024-07-01') & (daily_rets.index <= '2024-12-31')]

    print(f"\n  {'ETF':<6} {'Sector':<16}  {'AprMay':>6}  {'Jun26':>6}  {'Jul26':>6}  {'Δ Jul-Jun':>9}  {'Δ Jul-AM':>8}  {'Trend':>16}")
    print(f"  {'─'*6} {'─'*16}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*9}  {'─'*8}  {'─'*16}")

    for etf in available_etfs:
        if etf == 'WGMI':
            continue
        label = SECTOR_ETFS[etf][:16]
        c_am = apr_may_rets[etf].corr(apr_may_rets['WGMI']) if etf in apr_may_rets.columns else np.nan
        c_jun = jun_rets[etf].corr(jun_rets['WGMI']) if etf in jun_rets.columns else np.nan
        c_jul = jul_rets[etf].corr(jul_rets['WGMI']) if etf in jul_rets.columns and len(jul_rets) > 2 else np.nan
        d_jul_jun = c_jul - c_jun if not (np.isnan(c_jul) or np.isnan(c_jun)) else np.nan
        d_jul_am = c_jul - c_am if not (np.isnan(c_jul) or np.isnan(c_am)) else np.nan

        if not np.isnan(d_jul_jun):
            if d_jul_jun > 0.15: trend = "Re-coupling"
            elif d_jul_jun < -0.15: trend = "De-coupling"
            else: trend = "Stable"
        else:
            trend = "—"

        c_am_s = f"{c_am:>+5.2f}" if not np.isnan(c_am) else f"{'—':>6}"
        c_jun_s = f"{c_jun:>+5.2f}" if not np.isnan(c_jun) else f"{'—':>6}"
        c_jul_s = f"{c_jul:>+5.2f}" if not np.isnan(c_jul) else f"{'—':>6}"
        d_jj_s = f"{d_jul_jun:>+8.2f}" if not np.isnan(d_jul_jun) else f"{'—':>9}"
        d_ja_s = f"{d_jul_am:>+7.2f}" if not np.isnan(d_jul_am) else f"{'—':>8}"
        print(f"  {etf:<6} {label:<16}  {c_am_s}  {c_jun_s}  {c_jul_s}  {d_jj_s}  {d_ja_s}  {trend:>16}")

    spy_am = apr_may_rets[BENCHMARK].corr(apr_may_rets['WGMI'])
    spy_jun = jun_rets[BENCHMARK].corr(jun_rets['WGMI'])
    spy_jul = jul_rets[BENCHMARK].corr(jul_rets['WGMI']) if len(jul_rets) > 2 else np.nan
    d_jj = spy_jul - spy_jun if not np.isnan(spy_jul) else np.nan
    d_ja = spy_jul - spy_am if not np.isnan(spy_jul) else np.nan
    trend = "Re-coupling" if (not np.isnan(d_jj) and d_jj > 0.15) else "De-coupling" if (not np.isnan(d_jj) and d_jj < -0.15) else "Stable" if not np.isnan(d_jj) else "—"
    spy_jul_s = f"{spy_jul:>+5.2f}" if not np.isnan(spy_jul) else f"{'—':>6}"
    d_jj_s = f"{d_jj:>+8.2f}" if not np.isnan(d_jj) else f"{'—':>9}"
    d_ja_s = f"{d_ja:>+7.2f}" if not np.isnan(d_ja) else f"{'—':>8}"
    print(f"  {'SPY':<6} {'Benchmark':<16}  {spy_am:>+5.2f}  {spy_jun:>+5.2f}  {spy_jul_s}  {d_jj_s}  {d_ja_s}  {trend:>16}")
else:
    print("  WGMI not available — skipping correlation table")


# ==============================================================================
# TABLE F: ROTATION SCORECARD
# ==============================================================================
print(f"\n{'═' * W}")
print(f"  TABLE F: ROTATION SCORECARD — Ranked by June RS shift")
print(f"{'═' * W}")

print(f"  TABLE F: ROTATION SCORECARD — June vs July RS comparison (ranked by July)")

scored = []
for etf in available_etfs:
    jun_rs = period_return(rs_ratios[etf], '2026-06-01', '2026-06-30')
    jul_rs = period_return(rs_ratios[etf], '2026-07-01', '2026-07-16')
    h1_26_rs = period_return(rs_ratios[etf], '2026-01-01', '2026-05-31')
    jun_abs = period_return(closes[etf], '2026-06-01', '2026-06-30')
    jul_abs = period_return(closes[etf], '2026-07-01', '2026-07-16')

    r10 = rs_10d[etf][(rs_10d[etf].index >= '2026-07-01') & (rs_10d[etf].index <= '2026-07-16')]
    avg_rs10 = r10.mean() if len(r10) > 0 else 0

    scored.append({
        'etf': etf, 'sector': SECTOR_ETFS[etf],
        'jun_rs': jun_rs if not np.isnan(jun_rs) else 0,
        'jul_rs': jul_rs if not np.isnan(jul_rs) else 0,
        'h1_26_rs': h1_26_rs if not np.isnan(h1_26_rs) else 0,
        'jun_abs': jun_abs if not np.isnan(jun_abs) else 0,
        'jul_abs': jul_abs if not np.isnan(jul_abs) else 0,
        'rs_momentum': avg_rs10,
    })

scored.sort(key=lambda x: x['jul_rs'], reverse=True)

print(f"\n  {'Rank':>4}  {'ETF':<6} {'Sector':<16}  {'Jun RS':>7}  {'Jul RS':>7}  {'Δ':>6}  "
      f"{'Jun Abs':>7}  {'Jul Abs':>7}  {'Jul Verdict':>18}")
print(f"  {'─'*4}  {'─'*6} {'─'*16}  {'─'*7}  {'─'*7}  {'─'*6}  "
      f"{'─'*7}  {'─'*7}  {'─'*18}")

for i, s in enumerate(scored):
    delta_rs = s['jul_rs'] - s['jun_rs']
    if s['jul_rs'] > 0.02:
        verdict = "ROTATION INTO"
    elif s['jul_rs'] > 0:
        verdict = "~ mild inflow"
    elif s['jul_rs'] > -0.02:
        verdict = "~ mild outflow"
    else:
        verdict = "ROTATION OUT"

    print(f"  {i+1:>4}  {s['etf']:<6} {s['sector']:<16}  {s['jun_rs']:>+6.1%}  "
          f"{s['jul_rs']:>+6.1%}  {delta_rs:>+5.1%}  "
          f"{s['jun_abs']:>+6.1%}  {s['jul_abs']:>+6.1%}  {verdict:>18}")


# ==============================================================================
# TABLE G: EARLY WARNING — When was rotation detectable?
# ==============================================================================
print(f"\n{'═' * W}")
print(f"  TABLE G: EARLY WARNING — Day-by-day RS detection through May-June")
print(f"{'═' * W}")

DETECT_ETFS = ['WGMI', 'XLV', 'XLF', 'SMH', 'ITA', 'XLI']
DETECT_ETFS = [e for e in DETECT_ETFS if e in available_etfs]

# G1: Weekly RS snapshots (10d) through May-June
print(f"\n  G1: WGMI 10d RS Rate-of-Change — Weekly Snapshots")
print(f"      (When did crypto RS momentum turn negative?)\n")

wgmi_rs10 = rs_10d.get('WGMI')
if wgmi_rs10 is not None:
    snapshot_range = wgmi_rs10[(wgmi_rs10.index >= '2026-05-01') & (wgmi_rs10.index <= '2026-07-16')]
    fridays = snapshot_range[snapshot_range.index.dayofweek == 4]
    if len(fridays) == 0:
        fridays = snapshot_range.iloc[::5]

    print(f"      {'Date':<12}  {'RS_10d':>8}  {'RS_20d':>8}  {'Signal':>20}")
    print(f"      {'─'*12}  {'─'*8}  {'─'*8}  {'─'*20}")

    first_neg_date = None
    for date in fridays.index:
        r10 = wgmi_rs10.get(date, np.nan)
        r20 = rs_20d['WGMI'].get(date, np.nan) if 'WGMI' in rs_20d else np.nan

        if not np.isnan(r10) and not np.isnan(r20):
            if r10 < 0 and r20 < 0:
                sig = "⚠ BOTH NEGATIVE"
                if first_neg_date is None:
                    first_neg_date = date
            elif r10 < 0:
                sig = "⚠ 10d negative"
                if first_neg_date is None:
                    first_neg_date = date
            elif r20 < 0:
                sig = "~ 20d negative"
            elif r10 < r20:
                sig = "↘ decelerating"
            else:
                sig = "✓ positive"
        else:
            sig = "—"

        r10_s = f"{r10:>+7.2%}" if not np.isnan(r10) else f"{'—':>8}"
        r20_s = f"{r20:>+7.2%}" if not np.isnan(r20) else f"{'—':>8}"
        marker = " ◀" if date == first_neg_date else ""
        print(f"      {date.strftime('%Y-%m-%d'):<12}  {r10_s}  {r20_s}  {sig:>20}{marker}")

    if first_neg_date:
        print(f"\n      → First negative RS signal: {first_neg_date.strftime('%Y-%m-%d')}")
    else:
        print(f"\n      → WGMI RS never went negative on weekly snapshots")

# G2: Daily RS crossover — exact dates when RS_10d crossed zero
print(f"\n  G2: RS ZERO-CROSSOVER DATES (10d RS crosses from + to -)")
print(f"      (Earliest detectable rotation signal per sector)\n")

print(f"      {'ETF':<6} {'Sector':<16}  {'Last + date':>12}  {'First - date':>13}  {'Lead time':>12}")
print(f"      {'─'*6} {'─'*16}  {'─'*12}  {'─'*13}  {'─'*12}")

jun_1 = pd.Timestamp('2026-06-01')
for etf in DETECT_ETFS:
    label = SECTOR_ETFS[etf][:16]
    r10 = rs_10d[etf]
    may_jun = r10[(r10.index >= '2026-05-01') & (r10.index <= '2026-07-16')].dropna()

    if etf == 'WGMI':
        # For WGMI: when did RS go negative (outflow starts)
        crossovers = []
        for i in range(1, len(may_jun)):
            if may_jun.iloc[i-1] > 0 and may_jun.iloc[i] <= 0:
                crossovers.append((may_jun.index[i-1], may_jun.index[i]))
        if crossovers:
            last_pos, first_neg = crossovers[-1]
            days_before_jun_end = (pd.Timestamp('2026-06-30') - first_neg).days
            print(f"      {etf:<6} {label:<16}  {last_pos.strftime('%Y-%m-%d'):>12}  "
                  f"{first_neg.strftime('%Y-%m-%d'):>13}  {days_before_jun_end:>9}d early")
        else:
            neg_vals = may_jun[may_jun < 0]
            if len(neg_vals) > 0:
                print(f"      {etf:<6} {label:<16}  {'—':>12}  "
                      f"{neg_vals.index[0].strftime('%Y-%m-%d'):>13}  {'was neg':>12}")
            else:
                print(f"      {etf:<6} {label:<16}  {'—':>12}  {'never neg':>13}  {'—':>12}")
    else:
        # For gainers: when did RS go positive (inflow starts)
        crossovers = []
        for i in range(1, len(may_jun)):
            if may_jun.iloc[i-1] <= 0 and may_jun.iloc[i] > 0:
                crossovers.append((may_jun.index[i-1], may_jun.index[i]))
        if crossovers:
            last_neg, first_pos = crossovers[-1]
            days_after_jun_1 = (first_pos - jun_1).days
            print(f"      {etf:<6} {label:<16}  {last_neg.strftime('%Y-%m-%d'):>12}  "
                  f"{first_pos.strftime('%Y-%m-%d'):>13}  {'Jun+' + str(days_after_jun_1) + 'd':>12}")
        else:
            pos_vals = may_jun[may_jun > 0]
            if len(pos_vals) > 0:
                print(f"      {etf:<6} {label:<16}  {'—':>12}  "
                      f"{pos_vals.index[0].strftime('%Y-%m-%d'):>13}  {'was pos':>12}")
            else:
                print(f"      {etf:<6} {label:<16}  {'—':>12}  {'never pos':>13}  {'—':>12}")

# G3: What would you have seen on June 1?
print(f"\n  G3: SNAPSHOT — What was visible on key dates?")
print(f"      (Could you have acted on June 1? June 5? June 10?)\n")

snapshot_dates = ['2026-05-30', '2026-06-02', '2026-06-10', '2026-06-20',
                  '2026-06-30', '2026-07-07', '2026-07-14', '2026-07-16']

print(f"      {'Date':<12}", end="")
for etf in DETECT_ETFS:
    print(f"  {etf:>7}", end="")
print()
print(f"      {'─'*12}", end="")
for _ in DETECT_ETFS:
    print(f"  {'─'*7}", end="")
print()

for date_str in snapshot_dates:
    dt = pd.Timestamp(date_str)
    closest_dates = closes.index[closes.index <= dt]
    if len(closest_dates) == 0:
        continue
    actual_date = closest_dates[-1]

    print(f"      {actual_date.strftime('%Y-%m-%d'):<12}", end="")
    for etf in DETECT_ETFS:
        r10 = rs_10d[etf]
        val = r10.get(actual_date, np.nan)
        if not np.isnan(val):
            print(f"  {val:>+6.1%}", end="")
        else:
            print(f"  {'—':>7}", end="")
    print()

print(f"\n      Key: Values are 10d RS rate-of-change (ETF/SPY ratio momentum)")
print(f"      Negative WGMI + Positive XLV/XLF = rotation underway")


# ==============================================================================
# DIAGNOSIS
# ==============================================================================
print(f"\n{'═' * W}")
print(f"  DIAGNOSIS")
print(f"{'═' * W}")

wgmi = next((s for s in scored if s['etf'] == 'WGMI'), None)
jul_gainers = [s for s in scored if s['jul_rs'] > 0.02]
jul_losers = [s for s in scored if s['jul_rs'] < -0.02]
jun_gainers = [s for s in scored if s['jun_rs'] > 0.02]
jun_losers = [s for s in scored if s['jun_rs'] < -0.02]

print(f"\n  1. CRYPTO MINING (WGMI) — JUNE vs JULY:")
if wgmi:
    delta_rs = wgmi['jul_rs'] - wgmi['jun_rs']
    print(f"     June RS vs SPY:     {wgmi['jun_rs']:>+.1%}")
    print(f"     July RS vs SPY:     {wgmi['jul_rs']:>+.1%}  (Δ {delta_rs:>+.1%})")
    print(f"     June absolute:      {wgmi['jun_abs']:>+.1%}")
    print(f"     July absolute:      {wgmi['jul_abs']:>+.1%}")
    print(f"     RS momentum (10d):  {wgmi['rs_momentum']:>+.2%}")

    if wgmi['jul_rs'] < -0.05:
        print(f"     → STRONG UNDERPERFORMANCE in July")
    elif wgmi['jul_rs'] < -0.02:
        print(f"     → UNDERPERFORMANCE continues in July")
    elif wgmi['jul_rs'] < 0:
        print(f"     → MILD weakness in July")
    else:
        print(f"     → RECOVERY: WGMI kept pace with or beat SPY in July")

    if delta_rs > 0.03:
        print(f"     → IMPROVING: July RS is better than June by {delta_rs:+.1%}")
    elif delta_rs < -0.03:
        print(f"     → WORSENING: July RS is worse than June by {delta_rs:+.1%}")
    else:
        print(f"     → SAME REGIME: July RS similar to June")
else:
    print(f"     WGMI data not available")

print(f"\n  2. SECTORS GAINING IN JULY (RS > +2%):")
if jul_gainers:
    for g in jul_gainers:
        jun_ctx = f" (Jun: {g['jun_rs']:+.1%})"
        print(f"     {g['etf']:<6} {g['sector']:<16} Jul RS: {g['jul_rs']:>+.1%}{jun_ctx}")
else:
    print(f"     None — no sector significantly outperformed SPY in July")

print(f"\n  3. SECTORS LOSING IN JULY (RS < -2%):")
if jul_losers:
    for l in jul_losers:
        jun_ctx = f" (Jun: {l['jun_rs']:+.1%})"
        print(f"     {l['etf']:<6} {l['sector']:<16} Jul RS: {l['jul_rs']:>+.1%}{jun_ctx}")
else:
    print(f"     None — no sector significantly underperformed SPY in July")

print(f"\n  4. ROTATION VERDICT — JULY vs JUNE:")
crypto_out_jul = wgmi is not None and wgmi['jul_rs'] < -0.02
crypto_out_jun = wgmi is not None and wgmi['jun_rs'] < -0.02

if crypto_out_jul and crypto_out_jun:
    print(f"     CONTINUED ROTATION: Crypto mining underperformed in BOTH June and July")
    print(f"     → This is NOT a one-month blip — it's a sustained sector headwind")
elif crypto_out_jul and not crypto_out_jun:
    print(f"     NEW ROTATION: Crypto mining was ok in June but weakened in July")
    print(f"     → Fresh sector headwind — may need to reassess sector exposure")
elif not crypto_out_jul and crypto_out_jun:
    print(f"     RECOVERY: Crypto mining underperformed June but recovering in July")
    print(f"     → If model losses continue despite sector recovery, model decay is more likely")
else:
    print(f"     NO ROTATION: Crypto mining held up in both months")
    print(f"     → Losses are NOT sector-driven — look at model/vol calibration")

spy_jun = period_return(spy, '2026-06-01', '2026-06-30')
spy_jul = period_return(spy, '2026-07-01', '2026-07-16')
print(f"\n  5. MARKET CONTEXT:")
if not np.isnan(spy_jun):
    print(f"     SPY June return:  {spy_jun:+.1%}")
if not np.isnan(spy_jul):
    print(f"     SPY July return:  {spy_jul:+.1%}")
    if spy_jul > 0.03 and wgmi and wgmi['jul_abs'] < 0:
        print(f"     → Market rallying but crypto lagging = alpha/rotation problem")
    elif spy_jul > 0.03 and wgmi and wgmi['jul_abs'] > 0:
        print(f"     → Both market and crypto up — losses are model-specific")
    elif spy_jul < -0.03:
        print(f"     → Broad market selloff — beta exposure explains losses")
    else:
        print(f"     → Market roughly flat — losses are sector/model-specific")

n_jul_losers = len(jul_losers)
n_jul_gainers = len(jul_gainers)
n_neutral = len(available_etfs) - n_jul_losers - n_jul_gainers
print(f"\n  6. BREADTH (JULY):")
print(f"     Sectors gaining (RS > +2%):  {n_jul_gainers}/{len(available_etfs)}")
print(f"     Sectors neutral:             {n_neutral}/{len(available_etfs)}")
print(f"     Sectors losing  (RS < -2%):  {n_jul_losers}/{len(available_etfs)}")

print(f"\n  7. JUNE → JULY CHANGE SUMMARY:")
print(f"     {'Sector':<20} {'Jun RS':>7} → {'Jul RS':>7}  {'Direction':>12}")
print(f"     {'─'*20} {'─'*7}   {'─'*7}  {'─'*12}")
for s in scored:
    delta = s['jul_rs'] - s['jun_rs']
    if delta > 0.02: direction = "Improving ↑"
    elif delta < -0.02: direction = "Worsening ↓"
    else: direction = "Stable →"
    print(f"     {s['sector']:<20} {s['jun_rs']:>+6.1%} → {s['jul_rs']:>+6.1%}  {direction:>12}")


# ==============================================================================
# CHARTS
# ==============================================================================
if HAS_MPL:
    print(f"\n  Generating charts...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # ── Chart 1: Normalized ETF prices (rebased to 100) ──
    ax = axes[0, 0]
    for etf in available_etfs:
        series = closes[etf].dropna()
        if len(series) < 2:
            continue
        rebased = series / series.iloc[0] * 100
        lw = 3.0 if etf == 'WGMI' else 1.2
        alpha = 1.0 if etf == 'WGMI' else 0.6
        ax.plot(rebased.index, rebased.values, label=etf, linewidth=lw, alpha=alpha)
    spy_rb = spy.dropna()
    if len(spy_rb) > 0:
        spy_rb = spy_rb / spy_rb.iloc[0] * 100
        ax.plot(spy_rb.index, spy_rb.values, label='SPY', color='black',
                linewidth=2, linestyle='--', alpha=0.8)
    ax.axvline(x=pd.Timestamp('2026-06-01'), color='red', linestyle=':', alpha=0.5, label='Jun 1')
    ax.axvline(x=pd.Timestamp('2026-07-01'), color='blue', linestyle=':', alpha=0.5, label='Jul 1')
    ax.set_title('Sector ETFs — 2-Year Normalized Price (base=100)', fontweight='bold')
    ax.legend(fontsize=6, ncol=3, loc='upper left')
    ax.grid(True, alpha=0.3)

    # ── Chart 2: RS ratios vs SPY (rebased to 100) ──
    ax = axes[0, 1]
    for etf in available_etfs:
        rs = rs_ratios[etf].dropna()
        if len(rs) < 2:
            continue
        rebased = rs / rs.iloc[0] * 100
        lw = 3.0 if etf == 'WGMI' else 1.2
        ax.plot(rebased.index, rebased.values, label=etf, linewidth=lw,
                alpha=1.0 if etf == 'WGMI' else 0.6)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.4)
    ax.axvline(x=pd.Timestamp('2026-06-01'), color='red', linestyle=':', alpha=0.5)
    ax.axvline(x=pd.Timestamp('2026-07-01'), color='blue', linestyle=':', alpha=0.5)
    ax.set_title('Relative Strength vs SPY — 2-Year (base=100)', fontweight='bold')
    ax.legend(fontsize=6, ncol=3, loc='upper left')
    ax.grid(True, alpha=0.3)

    # ── Chart 3: June RS bar chart (ranked) ──
    ax = axes[1, 0]
    etf_names = [s['etf'] for s in scored]
    jun_rs_vals = [s['jun_rs'] * 100 for s in scored]
    colors = ['#e74c3c' if v < -2 else '#27ae60' if v > 2 else '#95a5a6'
              for v in jun_rs_vals]
    bars = ax.barh(etf_names, jun_rs_vals, color=colors, alpha=0.85)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.axvline(x=-2, color='red', linewidth=0.5, linestyle='--', alpha=0.4)
    ax.axvline(x=2, color='green', linewidth=0.5, linestyle='--', alpha=0.4)
    for bar, val in zip(bars, jun_rs_vals):
        xpos = val + (0.2 if val >= 0 else -0.2)
        ax.text(xpos, bar.get_y() + bar.get_height()/2,
                f'{val:+.1f}%', va='center', fontsize=8,
                ha='left' if val >= 0 else 'right')
    ax.set_title('June 2026 — RS vs SPY by Sector (%)', fontweight='bold')
    ax.set_xlabel('Relative Strength Change (%)')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    # ── Chart 4: WGMI RS 20d rolling ──
    ax = axes[1, 1]
    if 'WGMI' in rs_20d:
        wgmi_rs = rs_20d['WGMI'].dropna()
        ax.plot(wgmi_rs.index, wgmi_rs.values * 100, color='#e74c3c', linewidth=1.5,
                label='WGMI 20d RS RoC')
        ax.fill_between(wgmi_rs.index, wgmi_rs.values * 100, 0,
                        where=wgmi_rs.values < 0, color='#e74c3c', alpha=0.15)
        ax.fill_between(wgmi_rs.index, wgmi_rs.values * 100, 0,
                        where=wgmi_rs.values > 0, color='#27ae60', alpha=0.15)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=pd.Timestamp('2026-06-01'), color='red', linestyle=':', alpha=0.5)
    ax.axvline(x=pd.Timestamp('2026-07-01'), color='blue', linestyle=':', alpha=0.5)
    ax.set_title('WGMI/SPY — 20-day RS Rate of Change (%)', fontweight='bold')
    ax.set_ylabel('RS RoC (%)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/tmp/sector_rotation_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Chart saved: /tmp/sector_rotation_analysis.png")

print(f"\n{'═' * W}")
print(f"  Analysis complete.")
print(f"{'═' * W}")
