"""
July Spike Rate Analysis — Are spikes disappearing?
=====================================================
Monthly spike rates for deployed S3 tickers (Jan-Jul 2026).
Same spike definition as training: Future_Max from entry bar to 1 PM PST.

Answers: is the model calling false spikes, or are there no spikes to call?
"""

import numpy as np
import pandas as pd
import io
import pytz
from google.cloud import storage as gcs_storage

BUCKET_NAME = os.environ.get("GCS_BUCKET", "")
PST = pytz.timezone('America/Los_Angeles')
W = 100

DEPLOYED_TICKERS = ['APLD', 'ASTS', 'CIFR', 'CLSK', 'FLNC', 'HUT', 'KEEL', 'PL', 'RCAT', 'RIOT']
EXTRA_CONTEXT = ['MARA', 'MSTR', 'COIN', 'IREN', 'WULF']
ALL_TICKERS = DEPLOYED_TICKERS + EXTRA_CONTEXT

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

SPIKE_THRESHOLDS = [0.04, 0.06, 0.08, 0.10, 0.12, 0.16]

MONTHS = [
    ('Jan', '2026-01-01', '2026-01-31'),
    ('Feb', '2026-02-01', '2026-02-28'),
    ('Mar', '2026-03-01', '2026-03-31'),
    ('Apr', '2026-04-01', '2026-04-30'),
    ('May', '2026-05-01', '2026-05-31'),
    ('Jun', '2026-06-01', '2026-06-30'),
    ('Jul', '2026-07-01', '2026-07-16'),
]

ENTRY_START_MIN = 360  # 6:00 AM PST
ENTRY_END_MIN = 390    # 6:30 AM PST
EXIT_MIN = 780         # 1:00 PM PST (13:00)

gcs_client = gcs_storage.Client()
bucket = gcs_client.bucket(BUCKET_NAME)

print("=" * W)
print("  SPIKE RATE ANALYSIS — Are spikes disappearing in July?")
print("=" * W)

# ==============================================================================
# LOAD OHLCV & COMPUTE SPIKES
# ==============================================================================
print(f"\n  Loading OHLCV data for {len(ALL_TICKERS)} tickers...")

all_results = []

for ticker in ALL_TICKERS:
    gcs_tk = TICKER_FILE_MAP.get(ticker, ticker)
    blob_path = f"Data/Inference/{gcs_tk}_1m_full_historical_data.parquet"
    blob = bucket.blob(blob_path)
    if not blob.exists():
        print(f"    {ticker}: NOT FOUND")
        continue

    df = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df = df.sort_index()

    for c in ['Open', 'High', 'Low', 'Close']:
        if c in df.columns:
            df[c] = df[c].ffill()
    df['Volume'] = df.get('Volume', pd.Series(0, index=df.index)).fillna(0)

    df_pst = df.index.tz_convert(PST)
    mins = df_pst.hour * 60 + df_pst.minute

    # Future_Max: max High from next bar through 1:00 PM PST, only real bars
    is_before_exit = mins <= EXIT_MIN
    is_real = df['Volume'] > 0
    df['Safe_High'] = np.where(is_before_exit & is_real, df['High'], np.nan)

    dates = df_pst.date
    df['date'] = dates
    df['time_min'] = mins

    # Compute Future_Max per day: reverse cummax of Safe_High shifted by 1
    future_maxes = []
    for dt, grp in df.groupby('date'):
        sh = grp['Safe_High'].values.copy()
        fm = np.full_like(sh, np.nan)
        # For each bar, future max = max of Safe_High from next bar onward
        running_max = np.nan
        for i in range(len(sh) - 1, 0, -1):
            if not np.isnan(sh[i]):
                running_max = sh[i] if np.isnan(running_max) else max(running_max, sh[i])
            fm[i - 1] = running_max
        future_maxes.append(pd.Series(fm, index=grp.index))

    df['Future_Max'] = pd.concat(future_maxes)

    # Valid entry bars: 6:00-6:30 AM PST with Volume > 0
    df['is_entry'] = ((mins >= ENTRY_START_MIN) & (mins <= ENTRY_END_MIN) & (df['Volume'] > 0))
    entry_bars = df[df['is_entry']].copy()

    if len(entry_bars) == 0:
        print(f"    {ticker}: no valid entry bars")
        continue

    close_safe = entry_bars['Close'].replace(0, np.nan)
    entry_bars['max_gain'] = ((entry_bars['Future_Max'] / close_safe) - 1).clip(lower=0)

    # Compute daily max_gain (best entry bar per day)
    daily_best = entry_bars.groupby('date').agg(
        max_gain=('max_gain', 'max'),
        close=('Close', 'first'),
    )

    # Monthly spike rates
    daily_best['month'] = pd.to_datetime(daily_best.index).month
    daily_best['year'] = pd.to_datetime(daily_best.index).year
    daily_best['dt'] = pd.to_datetime(daily_best.index)

    for m_label, m_start, m_end in MONTHS:
        m_data = daily_best[(daily_best['dt'] >= m_start) & (daily_best['dt'] <= m_end)]
        n_days = len(m_data)
        if n_days == 0:
            row = {'ticker': ticker, 'month': m_label, 'n_days': 0}
            for thr in SPIKE_THRESHOLDS:
                key = f"spike_{int(thr*100)}"
                row[f'{key}_count'] = 0
                row[f'{key}_rate'] = 0
            row['avg_max_gain'] = 0
            row['med_max_gain'] = 0
            row['p90_max_gain'] = 0
            all_results.append(row)
            continue

        row = {
            'ticker': ticker,
            'month': m_label,
            'n_days': n_days,
            'avg_max_gain': float(m_data['max_gain'].mean()),
            'med_max_gain': float(m_data['max_gain'].median()),
            'p90_max_gain': float(m_data['max_gain'].quantile(0.90)) if n_days >= 3 else 0,
        }
        for thr in SPIKE_THRESHOLDS:
            key = f"spike_{int(thr*100)}"
            spike_days = int((m_data['max_gain'] >= thr).sum())
            row[f'{key}_count'] = spike_days
            row[f'{key}_rate'] = spike_days / n_days
        all_results.append(row)

    is_deployed = ticker in DEPLOYED_TICKERS
    tag = " [S3]" if is_deployed else ""
    print(f"    {ticker}{tag}: {len(entry_bars)} entry bars, {len(daily_best)} trading days")

rdf = pd.DataFrame(all_results)

# ==============================================================================
# TABLE 1: MONTHLY SPIKE RATES AT 8% THRESHOLD (deployed tickers)
# ==============================================================================
print(f"\n{'═' * W}")
print(f"  TABLE 1: MONTHLY 8% SPIKE RATES — Deployed S3 Tickers")
print(f"  (Rate = fraction of trading days where entry→1PM max gain >= 8%)")
print(f"{'═' * W}")

deployed_df = rdf[rdf['ticker'].isin(DEPLOYED_TICKERS)]
month_labels = [m[0] for m in MONTHS]

print(f"\n  {'Ticker':<8}", end="")
for ml in month_labels:
    print(f" {ml:>6}", end="")
print(f"  {'Apr→Jun':>7}  {'Jun→Jul':>7}  {'Trend':>8}")
print(f"  {'─'*8}", end="")
for _ in month_labels:
    print(f" {'─'*6}", end="")
print(f"  {'─'*7}  {'─'*7}  {'─'*8}")

ticker_trends = {}
for ticker in DEPLOYED_TICKERS:
    tk_data = deployed_df[deployed_df['ticker'] == ticker]
    print(f"  {ticker:<8}", end="")

    rates = {}
    for ml in month_labels:
        m_row = tk_data[tk_data['month'] == ml]
        if m_row.empty or m_row.iloc[0]['n_days'] == 0:
            print(f" {'—':>6}", end="")
            rates[ml] = None
        else:
            r = m_row.iloc[0]['spike_8_rate']
            n = m_row.iloc[0]['n_days']
            print(f" {r:>5.0%}", end="")
            rates[ml] = r

    apr_r = rates.get('Apr')
    jun_r = rates.get('Jun')
    jul_r = rates.get('Jul')

    if apr_r is not None and jun_r is not None:
        d1 = jun_r - apr_r
        print(f"  {d1:>+6.0%}", end="")
    else:
        print(f"  {'—':>7}", end="")

    if jun_r is not None and jul_r is not None:
        d2 = jul_r - jun_r
        print(f"  {d2:>+6.0%}", end="")
        if jul_r == 0 and jun_r == 0:
            trend = "DEAD"
        elif d2 < -0.10:
            trend = "CRASHING"
        elif d2 < -0.03:
            trend = "FALLING"
        elif d2 > 0.05:
            trend = "RISING"
        else:
            trend = "STABLE"
    else:
        print(f"  {'—':>7}", end="")
        trend = "?"

    ticker_trends[ticker] = trend
    print(f"  {trend:>8}")

# Portfolio average
print(f"  {'─'*8}", end="")
for _ in month_labels:
    print(f" {'─'*6}", end="")
print(f"  {'─'*7}  {'─'*7}  {'─'*8}")
print(f"  {'AVG':<8}", end="")
for ml in month_labels:
    m_data = deployed_df[deployed_df['month'] == ml]
    if len(m_data) > 0 and m_data['n_days'].sum() > 0:
        avg_r = m_data['spike_8_rate'].mean()
        print(f" {avg_r:>5.0%}", end="")
    else:
        print(f" {'—':>6}", end="")
print()


# ==============================================================================
# TABLE 2: MULTI-THRESHOLD VIEW (June vs July)
# ==============================================================================
print(f"\n{'═' * W}")
print(f"  TABLE 2: JUNE vs JULY — Spike Rates at Multiple Thresholds")
print(f"{'═' * W}")

print(f"\n  {'Ticker':<8}", end="")
for thr in SPIKE_THRESHOLDS:
    print(f"  {'Jun':>4} {'Jul':>4} {'Δ':>4}", end="")
print(f"  {'Jun avg':>7} {'Jul avg':>7}")
print(f"  {'─'*8}", end="")
for _ in SPIKE_THRESHOLDS:
    print(f"  {'─'*4} {'─'*4} {'─'*4}", end="")
print(f"  {'─'*7} {'─'*7}")

for ticker in DEPLOYED_TICKERS:
    tk_data = deployed_df[deployed_df['ticker'] == ticker]
    jun = tk_data[tk_data['month'] == 'Jun']
    jul = tk_data[tk_data['month'] == 'Jul']

    print(f"  {ticker:<8}", end="")
    for thr in SPIKE_THRESHOLDS:
        key = f"spike_{int(thr*100)}_rate"
        jr = jun.iloc[0][key] if not jun.empty else 0
        jlr = jul.iloc[0][key] if not jul.empty else 0
        delta = jlr - jr
        print(f"  {jr:>3.0%} {jlr:>3.0%} {delta:>+3.0%}", end="")

    jun_avg = jun.iloc[0]['avg_max_gain'] if not jun.empty else 0
    jul_avg = jul.iloc[0]['avg_max_gain'] if not jul.empty else 0
    print(f"  {jun_avg:>6.1%} {jul_avg:>6.1%}")


# ==============================================================================
# TABLE 3: AVERAGE MAX GAIN (monthly, shows spike magnitude trend)
# ==============================================================================
print(f"\n{'═' * W}")
print(f"  TABLE 3: AVERAGE MAX GAIN — Monthly (entry→1PM, best bar)")
print(f"  (How big are the spikes, not just how often)")
print(f"{'═' * W}")

print(f"\n  {'Ticker':<8}", end="")
for ml in month_labels:
    print(f" {ml:>6}", end="")
print(f"  {'Apr→Jul':>7}")
print(f"  {'─'*8}", end="")
for _ in month_labels:
    print(f" {'─'*6}", end="")
print(f"  {'─'*7}")

for ticker in DEPLOYED_TICKERS:
    tk_data = deployed_df[deployed_df['ticker'] == ticker]
    print(f"  {ticker:<8}", end="")

    gains = {}
    for ml in month_labels:
        m_row = tk_data[tk_data['month'] == ml]
        if m_row.empty or m_row.iloc[0]['n_days'] == 0:
            print(f" {'—':>6}", end="")
        else:
            g = m_row.iloc[0]['avg_max_gain']
            gains[ml] = g
            print(f" {g:>5.1%}", end="")

    apr_g = gains.get('Apr', None)
    jul_g = gains.get('Jul', None)
    if apr_g is not None and jul_g is not None:
        print(f"  {jul_g - apr_g:>+6.1%}")
    else:
        print(f"  {'—':>7}")

# Portfolio average
print(f"  {'─'*8}", end="")
for _ in month_labels:
    print(f" {'─'*6}", end="")
print(f"  {'─'*7}")
print(f"  {'AVG':<8}", end="")
for ml in month_labels:
    m_data = deployed_df[deployed_df['month'] == ml]
    if len(m_data) > 0:
        avg = m_data['avg_max_gain'].mean()
        print(f" {avg:>5.1%}", end="")
    else:
        print(f" {'—':>6}", end="")
print()


# ==============================================================================
# TABLE 4: CONTEXT TICKERS (MARA, MSTR, COIN, etc)
# ==============================================================================
print(f"\n{'═' * W}")
print(f"  TABLE 4: CONTEXT TICKERS — 8% Spike Rates (not deployed)")
print(f"{'═' * W}")

context_df = rdf[rdf['ticker'].isin(EXTRA_CONTEXT)]

print(f"\n  {'Ticker':<8}", end="")
for ml in month_labels:
    print(f" {ml:>6}", end="")
print()
print(f"  {'─'*8}", end="")
for _ in month_labels:
    print(f" {'─'*6}", end="")
print()

for ticker in EXTRA_CONTEXT:
    tk_data = context_df[context_df['ticker'] == ticker]
    print(f"  {ticker:<8}", end="")
    for ml in month_labels:
        m_row = tk_data[tk_data['month'] == ml]
        if m_row.empty or m_row.iloc[0]['n_days'] == 0:
            print(f" {'—':>6}", end="")
        else:
            r = m_row.iloc[0]['spike_8_rate']
            print(f" {r:>5.0%}", end="")
    print()


# ==============================================================================
# DIAGNOSIS
# ==============================================================================
print(f"\n{'═' * W}")
print(f"  DIAGNOSIS — Spike Rate vs Model Performance")
print(f"{'═' * W}")

apr_rates = deployed_df[deployed_df['month'] == 'Apr']['spike_8_rate']
may_rates = deployed_df[deployed_df['month'] == 'May']['spike_8_rate']
jun_rates = deployed_df[deployed_df['month'] == 'Jun']['spike_8_rate']
jul_rates = deployed_df[deployed_df['month'] == 'Jul']['spike_8_rate']

apr_avg = apr_rates.mean() if len(apr_rates) > 0 else 0
may_avg = may_rates.mean() if len(may_rates) > 0 else 0
jun_avg = jun_rates.mean() if len(jun_rates) > 0 else 0
jul_avg = jul_rates.mean() if len(jul_rates) > 0 else 0

print(f"\n  Portfolio avg 8% spike rate:")
print(f"    April:  {apr_avg:.1%}")
print(f"    May:    {may_avg:.1%}")
print(f"    June:   {jun_avg:.1%}")
print(f"    July:   {jul_avg:.1%}")

if jul_avg < 0.03:
    print(f"\n  → SPIKE DROUGHT: Portfolio avg below 3% — there are almost no 8%+ spikes to catch")
    print(f"    Models trained on ~{apr_avg:.0%} spike rate can't perform when rate drops to {jul_avg:.0%}")
elif jul_avg < jun_avg * 0.7:
    print(f"\n  → CONTINUED DECLINE: July spike rate is {(1 - jul_avg/jun_avg):.0%} lower than June")
    print(f"    The phenomenon models predict is actively disappearing")
elif jul_avg > jun_avg * 1.3:
    print(f"\n  → RECOVERY: July spike rate is higher than June")
    print(f"    If model still losing despite more spikes, this is a model quality issue")
else:
    print(f"\n  → STABLE: July spike rate similar to June")
    print(f"    Losses are not purely due to spike disappearance — model calibration matters")

dead_tickers = [t for t, tr in ticker_trends.items() if tr in ('DEAD', 'CRASHING')]
if dead_tickers:
    print(f"\n  TICKERS WITH COLLAPSING SPIKE RATES: {', '.join(dead_tickers)}")
    print(f"  → Consider removing from deployment — no spikes = no edge")

print(f"\n{'═' * W}")
print(f"  ANALYSIS COMPLETE")
print(f"{'═' * W}")
