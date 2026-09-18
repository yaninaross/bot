"""
Cross-Sector Spike Rate Analysis — EXPANDED (12 sectors, 100+ tickers)
========================================================================
Monthly 8%+ spike rates across all available sectors (Jan-Jul 2026).
Answers: is the spike collapse crypto-specific or market-wide?
"""

import numpy as np
import pandas as pd
import io
import pytz
from google.cloud import storage as gcs_storage

BUCKET_NAME = os.environ.get("GCS_BUCKET", "")
PST = pytz.timezone('America/Los_Angeles')
W = 120

SECTORS = {
    'crypto': {
        'tickers': ['RIOT', 'CLSK', 'HUT', 'MARA', 'CIFR', 'IREN', 'APLD',
                     'KEEL', 'WULF', 'COIN', 'MSTR', 'CORZ', 'BTBT', 'HOOD'],
        'label': 'Crypto/Blockchain',
    },
    'defense': {
        'tickers': ['LUNR', 'RKLB', 'KTOS', 'AVAV', 'PL', 'BKSY', 'RDW',
                     'LMT', 'NOC', 'RTX', 'LHX', 'GD', 'BWXT', 'SPIR',
                     'MNTS', 'NNDM'],
        'label': 'Defense/Aero',
    },
    'nuclear': {
        'tickers': ['SMR', 'OKLO', 'CCJ', 'CEG', 'NXE', 'NBIS', 'NNE'],
        'label': 'Nuclear',
    },
    'clean_energy': {
        'tickers': ['FCEL', 'PLUG', 'BE', 'FLNC', 'ENPH', 'FSLR', 'BEPC'],
        'label': 'Clean Energy',
    },
    'oil_gas': {
        'tickers': ['XOM', 'CVX', 'COP', 'EQT', 'KMI', 'LNG', 'WMB', 'NRG', 'TPL'],
        'label': 'Oil & Gas',
    },
    'utility': {
        'tickers': ['DUK', 'NEE', 'PEG', 'VST', 'TLN', 'GEV'],
        'label': 'Utility/Power',
    },
    'space_evtol': {
        'tickers': ['ASTS', 'ACHR', 'JOBY', 'RCAT', 'BBAI', 'SOUN', 'SMLR', 'VOYG'],
        'label': 'Space/eVTOL/AI',
    },
    'semiconductor': {
        'tickers': ['NVDA', 'AMD', 'MU', 'SMCI', 'ALAB', 'ARM', 'CRDO',
                     'MXL', 'UCTT', 'WOLF', 'AEHR', 'ACMR'],
        'label': 'Semiconductor',
    },
    'tech_infra': {
        'tickers': ['PLTR', 'VRT', 'POWL', 'MOD', 'NET', 'EQIX', 'MVIS',
                     'ETN', 'GE', 'HUBB', 'ONDS', 'NXT'],
        'label': 'Tech/Infra',
    },
    'ev_auto': {
        'tickers': ['TSLA', 'RIVN', 'LCID', 'NIO', 'LI', 'XPEV', 'F', 'GM', 'VWAGY'],
        'label': 'EV/Auto',
    },
    'biotech': {
        'tickers': ['MRNA', 'BEAM', 'NTLA', 'TWST', 'SLS', 'NEOV'],
        'label': 'Biotech/Genomic',
    },
    'fintech': {
        'tickers': ['SQ', 'UPST', 'ROKU'],
        'label': 'Fintech/Consumer',
    },
    'materials': {
        'tickers': ['MP', 'CLF', 'FCX', 'BHP', 'LIN', 'CAT'],
        'label': 'Materials/Indust',
    },
    'mega_tech': {
        'tickers': ['AMZN', 'BABA'],
        'label': 'Mega-cap Tech',
    },
}

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

SPIKE_THRESHOLDS = [0.04, 0.06, 0.08, 0.10]

MONTHS = [
    ('Jan', '2026-01-01', '2026-01-31'),
    ('Feb', '2026-02-01', '2026-02-28'),
    ('Mar', '2026-03-01', '2026-03-31'),
    ('Apr', '2026-04-01', '2026-04-30'),
    ('May', '2026-05-01', '2026-05-31'),
    ('Jun', '2026-06-01', '2026-06-30'),
    ('Jul', '2026-07-01', '2026-07-16'),
]

ENTRY_START_MIN = 360
ENTRY_END_MIN = 390
EXIT_MIN = 780

gcs_client = gcs_storage.Client()
bucket = gcs_client.bucket(BUCKET_NAME)

ticker_to_sector = {}
all_tickers = []
for sec, info in SECTORS.items():
    for t in info['tickers']:
        ticker_to_sector[t] = sec
        if t not in all_tickers:
            all_tickers.append(t)

print("=" * W)
print("  CROSS-SECTOR SPIKE RATE ANALYSIS — EXPANDED (12+ sectors)")
print("=" * W)
print(f"  Sectors: {len(SECTORS)}")
for sk, sv in SECTORS.items():
    print(f"    {sv['label']:<20} {len(sv['tickers']):>2} tickers: {', '.join(sv['tickers'][:8])}{'...' if len(sv['tickers']) > 8 else ''}")
print(f"  Total unique tickers: {len(all_tickers)}")

# ==============================================================================
# LOAD & COMPUTE
# ==============================================================================
print(f"\n  Loading OHLCV data from GCS...")

all_results = []
loaded_tickers = {}
skipped = []

for ticker in sorted(all_tickers):
    gcs_tk = TICKER_FILE_MAP.get(ticker, ticker)
    blob_path = f"Data/Inference/{gcs_tk}_1m_full_historical_data.parquet"
    blob = bucket.blob(blob_path)
    if not blob.exists():
        skipped.append(ticker)
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

    is_before_exit = mins <= EXIT_MIN
    is_real = df['Volume'] > 0
    df['Safe_High'] = np.where(is_before_exit & is_real, df['High'], np.nan)
    df['date'] = df_pst.date
    df['time_min'] = mins

    future_maxes = []
    for dt, grp in df.groupby('date'):
        sh = grp['Safe_High'].values.copy()
        fm = np.full_like(sh, np.nan)
        running_max = np.nan
        for i in range(len(sh) - 1, 0, -1):
            if not np.isnan(sh[i]):
                running_max = sh[i] if np.isnan(running_max) else max(running_max, sh[i])
            fm[i - 1] = running_max
        future_maxes.append(pd.Series(fm, index=grp.index))
    df['Future_Max'] = pd.concat(future_maxes)

    df['is_entry'] = ((mins >= ENTRY_START_MIN) & (mins <= ENTRY_END_MIN) & (df['Volume'] > 0))
    entry_bars = df[df['is_entry']].copy()

    if len(entry_bars) == 0:
        skipped.append(ticker)
        continue

    close_safe = entry_bars['Close'].replace(0, np.nan)
    entry_bars['max_gain'] = ((entry_bars['Future_Max'] / close_safe) - 1).clip(lower=0)

    daily_best = entry_bars.groupby('date').agg(
        max_gain=('max_gain', 'max'),
    )
    daily_best['dt'] = pd.to_datetime(daily_best.index)

    sec = ticker_to_sector.get(ticker, 'other')
    loaded_tickers[ticker] = sec

    for m_label, m_start, m_end in MONTHS:
        m_data = daily_best[(daily_best['dt'] >= m_start) & (daily_best['dt'] <= m_end)]
        n_days = len(m_data)
        row = {'ticker': ticker, 'sector': sec, 'month': m_label, 'n_days': n_days}
        if n_days == 0:
            for thr in SPIKE_THRESHOLDS:
                row[f'spike_{int(thr*100)}_rate'] = 0
                row[f'spike_{int(thr*100)}_count'] = 0
            row['avg_max_gain'] = 0
        else:
            for thr in SPIKE_THRESHOLDS:
                spike_days = int((m_data['max_gain'] >= thr).sum())
                row[f'spike_{int(thr*100)}_rate'] = spike_days / n_days
                row[f'spike_{int(thr*100)}_count'] = spike_days
            row['avg_max_gain'] = float(m_data['max_gain'].mean())
        all_results.append(row)

    print(f"    {ticker:<6} [{sec:<14}] {len(daily_best):>5} days")

rdf = pd.DataFrame(all_results)
month_labels = [m[0] for m in MONTHS]

print(f"\n  Loaded: {len(loaded_tickers)} tickers across {len(set(loaded_tickers.values()))} sectors")
if skipped:
    print(f"  Skipped (no OHLCV): {', '.join(skipped)}")
for sec in SECTORS:
    sec_tickers = [t for t, s in loaded_tickers.items() if s == sec]
    if sec_tickers:
        print(f"    {SECTORS[sec]['label']:<20}: {len(sec_tickers):>2} — {', '.join(sorted(sec_tickers))}")


# ==============================================================================
# TABLE 1: SECTOR-LEVEL 8% SPIKE RATES BY MONTH (sorted by Jul drop)
# ==============================================================================
print(f"\n{'═' * W}")
print(f"  TABLE 1: SECTOR-LEVEL 8% SPIKE RATES — sorted by June→July change")
print(f"{'═' * W}")

sector_monthly = {}
sector_rows = []
for sec_key in SECTORS:
    sec_data = rdf[rdf['sector'] == sec_key]
    if len(sec_data) == 0:
        continue
    rates = {}
    for ml in month_labels:
        m_sec = sec_data[sec_data['month'] == ml]
        if len(m_sec) > 0 and m_sec['n_days'].sum() > 0:
            rates[ml] = m_sec['spike_8_rate'].mean()
        else:
            rates[ml] = None
    sector_monthly[sec_key] = rates
    jun = rates.get('Jun') or 0
    jul = rates.get('Jul') or 0
    sector_rows.append((sec_key, rates, jul - jun))

sector_rows.sort(key=lambda x: x[2])

header_months = ['Apr', 'May', 'Jun', 'Jul']
print(f"\n  {'Sector':<20} {'#':>2}", end="")
for ml in header_months:
    print(f" {ml:>5}", end="")
print(f"  {'Jun→Jul':>7}  {'%chg':>6}  {'Trend':>8}")
print(f"  {'─'*20} {'─'*2}", end="")
for _ in header_months:
    print(f" {'─'*5}", end="")
print(f"  {'─'*7}  {'─'*6}  {'─'*8}")

for sec_key, rates, _ in sector_rows:
    label = SECTORS[sec_key]['label']
    n_tk = len([t for t, s in loaded_tickers.items() if s == sec_key])
    print(f"  {label:<20} {n_tk:>2}", end="")

    for ml in header_months:
        r = rates.get(ml)
        if r is not None:
            print(f" {r:>4.0%}", end="")
        else:
            print(f" {'—':>5}", end="")

    jun_r = rates.get('Jun') or 0
    jul_r = rates.get('Jul') or 0
    delta = jul_r - jun_r
    pct_chg = ((jul_r / jun_r) - 1) if jun_r > 0.001 else (-1 if jul_r < jun_r else 0)

    print(f"  {delta:>+6.0%}", end="")
    print(f"  {pct_chg:>+5.0%}", end="")

    if pct_chg < -0.5:
        trend = "CRASH"
    elif pct_chg < -0.2:
        trend = "DOWN"
    elif pct_chg > 0.2:
        trend = "UP"
    elif abs(pct_chg) <= 0.2:
        trend = "FLAT"
    else:
        trend = "?"
    print(f"  {trend:>8}")

# All sectors combined
n_total = len(loaded_tickers)
print(f"  {'─'*20} {'─'*2}", end="")
for _ in header_months:
    print(f" {'─'*5}", end="")
print(f"  {'─'*7}  {'─'*6}  {'─'*8}")
print(f"  {'ALL SECTORS':<20} {n_total:>2}", end="")
for ml in header_months:
    m_all = rdf[rdf['month'] == ml]
    if len(m_all) > 0:
        print(f" {m_all['spike_8_rate'].mean():>4.0%}", end="")
    else:
        print(f" {'—':>5}", end="")
print()


# ==============================================================================
# TABLE 2: SECTOR-LEVEL AVG MAX GAIN BY MONTH
# ==============================================================================
print(f"\n{'═' * W}")
print(f"  TABLE 2: SECTOR-LEVEL AVG MAX GAIN (entry→1PM)")
print(f"{'═' * W}")

print(f"\n  {'Sector':<20} {'#':>2}", end="")
for ml in header_months:
    print(f" {ml:>5}", end="")
print(f"  {'Jun→Jul':>7}")
print(f"  {'─'*20} {'─'*2}", end="")
for _ in header_months:
    print(f" {'─'*5}", end="")
print(f"  {'─'*7}")

for sec_key, _, _ in sector_rows:
    label = SECTORS[sec_key]['label']
    sec_data = rdf[rdf['sector'] == sec_key]
    n_tk = len([t for t, s in loaded_tickers.items() if s == sec_key])

    gains = {}
    print(f"  {label:<20} {n_tk:>2}", end="")
    for ml in header_months:
        m_sec = sec_data[sec_data['month'] == ml]
        if len(m_sec) > 0 and m_sec['n_days'].sum() > 0:
            avg_g = m_sec['avg_max_gain'].mean()
            gains[ml] = avg_g
            print(f" {avg_g:>4.1%}", end="")
        else:
            print(f" {'—':>5}", end="")

    jun_g = gains.get('Jun')
    jul_g = gains.get('Jul')
    if jun_g is not None and jul_g is not None:
        print(f"  {jul_g - jun_g:>+6.1%}")
    else:
        print(f"  {'—':>7}")


# ==============================================================================
# TABLE 3: PER-TICKER DETAIL — Only sectors where crash happened
# ==============================================================================
print(f"\n{'═' * W}")
print(f"  TABLE 3: PER-TICKER 8% SPIKE RATES — by Sector (Apr-Jul)")
print(f"{'═' * W}")

for sec_key, _, sort_delta in sector_rows:
    label = SECTORS[sec_key]['label']
    sec_data = rdf[rdf['sector'] == sec_key]
    sec_tickers = sorted(sec_data['ticker'].unique())

    if len(sec_tickers) == 0:
        continue

    print(f"\n  ── {label} ({len(sec_tickers)} tickers) ──")
    print(f"  {'Ticker':<8}", end="")
    for ml in header_months:
        print(f" {ml:>5}", end="")
    print(f"  {'Jun→Jul':>7}  {'Status':>10}")
    print(f"  {'─'*8}", end="")
    for _ in header_months:
        print(f" {'─'*5}", end="")
    print(f"  {'─'*7}  {'─'*10}")

    for ticker in sec_tickers:
        tk_data = sec_data[sec_data['ticker'] == ticker]
        print(f"  {ticker:<8}", end="")

        rates = {}
        for ml in header_months:
            m_row = tk_data[tk_data['month'] == ml]
            if m_row.empty or m_row.iloc[0]['n_days'] == 0:
                print(f" {'—':>5}", end="")
            else:
                r = m_row.iloc[0]['spike_8_rate']
                rates[ml] = r
                print(f" {r:>4.0%}", end="")

        jun_r = rates.get('Jun')
        jul_r = rates.get('Jul')
        if jun_r is not None and jul_r is not None:
            delta = jul_r - jun_r
            print(f"  {delta:>+6.0%}", end="")
            if jul_r == 0:
                status = "DEAD"
            elif delta < -0.15:
                status = "CRASHING"
            elif delta < -0.05:
                status = "FALLING"
            elif delta > 0.05:
                status = "RISING"
            else:
                status = "STABLE"
        else:
            print(f"  {'—':>7}", end="")
            status = "?"
        print(f"  {status:>10}")


# ==============================================================================
# TABLE 4: SECTOR x THRESHOLD — June vs July
# ==============================================================================
print(f"\n{'═' * W}")
print(f"  TABLE 4: SECTOR x THRESHOLD — June vs July Spike Rates")
print(f"{'═' * W}")

print(f"\n  {'Sector':<20}", end="")
for thr in SPIKE_THRESHOLDS:
    pct = f"{int(thr*100)}%"
    print(f"  {pct+'_Jun':>6} {pct+'_Jul':>6} {'Δ':>5}", end="")
print()
print(f"  {'─'*20}", end="")
for _ in SPIKE_THRESHOLDS:
    print(f"  {'─'*6} {'─'*6} {'─'*5}", end="")
print()

for sec_key, _, _ in sector_rows:
    label = SECTORS[sec_key]['label']
    sec_data = rdf[rdf['sector'] == sec_key]
    jun = sec_data[sec_data['month'] == 'Jun']
    jul = sec_data[sec_data['month'] == 'Jul']

    print(f"  {label:<20}", end="")
    for thr in SPIKE_THRESHOLDS:
        key = f'spike_{int(thr*100)}_rate'
        jr = jun[key].mean() if len(jun) > 0 else 0
        jlr = jul[key].mean() if len(jul) > 0 else 0
        delta = jlr - jr
        print(f"  {jr:>5.0%} {jlr:>5.0%} {delta:>+4.0%}", end="")
    print()


# ==============================================================================
# DIAGNOSIS: HEATMAP + VERDICT
# ==============================================================================
print(f"\n{'═' * W}")
print(f"  DIAGNOSIS — Cross-sector spike collapse analysis")
print(f"{'═' * W}")

print(f"\n  SECTOR RANKING: 8% Spike Rate June → July (worst to best)")
print(f"  {'─'*80}")

rankings = []
for sec_key, rates, _ in sector_rows:
    label = SECTORS[sec_key]['label']
    jun_r = rates.get('Jun') or 0
    jul_r = rates.get('Jul') or 0
    delta = jul_r - jun_r
    pct_chg = ((jul_r / jun_r) - 1) if jun_r > 0.001 else 0
    n_tk = len([t for t, s in loaded_tickers.items() if s == sec_key])
    rankings.append((label, jun_r, jul_r, delta, pct_chg, n_tk))

for label, jun_r, jul_r, delta, pct_chg, n_tk in rankings:
    bar_len = int(abs(pct_chg) * 20)
    bar = '█' * min(bar_len, 30)
    direction = '↓' if delta < 0 else ('↑' if delta > 0 else '→')
    print(f"    {label:<20} {jun_r:>4.0%} → {jul_r:>4.0%}  ({delta:>+5.0%}, {pct_chg:>+6.0%})  {direction} {bar}  [{n_tk}tk]")

# Groupings
crashed = [(l, j, jl, d, p, n) for l, j, jl, d, p, n in rankings if p < -0.4]
down = [(l, j, jl, d, p, n) for l, j, jl, d, p, n in rankings if -0.4 <= p < -0.15]
flat_up = [(l, j, jl, d, p, n) for l, j, jl, d, p, n in rankings if p >= -0.15]

print(f"\n  CRASH ZONE (>40% drop in spike rate):")
if crashed:
    for l, j, jl, d, p, n in crashed:
        print(f"    {l:<20} dropped {p:+.0%}")
else:
    print(f"    None")

print(f"\n  MODERATE DECLINE (15-40% drop):")
if down:
    for l, j, jl, d, p, n in down:
        print(f"    {l:<20} dropped {p:+.0%}")
else:
    print(f"    None")

print(f"\n  HOLDING / RISING (<15% drop or positive):")
if flat_up:
    for l, j, jl, d, p, n in flat_up:
        print(f"    {l:<20} changed {p:+.0%}")
else:
    print(f"    None")

# Crypto vs all others
crypto_rates = sector_monthly.get('crypto', {})
crypto_jun = crypto_rates.get('Jun', 0) or 0
crypto_jul = crypto_rates.get('Jul', 0) or 0
other_jun_vals = [sector_monthly[s].get('Jun', 0) or 0 for s in SECTORS if s != 'crypto' and s in sector_monthly]
other_jul_vals = [sector_monthly[s].get('Jul', 0) or 0 for s in SECTORS if s != 'crypto' and s in sector_monthly]
oc_jun = np.mean(other_jun_vals)
oc_jul = np.mean(other_jul_vals)

crypto_drop = ((crypto_jul / crypto_jun) - 1) if crypto_jun > 0.001 else 0
other_drop = ((oc_jul / oc_jun) - 1) if oc_jun > 0.001 else 0

print(f"\n  CRYPTO vs ALL OTHER SECTORS:")
print(f"    Crypto:        {crypto_jun:>4.0%} → {crypto_jul:>4.0%}  ({crypto_drop:>+.0%})")
print(f"    All others:    {oc_jun:>4.0%} → {oc_jul:>4.0%}  ({other_drop:>+.0%})")

n_crashed = len(crashed)
n_total_sectors = len(rankings)

print(f"\n  ┌──────────────────────────────────────────────────────────────────┐")
if n_crashed >= n_total_sectors * 0.6:
    print(f"  │  VERDICT: BROAD MARKET VOLATILITY COMPRESSION                  │")
    print(f"  │  {n_crashed}/{n_total_sectors} sectors crashed >40%. This is market-wide.               │")
    print(f"  │  → VIX/vol regime filter would catch this universally          │")
elif n_crashed >= 2 and crypto_drop < other_drop:
    print(f"  │  VERDICT: CRYPTO HIT HARDER in a broad downturn               │")
    print(f"  │  Multiple sectors down, but crypto fell {crypto_drop:.0%} vs others {other_drop:.0%}    │")
    print(f"  │  → Sector rotation + vol compression both at play             │")
elif n_crashed <= 2 and crypto_drop < -0.4:
    print(f"  │  VERDICT: CRYPTO-SPECIFIC COLLAPSE                            │")
    print(f"  │  Only {n_crashed} sectors crashed — rest held up                       │")
    print(f"  │  → Diversify into sectors still showing spikes                │")
else:
    print(f"  │  VERDICT: MIXED — sector-dependent pattern                    │")
    print(f"  │  {n_crashed} sectors crashed, {len(flat_up)} holding. Not uniform.                 │")
print(f"  └──────────────────────────────────────────────────────────────────┘")

# Which sectors still have July spikes?
print(f"\n  SECTORS WITH JULY 8% SPIKE RATE > 5%:")
any_alive = False
for label, jun_r, jul_r, delta, pct_chg, n_tk in sorted(rankings, key=lambda x: -x[2]):
    if jul_r > 0.05:
        n_alive = sum(1 for _, row in rdf[(rdf['month'] == 'Jul')].iterrows()
                       if row['spike_8_rate'] > 0 and row['sector'] == [k for k, v in SECTORS.items() if v['label'] == label][0])
        print(f"    {label:<20} {jul_r:>4.0%} avg spike rate")
        any_alive = True
if not any_alive:
    print(f"    NONE — spikes are dead everywhere")

# Top individual ticker survivors
print(f"\n  TOP INDIVIDUAL TICKERS — July 8% spike rate ≥ 18%:")
jul_data = rdf[(rdf['month'] == 'Jul') & (rdf['spike_8_rate'] >= 0.18)]
if len(jul_data) > 0:
    for _, row in jul_data.sort_values('spike_8_rate', ascending=False).iterrows():
        jun_row = rdf[(rdf['ticker'] == row['ticker']) & (rdf['month'] == 'Jun')]
        jun_rate = jun_row.iloc[0]['spike_8_rate'] if len(jun_row) > 0 else 0
        sec_label = SECTORS.get(row['sector'], {}).get('label', row['sector'])
        print(f"    {row['ticker']:<8} [{sec_label:<18}]  Jul: {row['spike_8_rate']:>4.0%}  (Jun: {jun_rate:>4.0%})")
else:
    print(f"    NONE")

# Bottom — tickers that went from high to zero
print(f"\n  BIGGEST COLLAPSES — Jun ≥ 20% → Jul = 0%:")
for _, row in rdf[rdf['month'] == 'Jul'].iterrows():
    if row['spike_8_rate'] == 0:
        jun_row = rdf[(rdf['ticker'] == row['ticker']) & (rdf['month'] == 'Jun')]
        if len(jun_row) > 0 and jun_row.iloc[0]['spike_8_rate'] >= 0.20:
            sec_label = SECTORS.get(row['sector'], {}).get('label', row['sector'])
            print(f"    {row['ticker']:<8} [{sec_label:<18}]  Jun: {jun_row.iloc[0]['spike_8_rate']:>4.0%} → Jul: 0%")

print(f"\n{'═' * W}")
print(f"  ANALYSIS COMPLETE — {len(loaded_tickers)} tickers, {len(set(loaded_tickers.values()))} sectors")
print(f"{'═' * W}")
