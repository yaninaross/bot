"""
Exp 400 — Downside Spike (Dump) Target Analysis
=================================================
Computes Future_Min (mirror of Future_Max) to analyze downside spike
frequency, magnitude, and distribution across all GCS tickers.
Goal: determine the right binary target threshold for Exp 400 short-side model.
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

DOWN_THRESHOLDS = [0.04, 0.06, 0.08, 0.10, 0.12, 0.15]
UP_THRESHOLDS   = [0.04, 0.06, 0.08, 0.10]

MONTHS = [
    ('Jan', '2026-01-01', '2026-01-31'),
    ('Feb', '2026-02-01', '2026-02-28'),
    ('Mar', '2026-03-01', '2026-03-31'),
    ('Apr', '2026-04-01', '2026-04-30'),
    ('May', '2026-05-01', '2026-05-31'),
    ('Jun', '2026-06-01', '2026-06-30'),
    ('Jul', '2026-07-01', '2026-07-19'),
]

FOCUS_MONTHS = ['May', 'Jun', 'Jul']

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
print("  EXP 400 — DOWNSIDE SPIKE (DUMP) TARGET ANALYSIS")
print("=" * W)
print(f"  Sectors: {len(SECTORS)}")
for sk, sv in SECTORS.items():
    print(f"    {sv['label']:<20} {len(sv['tickers']):>2} tickers: {', '.join(sv['tickers'][:8])}{'...' if len(sv['tickers']) > 8 else ''}")
print(f"  Total unique tickers: {len(all_tickers)}")
print(f"  Down thresholds: {', '.join(f'{t:.0%}' for t in DOWN_THRESHOLDS)}")
print(f"  Up thresholds (comparison): {', '.join(f'{t:.0%}' for t in UP_THRESHOLDS)}")

# ==============================================================================
# LOAD & COMPUTE — Future_Min (downside) AND Future_Max (upside)
# ==============================================================================
print(f"\n  Loading OHLCV data from GCS...")

all_results = []
daily_drop_distributions = []
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
    df['Safe_Low']  = np.where(is_before_exit & is_real, df['Low'],  np.nan)
    df['date'] = df_pst.date
    df['time_min'] = mins

    future_maxes = []
    future_mins  = []
    for dt, grp in df.groupby('date'):
        sh = grp['Safe_High'].values.copy()
        sl = grp['Safe_Low'].values.copy()
        fm_hi = np.full_like(sh, np.nan)
        fm_lo = np.full_like(sl, np.nan)

        running_max = np.nan
        running_min = np.nan
        for i in range(len(sh) - 1, 0, -1):
            if not np.isnan(sh[i]):
                running_max = sh[i] if np.isnan(running_max) else max(running_max, sh[i])
            if not np.isnan(sl[i]):
                running_min = sl[i] if np.isnan(running_min) else min(running_min, sl[i])
            fm_hi[i - 1] = running_max
            fm_lo[i - 1] = running_min

        future_maxes.append(pd.Series(fm_hi, index=grp.index))
        future_mins.append(pd.Series(fm_lo, index=grp.index))

    df['Future_Max'] = pd.concat(future_maxes)
    df['Future_Min'] = pd.concat(future_mins)

    df['is_entry'] = ((mins >= ENTRY_START_MIN) & (mins <= ENTRY_END_MIN) & (df['Volume'] > 0))
    entry_bars = df[df['is_entry']].copy()

    if len(entry_bars) == 0:
        skipped.append(ticker)
        continue

    close_safe = entry_bars['Close'].replace(0, np.nan)
    entry_bars['max_gain'] = ((entry_bars['Future_Max'] / close_safe) - 1).clip(lower=0)
    entry_bars['max_drop'] = (1 - (entry_bars['Future_Min'] / close_safe)).clip(lower=0)

    daily_best = entry_bars.groupby('date').agg(
        max_gain=('max_gain', 'max'),
        max_drop=('max_drop', 'max'),
    )
    daily_best['dt'] = pd.to_datetime(daily_best.index)

    sec = ticker_to_sector.get(ticker, 'other')
    loaded_tickers[ticker] = sec

    for _, drow in daily_best.iterrows():
        daily_drop_distributions.append({
            'ticker': ticker, 'sector': sec, 'date': drow['dt'],
            'max_drop': drow['max_drop'], 'max_gain': drow['max_gain'],
        })

    for m_label, m_start, m_end in MONTHS:
        m_data = daily_best[(daily_best['dt'] >= m_start) & (daily_best['dt'] <= m_end)]
        n_days = len(m_data)
        row = {'ticker': ticker, 'sector': sec, 'month': m_label, 'n_days': n_days}
        if n_days == 0:
            for thr in DOWN_THRESHOLDS:
                row[f'dump_{int(thr*100)}_rate'] = 0
                row[f'dump_{int(thr*100)}_count'] = 0
            for thr in UP_THRESHOLDS:
                row[f'spike_{int(thr*100)}_rate'] = 0
            row['avg_max_drop'] = 0
            row['avg_max_gain'] = 0
        else:
            for thr in DOWN_THRESHOLDS:
                dump_days = int((m_data['max_drop'] >= thr).sum())
                row[f'dump_{int(thr*100)}_rate'] = dump_days / n_days
                row[f'dump_{int(thr*100)}_count'] = dump_days
            for thr in UP_THRESHOLDS:
                spike_days = int((m_data['max_gain'] >= thr).sum())
                row[f'spike_{int(thr*100)}_rate'] = spike_days / n_days
            row['avg_max_drop'] = float(m_data['max_drop'].mean())
            row['avg_max_gain'] = float(m_data['max_gain'].mean())
        all_results.append(row)

    print(f"    {ticker:<6} [{sec:<14}] {len(daily_best):>5} days  "
          f"avg_drop={daily_best['max_drop'].mean():.1%}  avg_gain={daily_best['max_gain'].mean():.1%}")

rdf = pd.DataFrame(all_results)
dist_df = pd.DataFrame(daily_drop_distributions)
month_labels = [m[0] for m in MONTHS]

print(f"\n  Loaded: {len(loaded_tickers)} tickers across {len(set(loaded_tickers.values()))} sectors")
if skipped:
    print(f"  Skipped (no OHLCV): {', '.join(skipped)}")


# ==============================================================================
# TABLE 1: DOWNSIDE DUMP RATES BY SECTOR × MONTH (at all thresholds)
# ==============================================================================
print(f"\n{'═' * W}")
print(f"  TABLE 1: SECTOR-LEVEL DUMP RATES BY MONTH (% of days with drop ≥ threshold)")
print(f"{'═' * W}")

for thr in DOWN_THRESHOLDS:
    key = f'dump_{int(thr*100)}_rate'
    pct = f"{int(thr*100)}%"
    print(f"\n  ── {pct} DOWNSIDE THRESHOLD ──")
    print(f"  {'Sector':<20} {'#':>2}", end="")
    for ml in FOCUS_MONTHS:
        print(f" {ml:>5}", end="")
    print(f"  {'May→Jul':>7}")
    print(f"  {'─'*20} {'─'*2}", end="")
    for _ in FOCUS_MONTHS:
        print(f" {'─'*5}", end="")
    print(f"  {'─'*7}")

    sector_items = []
    for sec_key in SECTORS:
        sec_data = rdf[rdf['sector'] == sec_key]
        if len(sec_data) == 0:
            continue
        rates = {}
        for ml in FOCUS_MONTHS:
            m_sec = sec_data[sec_data['month'] == ml]
            if len(m_sec) > 0 and m_sec['n_days'].sum() > 0:
                rates[ml] = m_sec[key].mean()
            else:
                rates[ml] = None
        jul_r = rates.get('Jul') or 0
        sector_items.append((sec_key, rates, jul_r))

    sector_items.sort(key=lambda x: -x[2])

    for sec_key, rates, _ in sector_items:
        label = SECTORS[sec_key]['label']
        n_tk = len([t for t, s in loaded_tickers.items() if s == sec_key])
        print(f"  {label:<20} {n_tk:>2}", end="")
        for ml in FOCUS_MONTHS:
            r = rates.get(ml)
            if r is not None:
                print(f" {r:>4.0%}", end="")
            else:
                print(f" {'—':>5}", end="")
        may_r = rates.get('May') or 0
        jul_r = rates.get('Jul') or 0
        print(f"  {jul_r - may_r:>+6.0%}")

    all_rates = {}
    for ml in FOCUS_MONTHS:
        m_all = rdf[rdf['month'] == ml]
        if len(m_all) > 0:
            all_rates[ml] = m_all[key].mean()
    print(f"  {'ALL SECTORS':<20} {len(loaded_tickers):>2}", end="")
    for ml in FOCUS_MONTHS:
        r = all_rates.get(ml)
        if r is not None:
            print(f" {r:>4.0%}", end="")
        else:
            print(f" {'—':>5}", end="")
    may_all = all_rates.get('May', 0) or 0
    jul_all = all_rates.get('Jul', 0) or 0
    print(f"  {jul_all - may_all:>+6.0%}")


# ==============================================================================
# TABLE 2: UPSIDE vs DOWNSIDE COMPARISON — 8% threshold, side by side
# ==============================================================================
print(f"\n{'═' * W}")
print(f"  TABLE 2: UPSIDE 8% SPIKE vs DOWNSIDE 8% DUMP — Side-by-Side")
print(f"{'═' * W}")

print(f"\n  {'Sector':<20} {'#':>2}", end="")
for ml in FOCUS_MONTHS:
    print(f"  {'↑'+ml:>5} {'↓'+ml:>5} {'D/U':>5}", end="")
print()
print(f"  {'─'*20} {'─'*2}", end="")
for _ in FOCUS_MONTHS:
    print(f"  {'─'*5} {'─'*5} {'─'*5}", end="")
print()

sector_du_ratios = []
for sec_key in SECTORS:
    sec_data = rdf[rdf['sector'] == sec_key]
    if len(sec_data) == 0:
        continue
    label = SECTORS[sec_key]['label']
    n_tk = len([t for t, s in loaded_tickers.items() if s == sec_key])
    print(f"  {label:<20} {n_tk:>2}", end="")

    jul_ratio = None
    for ml in FOCUS_MONTHS:
        m_sec = sec_data[sec_data['month'] == ml]
        if len(m_sec) > 0 and m_sec['n_days'].sum() > 0:
            up_r = m_sec['spike_8_rate'].mean()
            dn_r = m_sec['dump_8_rate'].mean()
            ratio = (dn_r / up_r) if up_r > 0.001 else (99.0 if dn_r > 0 else 0)
            print(f"  {up_r:>4.0%} {dn_r:>4.0%} {ratio:>4.1f}x", end="")
            if ml == 'Jul':
                jul_ratio = ratio
        else:
            print(f"  {'—':>5} {'—':>5} {'—':>5}", end="")
    print()
    sector_du_ratios.append((sec_key, label, jul_ratio or 0))

# All sectors row
print(f"  {'─'*20} {'─'*2}", end="")
for _ in FOCUS_MONTHS:
    print(f"  {'─'*5} {'─'*5} {'─'*5}", end="")
print()
print(f"  {'ALL SECTORS':<20} {len(loaded_tickers):>2}", end="")
for ml in FOCUS_MONTHS:
    m_all = rdf[rdf['month'] == ml]
    if len(m_all) > 0:
        up_r = m_all['spike_8_rate'].mean()
        dn_r = m_all['dump_8_rate'].mean()
        ratio = (dn_r / up_r) if up_r > 0.001 else 99.0
        print(f"  {up_r:>4.0%} {dn_r:>4.0%} {ratio:>4.1f}x", end="")
    else:
        print(f"  {'—':>5} {'—':>5} {'—':>5}", end="")
print()

print(f"\n  D/U > 1.0 means more downside dumps than upside spikes (short-side opportunity)")
sector_du_ratios.sort(key=lambda x: -x[2])
print(f"\n  July Dump/Spike Ratio Ranking:")
for _, label, ratio in sector_du_ratios:
    bar = '█' * min(int(ratio * 5), 40)
    marker = " ← SHORT OPPORTUNITY" if ratio > 1.5 else ""
    print(f"    {label:<20} {ratio:>5.1f}x  {bar}{marker}")


# ==============================================================================
# TABLE 3: THRESHOLD SELECTION — Positive Rate per Threshold
# ==============================================================================
print(f"\n{'═' * W}")
print(f"  TABLE 3: THRESHOLD SELECTION — Positive Rate (target: 8-15%)")
print(f"{'═' * W}")

print(f"\n  Overall positive rate = % of ticker-days where max_drop ≥ threshold")
print(f"\n  {'Threshold':>10}", end="")
for ml in FOCUS_MONTHS:
    print(f"  {ml:>6}", end="")
print(f"  {'Avg':>6}  {'Status':>12}")
print(f"  {'─'*10}", end="")
for _ in FOCUS_MONTHS:
    print(f"  {'─'*6}", end="")
print(f"  {'─'*6}  {'─'*12}")

recommended_thr = None
for thr in DOWN_THRESHOLDS:
    key = f'dump_{int(thr*100)}_rate'
    pct = f"-{int(thr*100)}%"
    rates_by_month = []
    print(f"  {pct:>10}", end="")
    for ml in FOCUS_MONTHS:
        m_all = rdf[rdf['month'] == ml]
        if len(m_all) > 0 and m_all['n_days'].sum() > 0:
            r = m_all[key].mean()
            rates_by_month.append(r)
            print(f"  {r:>5.1%}", end="")
        else:
            print(f"  {'—':>6}", end="")

    if rates_by_month:
        avg_r = np.mean(rates_by_month)
        print(f"  {avg_r:>5.1%}", end="")
        if 0.08 <= avg_r <= 0.15:
            print(f"  {'★ TARGET':>12}")
            if recommended_thr is None:
                recommended_thr = thr
        elif avg_r > 0.15:
            print(f"  {'TOO HIGH':>12}")
        else:
            print(f"  {'TOO LOW':>12}")
    else:
        print(f"  {'—':>6}  {'—':>12}")

# Per-sector breakdown at candidate thresholds
candidate_thresholds = [t for t in DOWN_THRESHOLDS if t >= 0.06 and t <= 0.12]
print(f"\n  Per-sector positive rates at candidate thresholds (Jul only):")
print(f"\n  {'Sector':<20}", end="")
for thr in candidate_thresholds:
    print(f" {f'-{int(thr*100)}%':>6}", end="")
print()
print(f"  {'─'*20}", end="")
for _ in candidate_thresholds:
    print(f" {'─'*6}", end="")
print()

for sec_key in SECTORS:
    sec_data = rdf[(rdf['sector'] == sec_key) & (rdf['month'] == 'Jul')]
    if len(sec_data) == 0 or sec_data['n_days'].sum() == 0:
        continue
    label = SECTORS[sec_key]['label']
    print(f"  {label:<20}", end="")
    for thr in candidate_thresholds:
        key = f'dump_{int(thr*100)}_rate'
        r = sec_data[key].mean()
        marker = "★" if 0.08 <= r <= 0.15 else " "
        print(f" {r:>4.0%}{marker} ", end="")
    print()


# ==============================================================================
# TABLE 4: PER-TICKER DETAIL — Top short-side candidates
# ==============================================================================
print(f"\n{'═' * W}")
print(f"  TABLE 4: PER-TICKER DUMP RATES — Best Short-Side Candidates (Jul)")
print(f"{'═' * W}")

ref_thr = recommended_thr or 0.08
ref_key = f'dump_{int(ref_thr*100)}_rate'
ref_pct = f"{int(ref_thr*100)}%"

jul_data = rdf[(rdf['month'] == 'Jul') & (rdf['n_days'] > 0)].copy()
jul_data = jul_data.sort_values(ref_key, ascending=False)

print(f"\n  Using reference threshold: -{ref_pct}")
print(f"\n  {'Ticker':<8} {'Sector':<20} {'Dump%':>5} {'#Dmp':>4} {'Days':>4}"
      f"  {'AvgDrop':>7} {'AvgGain':>7}  {'D/U':>5}  {'Status':>10}")
print(f"  {'─'*8} {'─'*20} {'─'*5} {'─'*4} {'─'*4}"
      f"  {'─'*7} {'─'*7}  {'─'*5}  {'─'*10}")

n_shown = 0
for _, row in jul_data.iterrows():
    dump_rate = row[ref_key]
    dump_count = row.get(f'dump_{int(ref_thr*100)}_count', 0)
    spike_rate = row.get('spike_8_rate', 0)
    du_ratio = (dump_rate / spike_rate) if spike_rate > 0.001 else (99.0 if dump_rate > 0 else 0)
    sec_label = SECTORS.get(row['sector'], {}).get('label', row['sector'])

    if n_shown >= 40 and dump_rate < 0.05:
        break

    if dump_rate > 0.20:
        status = "HIGH DUMP"
    elif dump_rate > 0.10:
        status = "MODERATE"
    elif dump_rate > 0.05:
        status = "LOW"
    else:
        status = "RARE"

    print(f"  {row['ticker']:<8} {sec_label:<20} {dump_rate:>4.0%} {dump_count:>4.0f} {row['n_days']:>4.0f}"
          f"  {row['avg_max_drop']:>6.1%} {row['avg_max_gain']:>6.1%}  {du_ratio:>4.1f}x  {status:>10}")
    n_shown += 1

# Summary: how many tickers at each dump rate level
print(f"\n  TICKER DISTRIBUTION BY DUMP FREQUENCY (Jul, -{ref_pct} threshold):")
for label, lo, hi in [("HIGH (>20%)", 0.20, 1.01), ("MODERATE (10-20%)", 0.10, 0.20),
                       ("LOW (5-10%)", 0.05, 0.10), ("RARE (<5%)", 0.0, 0.05)]:
    count = len(jul_data[(jul_data[ref_key] >= lo) & (jul_data[ref_key] < hi)])
    tickers = jul_data[(jul_data[ref_key] >= lo) & (jul_data[ref_key] < hi)]['ticker'].tolist()
    tk_str = ', '.join(tickers[:10])
    if len(tickers) > 10:
        tk_str += f"... +{len(tickers)-10}"
    print(f"    {label:<20} {count:>3} tickers  {tk_str}")


# ==============================================================================
# TABLE 5: DISTRIBUTION ANALYSIS — Percentiles of daily max_drop
# ==============================================================================
print(f"\n{'═' * W}")
print(f"  TABLE 5: DISTRIBUTION ANALYSIS — Percentiles of Daily Max Drop")
print(f"{'═' * W}")

focus_dist = dist_df[dist_df['date'] >= '2026-05-01'].copy()

print(f"\n  A. By Sector (May-Jul 2026)")
print(f"\n  {'Sector':<20} {'P50':>5} {'P75':>5} {'P90':>5} {'P95':>5} {'P99':>5}"
      f"  {'↑P50':>5} {'↑P90':>5}  {'Skew':>5}")
print(f"  {'─'*20} {'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*5}"
      f"  {'─'*5} {'─'*5}  {'─'*5}")

for sec_key in SECTORS:
    sec_dist = focus_dist[focus_dist['sector'] == sec_key]
    if len(sec_dist) < 10:
        continue
    label = SECTORS[sec_key]['label']
    drops = sec_dist['max_drop']
    gains = sec_dist['max_gain']
    p50 = drops.quantile(0.50)
    p75 = drops.quantile(0.75)
    p90 = drops.quantile(0.90)
    p95 = drops.quantile(0.95)
    p99 = drops.quantile(0.99)
    g_p50 = gains.quantile(0.50)
    g_p90 = gains.quantile(0.90)
    skew = (p90 / g_p90) if g_p90 > 0.001 else 0
    print(f"  {label:<20} {p50:>4.1%} {p75:>4.1%} {p90:>4.1%} {p95:>4.1%} {p99:>4.1%}"
          f"  {g_p50:>4.1%} {g_p90:>4.1%}  {skew:>4.1f}x")

# Overall
drops_all = focus_dist['max_drop']
gains_all = focus_dist['max_gain']
print(f"  {'─'*20} {'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*5}"
      f"  {'─'*5} {'─'*5}  {'─'*5}")
p50 = drops_all.quantile(0.50)
p75 = drops_all.quantile(0.75)
p90 = drops_all.quantile(0.90)
p95 = drops_all.quantile(0.95)
p99 = drops_all.quantile(0.99)
g_p50 = gains_all.quantile(0.50)
g_p90 = gains_all.quantile(0.90)
skew = (p90 / g_p90) if g_p90 > 0.001 else 0
print(f"  {'ALL SECTORS':<20} {p50:>4.1%} {p75:>4.1%} {p90:>4.1%} {p95:>4.1%} {p99:>4.1%}"
      f"  {g_p50:>4.1%} {g_p90:>4.1%}  {skew:>4.1f}x")
print(f"\n  Skew = P90_drop / P90_gain. >1.0 means downside tails are fatter than upside.")

# B. Month-over-month trend
print(f"\n  B. Monthly Trend — Overall Distribution")
print(f"\n  {'Month':<6} {'P50':>5} {'P75':>5} {'P90':>5} {'P95':>5}"
      f"  {'↑P50':>5} {'↑P90':>5}  {'Skew':>5}")
print(f"  {'─'*6} {'─'*5} {'─'*5} {'─'*5} {'─'*5}"
      f"  {'─'*5} {'─'*5}  {'─'*5}")

for m_label, m_start, m_end in MONTHS:
    m_dist = dist_df[(dist_df['date'] >= m_start) & (dist_df['date'] <= m_end)]
    if len(m_dist) < 10:
        continue
    drops = m_dist['max_drop']
    gains = m_dist['max_gain']
    p50 = drops.quantile(0.50)
    p75 = drops.quantile(0.75)
    p90 = drops.quantile(0.90)
    p95 = drops.quantile(0.95)
    g_p50 = gains.quantile(0.50)
    g_p90 = gains.quantile(0.90)
    skew = (p90 / g_p90) if g_p90 > 0.001 else 0
    marker = " ← " if m_label in FOCUS_MONTHS else "   "
    print(f"  {m_label:<6} {p50:>4.1%} {p75:>4.1%} {p90:>4.1%} {p95:>4.1%}"
          f"  {g_p50:>4.1%} {g_p90:>4.1%}  {skew:>4.1f}x{marker}")


# ==============================================================================
# DIAGNOSIS — Threshold Recommendation
# ==============================================================================
print(f"\n{'═' * W}")
print(f"  DIAGNOSIS — Exp 400 Target Threshold Recommendation")
print(f"{'═' * W}")

# Compute overall rates for each threshold (May-Jul focus)
print(f"\n  A. OVERALL DUMP FREQUENCY (May-Jul avg)")
best_thr = None
best_gap = float('inf')
for thr in DOWN_THRESHOLDS:
    key = f'dump_{int(thr*100)}_rate'
    focus_data = rdf[rdf['month'].isin(FOCUS_MONTHS)]
    if len(focus_data) > 0:
        overall_rate = focus_data[key].mean()
        target_mid = 0.115
        gap = abs(overall_rate - target_mid)
        if gap < best_gap:
            best_gap = gap
            best_thr = thr
        in_range = "★ IN RANGE" if 0.08 <= overall_rate <= 0.15 else ""
        print(f"    -{int(thr*100):>2}%  →  {overall_rate:>5.1%}  {in_range}")

print(f"\n  B. DUMP vs SPIKE ASYMMETRY (Jul)")
jul_focus = rdf[rdf['month'] == 'Jul']
if len(jul_focus) > 0:
    for thr in [0.06, 0.08, 0.10]:
        dump_r = jul_focus[f'dump_{int(thr*100)}_rate'].mean()
        spike_r = jul_focus[f'spike_{int(thr*100)}_rate'].mean()
        ratio = (dump_r / spike_r) if spike_r > 0.001 else 99.0
        print(f"    {int(thr*100)}%:  Upside {spike_r:>5.1%}  |  Downside {dump_r:>5.1%}  |  Ratio {ratio:.1f}x")

# Count tickers with enough dumps to train
print(f"\n  C. TRAINABLE TICKERS (tickers with ≥3 dump days in last 3 months)")
for thr in candidate_thresholds:
    key = f'dump_{int(thr*100)}_count'
    focus_data = rdf[rdf['month'].isin(FOCUS_MONTHS)]
    if len(focus_data) > 0:
        ticker_totals = focus_data.groupby('ticker')[key].sum()
        trainable = (ticker_totals >= 3).sum()
        total = len(ticker_totals)
        print(f"    -{int(thr*100):>2}%  →  {trainable:>3}/{total} tickers ({trainable/total:.0%})")

# Best tickers for short-side
if best_thr is not None:
    ref_key2 = f'dump_{int(best_thr*100)}_rate'
    print(f"\n  D. TOP 15 SHORT-SIDE CANDIDATES (by Jul dump rate at -{int(best_thr*100)}%)")
    jul_top = rdf[(rdf['month'] == 'Jul') & (rdf['n_days'] > 0)].sort_values(ref_key2, ascending=False).head(15)
    for i, (_, row) in enumerate(jul_top.iterrows(), 1):
        sec_label = SECTORS.get(row['sector'], {}).get('label', row['sector'])
        print(f"    {i:>2}. {row['ticker']:<8} [{sec_label:<18}]  dump_rate={row[ref_key2]:>4.0%}  avg_drop={row['avg_max_drop']:.1%}")

# Final recommendation
print(f"\n  ┌{'─'*78}┐")
if best_thr:
    print(f"  │  RECOMMENDED THRESHOLD: -{int(best_thr*100)}%{' '*60}│")
    focus_rate = rdf[rdf['month'].isin(FOCUS_MONTHS)][f'dump_{int(best_thr*100)}_rate'].mean()
    jul_rate = rdf[rdf['month'] == 'Jul'][f'dump_{int(best_thr*100)}_rate'].mean() if len(rdf[rdf['month'] == 'Jul']) > 0 else 0
    print(f"  │  May-Jul avg positive rate: {focus_rate:.1%}{' '*(48-len(f'{focus_rate:.1%}'))}│")
    print(f"  │  Jul positive rate: {jul_rate:.1%}{' '*(57-len(f'{jul_rate:.1%}'))}│")
    print(f"  │{' '*78}│")
    print(f"  │  For Exp 400 training:                                                    │")
    print(f"  │    Future_Min = min(Low) from entry bar → 1:00 PM PST                     │")
    print(f"  │    Dump = 1 if (1 - Future_Min/Close) >= {best_thr:.2f}{' '*(36-len(f'{best_thr:.2f}'))}│")
    print(f"  │    Same 15-seed convention: seed = 400*100 + 01..15                        │")
else:
    print(f"  │  NO CLEAR THRESHOLD — check data quality                                  │")
print(f"  └{'─'*78}┘")

print(f"\n{'═' * W}")
print(f"  ANALYSIS COMPLETE — {len(loaded_tickers)} tickers, {len(set(loaded_tickers.values()))} sectors")
print(f"{'═' * W}")
