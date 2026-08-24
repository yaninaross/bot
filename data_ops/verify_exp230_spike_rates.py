"""
verify_exp230_spike_rates.py — Spike Rate Verification for Exp 230 Teacher Set
================================================================================
Scans ALL downloaded tickers (~87) for 10% intraday spike rates using the same
methodology as training label generation. Ranks the full universe and recommends
the best teacher set for Exp 230.

Colab-ready (no if __name__).
"""

import gc, ctypes, io, warnings, sys, subprocess
import numpy as np
import pandas as pd
import pytz

warnings.filterwarnings('ignore')

def _release_memory():
    gc.collect(); gc.collect()
    try: ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception: pass

try:
    from google.cloud import storage as gcs_storage
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
        "google-cloud-storage", "pyarrow", "--quiet"])
    from google.cloud import storage as gcs_storage

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

# ==============================================================================
# CONFIG
# ==============================================================================
BUCKET_NAME    = "time-series-bucket-ys"
INFERENCE_V2   = "Data/Inference_v2"
INFERENCE_V1   = "Data/Inference"

ENTRY_MINUTE   = 375     # 6:15 PT
MARKET_OPEN    = 390     # 6:30 PT (fallback)
EOD_MINUTE     = 780     # 13:00 PT
SPIKE_THRESHOLD = 0.10
MIN_SPIKE_10PCT_PER_MONTH = 2.0

DATA_START   = '2025-07-01'
RECENT_START = '2026-04-01'

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
PST = pytz.timezone('America/Los_Angeles')

ALL_DOWNLOADED = [
    'ACHR', 'ACMR', 'AEHR', 'ALAB', 'AMD', 'APLD', 'ARM', 'ASTS', 'AVAV',
    'BBAI', 'BE', 'BEAM', 'BGDE', 'BHP', 'BKSY', 'BTBT',
    'CAT', 'CCJ', 'CEG', 'CIFR', 'CLSK', 'COIN', 'COP', 'CORZ', 'CRDO', 'CVX',
    'DUK', 'EQIX', 'ETN', 'FCEL', 'FCX', 'FLNC',
    'GD', 'GE', 'GEV', 'HOOD', 'HUT', 'IREN',
    'JOBY', 'KEEL', 'KTOS',
    'LHX', 'LIN', 'LMT', 'LUNR',
    'MARA', 'MNTS', 'MOD', 'MP', 'MRNA', 'MSTR', 'MU', 'MVIS', 'MXL',
    'NBIS', 'NEE', 'NEOV', 'NNE', 'NNDM', 'NOC', 'NTLA', 'NVDA',
    'OKLO', 'ONDS', 'PL', 'PLTR', 'PLUG', 'POWL',
    'RCAT', 'RDW', 'RIOT', 'RKLB', 'RTX',
    'SLS', 'SMCI', 'SMR', 'SOUN', 'SPIR',
    'TSLA', 'TWST', 'UCTT', 'VOYG', 'VRT', 'VST', 'WOLF', 'WULF', 'XOM',
]

EXP180_TEACHERS = {
    'CIFR', 'IREN', 'KEEL', 'HUT', 'APLD', 'WULF', 'RIOT', 'CLSK', 'BTBT',
    'ALAB', 'MARA', 'RCAT', 'ASTS', 'LUNR', 'BBAI', 'RKLB', 'SPIR', 'BKSY',
    'MNTS', 'PLUG', 'BE', 'FCEL', 'FLNC', 'OKLO', 'SMR', 'MVIS', 'BGDE',
    'RDW', 'PL', 'VOYG', 'SMLR',
}

ALL_CANDIDATES = sorted(set(ALL_DOWNLOADED))

W = 100

client = gcs_storage.Client()
bucket = client.bucket(BUCKET_NAME)


# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_ohlcv_from_gcs(ticker):
    file_ticker = TICKER_FILE_MAP.get(ticker, ticker)
    paths = [
        f"{INFERENCE_V2}/{file_ticker}_master_inference.parquet",
        f"{INFERENCE_V1}/{file_ticker}_1m_full_historical_data.parquet",
    ]
    for blob_path in paths:
        blob = bucket.blob(blob_path)
        if not blob.exists():
            continue
        raw = blob.download_as_bytes()
        df = pd.read_parquet(io.BytesIO(raw))
        del raw
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        if dt_cols:
            df['Datetime'] = pd.to_datetime(df[dt_cols[0]], utc=True)
        else:
            num_col = next((c for c in df.columns
                           if c in ('Datetime', 'Date', 'Time', 'Timestamp', 'Level_0')),
                          df.columns[0])
            df['Datetime'] = pd.to_datetime(df[num_col], unit='ms', utc=True)
        df = df.set_index('Datetime')
        df = df[~df.index.duplicated(keep='first')]
        std_map = {'open': 'Open', 'high': 'High', 'low': 'Low',
                   'close': 'Close', 'volume': 'Volume'}
        lower_map = {k.lower(): v for k, v in std_map.items()}
        df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip()) for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()
        df = df[df.index.notna()].sort_index()
        cutoff = pd.Timestamp(DATA_START, tz='UTC')
        df = df[df.index >= cutoff]
        keep = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
        if len(keep) < 4:
            continue
        return df[keep]
    return None


# ==============================================================================
# SPIKE COMPUTATION
# ==============================================================================
def compute_daily_spikes(df):
    df_pst = df.copy()
    df_pst.index = df_pst.index.tz_convert(PST)
    times = (df_pst.index.hour * 60 + df_pst.index.minute).values
    dates = df_pst.index.date

    unique_dates = sorted(set(dates))
    daily_spikes = []

    for d in unique_dates:
        day_mask = dates == d
        day_times  = times[day_mask]
        day_highs  = df_pst['High'].values[day_mask]
        day_closes = df_pst['Close'].values[day_mask]

        entry_mask = day_times == ENTRY_MINUTE
        if entry_mask.sum() == 0:
            entry_mask = day_times == MARKET_OPEN
            if entry_mask.sum() == 0:
                continue

        entry_idx = np.where(entry_mask)[0][0]
        entry_price = day_closes[entry_idx]
        if entry_price <= 0:
            continue

        post_entry = (day_times > ENTRY_MINUTE) & (day_times <= EOD_MINUTE)
        if post_entry.sum() == 0:
            continue

        max_high = day_highs[post_entry].max()
        spike_up = (max_high - entry_price) / entry_price

        daily_spikes.append({
            'date': d,
            'entry_price': entry_price,
            'max_high': max_high,
            'spike_up': spike_up,
        })

    return pd.DataFrame(daily_spikes)


def compute_spike_metrics(daily_df):
    if daily_df is None or len(daily_df) == 0:
        return {'n_days': 0, 'spike_10_count': 0, 'spike_10_per_mo': 0.0,
                'avg_spike': 0.0, 'max_spike': 0.0}
    n_days = len(daily_df)
    n_months = max(1.0, n_days / 21.0)
    spike_10_count = int((daily_df['spike_up'] >= SPIKE_THRESHOLD).sum())
    spike_10_per_mo = spike_10_count / n_months
    avg_spike = daily_df['spike_up'].mean()
    max_spike = daily_df['spike_up'].max()
    return {
        'n_days': n_days,
        'spike_10_count': spike_10_count,
        'spike_10_per_mo': round(spike_10_per_mo, 2),
        'avg_spike': round(avg_spike * 100, 2),
        'max_spike': round(max_spike * 100, 2),
    }


# ==============================================================================
# MAIN LOOP
# ==============================================================================
print("=" * W)
print("  SPIKE RATE VERIFICATION — FULL UNIVERSE SCAN")
print(f"  {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')}")
print(f"  Tickers: {len(ALL_CANDIDATES)}")
print(f"  Threshold:  {SPIKE_THRESHOLD*100:.0f}% spike, {MIN_SPIKE_10PCT_PER_MONTH}/mo min")
print(f"  Data range: {DATA_START} → now  |  Recent: {RECENT_START} →")
print("=" * W)

results = []
recent_cutoff = pd.Timestamp(RECENT_START).date() if isinstance(RECENT_START, str) \
                else RECENT_START

for i, ticker in enumerate(ALL_CANDIDATES):
    was_teacher = 'T180' if ticker in EXP180_TEACHERS else '    '
    print(f"  [{i+1:2d}/{len(ALL_CANDIDATES)}] {ticker:5s} {was_teacher} ...", end=" ", flush=True)

    df = load_ohlcv_from_gcs(ticker)
    if df is None:
        print("NO DATA")
        results.append({
            'Ticker': ticker, 'In180': ticker in EXP180_TEACHERS,
            'full_days': 0, 'full_10_per_mo': 0.0,
            'rcnt_days': 0, 'rcnt_10_per_mo': 0.0,
            'avg_spike': 0.0, 'max_spike': 0.0,
        })
        continue

    n_bars = len(df)
    daily = compute_daily_spikes(df)
    del df
    _release_memory()

    if len(daily) < 5:
        print(f"{n_bars:,} bars, {len(daily)} days — TOO FEW")
        results.append({
            'Ticker': ticker, 'In180': ticker in EXP180_TEACHERS,
            'full_days': len(daily), 'full_10_per_mo': 0.0,
            'rcnt_days': 0, 'rcnt_10_per_mo': 0.0,
            'avg_spike': 0.0, 'max_spike': 0.0,
        })
        del daily
        _release_memory()
        continue

    full = compute_spike_metrics(daily)
    recent_daily = daily[daily['date'] >= recent_cutoff]
    rcnt = compute_spike_metrics(recent_daily) if len(recent_daily) >= 5 else \
           {'n_days': len(recent_daily), 'spike_10_count': 0, 'spike_10_per_mo': 0.0,
            'avg_spike': 0.0, 'max_spike': 0.0}

    results.append({
        'Ticker': ticker,
        'In180': ticker in EXP180_TEACHERS,
        'full_days': full['n_days'],
        'full_10_per_mo': full['spike_10_per_mo'],
        'rcnt_days': rcnt['n_days'],
        'rcnt_10_per_mo': rcnt['spike_10_per_mo'],
        'avg_spike': full['avg_spike'],
        'max_spike': full['max_spike'],
    })

    del daily, recent_daily
    _release_memory()

    print(f"{n_bars:>9,} bars | {full['n_days']:3d}d | "
          f"full={full['spike_10_per_mo']:5.2f}/mo  rcnt={rcnt['spike_10_per_mo']:5.2f}/mo")


# ==============================================================================
# CLASSIFICATION
# ==============================================================================
df_r = pd.DataFrame(results)

def classify(row):
    r = row['rcnt_10_per_mo']
    f = row['full_10_per_mo']
    if r >= MIN_SPIKE_10PCT_PER_MONTH:
        return 'KEEP'
    elif r >= 1.0 or f >= MIN_SPIKE_10PCT_PER_MONTH:
        return 'BORDERLINE'
    else:
        return 'DROP'

df_r['Verdict'] = df_r.apply(classify, axis=1)
df_r['Trend'] = df_r['rcnt_10_per_mo'] - df_r['full_10_per_mo']
df_r['Trend_Label'] = df_r['Trend'].apply(
    lambda x: 'ACCEL' if x > 0.5 else ('DECEL' if x < -0.5 else 'FLAT'))
df_r = df_r.sort_values('rcnt_10_per_mo', ascending=False)


# ==============================================================================
# TABLE A — FULL UNIVERSE RANKED (KEEP + BORDERLINE only)
# ==============================================================================
keepers = df_r[df_r['Verdict'] == 'KEEP'].copy()
borderlines = df_r[df_r['Verdict'] == 'BORDERLINE'].copy()
drops = df_r[df_r['Verdict'] == 'DROP'].copy()
show = pd.concat([keepers, borderlines]).sort_values('rcnt_10_per_mo', ascending=False)

print()
print("=" * W)
print("  TABLE A — QUALIFYING TICKERS (≥1.0/mo recent OR ≥2.0/mo full)")
print("=" * W)
header = f"  {'Ticker':6s} {'180?':4s} {'FDays':>5s} {'F10/mo':>7s} {'RDays':>5s} {'R10/mo':>7s} " \
         f"{'AvgSp%':>7s} {'MaxSp%':>7s} {'Trend':>6s} {'Verdict':>10s}"
print(header)
print("  " + "-" * (len(header) - 2))

for _, row in show.iterrows():
    v = row['Verdict']
    t180 = 'yes' if row['In180'] else '   '
    flag = ' ***' if v == 'KEEP' else ' ??'
    print(f"  {row['Ticker']:6s} {t180:4s} "
          f"{row['full_days']:5.0f} {row['full_10_per_mo']:7.2f} "
          f"{row['rcnt_days']:5.0f} {row['rcnt_10_per_mo']:7.2f} "
          f"{row['avg_spike']:7.2f} {row['max_spike']:7.2f} "
          f"{row['Trend_Label']:>6s} {v:>10s}{flag}")


# ==============================================================================
# TABLE B — RECOMMENDED EXP 230 TEACHERS
# ==============================================================================
print()
print("=" * W)
print("  TABLE B — RECOMMENDED EXP 230 TEACHERS (Verdict = KEEP)")
print("=" * W)

if len(keepers) > 0:
    for _, row in keepers.iterrows():
        t180 = 'was in 180' if row['In180'] else 'NEW'
        print(f"  {row['Ticker']:6s}  R10/mo={row['rcnt_10_per_mo']:5.2f}  "
              f"F10/mo={row['full_10_per_mo']:5.2f}  {row['Trend_Label']:>5s}  ({t180})")
else:
    print("  (none)")


# ==============================================================================
# TABLE C — BORDERLINE (MANUAL REVIEW)
# ==============================================================================
if len(borderlines) > 0:
    print()
    print("=" * W)
    print("  TABLE C — BORDERLINE (manual review needed)")
    print("=" * W)
    for _, row in borderlines.iterrows():
        t180 = 'was in 180' if row['In180'] else 'NEW'
        print(f"  {row['Ticker']:6s}  R10/mo={row['rcnt_10_per_mo']:5.2f}  "
              f"F10/mo={row['full_10_per_mo']:5.2f}  {row['Trend_Label']:>5s}  ({t180})")


# ==============================================================================
# TABLE D — ALL DROPS (for reference)
# ==============================================================================
print()
print("=" * W)
print("  TABLE D — ALL DROPS (<1.0/mo recent AND <2.0/mo full)")
print("=" * W)
drops_sorted = drops.sort_values('rcnt_10_per_mo', ascending=False)
for _, row in drops_sorted.iterrows():
    t180 = '*' if row['In180'] else ' '
    print(f"  {t180} {row['Ticker']:6s}  R10/mo={row['rcnt_10_per_mo']:5.2f}  "
          f"F10/mo={row['full_10_per_mo']:5.2f}  AvgSp={row['avg_spike']:5.2f}%")


# ==============================================================================
# SUMMARY
# ==============================================================================
print()
print("=" * W)
print("  SUMMARY")
print("=" * W)
print(f"  Total analyzed:    {len(df_r):>3d}")
print(f"  KEEP  (≥{MIN_SPIKE_10PCT_PER_MONTH}/mo):  {len(keepers):>3d}")
print(f"  BORDERLINE:        {len(borderlines):>3d}")
print(f"  DROP:              {len(drops):>3d}")
print()
teacher_list = sorted(keepers['Ticker'].tolist())
print(f"  Recommended Exp 230 teachers ({len(teacher_list)} tickers):")
for chunk_start in range(0, len(teacher_list), 10):
    chunk = teacher_list[chunk_start:chunk_start+10]
    print(f"    {', '.join(chunk)}")
print()
print("  Copy-paste for training script:")
print(f"    TRAIN_TEACHERS = {teacher_list}")
print()
if len(borderlines) > 0:
    border_list = sorted(borderlines['Ticker'].tolist())
    print(f"  If including borderlines ({len(border_list)} more):")
    combined = sorted(teacher_list + border_list)
    print(f"    TRAIN_TEACHERS = {combined}")
    print()
print("=" * W)
