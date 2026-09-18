# %% ── Where Did The Money Go? ─────────────────────────────────────────────────
#
# PM spikes disappeared — but are markets actually flat?
# This analysis maps WHERE the opportunity moved:
#
# 1. Full-day range analysis: are stocks still moving 3%+ intraday?
# 2. Time-of-day heatmap: WHEN do the big moves happen?
#    - Pre-market (4:00-6:25 AM PT)
#    - Market open (6:25-7:00 AM PT) — our entry window
#    - Morning (7:00-9:00 AM PT)
#    - Midday (9:00-11:00 AM PT)
#    - Afternoon (11:00 AM - 1:00 PM PT)
# 3. Move character: fast spikes vs slow drifts?
# 4. Direction: up-bias, down-bias, or balanced?
# 5. Which tickers/sectors still have tradeable moves?
#
# Data source: GCS master_inference parquets (1-min bars)
# Colab-ready. No GPU needed.

import os, sys, gc, ctypes, subprocess, warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import pytz

warnings.filterwarnings('ignore')

try:
    from google.cloud import storage as gcs_storage
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
        "google-cloud-storage", "--quiet", "--break-system-packages"])
    from google.cloud import storage as gcs_storage

try:
    from google.cloud import bigquery
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
        "google-cloud-bigquery", "--quiet", "--break-system-packages"])
    from google.cloud import bigquery

try:
    from google.colab import auth
    _IN_COLAB = True
    try:
        auth.authenticate_user()
    except Exception:
        pass
except ImportError:
    _IN_COLAB = False

import matplotlib
if not _IN_COLAB:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ==============================================================================
# CONFIG
# ==============================================================================
BUCKET_NAME   = os.environ.get("GCS_BUCKET", "")
INFERENCE_DIR = "Data/Inference_v2"
PST           = pytz.timezone('America/Los_Angeles')

ANALYSIS_START = '2026-07-01'
ANALYSIS_END   = '2026-09-07'
TAIL_DAYS = 75

ENTRY_MINUTE = 385   # 6:25 AM PT

SPIKE_THRESHOLD_PCT = 3.0

PERIODS = [
    ('P1: Jul 1-15',    '2026-07-01', '2026-07-15'),
    ('P2: Jul 16-31',   '2026-07-16', '2026-07-31'),
    ('P3: Aug 1-15',    '2026-08-01', '2026-08-15'),
    ('P4: Aug 16-Sep7', '2026-08-16', '2026-09-07'),
]

# Time-of-day segments (in minutes from midnight PT)
SEGMENTS = [
    ('Pre-Market',   240, 385),   # 4:00 - 6:25 AM PT
    ('Open_30m',     385, 415),   # 6:25 - 6:55 AM PT (first 30 min after entry)
    ('Open_60m',     385, 445),   # 6:25 - 7:25 AM PT (first 60 min after entry)
    ('Morning',      445, 540),   # 7:25 - 9:00 AM PT
    ('Midday',       540, 660),   # 9:00 - 11:00 AM PT
    ('Afternoon',    660, 780),   # 11:00 AM - 1:00 PM PT (market close)
]

# For MFE peak timing analysis: finer-grained windows after entry
MFE_WINDOWS = [
    ('0-5m',    385, 390),
    ('5-15m',   390, 400),
    ('15-30m',  400, 415),
    ('30-60m',  415, 445),
    ('60-120m', 445, 505),
    ('120m+',   505, 780),
]

W = 120

gcs_client = gcs_storage.Client()

TICKER_SECTOR = {
    'CIFR': 'Crypto', 'IREN': 'Crypto', 'KEEL': 'Crypto', 'HUT': 'Crypto',
    'APLD': 'Crypto', 'WULF': 'Crypto', 'RIOT': 'Crypto', 'CLSK': 'Crypto',
    'BTBT': 'Crypto', 'MARA': 'Crypto', 'CORZ': 'Crypto', 'MSTR': 'Crypto',
    'BITF': 'Crypto', 'BGDE': 'Crypto', 'MIGI': 'Crypto',
    'RCAT': 'Defense', 'ASTS': 'Defense', 'LUNR': 'Defense', 'BBAI': 'Defense',
    'RKLB': 'Defense', 'SPIR': 'Defense', 'BKSY': 'Defense', 'MNTS': 'Defense',
    'PL': 'Defense', 'RDW': 'Defense', 'VOYG': 'Defense', 'KTOS': 'Defense',
    'GD': 'Defense', 'GE': 'Defense', 'GEV': 'Defense', 'LHX': 'Defense',
    'LMT': 'Defense', 'NOC': 'Defense', 'RTX': 'Defense', 'JOBY': 'Defense',
    'AVAV': 'Defense', 'ACHR': 'Defense', 'DFNS': 'Defense',
    'PLUG': 'CleanEnergy', 'BE': 'CleanEnergy', 'FCEL': 'CleanEnergy',
    'FLNC': 'CleanEnergy',
    'OKLO': 'Nuclear', 'SMR': 'Nuclear', 'NNE': 'Nuclear', 'CCJ': 'Nuclear',
    'CEG': 'Nuclear', 'VST': 'Nuclear',
    'AMD': 'Semis', 'NVDA': 'Semis', 'SOUN': 'Semis', 'VRT': 'Semis',
    'ALAB': 'Semis', 'MVIS': 'Semis', 'AEHR': 'Semis', 'ACMR': 'Semis',
    'MXL': 'Semis', 'AAOI': 'Semis', 'MRVL': 'Semis', 'WDC': 'Semis',
    'CRDO': 'Semis', 'SMCI': 'Semis', 'WOLF': 'Semis', 'AXTI': 'Semis',
    'MU': 'Semis',
    'PLTR': 'Tech', 'TSLA': 'Tech', 'HOOD': 'Tech', 'SLS': 'Tech',
    'NBIS': 'Tech', 'NTLA': 'Tech',
    'BHP': 'Energy', 'CAT': 'Energy', 'COP': 'Energy', 'CVX': 'Energy',
    'XOM': 'Energy',
    'DUK': 'Utilities', 'EQIX': 'Utilities', 'ETN': 'Utilities',
    'LIN': 'Utilities', 'NEE': 'Utilities',
}

SECTOR_COLORS = {
    'Crypto': '#FF6F00', 'Defense': '#1565C0', 'CleanEnergy': '#2E7D32',
    'Nuclear': '#7B1FA2', 'Semis': '#C62828', 'Tech': '#00838F',
    'Energy': '#5D4037', 'Utilities': '#757575', 'Unknown': '#BDBDBD',
}

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
EXCLUDE_TICKERS = {'SPY', 'VIX', 'TLT', 'IWM', 'XLE', 'ITA', 'BTCUSD', 'ETHUSD'}

_NEEDED_COLS = {'Open', 'High', 'Low', 'Close', 'Volume',
                'open', 'high', 'low', 'close', 'volume',
                'Datetime', 'Date', 'Time', 'Timestamp', 'Level_0'}


# ==============================================================================
# DATA LOADING (same as resilience script)
# ==============================================================================
def load_master(ticker, tail_days=None):
    clean = ticker.replace("I:", "").replace("X:", "")
    file_ticker = TICKER_FILE_MAP.get(clean, clean)
    blob_path = f"{INFERENCE_DIR}/{file_ticker}_master_inference.parquet"
    local_path = f"/tmp/_wheremoney_{file_ticker}.parquet"
    try:
        gcs_client.bucket(BUCKET_NAME).blob(blob_path).download_to_filename(local_path)
        import pyarrow.parquet as pq
        schema = pq.read_schema(local_path)
        all_cols = [f.name for f in schema]
        use_cols = [c for c in all_cols if c in _NEEDED_COLS]
        if not use_cols:
            use_cols = None
        df = pd.read_parquet(local_path, columns=use_cols)
        os.remove(local_path)
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        if dt_cols:
            df['Datetime'] = pd.to_datetime(df[dt_cols[0]], utc=True)
        else:
            num_col = next(
                (c for c in df.columns
                 if c in ('Datetime', 'Date', 'Time', 'Timestamp', 'Level_0')),
                df.columns[0])
            df['Datetime'] = pd.to_datetime(df[num_col], unit='ms', utc=True)
        df = df.set_index('Datetime')
        df = df[~df.index.duplicated(keep='first')]
        std_map = {'open': 'Open', 'high': 'High', 'low': 'Low',
                   'close': 'Close', 'volume': 'Volume'}
        lower_map = {k.lower(): v for k, v in std_map.items()}
        df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip())
                      for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()
        df = df[df.index.notna()].sort_index()
        if tail_days is not None:
            cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=tail_days)
            df = df[df.index >= cutoff]
            if df.empty:
                return None
        return df
    except Exception:
        if os.path.exists(local_path):
            os.remove(local_path)
        return None


def list_gcs_inference_tickers():
    bucket = gcs_client.bucket(BUCKET_NAME)
    prefix = f"{INFERENCE_DIR}/"
    blobs = bucket.list_blobs(prefix=prefix)
    tickers = []
    for blob in blobs:
        name = blob.name.split('/')[-1]
        if name.endswith('_master_inference.parquet'):
            tk = name.replace('_master_inference.parquet', '')
            if tk not in EXCLUDE_TICKERS:
                tickers.append(tk)
    return sorted(set(tickers))


# ==============================================================================
# COMPUTE DAILY METRICS — FULL TIME-OF-DAY DECOMPOSITION
# ==============================================================================
def compute_full_day_metrics(df, ticker):
    try:
        df_pst = df.index.tz_convert(PST)
    except Exception:
        return []
    mins = df_pst.hour * 60 + df_pst.minute
    df = df.copy()
    df['_min'] = mins
    df['_date'] = df_pst.date

    bt_start = pd.Timestamp(ANALYSIS_START).date()
    bt_end = pd.Timestamp(ANALYSIS_END).date()

    results = []
    for date_val, day_df in df.groupby('_date'):
        if date_val < bt_start or date_val > bt_end:
            continue
        if len(day_df) < 30:
            continue

        # Need entry price as reference
        entry_bars = day_df[day_df['_min'] == ENTRY_MINUTE]
        if len(entry_bars) == 0:
            entry_bars = day_df[(day_df['_min'] >= ENTRY_MINUTE - 2) &
                                (day_df['_min'] <= ENTRY_MINUTE + 2)]
        if len(entry_bars) == 0:
            continue
        entry_price = float(entry_bars.iloc[0]['Close'])
        if entry_price <= 0:
            continue

        # Full-day range (all available bars)
        full_high = day_df['High'].max()
        full_low = day_df['Low'].min()
        full_range_pct = (full_high - full_low) / entry_price * 100
        full_open = float(day_df.iloc[0]['Open'])
        full_close = float(day_df.iloc[-1]['Close'])
        full_return = (full_close - full_open) / full_open * 100 if full_open > 0 else 0

        row = {
            'Ticker': ticker,
            'Date': str(date_val),
            'Date_dt': pd.Timestamp(date_val),
            'Entry_Price': entry_price,
            'Full_Range_Pct': full_range_pct,
            'Full_Return': full_return,
            'Full_Range_3pct': full_range_pct >= SPIKE_THRESHOLD_PCT,
            'Full_Range_6pct': full_range_pct >= 6.0,
            'Sector': TICKER_SECTOR.get(ticker, 'Unknown'),
        }

        # Per-segment MFE (max move from entry price within each segment)
        for seg_name, seg_start, seg_end in SEGMENTS:
            seg_df = day_df[(day_df['_min'] >= seg_start) & (day_df['_min'] < seg_end)]
            if len(seg_df) >= 2:
                seg_mfe_up = (seg_df['High'].max() - entry_price) / entry_price * 100
                seg_mfe_dn = (entry_price - seg_df['Low'].min()) / entry_price * 100
                seg_range = (seg_df['High'].max() - seg_df['Low'].min()) / entry_price * 100
                seg_open_p = float(seg_df.iloc[0]['Open'])
                seg_close_p = float(seg_df.iloc[-1]['Close'])
                seg_return = (seg_close_p - seg_open_p) / seg_open_p * 100 if seg_open_p > 0 else 0
            else:
                seg_mfe_up = seg_mfe_dn = seg_range = seg_return = 0

            row[f'{seg_name}_MFE_Up'] = seg_mfe_up
            row[f'{seg_name}_MFE_Dn'] = seg_mfe_dn
            row[f'{seg_name}_Range'] = seg_range
            row[f'{seg_name}_Return'] = seg_return
            row[f'{seg_name}_3pct_Up'] = seg_mfe_up >= SPIKE_THRESHOLD_PCT
            row[f'{seg_name}_3pct_Dn'] = seg_mfe_dn >= SPIKE_THRESHOLD_PCT

        # MFE peak timing: WHEN does the max move happen after entry?
        after_entry = day_df[day_df['_min'] > ENTRY_MINUTE]
        if len(after_entry) >= 5:
            # Find when LONG MFE peaks
            cummax_high = after_entry['High'].cummax()
            mfe_long_vals = (cummax_high - entry_price) / entry_price * 100
            if mfe_long_vals.max() >= 1.0:
                peak_idx = mfe_long_vals.idxmax()
                peak_min = after_entry.loc[peak_idx, '_min']
                row['MFE_Long_Peak_Min'] = peak_min
                row['MFE_Long_Peak_Pct'] = mfe_long_vals.max()
            else:
                row['MFE_Long_Peak_Min'] = np.nan
                row['MFE_Long_Peak_Pct'] = mfe_long_vals.max()

            # Find when SHORT MFE peaks
            cummin_low = after_entry['Low'].cummin()
            mfe_short_vals = (entry_price - cummin_low) / entry_price * 100
            if mfe_short_vals.max() >= 1.0:
                peak_idx_s = mfe_short_vals.idxmax()
                peak_min_s = after_entry.loc[peak_idx_s, '_min']
                row['MFE_Short_Peak_Min'] = peak_min_s
                row['MFE_Short_Peak_Pct'] = mfe_short_vals.max()
            else:
                row['MFE_Short_Peak_Min'] = np.nan
                row['MFE_Short_Peak_Pct'] = mfe_short_vals.max()

            # Speed: how fast does the move develop?
            # Measure MFE at 5m, 15m, 30m, 60m to see if moves are fast spikes or slow grinds
            for win_name, win_start, win_end in MFE_WINDOWS:
                win_df = day_df[(day_df['_min'] >= win_start) & (day_df['_min'] < win_end)]
                if len(win_df) >= 1:
                    win_up = (win_df['High'].max() - entry_price) / entry_price * 100
                    win_dn = (entry_price - win_df['Low'].min()) / entry_price * 100
                else:
                    win_up = win_dn = 0
                row[f'MFE_Up_{win_name}'] = win_up
                row[f'MFE_Dn_{win_name}'] = win_dn
        else:
            row['MFE_Long_Peak_Min'] = np.nan
            row['MFE_Long_Peak_Pct'] = 0
            row['MFE_Short_Peak_Min'] = np.nan
            row['MFE_Short_Peak_Pct'] = 0
            for win_name, _, _ in MFE_WINDOWS:
                row[f'MFE_Up_{win_name}'] = 0
                row[f'MFE_Dn_{win_name}'] = 0

        results.append(row)
    return results


# ==============================================================================
# SCAN ALL TICKERS
# ==============================================================================
print(f"\n{'='*W}")
print(f"  WHERE DID THE MONEY GO?")
print(f"  Full time-of-day opportunity mapping across 105 tickers")
print(f"  Jul 1 - Sep 7, 2026")
print(f"{'='*W}")

all_gcs_tickers = list_gcs_inference_tickers()
print(f"\n  GCS tickers: {len(all_gcs_tickers)}")

print(f"\n  Scanning {len(all_gcs_tickers)} tickers...")

all_daily = []
for i, ticker in enumerate(all_gcs_tickers):
    if ticker in EXCLUDE_TICKERS:
        continue
    df = load_master(ticker, tail_days=TAIL_DAYS)
    if df is None:
        continue
    metrics = compute_full_day_metrics(df, ticker)
    if metrics:
        all_daily.extend(metrics)
    del df
    gc.collect()
    if (i + 1) % 25 == 0 or i == len(all_gcs_tickers) - 1:
        print(f"    {i+1}/{len(all_gcs_tickers)} tickers, {len(all_daily)} observations")

df_all = pd.DataFrame(all_daily)
df_all['Date_dt'] = pd.to_datetime(df_all['Date'])
df_all['Week'] = df_all['Date_dt'].dt.isocalendar().week.astype(int)

print(f"\n  Total observations: {len(df_all)}")
print(f"  Tickers: {df_all['Ticker'].nunique()}")
print(f"  Date range: {df_all['Date'].min()} → {df_all['Date'].max()}")

# Assign periods
def get_period(date_str):
    for label, p_start, p_end in PERIODS:
        if p_start <= date_str <= p_end:
            return label
    return None

df_all['Period'] = df_all['Date'].apply(get_period)
df_all = df_all[df_all['Period'].notna()].copy()


# ==============================================================================
# PHASE 1: ARE MARKETS FLAT?
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  PHASE 1: ARE MARKETS FLAT?")
print(f"  Full-day range (high-low) as % of entry price")
print(f"{'='*W}\n")

for label, _, _ in PERIODS:
    p_df = df_all[df_all['Period'] == label]
    n_obs = len(p_df)
    avg_range = p_df['Full_Range_Pct'].mean()
    med_range = p_df['Full_Range_Pct'].median()
    pct_3 = p_df['Full_Range_3pct'].mean() * 100
    pct_6 = p_df['Full_Range_6pct'].mean() * 100
    avg_ret = p_df['Full_Return'].mean()
    print(f"  {label:>18}  N={n_obs:>5}  "
          f"Avg Range: {avg_range:>5.2f}%  Med: {med_range:>5.2f}%  "
          f"≥3%: {pct_3:>5.1f}%  ≥6%: {pct_6:>5.1f}%  "
          f"Avg Ret: {avg_ret:>+5.2f}%")

print(f"\n  ── By Sector (P1 vs P4) ──\n")
print(f"  {'Sector':<12}  "
      f"{'P1 AvgRange':>11} {'P1 ≥3%':>7}  "
      f"{'P4 AvgRange':>11} {'P4 ≥3%':>7}  {'Change':>8}")
print(f"  {'─'*75}")

for sector in sorted(df_all['Sector'].unique()):
    p1 = df_all[(df_all['Sector'] == sector) & (df_all['Period'] == 'P1: Jul 1-15')]
    p4 = df_all[(df_all['Sector'] == sector) & (df_all['Period'] == 'P4: Aug 16-Sep7')]
    if len(p1) < 10 or len(p4) < 10:
        continue
    p1_range = p1['Full_Range_Pct'].mean()
    p4_range = p4['Full_Range_Pct'].mean()
    p1_3 = p1['Full_Range_3pct'].mean() * 100
    p4_3 = p4['Full_Range_3pct'].mean() * 100
    change = p4_range - p1_range
    print(f"  {sector:<12}  "
          f"{p1_range:>10.2f}% {p1_3:>6.1f}%  "
          f"{p4_range:>10.2f}% {p4_3:>6.1f}%  "
          f"{change:>+7.2f}%")


# ==============================================================================
# PHASE 2: WHERE DO THE MOVES HAPPEN? (Time-of-day heatmap)
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  PHASE 2: WHERE DO THE MOVES HAPPEN?")
print(f"  % of ticker-days with 3%+ MFE (from entry) in each time segment")
print(f"{'='*W}\n")

seg_names = [s[0] for s in SEGMENTS]

print(f"  {'Period':>18}", end="")
for seg in seg_names:
    print(f"  {seg:>12}", end="")
print()
print(f"  {'─'*(20 + 14*len(seg_names))}")

for label, _, _ in PERIODS:
    p_df = df_all[df_all['Period'] == label]
    print(f"  {label:>18}", end="")
    for seg in seg_names:
        up_col = f'{seg}_3pct_Up'
        dn_col = f'{seg}_3pct_Dn'
        if up_col in p_df.columns:
            either = ((p_df[up_col]) | (p_df[dn_col])).mean() * 100
            print(f"  {either:>11.1f}%", end="")
        else:
            print(f"  {'N/A':>12}", end="")
    print()

print(f"\n  ── 3%+ MFE UP (long opportunity) by segment ──\n")
print(f"  {'Period':>18}", end="")
for seg in seg_names:
    print(f"  {seg:>12}", end="")
print()
print(f"  {'─'*(20 + 14*len(seg_names))}")

for label, _, _ in PERIODS:
    p_df = df_all[df_all['Period'] == label]
    print(f"  {label:>18}", end="")
    for seg in seg_names:
        up_col = f'{seg}_3pct_Up'
        if up_col in p_df.columns:
            rate = p_df[up_col].mean() * 100
            print(f"  {rate:>11.1f}%", end="")
        else:
            print(f"  {'N/A':>12}", end="")
    print()

print(f"\n  ── 3%+ MFE DOWN (short opportunity) by segment ──\n")
print(f"  {'Period':>18}", end="")
for seg in seg_names:
    print(f"  {seg:>12}", end="")
print()
print(f"  {'─'*(20 + 14*len(seg_names))}")

for label, _, _ in PERIODS:
    p_df = df_all[df_all['Period'] == label]
    print(f"  {label:>18}", end="")
    for seg in seg_names:
        dn_col = f'{seg}_3pct_Dn'
        if dn_col in p_df.columns:
            rate = p_df[dn_col].mean() * 100
            print(f"  {rate:>11.1f}%", end="")
        else:
            print(f"  {'N/A':>12}", end="")
    print()


# ==============================================================================
# PHASE 3: WHEN DOES MFE PEAK? (timing of max move after entry)
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  PHASE 3: WHEN DOES THE MAX MOVE HAPPEN AFTER ENTRY?")
print(f"  Distribution of MFE peak timing (minutes after 6:25 AM)")
print(f"{'='*W}\n")

for label, _, _ in PERIODS:
    p_df = df_all[df_all['Period'] == label]

    # LONG peaks
    long_peaks = p_df['MFE_Long_Peak_Min'].dropna()
    if len(long_peaks) > 10:
        long_offsets = long_peaks - ENTRY_MINUTE
        bins = [0, 5, 15, 30, 60, 120, 400]
        bin_labels = ['0-5m', '5-15m', '15-30m', '30-60m', '60-120m', '120m+']
        counts = pd.cut(long_offsets, bins=bins, labels=bin_labels).value_counts().sort_index()
        total = counts.sum()
        print(f"  {label} — LONG MFE peak timing (N={len(long_peaks)}):")
        for bl in bin_labels:
            c = counts.get(bl, 0)
            pct = c / total * 100 if total > 0 else 0
            bar = '█' * int(pct / 2)
            print(f"    {bl:>8}: {c:>5} ({pct:>5.1f}%)  {bar}")
        med_offset = long_offsets.median()
        print(f"    Median peak: {med_offset:.0f} min after entry "
              f"({(ENTRY_MINUTE + med_offset)//60:.0f}:{(ENTRY_MINUTE + med_offset)%60:02.0f} PT)")
        print()

    # SHORT peaks
    short_peaks = p_df['MFE_Short_Peak_Min'].dropna()
    if len(short_peaks) > 10:
        short_offsets = short_peaks - ENTRY_MINUTE
        counts_s = pd.cut(short_offsets, bins=bins, labels=bin_labels).value_counts().sort_index()
        total_s = counts_s.sum()
        print(f"  {label} — SHORT MFE peak timing (N={len(short_peaks)}):")
        for bl in bin_labels:
            c = counts_s.get(bl, 0)
            pct = c / total_s * 100 if total_s > 0 else 0
            bar = '█' * int(pct / 2)
            print(f"    {bl:>8}: {c:>5} ({pct:>5.1f}%)  {bar}")
        med_offset_s = short_offsets.median()
        print(f"    Median peak: {med_offset_s:.0f} min after entry "
              f"({(ENTRY_MINUTE + med_offset_s)//60:.0f}:{(ENTRY_MINUTE + med_offset_s)%60:02.0f} PT)")
        print()


# ==============================================================================
# PHASE 4: FAST SPIKES VS SLOW DRIFTS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  PHASE 4: MOVE CHARACTER — FAST SPIKES VS SLOW DRIFTS")
print(f"  Average MFE accumulated by time window after entry")
print(f"{'='*W}\n")

win_names = [w[0] for w in MFE_WINDOWS]

print(f"  {'Period':>18}", end="")
for wn in win_names:
    print(f"  {'Up_'+wn:>10}", end="")
print()
print(f"  {'─'*(20 + 12*len(win_names))}")

for label, _, _ in PERIODS:
    p_df = df_all[df_all['Period'] == label]
    print(f"  {label:>18}", end="")
    for wn in win_names:
        col = f'MFE_Up_{wn}'
        if col in p_df.columns:
            avg = p_df[col].mean()
            print(f"  {avg:>9.2f}%", end="")
    print()

print()
print(f"  {'Period':>18}", end="")
for wn in win_names:
    print(f"  {'Dn_'+wn:>10}", end="")
print()
print(f"  {'─'*(20 + 12*len(win_names))}")

for label, _, _ in PERIODS:
    p_df = df_all[df_all['Period'] == label]
    print(f"  {label:>18}", end="")
    for wn in win_names:
        col = f'MFE_Dn_{wn}'
        if col in p_df.columns:
            avg = p_df[col].mean()
            print(f"  {avg:>9.2f}%", end="")
    print()

# Spike speed ratio: what % of final 60m MFE is captured in first 15m?
print(f"\n  ── Spike Speed: % of 60m MFE captured in first 15 minutes ──\n")
for label, _, _ in PERIODS:
    p_df = df_all[df_all['Period'] == label]
    mfe_15m = p_df['MFE_Up_5-15m'].values
    mfe_60m_up = p_df['MFE_Up_30-60m'].values
    # Use cumulative: 0-5 + 5-15 vs full 60m window
    early_up = p_df['MFE_Up_0-5m'] + p_df['MFE_Up_5-15m']
    late_up = p_df['MFE_Up_30-60m'] + p_df['MFE_Up_60-120m']
    # Better: compare avg MFE at 15m vs at 60m
    col_15 = 'MFE_Up_5-15m'
    col_60 = 'MFE_Up_30-60m'
    avg_early = early_up.mean()
    avg_late = late_up.mean()
    ratio = avg_early / avg_late * 100 if avg_late > 0 else 0
    print(f"  {label:>18}  "
          f"First 15m avg: {avg_early:.2f}%  "
          f"30-120m avg: {avg_late:.2f}%  "
          f"Early/Late ratio: {ratio:.0f}%")


# ==============================================================================
# PHASE 5: TOP TICKERS WITH INTRADAY OPPORTUNITY IN P4
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  PHASE 5: WHERE IS THE MONEY NOW? (P4: Aug 16 - Sep 7)")
print(f"  Tickers with most 3%+ intraday moves in the collapse period")
print(f"{'='*W}\n")

p4_df = df_all[df_all['Period'] == 'P4: Aug 16-Sep7']

# Full-day range ranking
ticker_p4 = p4_df.groupby('Ticker').agg({
    'Full_Range_Pct': 'mean',
    'Full_Range_3pct': 'mean',
    'Full_Return': 'mean',
}).reset_index()
ticker_p4.columns = ['Ticker', 'Avg_Range', 'Pct_3plus', 'Avg_Return']
ticker_p4['Sector'] = ticker_p4['Ticker'].map(lambda t: TICKER_SECTOR.get(t, 'Unknown'))

# Add segment detail
for seg in seg_names:
    up_col = f'{seg}_3pct_Up'
    dn_col = f'{seg}_3pct_Dn'
    if up_col in p4_df.columns:
        seg_up = p4_df.groupby('Ticker')[up_col].mean().reset_index()
        seg_up.columns = ['Ticker', f'{seg}_Up_Rate']
        ticker_p4 = ticker_p4.merge(seg_up, on='Ticker', how='left')
        seg_dn = p4_df.groupby('Ticker')[dn_col].mean().reset_index()
        seg_dn.columns = ['Ticker', f'{seg}_Dn_Rate']
        ticker_p4 = ticker_p4.merge(seg_dn, on='Ticker', how='left')

ticker_p4 = ticker_p4.sort_values('Avg_Range', ascending=False)

print(f"  {'Ticker':<7} {'Sector':<12} {'AvgRange':>8} {'≥3% Days':>8} {'AvgRet':>7}  "
      f"{'PM_Up':>6} {'Open30_Up':>9} {'Morn_Up':>7} {'Mid_Up':>7}  "
      f"{'PM_Dn':>6} {'Open30_Dn':>9} {'Morn_Dn':>7} {'Mid_Dn':>7}")
print(f"  {'─'*120}")

for _, row in ticker_p4.head(30).iterrows():
    pm_up = row.get('Pre-Market_Up_Rate', 0) * 100
    o30_up = row.get('Open_30m_Up_Rate', 0) * 100
    morn_up = row.get('Morning_Up_Rate', 0) * 100
    mid_up = row.get('Midday_Up_Rate', 0) * 100
    pm_dn = row.get('Pre-Market_Dn_Rate', 0) * 100
    o30_dn = row.get('Open_30m_Dn_Rate', 0) * 100
    morn_dn = row.get('Morning_Dn_Rate', 0) * 100
    mid_dn = row.get('Midday_Dn_Rate', 0) * 100

    print(f"  {row['Ticker']:<7} {row['Sector']:<12} {row['Avg_Range']:>7.2f}% {row['Pct_3plus']*100:>7.1f}% "
          f"{row['Avg_Return']:>+6.2f}%  "
          f"{pm_up:>5.0f}% {o30_up:>8.0f}% {morn_up:>6.0f}% {mid_up:>6.0f}%  "
          f"{pm_dn:>5.0f}% {o30_dn:>8.0f}% {morn_dn:>6.0f}% {mid_dn:>6.0f}%")


# ==============================================================================
# PHASE 6: TIME SHIFT DETECTION
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  PHASE 6: DID THE OPPORTUNITY SHIFT TO LATER IN THE DAY?")
print(f"  Comparing fraction of 3%+ moves by time segment across periods")
print(f"{'='*W}\n")

# For each period, what % of all 3%+ moves (any segment) come from each segment?
print(f"  {'Period':>18}", end="")
for seg in seg_names:
    print(f"  {seg:>12}", end="")
print(f"  {'Total 3%+':>10}")
print(f"  {'─'*(20 + 14*len(seg_names) + 12)}")

for label, _, _ in PERIODS:
    p_df = df_all[df_all['Period'] == label]
    total_3_any = 0
    seg_counts = {}
    for seg in seg_names:
        up_col = f'{seg}_3pct_Up'
        dn_col = f'{seg}_3pct_Dn'
        if up_col in p_df.columns:
            n_3 = ((p_df[up_col]) | (p_df[dn_col])).sum()
            seg_counts[seg] = n_3
            total_3_any += n_3

    print(f"  {label:>18}", end="")
    for seg in seg_names:
        n = seg_counts.get(seg, 0)
        pct = n / total_3_any * 100 if total_3_any > 0 else 0
        print(f"  {pct:>11.1f}%", end="")
    print(f"  {total_3_any:>10.0f}")

# Per-sector time shift in P4
print(f"\n  ── Sector-level: where do P4 moves happen? (% of 3%+ moves by segment) ──\n")
print(f"  {'Sector':<12}", end="")
for seg in seg_names:
    print(f"  {seg:>12}", end="")
print()
print(f"  {'─'*(14 + 14*len(seg_names))}")

for sector in sorted(df_all['Sector'].unique()):
    p4_sec = df_all[(df_all['Sector'] == sector) & (df_all['Period'] == 'P4: Aug 16-Sep7')]
    if len(p4_sec) < 20:
        continue
    total_3 = 0
    seg_counts = {}
    for seg in seg_names:
        up_col = f'{seg}_3pct_Up'
        dn_col = f'{seg}_3pct_Dn'
        if up_col in p4_sec.columns:
            n_3 = ((p4_sec[up_col]) | (p4_sec[dn_col])).sum()
            seg_counts[seg] = n_3
            total_3 += n_3
    print(f"  {sector:<12}", end="")
    for seg in seg_names:
        n = seg_counts.get(seg, 0)
        pct = n / total_3 * 100 if total_3 > 0 else 0
        print(f"  {pct:>11.1f}%", end="")
    print()


# ==============================================================================
# CHARTS
# ==============================================================================
print(f"\n\n  Generating charts...")

fig, axes = plt.subplots(2, 3, figsize=(24, 14))
fig.suptitle('Where Did The Money Go? — Full Day Opportunity Map\n'
             'Jul 1 - Sep 7, 2026 | 105 tickers',
             fontsize=14, fontweight='bold')

# Chart 1: Full-day range by period (distribution)
ax = axes[0, 0]
period_labels_short = ['P1', 'P2', 'P3', 'P4']
period_colors = ['#1565C0', '#2E7D32', '#FF6F00', '#C62828']
data_ranges = []
for label, _, _ in PERIODS:
    p_df = df_all[df_all['Period'] == label]
    data_ranges.append(p_df['Full_Range_Pct'].values)

bp = ax.boxplot(data_ranges, labels=period_labels_short, patch_artist=True,
                showfliers=False, whis=[5, 95])
for patch, color in zip(bp['boxes'], period_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.axhline(3, color='red', linewidth=1, linestyle='--', alpha=0.5, label='3% threshold')
ax.axhline(6, color='darkred', linewidth=1, linestyle='--', alpha=0.3, label='6% threshold')
ax.set_ylabel('Full-Day Range (%)')
ax.set_title('Are Markets Flat?\nFull-day high-low range distribution')
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3)

# Chart 2: Time-of-day heatmap — 3%+ MFE rate by segment and period
ax = axes[0, 1]
heatmap_data = []
for label, _, _ in PERIODS:
    p_df = df_all[df_all['Period'] == label]
    row_data = []
    for seg in seg_names:
        up_col = f'{seg}_3pct_Up'
        dn_col = f'{seg}_3pct_Dn'
        if up_col in p_df.columns:
            rate = ((p_df[up_col]) | (p_df[dn_col])).mean() * 100
        else:
            rate = 0
        row_data.append(rate)
    heatmap_data.append(row_data)

hm = np.array(heatmap_data)
im = ax.imshow(hm, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(seg_names)))
ax.set_xticklabels(seg_names, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(PERIODS)))
ax.set_yticklabels(period_labels_short)
for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
        ax.text(j, i, f'{hm[i,j]:.1f}%', ha='center', va='center', fontsize=8,
                color='white' if hm[i,j] > hm.max() * 0.6 else 'black')
ax.set_title('3%+ Move Rate by Time Segment\n(higher = more opportunity)')
plt.colorbar(im, ax=ax, label='% of days with 3%+ move')

# Chart 3: MFE peak timing shift
ax = axes[0, 2]
bin_labels = ['0-5m', '5-15m', '15-30m', '30-60m', '60-120m', '120m+']
bins = [0, 5, 15, 30, 60, 120, 400]
x_pos = np.arange(len(bin_labels))
bar_w = 0.2

for pi, (label, _, _) in enumerate(PERIODS):
    p_df = df_all[df_all['Period'] == label]
    long_peaks = p_df['MFE_Long_Peak_Min'].dropna() - ENTRY_MINUTE
    if len(long_peaks) > 5:
        counts = pd.cut(long_peaks, bins=bins, labels=bin_labels).value_counts().sort_index()
        pcts = counts / counts.sum() * 100
        ax.bar(x_pos + pi * bar_w - 0.3, [pcts.get(bl, 0) for bl in bin_labels],
               bar_w, label=period_labels_short[pi], color=period_colors[pi], alpha=0.7)

ax.set_xticks(x_pos)
ax.set_xticklabels(bin_labels, rotation=45, fontsize=8)
ax.set_ylabel('% of MFE peaks')
ax.set_title('When Does LONG MFE Peak?\n(timing shift detection)')
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3)

# Chart 4: MFE accumulation curve (speed of moves)
ax = axes[1, 0]
cumulative_windows = ['0-5m', '5-15m', '15-30m', '30-60m', '60-120m', '120m+']
x_times = [2.5, 10, 22.5, 45, 90, 200]

for pi, (label, _, _) in enumerate(PERIODS):
    p_df = df_all[df_all['Period'] == label]
    cum_mfe = []
    running = 0
    for wn in cumulative_windows:
        col = f'MFE_Up_{wn}'
        if col in p_df.columns:
            running = p_df[col].mean()
        cum_mfe.append(running)
    ax.plot(x_times, cum_mfe, 'o-', color=period_colors[pi],
            linewidth=2, label=period_labels_short[pi], alpha=0.8)

ax.axhline(3, color='red', linewidth=1, linestyle='--', alpha=0.5)
ax.set_xlabel('Minutes after entry')
ax.set_ylabel('Avg MFE UP (%)')
ax.set_title('How Fast Do Moves Develop?\nMFE by time window')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xscale('log')
ax.set_xticks(x_times)
ax.set_xticklabels(['5m', '15m', '30m', '60m', '120m', '200m+'], fontsize=8)

# Chart 5: Sector opportunity in P4 (full-day range)
ax = axes[1, 1]
p4_sectors = p4_df.groupby('Sector').agg({
    'Full_Range_Pct': 'mean',
    'Full_Range_3pct': 'mean',
}).reset_index().sort_values('Full_Range_Pct', ascending=True)

colors_bar = [SECTOR_COLORS.get(s, '#333') for s in p4_sectors['Sector']]
y = np.arange(len(p4_sectors))
ax.barh(y, p4_sectors['Full_Range_Pct'], color=colors_bar, alpha=0.8)
ax.set_yticks(y)
ax.set_yticklabels(p4_sectors['Sector'])
ax.axvline(3, color='red', linewidth=1, linestyle='--', alpha=0.5)
for i, (_, row) in enumerate(p4_sectors.iterrows()):
    ax.text(row['Full_Range_Pct'] + 0.1, i,
            f"{row['Full_Range_3pct']*100:.0f}% ≥3%", fontsize=8, va='center')
ax.set_xlabel('Avg Full-Day Range (%)')
ax.set_title('P4 Sector Opportunity\n(Avg daily range, Aug 16-Sep 7)')
ax.grid(axis='x', alpha=0.3)

# Chart 6: Top 15 tickers in P4 — segment breakdown
ax = axes[1, 2]
top15_p4 = ticker_p4.head(15)
y_pos = np.arange(len(top15_p4))
seg_plot = ['Pre-Market', 'Open_30m', 'Morning', 'Midday', 'Afternoon']
seg_colors_plot = ['#1565C0', '#FF6F00', '#2E7D32', '#7B1FA2', '#C62828']
bar_h = 0.15

for si, (seg, color) in enumerate(zip(seg_plot, seg_colors_plot)):
    range_col = f'{seg}_Up_Rate'
    if range_col in top15_p4.columns:
        vals = top15_p4[range_col].fillna(0).values * 100
        ax.barh(y_pos + si * bar_h - 0.3, vals, bar_h,
                label=seg, color=color, alpha=0.7)

ax.set_yticks(y_pos)
ax.set_yticklabels(top15_p4['Ticker'], fontsize=8)
ax.set_xlabel('% days with 3%+ UP move')
ax.set_title('Top 15 P4 Tickers\nWhen do UP moves happen?')
ax.legend(fontsize=6, loc='lower right')
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()

plt.tight_layout(rect=[0, 0, 1, 0.95])
chart_path = '/tmp/where_money_went.png'
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"\n  Chart saved: {chart_path}")
if _IN_COLAB:
    plt.show()
plt.close()


# ==============================================================================
# EXECUTIVE SUMMARY
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  EXECUTIVE SUMMARY: WHERE DID THE MONEY GO?")
print(f"{'='*W}")

# Compute key summary stats
p1_df = df_all[df_all['Period'] == 'P1: Jul 1-15']
p4_df_all = df_all[df_all['Period'] == 'P4: Aug 16-Sep7']

p1_range = p1_df['Full_Range_Pct'].mean()
p4_range = p4_df_all['Full_Range_Pct'].mean()
range_change = (p4_range - p1_range) / p1_range * 100

p1_3pct = p1_df['Full_Range_3pct'].mean() * 100
p4_3pct = p4_df_all['Full_Range_3pct'].mean() * 100

# Find which segment gained most share P1→P4
seg_share_p1 = {}
seg_share_p4 = {}
for seg in seg_names:
    up_col = f'{seg}_3pct_Up'
    dn_col = f'{seg}_3pct_Dn'
    if up_col in p1_df.columns:
        seg_share_p1[seg] = ((p1_df[up_col]) | (p1_df[dn_col])).mean() * 100
        seg_share_p4[seg] = ((p4_df_all[up_col]) | (p4_df_all[dn_col])).mean() * 100

# Best segment in P4
best_seg_p4 = max(seg_share_p4, key=seg_share_p4.get) if seg_share_p4 else 'N/A'

# Time shift?
for seg in seg_names:
    delta = seg_share_p4.get(seg, 0) - seg_share_p1.get(seg, 0)
    if delta > 3:
        shift_gaining = seg
        break
else:
    shift_gaining = 'None'

print(f"""
  1. MARKET FLATNESS:
     - P1 avg full-day range: {p1_range:.2f}%   |  P4: {p4_range:.2f}%  ({range_change:+.0f}%)
     - P1 days with ≥3% range: {p1_3pct:.1f}%  |  P4: {p4_3pct:.1f}%
     - Verdict: {"Markets compressed significantly" if range_change < -15
                 else "Markets compressed moderately" if range_change < -5
                 else "Markets are NOT flat — ranges similar" if range_change > -5
                 else "Markets slightly compressed"}

  2. WHERE DO 3%+ MOVES HAPPEN? (P4 rates by segment)""")
for seg in seg_names:
    rate = seg_share_p4.get(seg, 0)
    p1_rate = seg_share_p1.get(seg, 0)
    delta = rate - p1_rate
    marker = '←←' if delta > 3 else '←' if delta > 0 else '→' if delta < -3 else ''
    print(f"     {seg:>12}: {rate:>5.1f}% (was {p1_rate:.1f}%, {delta:+.1f}%) {marker}")

print(f"""
  3. TIME SHIFT DETECTION:
     - Segment gaining share: {shift_gaining}
     - Best segment in P4: {best_seg_p4} ({seg_share_p4.get(best_seg_p4, 0):.1f}%)

  4. MOVE CHARACTER:
     - Check Phase 4 tables above for fast-spike vs slow-drift comparison

  5. IMPLICATIONS FOR STRATEGY:
""")

if range_change > -15:
    print(f"     Markets are NOT dead — daily ranges still support 3%+ moves.")
    print(f"     The opportunity may have shifted to {best_seg_p4} segment.")
    print(f"     Consider: can the model detect moves in this window?")
else:
    print(f"     Markets genuinely compressed — less opportunity overall.")
    print(f"     Best remaining segment: {best_seg_p4}.")
    print(f"     Reducing exposure is the right call until vol returns.")


# ==============================================================================
# CLEANUP
# ==============================================================================
gc.collect()
try:
    ctypes.CDLL("libc.so.6").malloc_trim(0)
except Exception:
    pass

print(f"\n{'='*W}")
print(f"  WHERE DID THE MONEY GO? — ANALYSIS COMPLETE")
print(f"  Chart: {chart_path}")
print(f"{'='*W}")
