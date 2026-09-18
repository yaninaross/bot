# %% ── Spike Resilience Analysis ──────────────────────────────────────────────
#
# Which tickers had GROWING or STABLE PM spikes through Jul 1 - Aug 15?
# These are the resilient assets — spikes didn't compress before the
# late-August regime collapse.
#
# Periods:
#   P1: Jul 1-15    (baseline)
#   P2: Jul 16-31   (mid-summer)
#   P3: Aug 1-15    (pre-collapse)
#   P4: Aug 16-Sep 7 (collapse — for context)
#
# Data source: GCS master_inference parquets
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
BQ_WF_TABLE   = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"
PST           = pytz.timezone('America/Los_Angeles')

ANALYSIS_START = '2026-07-01'
ANALYSIS_END   = '2026-09-07'
TAIL_DAYS = 75

PREMARKET_START_MIN = 360   # 6:00 AM PT
PREMARKET_END_MIN   = 390   # 6:30 AM PT
ENTRY_MINUTE        = 385   # 6:25 AM PT

SPIKE_THRESHOLD_PCT = 3.0

PERIODS = [
    ('P1: Jul 1-15',   '2026-07-01', '2026-07-15'),
    ('P2: Jul 16-31',  '2026-07-16', '2026-07-31'),
    ('P3: Aug 1-15',   '2026-08-01', '2026-08-15'),
    ('P4: Aug 16-Sep7', '2026-08-16', '2026-09-07'),
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
# DATA LOADING
# ==============================================================================
def load_master(ticker, tail_days=None):
    clean = ticker.replace("I:", "").replace("X:", "")
    file_ticker = TICKER_FILE_MAP.get(clean, clean)
    blob_path = f"{INFERENCE_DIR}/{file_ticker}_master_inference.parquet"
    local_path = f"/tmp/_resilience_{file_ticker}.parquet"
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


def compute_daily_metrics(df, ticker):
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

        pm = day_df[(day_df['_min'] >= PREMARKET_START_MIN) &
                     (day_df['_min'] < PREMARKET_END_MIN)]
        if len(pm) < 5:
            continue

        entry_bars = day_df[day_df['_min'] == ENTRY_MINUTE]
        if len(entry_bars) == 0:
            entry_bars = day_df[(day_df['_min'] >= ENTRY_MINUTE - 2) &
                                (day_df['_min'] <= ENTRY_MINUTE + 2)]
        if len(entry_bars) == 0:
            continue

        entry_price = float(entry_bars.iloc[0]['Close'])
        if entry_price <= 0:
            continue

        after_entry = day_df[day_df['_min'] > ENTRY_MINUTE]
        if len(after_entry) < 10:
            continue

        # 60-minute MFE
        after_60m = day_df[(day_df['_min'] > ENTRY_MINUTE) &
                           (day_df['_min'] <= ENTRY_MINUTE + 60)]
        if len(after_60m) >= 3:
            mfe_long = (max(after_60m['High'].values) - entry_price) / entry_price * 100
            mfe_short = (entry_price - min(after_60m['Low'].values)) / entry_price * 100
        else:
            highs_after = after_entry['High'].values
            lows_after = after_entry['Low'].values
            mfe_long = (max(highs_after) - entry_price) / entry_price * 100
            mfe_short = (entry_price - min(lows_after)) / entry_price * 100

        # Pre-market spike magnitude
        pm_open = float(pm.iloc[0]['Open'])
        if pm_open > 0:
            pm_spike_up = (pm['High'].max() - pm_open) / pm_open * 100
            pm_spike_down = (pm_open - pm['Low'].min()) / pm_open * 100
            pm_spike_mag = max(pm_spike_up, pm_spike_down)
        else:
            pm_spike_up = pm_spike_down = pm_spike_mag = 0

        close_eod = float(after_entry.iloc[-1]['Close'])
        eod_return = (close_eod - entry_price) / entry_price * 100

        results.append({
            'Ticker': ticker,
            'Date': str(date_val),
            'Date_dt': pd.Timestamp(date_val),
            'MFE_Long': mfe_long,
            'MFE_Short': mfe_short,
            'PM_Spike_Mag': pm_spike_mag,
            'PM_Spike_Up': pm_spike_up,
            'PM_Spike_Down': pm_spike_down,
            'PM_Spike_3pct': pm_spike_mag >= SPIKE_THRESHOLD_PCT,
            'Spike_Long': mfe_long >= SPIKE_THRESHOLD_PCT,
            'Spike_Short': mfe_short >= SPIKE_THRESHOLD_PCT,
            'EOD_Return': eod_return,
            'Sector': TICKER_SECTOR.get(ticker, 'Unknown'),
        })
    return results


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
# SCAN ALL TICKERS
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SPIKE RESILIENCE ANALYSIS")
print(f"  Which tickers had GROWING PM spikes through Jul 1 - Aug 15?")
print(f"  Periods: P1 Jul 1-15 | P2 Jul 16-31 | P3 Aug 1-15 | P4 Aug 16-Sep 7")
print(f"  Metrics: PM spike magnitude (360-comparable) + 60m MFE")
print(f"{'='*W}")

all_gcs_tickers = list_gcs_inference_tickers()
print(f"\n  GCS tickers: {len(all_gcs_tickers)}")

# Get deployed tickers
bq_client = bigquery.Client()
q_deploy = f"""
SELECT DISTINCT ticker FROM `{BQ_WF_TABLE}`
WHERE (model_version LIKE 'spike_transformer_long_%'
    OR model_version LIKE 'spike_transformer_short_%')
  AND survived_a = TRUE AND val_b_avg_pnl >= 0.018
"""
df_dep = bq_client.query(q_deploy).to_dataframe()
ALL_DEPLOY = set(df_dep['ticker'].tolist()) - EXCLUDE_TICKERS
print(f"  Deployed tickers: {len(ALL_DEPLOY)}")

print(f"\n  Scanning {len(all_gcs_tickers)} tickers...")

all_daily = []
for i, ticker in enumerate(all_gcs_tickers):
    if ticker in EXCLUDE_TICKERS:
        continue
    df = load_master(ticker, tail_days=TAIL_DAYS)
    if df is None:
        continue
    metrics = compute_daily_metrics(df, ticker)
    if metrics:
        all_daily.extend(metrics)
    del df
    gc.collect()
    if (i + 1) % 25 == 0 or i == len(all_gcs_tickers) - 1:
        print(f"    {i+1}/{len(all_gcs_tickers)} tickers, {len(all_daily)} observations")

df_all = pd.DataFrame(all_daily)
df_all['Date_dt'] = pd.to_datetime(df_all['Date'])
df_all['Deployed'] = df_all['Ticker'].isin(ALL_DEPLOY)
df_all['Week'] = df_all['Date_dt'].dt.isocalendar().week.astype(int)
df_all['Year_Week'] = (df_all['Date_dt'].dt.isocalendar().year.astype(str) + '-W' +
                        df_all['Date_dt'].dt.isocalendar().week.astype(str).str.zfill(2))

print(f"\n  Total observations: {len(df_all)}")
print(f"  Tickers: {df_all['Ticker'].nunique()}")
print(f"  Date range: {df_all['Date'].min()} → {df_all['Date'].max()}")


# ==============================================================================
# COMPUTE PER-TICKER PERIOD METRICS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  PER-TICKER RESILIENCE SCORING")
print(f"{'='*W}")

MIN_DAYS_PER_PERIOD = 5

ticker_resilience = []

for ticker in sorted(df_all['Ticker'].unique()):
    tk_df = df_all[df_all['Ticker'] == ticker]
    sector = TICKER_SECTOR.get(ticker, 'Unknown')
    deployed = 'Y' if ticker in ALL_DEPLOY else 'N'

    period_stats = {}
    has_all_periods = True

    for label, p_start, p_end in PERIODS:
        p_df = tk_df[(tk_df['Date'] >= p_start) & (tk_df['Date'] <= p_end)]
        if len(p_df) < MIN_DAYS_PER_PERIOD:
            has_all_periods = False
            break
        period_stats[label] = {
            'n_days': len(p_df),
            'pm_spike_rate': p_df['PM_Spike_3pct'].mean() * 100,
            'avg_pm_mag': p_df['PM_Spike_Mag'].mean(),
            'med_pm_mag': p_df['PM_Spike_Mag'].median(),
            'mfe_long_rate': p_df['Spike_Long'].mean() * 100,
            'mfe_short_rate': p_df['Spike_Short'].mean() * 100,
            'avg_mfe_long': p_df['MFE_Long'].mean(),
            'avg_mfe_short': p_df['MFE_Short'].mean(),
        }

    if not has_all_periods:
        continue

    p1 = period_stats['P1: Jul 1-15']
    p2 = period_stats['P2: Jul 16-31']
    p3 = period_stats['P3: Aug 1-15']
    p4 = period_stats['P4: Aug 16-Sep7']

    # PM spike resilience: trend from P1 to P3 (before collapse)
    pm_rates = [p1['pm_spike_rate'], p2['pm_spike_rate'], p3['pm_spike_rate']]
    pm_mags = [p1['avg_pm_mag'], p2['avg_pm_mag'], p3['avg_pm_mag']]

    # 60m MFE resilience
    mfe_l_rates = [p1['mfe_long_rate'], p2['mfe_long_rate'], p3['mfe_long_rate']]
    mfe_s_rates = [p1['mfe_short_rate'], p2['mfe_short_rate'], p3['mfe_short_rate']]
    mfe_l_avgs = [p1['avg_mfe_long'], p2['avg_mfe_long'], p3['avg_mfe_long']]
    mfe_s_avgs = [p1['avg_mfe_short'], p2['avg_mfe_short'], p3['avg_mfe_short']]

    # Resilience score: higher is better (spike activity maintained or grew)
    # PM spike: average rate across P1-P3 + bonus if growing
    pm_avg = np.mean(pm_rates)
    pm_trend = pm_rates[2] - pm_rates[0]  # P3 - P1

    # MFE: average rate across P1-P3 + bonus if growing
    mfe_l_avg = np.mean(mfe_l_rates)
    mfe_l_trend = mfe_l_rates[2] - mfe_l_rates[0]
    mfe_s_avg = np.mean(mfe_s_rates)
    mfe_s_trend = mfe_s_rates[2] - mfe_s_rates[0]

    # Combined resilience: weight PM spike (what model detects) + MFE (opportunity)
    # Positive trend = resilient, negative = compressing
    resilience_pm = pm_avg + max(pm_trend, 0) * 2
    resilience_mfe = (mfe_l_avg + mfe_s_avg) / 2 + max(mfe_l_trend + mfe_s_trend, 0)

    # P4 collapse magnitude (how much did it drop?)
    p4_pm_drop = p4['pm_spike_rate'] - np.mean(pm_rates)
    p4_mfe_l_drop = p4['avg_mfe_long'] - np.mean(mfe_l_avgs)

    ticker_resilience.append({
        'Ticker': ticker,
        'Sector': sector,
        'Deployed': deployed,
        'PM_P1': pm_rates[0], 'PM_P2': pm_rates[1], 'PM_P3': pm_rates[2],
        'PM_P4': p4['pm_spike_rate'],
        'PM_Avg_P1P3': pm_avg, 'PM_Trend': pm_trend,
        'MFE_L_P1': mfe_l_rates[0], 'MFE_L_P2': mfe_l_rates[1], 'MFE_L_P3': mfe_l_rates[2],
        'MFE_L_P4': p4['mfe_long_rate'],
        'MFE_L_Avg': mfe_l_avg, 'MFE_L_Trend': mfe_l_trend,
        'MFE_S_P1': mfe_s_rates[0], 'MFE_S_P2': mfe_s_rates[1], 'MFE_S_P3': mfe_s_rates[2],
        'MFE_S_P4': p4['mfe_short_rate'],
        'MFE_S_Avg': mfe_s_avg, 'MFE_S_Trend': mfe_s_trend,
        'AvgMag_P1': pm_mags[0], 'AvgMag_P2': pm_mags[1], 'AvgMag_P3': pm_mags[2],
        'AvgMag_P4': p4['avg_pm_mag'],
        'AvgMFE_L_P1': mfe_l_avgs[0], 'AvgMFE_L_P2': mfe_l_avgs[1],
        'AvgMFE_L_P3': mfe_l_avgs[2], 'AvgMFE_L_P4': p4['avg_mfe_long'],
        'Resilience_PM': resilience_pm,
        'Resilience_MFE': resilience_mfe,
        'P4_PM_Drop': p4_pm_drop,
        'P4_MFE_L_Drop': p4_mfe_l_drop,
    })

df_res = pd.DataFrame(ticker_resilience)


# ==============================================================================
# TOP TICKERS: PM SPIKES GREW OR STABLE (P1→P3)
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TOP TICKERS: PM SPIKE RATE GREW OR STABLE (P1 → P3)")
print(f"  Sorted by PM_Trend (P3 - P1), then PM_Avg")
print(f"{'='*W}\n")

# Filter: any PM activity in P1-P3 period
active = df_res[df_res['PM_Avg_P1P3'] > 0].copy()
active = active.sort_values(['PM_Trend', 'PM_Avg_P1P3'], ascending=[False, False])

print(f"  {'Ticker':<7} {'Sector':<12} {'Dep':>3}  "
      f"{'PM_P1':>6} {'PM_P2':>6} {'PM_P3':>6} {'PM_P4':>6}  "
      f"{'Trend':>6} {'AvgP1P3':>7}  "
      f"{'MFE_L_P3':>8} {'MFE_S_P3':>8}")
print(f"  {'─'*100}")

for _, row in active.head(40).iterrows():
    trend_marker = '+' if row['PM_Trend'] > 0 else '=' if row['PM_Trend'] == 0 else '-'
    p4_marker = '!' if row['PM_P4'] == 0 else ' '
    print(f"  {row['Ticker']:<7} {row['Sector']:<12} {row['Deployed']:>3}  "
          f"{row['PM_P1']:>5.1f}% {row['PM_P2']:>5.1f}% {row['PM_P3']:>5.1f}% {row['PM_P4']:>5.1f}%{p4_marker} "
          f"{row['PM_Trend']:>+5.1f}% {row['PM_Avg_P1P3']:>6.1f}%  "
          f"{row['MFE_L_P3']:>7.1f}% {row['MFE_S_P3']:>7.1f}%")


# ==============================================================================
# TOP TICKERS: 60m MFE GREW (P1→P3)
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TOP TICKERS: 60m MFE SPIKE RATE GREW (P1 → P3)")
print(f"  Sorted by combined LONG+SHORT MFE trend")
print(f"{'='*W}\n")

df_res['MFE_Combined_Trend'] = df_res['MFE_L_Trend'] + df_res['MFE_S_Trend']
mfe_active = df_res[df_res['MFE_L_Avg'] + df_res['MFE_S_Avg'] > 10].copy()
mfe_active = mfe_active.sort_values('MFE_Combined_Trend', ascending=False)

print(f"  {'Ticker':<7} {'Sector':<12} {'Dep':>3}  "
      f"{'MFEL_P1':>7} {'MFEL_P2':>7} {'MFEL_P3':>7} {'MFEL_P4':>7}  "
      f"{'MFES_P1':>7} {'MFES_P2':>7} {'MFES_P3':>7} {'MFES_P4':>7}  "
      f"{'LTrend':>7} {'STrend':>7}")
print(f"  {'─'*115}")

for _, row in mfe_active.head(30).iterrows():
    print(f"  {row['Ticker']:<7} {row['Sector']:<12} {row['Deployed']:>3}  "
          f"{row['MFE_L_P1']:>6.1f}% {row['MFE_L_P2']:>6.1f}% "
          f"{row['MFE_L_P3']:>6.1f}% {row['MFE_L_P4']:>6.1f}%  "
          f"{row['MFE_S_P1']:>6.1f}% {row['MFE_S_P2']:>6.1f}% "
          f"{row['MFE_S_P3']:>6.1f}% {row['MFE_S_P4']:>6.1f}%  "
          f"{row['MFE_L_Trend']:>+6.1f}% {row['MFE_S_Trend']:>+6.1f}%")


# ==============================================================================
# RESILIENCE SCORECARD: best tickers that SURVIVED P4
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  RESILIENCE SCORECARD: TICKERS THAT SURVIVED THE COLLAPSE (P4)")
print(f"  Requirement: PM or 60m MFE spike rate > 0 in P4")
print(f"{'='*W}\n")

# Tickers that still had PM spikes in P4
pm_survivors = df_res[df_res['PM_P4'] > 0].sort_values('PM_P4', ascending=False)

print(f"  ── PM Spike Survivors (had 3%+ PM spike in Aug 16 - Sep 7) ──\n")
print(f"  {'Ticker':<7} {'Sector':<12} {'Dep':>3}  "
      f"{'PM_P1':>6} {'PM_P2':>6} {'PM_P3':>6} {'PM_P4':>6}  "
      f"{'P1→P3_Trend':>11} {'P3→P4_Drop':>10}")
print(f"  {'─'*85}")

for _, row in pm_survivors.iterrows():
    trend = row['PM_Trend']
    drop = row['PM_P4'] - row['PM_P3']
    print(f"  {row['Ticker']:<7} {row['Sector']:<12} {row['Deployed']:>3}  "
          f"{row['PM_P1']:>5.1f}% {row['PM_P2']:>5.1f}% {row['PM_P3']:>5.1f}% {row['PM_P4']:>5.1f}%  "
          f"{trend:>+10.1f}% {drop:>+9.1f}%")


# ==============================================================================
# SECTOR AGGREGATION: resilience by sector
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SECTOR RESILIENCE SUMMARY")
print(f"{'='*W}\n")

print(f"  {'Sector':<12} {'N':>3}  "
      f"{'PM_P1':>6} {'PM_P2':>6} {'PM_P3':>6} {'PM_P4':>6}  "
      f"{'Trend':>6}  "
      f"{'MFE_L_P1':>8} {'MFE_L_P3':>8} {'MFE_L_P4':>8}  {'L_Trend':>8}")
print(f"  {'─'*105}")

for sector in sorted(df_res['Sector'].unique()):
    sec = df_res[df_res['Sector'] == sector]
    n = len(sec)
    print(f"  {sector:<12} {n:>3}  "
          f"{sec['PM_P1'].mean():>5.1f}% {sec['PM_P2'].mean():>5.1f}% "
          f"{sec['PM_P3'].mean():>5.1f}% {sec['PM_P4'].mean():>5.1f}%  "
          f"{sec['PM_Trend'].mean():>+5.1f}%  "
          f"{sec['MFE_L_P1'].mean():>7.1f}% {sec['MFE_L_P3'].mean():>7.1f}% "
          f"{sec['MFE_L_P4'].mean():>7.1f}%  "
          f"{sec['MFE_L_Trend'].mean():>+7.1f}%")


# ==============================================================================
# WEEKLY DETAIL: top 15 resilient tickers
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  WEEKLY PM SPIKE DETAIL — TOP 15 MOST RESILIENT TICKERS")
print(f"{'='*W}\n")

# Pick top 15 by resilience score
top15 = df_res.sort_values('Resilience_PM', ascending=False).head(15)['Ticker'].tolist()

weeks = sorted(df_all['Year_Week'].unique())

print(f"  {'Ticker':<7} {'Sector':<10} {'Dep':>3}", end="")
for wk in weeks:
    w_num = wk.split('-W')[1]
    print(f"  W{w_num:>2}", end="")
print()
print(f"  {'─'*(20 + 5*len(weeks))}")

for tk in top15:
    tk_df = df_all[df_all['Ticker'] == tk]
    sector = TICKER_SECTOR.get(tk, 'Unk')[:8]
    dep = 'Y' if tk in ALL_DEPLOY else 'N'
    print(f"  {tk:<7} {sector:<10} {dep:>3}", end="")
    for wk in weeks:
        wk_data = tk_df[tk_df['Year_Week'] == wk]
        if len(wk_data) == 0:
            print(f"  {'·':>4}", end="")
        else:
            n_pm = wk_data['PM_Spike_3pct'].sum()
            n_days = len(wk_data)
            if n_pm == 0:
                print(f"  {'0':>4}", end="")
            else:
                pct = n_pm / n_days * 100
                print(f"  {pct:>3.0f}%", end="")
    print()


# ==============================================================================
# CHARTS
# ==============================================================================
print(f"\n\n  Generating charts...")

fig, axes = plt.subplots(2, 3, figsize=(24, 14))
fig.suptitle('Spike Resilience Analysis: Jul 1 - Sep 7, 2026\n'
             'Which tickers maintained PM spike activity through mid-August?',
             fontsize=14, fontweight='bold')

# Chart 1: PM spike rate by period — top 20 tickers (stacked)
ax = axes[0, 0]
top20_pm = df_res.sort_values('PM_Avg_P1P3', ascending=False).head(20)
y_pos = np.arange(len(top20_pm))
period_labels = ['P1: Jul 1-15', 'P2: Jul 16-31', 'P3: Aug 1-15', 'P4: Aug 16+']
colors_p = ['#1565C0', '#2E7D32', '#FF6F00', '#C62828']
bar_h = 0.2

for pi, (col, label, color) in enumerate(zip(
        ['PM_P1', 'PM_P2', 'PM_P3', 'PM_P4'], period_labels, colors_p)):
    ax.barh(y_pos + pi * bar_h - 0.3, top20_pm[col].values, bar_h,
            label=label, color=color, alpha=0.8)

ax.set_yticks(y_pos)
dep_labels = [f"{r['Ticker']} ({'D' if r['Deployed']=='Y' else 'S'})"
              for _, r in top20_pm.iterrows()]
ax.set_yticklabels(dep_labels, fontsize=8)
ax.set_xlabel('PM 3%+ Spike Rate (%)')
ax.set_title('PM Spike Rate by Period\n(Top 20 by avg P1-P3)')
ax.legend(fontsize=7, loc='lower right')
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()

# Chart 2: Trend scatter — PM trend (P1→P3) vs P4 collapse
ax = axes[0, 1]
for sector in sorted(df_res['Sector'].unique()):
    sub = df_res[df_res['Sector'] == sector]
    color = SECTOR_COLORS.get(sector, '#333')
    dep = sub[sub['Deployed'] == 'Y']
    ndep = sub[sub['Deployed'] == 'N']
    if len(dep) > 0:
        ax.scatter(dep['PM_Trend'], dep['PM_P4'], color=color,
                  marker='o', s=50, alpha=0.7, label=f'{sector}')
    if len(ndep) > 0:
        ax.scatter(ndep['PM_Trend'], ndep['PM_P4'], color=color,
                  marker='^', s=50, alpha=0.5)

ax.axhline(0, color='red', linewidth=1, linestyle='--', alpha=0.5)
ax.axvline(0, color='red', linewidth=1, linestyle='--', alpha=0.5)
ax.set_xlabel('PM Spike Trend P1→P3 (positive = growing)')
ax.set_ylabel('PM Spike Rate in P4 (collapse period)')
ax.set_title('Resilience Map\nRight+Up = survived, Left+Down = collapsed')
ax.legend(fontsize=6, ncol=2)
ax.grid(alpha=0.3)

# Chart 3: Sector resilience comparison
ax = axes[0, 2]
sector_data = []
for sector in sorted(df_res['Sector'].unique()):
    sub = df_res[df_res['Sector'] == sector]
    if len(sub) < 2:
        continue
    sector_data.append({
        'Sector': sector,
        'PM_P1P3': sub['PM_Avg_P1P3'].mean(),
        'PM_P4': sub['PM_P4'].mean(),
        'MFE_L_P3': sub['MFE_L_P3'].mean(),
        'MFE_L_P4': sub['MFE_L_P4'].mean(),
    })

sd = pd.DataFrame(sector_data).sort_values('PM_P1P3', ascending=True)
y = np.arange(len(sd))
ax.barh(y - 0.15, sd['PM_P1P3'], 0.3, label='PM avg P1-P3', color='#1565C0', alpha=0.8)
ax.barh(y + 0.15, sd['PM_P4'], 0.3, label='PM P4 (collapse)', color='#C62828', alpha=0.8)
ax.set_yticks(y)
ax.set_yticklabels(sd['Sector'], fontsize=9)
ax.set_xlabel('PM 3%+ Spike Rate (%)')
ax.set_title('Sector PM Spike: Pre-Collapse vs Collapse')
ax.legend(fontsize=8)
ax.grid(axis='x', alpha=0.3)

# Chart 4: Top 10 resilient — weekly PM spike timeline
ax = axes[1, 0]
top10 = df_res.sort_values('Resilience_PM', ascending=False).head(10)['Ticker'].tolist()
week_dates = sorted(df_all['Year_Week'].unique())

for tk in top10:
    tk_df = df_all[df_all['Ticker'] == tk]
    weekly_pm = []
    for wk in week_dates:
        wk_data = tk_df[tk_df['Year_Week'] == wk]
        if len(wk_data) == 0:
            weekly_pm.append(0)
        else:
            weekly_pm.append(wk_data['PM_Spike_3pct'].mean() * 100)
    sector = TICKER_SECTOR.get(tk, 'Unknown')
    color = SECTOR_COLORS.get(sector, '#333')
    ax.plot(range(len(week_dates)), weekly_pm, 'o-', color=color,
            linewidth=1.5, markersize=4, label=tk, alpha=0.7)

ax.axvline(len(week_dates) - 3, color='red', linewidth=2, linestyle='--',
           alpha=0.5, label='P4 start')
ax.set_xticks(range(len(week_dates)))
ax.set_xticklabels([w.split('-W')[1] for w in week_dates], fontsize=7, rotation=45)
ax.set_ylabel('PM 3%+ Spike Rate (%)')
ax.set_title('Top 10 Resilient — Weekly PM Spike')
ax.legend(fontsize=6, ncol=2)
ax.grid(alpha=0.3)
ax.set_xlabel('Week Number')

# Chart 5: Top 10 resilient — weekly 60m MFE LONG
ax = axes[1, 1]
for tk in top10:
    tk_df = df_all[df_all['Ticker'] == tk]
    weekly_mfe = []
    for wk in week_dates:
        wk_data = tk_df[tk_df['Year_Week'] == wk]
        if len(wk_data) == 0:
            weekly_mfe.append(0)
        else:
            weekly_mfe.append(wk_data['MFE_Long'].mean())
    sector = TICKER_SECTOR.get(tk, 'Unknown')
    color = SECTOR_COLORS.get(sector, '#333')
    ax.plot(range(len(week_dates)), weekly_mfe, 'o-', color=color,
            linewidth=1.5, markersize=4, label=tk, alpha=0.7)

ax.axhline(3, color='red', linewidth=1, linestyle='--', alpha=0.5, label='3% spike')
ax.axvline(len(week_dates) - 3, color='red', linewidth=2, linestyle='--', alpha=0.5)
ax.set_xticks(range(len(week_dates)))
ax.set_xticklabels([w.split('-W')[1] for w in week_dates], fontsize=7, rotation=45)
ax.set_ylabel('Avg 60m MFE LONG (%)')
ax.set_title('Top 10 Resilient — Weekly 60m MFE LONG')
ax.legend(fontsize=6, ncol=2)
ax.grid(alpha=0.3)
ax.set_xlabel('Week Number')

# Chart 6: Period comparison — avg PM magnitude
ax = axes[1, 2]
top15_chart = df_res.sort_values('Resilience_PM', ascending=False).head(15)
y_pos_c = np.arange(len(top15_chart))
for pi, (col, label, color) in enumerate(zip(
        ['AvgMag_P1', 'AvgMag_P2', 'AvgMag_P3', 'AvgMag_P4'],
        period_labels, colors_p)):
    ax.barh(y_pos_c + pi * 0.2 - 0.3, top15_chart[col].values, 0.2,
            label=label, color=color, alpha=0.8)

ax.set_yticks(y_pos_c)
ax.set_yticklabels(top15_chart['Ticker'], fontsize=8)
ax.set_xlabel('Avg PM Spike Magnitude (%)')
ax.set_title('PM Magnitude by Period\n(Top 15 resilient)')
ax.legend(fontsize=7, loc='lower right')
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()

plt.tight_layout(rect=[0, 0, 1, 0.95])
chart_path = '/tmp/spike_resilience.png'
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"\n  Chart saved: {chart_path}")
if _IN_COLAB:
    plt.show()
plt.close()


# ==============================================================================
# EXECUTIVE SUMMARY
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  EXECUTIVE SUMMARY: SPIKE RESILIENCE")
print(f"{'='*W}")

# Tickers with PM growing P1→P3
pm_growers = df_res[df_res['PM_Trend'] > 0].sort_values('PM_Trend', ascending=False)
pm_stable = df_res[(df_res['PM_Trend'] >= -1) & (df_res['PM_Avg_P1P3'] > 2)]
pm_survivors_p4 = df_res[df_res['PM_P4'] > 0]

# Sectors with growing PM
sector_trends = df_res.groupby('Sector').agg({
    'PM_Trend': 'mean',
    'PM_Avg_P1P3': 'mean',
    'PM_P4': 'mean',
    'MFE_L_Trend': 'mean',
}).sort_values('PM_Trend', ascending=False)

growing_sectors = sector_trends[sector_trends['PM_Trend'] > 0].index.tolist()
stable_sectors = sector_trends[(sector_trends['PM_Trend'] >= -1) &
                                (sector_trends['PM_Avg_P1P3'] > 1)].index.tolist()

print(f"""
  1. PM SPIKE GROWERS (P1 → P3, before collapse):
     - {len(pm_growers)} tickers with INCREASING PM spike rate
     - Top: {', '.join(pm_growers.head(5)['Ticker'].tolist())}

  2. PM SPIKE STABLE (maintained ≥2% through P1-P3):
     - {len(pm_stable)} tickers
     - Top: {', '.join(pm_stable.sort_values('PM_Avg_P1P3', ascending=False).head(5)['Ticker'].tolist())}

  3. PM SPIKE SURVIVORS (still had PM spikes in P4):
     - {len(pm_survivors_p4)} tickers
     - {', '.join(pm_survivors_p4.sort_values('PM_P4', ascending=False).head(10)['Ticker'].tolist())}

  4. SECTOR TRENDS (P1 → P3):
     - Growing: {', '.join(growing_sectors) if growing_sectors else 'None'}
     - Stable: {', '.join(stable_sectors) if stable_sectors else 'None'}
     - Collapsed: {', '.join(sector_trends[sector_trends['PM_Trend'] < -2].index.tolist())}

  5. KEY FINDING:
""")

if len(pm_growers) > 10:
    print(f"     Many tickers ({len(pm_growers)}) had GROWING spikes through mid-August.")
    print(f"     The compression hit market-wide in P4 — not a gradual per-ticker decline.")
elif len(pm_growers) > 3:
    print(f"     A handful of tickers ({len(pm_growers)}) maintained growing spikes.")
    print(f"     These may be worth concentrating on during compressed regimes.")
else:
    print(f"     Almost all tickers were already compressing before the P4 collapse.")
    print(f"     The regime shift was building gradually, not a sudden break.")


# ==============================================================================
# CLEANUP
# ==============================================================================
gc.collect()
try:
    ctypes.CDLL("libc.so.6").malloc_trim(0)
except Exception:
    pass

print(f"\n{'='*W}")
print(f"  SPIKE RESILIENCE ANALYSIS COMPLETE")
print(f"  Chart: {chart_path}")
print(f"{'='*W}")
