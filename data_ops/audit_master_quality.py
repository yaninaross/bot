"""
audit_master_quality.py — Check which master_inference files have real microstructure data
==========================================================================================
For each master file in GCS, reports:
  - Total rows, date range
  - Whether micro columns (Spread, OBI, Volume_Delta, etc.) have non-zero data
  - Date from which micro data starts (if present)
  - Valid scoring rows post-cutoff (what the scoring script actually uses)

Outputs a summary table categorizing files as COMPLETE (OHLCV+micro) or OHLCV-ONLY.
"""

import pandas as pd
import numpy as np
from google.cloud import storage

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

client = storage.Client()
bucket = client.bucket(os.environ.get("GCS_BUCKET", ""))

MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']

CONTEXT_SKIP = {'SPY', 'VIX', 'TLT', 'IWM', 'SMH', 'BTCUSD', 'ITA', 'XLE'}

CUTOFF_160 = pd.Timestamp('2025-03-01', tz='UTC')

# ==============================================================================
# Discover all master_inference files
# ==============================================================================
print("Scanning GCS for master_inference files...")
master_blobs = list(bucket.list_blobs(prefix='Data/Inference_v2/'))
tickers = []
for b in master_blobs:
    if b.name.endswith('_master_inference.parquet'):
        ticker = b.name.replace('Data/Inference_v2/', '').replace('_master_inference.parquet', '')
        if ticker and ticker not in CONTEXT_SKIP:
            tickers.append(ticker)
tickers = sorted(set(tickers))
print(f"Found {len(tickers)} master files\n")

# ==============================================================================
# Audit each file
# ==============================================================================
results = []

for ticker in tickers:
    uri = f"gs://{GCS_BUCKET}/Data/Inference_v2/{ticker}_master_inference.parquet"
    try:
        df = pd.read_parquet(uri)
        df = df.reset_index()

        # Find datetime column
        dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        if dt_cols:
            df['Datetime'] = pd.to_datetime(df[dt_cols[0]], utc=True)
        else:
            num_col = next((c for c in df.columns
                           if c in ('Datetime', 'Date', 'Time', 'Timestamp', 'Level_0')),
                          df.columns[0])
            df['Datetime'] = pd.to_datetime(df[num_col], unit='ms', utc=True)
        df = df.set_index('Datetime').sort_index()

        total_rows = len(df)
        date_start = df.index.min()
        date_end = df.index.max()

        # Check micro columns
        micro_present = []
        micro_nonzero_pct = {}
        for col in MICRO_COLS:
            if col in df.columns:
                nz = (df[col] != 0).sum()
                pct = nz / len(df) * 100 if len(df) > 0 else 0
                micro_nonzero_pct[col] = pct
                if nz > 0:
                    micro_present.append(col)

        # Find first date with non-zero micro data
        micro_start = None
        if micro_present and 'Volume_Delta' in df.columns:
            micro_mask = df['Volume_Delta'] != 0
            if micro_mask.any():
                micro_start = df.index[micro_mask][0]

        # Classify
        vd_pct = micro_nonzero_pct.get('Volume_Delta', 0)
        if vd_pct > 5:
            quality = 'COMPLETE'
        elif vd_pct > 0:
            quality = 'PARTIAL'
        else:
            quality = 'OHLCV-ONLY'

        # Count valid scoring rows post-cutoff
        post_cutoff = df[df.index > CUTOFF_160]
        if 'Volume' in df.columns:
            valid_rows = len(post_cutoff[post_cutoff['Volume'] > 0])
        else:
            valid_rows = len(post_cutoff)

        results.append({
            'Ticker': ticker,
            'Total_Rows': total_rows,
            'Start': date_start.strftime('%Y-%m-%d'),
            'End': date_end.strftime('%Y-%m-%d'),
            'Micro_Start': micro_start.strftime('%Y-%m-%d') if micro_start else '-',
            'VD_Pct': round(vd_pct, 1),
            'OBI_Pct': round(micro_nonzero_pct.get('OBI', 0), 1),
            'Quality': quality,
            'Scoring_Rows_160': valid_rows,
        })

        print(f"  {ticker:8s}  {total_rows:>9,} rows  {date_start.strftime('%Y-%m-%d')} → {date_end.strftime('%Y-%m-%d')}  "
              f"micro_start={micro_start.strftime('%Y-%m-%d') if micro_start else 'NONE':>10s}  "
              f"VD={vd_pct:5.1f}%  {quality}")

    except Exception as e:
        print(f"  {ticker:8s}  ERROR: {e}")
        results.append({
            'Ticker': ticker, 'Total_Rows': 0, 'Start': '-', 'End': '-',
            'Micro_Start': '-', 'VD_Pct': 0, 'OBI_Pct': 0,
            'Quality': 'ERROR', 'Scoring_Rows_160': 0,
        })

# ==============================================================================
# Summary
# ==============================================================================
df_results = pd.DataFrame(results)

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
for q in ['COMPLETE', 'PARTIAL', 'OHLCV-ONLY', 'ERROR']:
    subset = df_results[df_results['Quality'] == q]
    if len(subset) > 0:
        print(f"\n{q}: {len(subset)} tickers")
        for _, r in subset.iterrows():
            print(f"  {r['Ticker']:8s}  rows={r['Total_Rows']:>9,}  "
                  f"range={r['Start']}→{r['End']}  "
                  f"micro_from={r['Micro_Start']:>10s}  "
                  f"scoring_rows={r['Scoring_Rows_160']:>6,}")

print(f"\nTotal: {len(df_results)} tickers")
print(f"  COMPLETE:   {len(df_results[df_results['Quality']=='COMPLETE'])}")
print(f"  PARTIAL:    {len(df_results[df_results['Quality']=='PARTIAL'])}")
print(f"  OHLCV-ONLY: {len(df_results[df_results['Quality']=='OHLCV-ONLY'])}")
