"""
audit_master_coverage.py — Data coverage audit for Exp 400 teacher candidates
===============================================================================
For each ticker in the Exp 400 candidate pool (~117 across 14 sectors):
  - Checks if master_inference file exists on GCS
  - Reports OHLCV data range (first/last Close)
  - Reports micro feature range (first/last non-zero Spread/OBI/etc.)
  - Flags gaps vs training requirements (2022-02-01 through 2026-01-01+)

Colab-ready. No GPU needed.
"""

import io
import gc
import sys
import subprocess
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

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

client = gcs_storage.Client()
bucket = client.bucket("time-series-bucket-ys")

# ==============================================================================
# CONFIG
# ==============================================================================
BUCKET_NAME = "time-series-bucket-ys"
GCS_MASTER_DIR = "Data/Inference_v2"

TRAIN_START = pd.Timestamp("2022-02-01", tz='UTC')
LAST_FOLD_CUTOFF = pd.Timestamp("2026-01-01", tz='UTC')

MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

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

# ==============================================================================
# BUILD TICKER LIST
# ==============================================================================
ticker_to_sector = {}
all_tickers = []
for sec, info in SECTORS.items():
    for t in info['tickers']:
        ticker_to_sector[t] = sec
        if t not in all_tickers:
            all_tickers.append(t)

W = 120
print("=" * W)
print("  MASTER INFERENCE DATA COVERAGE AUDIT — Exp 400 Teacher Candidates")
print("=" * W)
print(f"  Tickers to audit: {len(all_tickers)} across {len(SECTORS)} sectors")
print(f"  Training range needed: {TRAIN_START.strftime('%Y-%m-%d')} → {LAST_FOLD_CUTOFF.strftime('%Y-%m-%d')}+")
print(f"  Master dir: gs://{BUCKET_NAME}/{GCS_MASTER_DIR}/")
print()

# ==============================================================================
# AUDIT EACH TICKER
# ==============================================================================
results = []
no_master = []

for ticker in sorted(all_tickers):
    file_ticker = TICKER_FILE_MAP.get(ticker, ticker)
    blob_path = f"{GCS_MASTER_DIR}/{file_ticker}_master_inference.parquet"
    blob = bucket.blob(blob_path)

    if not blob.exists():
        no_master.append(ticker)
        results.append({
            'ticker': ticker,
            'sector': ticker_to_sector[ticker],
            'has_master': False,
            'ohlcv_start': None, 'ohlcv_end': None,
            'micro_start': None, 'micro_end': None,
            'n_rows': 0,
            'flags': ['NO_MASTER'],
        })
        print(f"  {ticker:<8} NO MASTER FILE")
        continue

    try:
        data = blob.download_as_bytes()
        df = pd.read_parquet(io.BytesIO(data))
        del data

        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df = df.sort_index()

        n_rows = len(df)

        ohlcv_start = None
        ohlcv_end = None
        if 'Close' in df.columns:
            valid_close = df['Close'].dropna()
            if len(valid_close) > 0:
                ohlcv_start = valid_close.index.min()
                ohlcv_end = valid_close.index.max()

        micro_start = None
        micro_end = None
        available_micro = [c for c in MICRO_COLS if c in df.columns]
        if available_micro:
            micro_mask = (df[available_micro].abs() > 0).any(axis=1)
            micro_rows = df[micro_mask]
            if len(micro_rows) > 0:
                micro_start = micro_rows.index.min()
                micro_end = micro_rows.index.max()

        flags = []
        if ohlcv_start is not None and ohlcv_start > TRAIN_START:
            flags.append('OHLCV_LATE')
        if ohlcv_end is not None and ohlcv_end < LAST_FOLD_CUTOFF:
            flags.append('OHLCV_SHORT')
        if micro_start is None:
            flags.append('NO_MICRO')
        elif micro_start > TRAIN_START:
            gap_months = (micro_start - TRAIN_START).days / 30.4
            flags.append(f'MICRO_LATE({gap_months:.0f}mo)')
        if micro_end is not None and ohlcv_end is not None and micro_end < ohlcv_end - pd.Timedelta(days=7):
            gap_days = (ohlcv_end - micro_end).days
            flags.append(f'MICRO_ENDS_EARLY({gap_days}d)')

        results.append({
            'ticker': ticker,
            'sector': ticker_to_sector[ticker],
            'has_master': True,
            'ohlcv_start': ohlcv_start,
            'ohlcv_end': ohlcv_end,
            'micro_start': micro_start,
            'micro_end': micro_end,
            'n_rows': n_rows,
            'flags': flags,
        })

        flag_str = '  '.join(flags) if flags else 'OK'
        o_s = ohlcv_start.strftime('%Y-%m-%d') if ohlcv_start else '—'
        o_e = ohlcv_end.strftime('%Y-%m-%d') if ohlcv_end else '—'
        m_s = micro_start.strftime('%Y-%m-%d') if micro_start else '—'
        m_e = micro_end.strftime('%Y-%m-%d') if micro_end else '—'
        print(f"  {ticker:<8} OHLCV: {o_s}→{o_e}  MICRO: {m_s}→{m_e}  rows={n_rows:>9,}  {flag_str}")

        del df
        gc.collect()

    except Exception as e:
        print(f"  {ticker:<8} ERROR: {e}")
        results.append({
            'ticker': ticker,
            'sector': ticker_to_sector[ticker],
            'has_master': False,
            'ohlcv_start': None, 'ohlcv_end': None,
            'micro_start': None, 'micro_end': None,
            'n_rows': 0,
            'flags': [f'ERROR:{e}'],
        })

# ==============================================================================
# TABLE 1: SUMMARY BY SECTOR
# ==============================================================================
print(f"\n{'=' * W}")
print(f"  TABLE 1: SECTOR COVERAGE SUMMARY")
print(f"{'=' * W}")
print(f"  {'Sector':<20} {'Total':>5} {'Master':>6} {'Full':>4} {'MicroLate':>9} {'NoMicro':>7} {'NoMaster':>8}")
print(f"  {'─'*20} {'─'*5} {'─'*6} {'─'*4} {'─'*9} {'─'*7} {'─'*8}")

for sec_key in sorted(SECTORS.keys()):
    sec_results = [r for r in results if r['sector'] == sec_key]
    total = len(sec_results)
    has_master = sum(1 for r in sec_results if r['has_master'])
    full_ok = sum(1 for r in sec_results if r['has_master'] and not r['flags'])
    micro_late = sum(1 for r in sec_results if any('MICRO_LATE' in f for f in r['flags']))
    no_micro = sum(1 for r in sec_results if 'NO_MICRO' in r['flags'])
    no_master_cnt = sum(1 for r in sec_results if not r['has_master'])
    label = SECTORS[sec_key]['label']
    print(f"  {label:<20} {total:>5} {has_master:>6} {full_ok:>4} {micro_late:>9} {no_micro:>7} {no_master_cnt:>8}")

total_all = len(results)
has_master_all = sum(1 for r in results if r['has_master'])
full_ok_all = sum(1 for r in results if r['has_master'] and not r['flags'])
micro_late_all = sum(1 for r in results if any('MICRO_LATE' in f for f in r['flags']))
no_micro_all = sum(1 for r in results if 'NO_MICRO' in r['flags'])
no_master_all = len(no_master)
print(f"  {'─'*20} {'─'*5} {'─'*6} {'─'*4} {'─'*9} {'─'*7} {'─'*8}")
print(f"  {'ALL':<20} {total_all:>5} {has_master_all:>6} {full_ok_all:>4} {micro_late_all:>9} {no_micro_all:>7} {no_master_all:>8}")

# ==============================================================================
# TABLE 2: TICKERS WITH FLAGS (problems only)
# ==============================================================================
flagged = [r for r in results if r['flags']]
if flagged:
    print(f"\n{'=' * W}")
    print(f"  TABLE 2: TICKERS WITH DATA GAPS ({len(flagged)} flagged)")
    print(f"{'=' * W}")
    print(f"  {'Ticker':<8} {'Sector':<18} {'OHLCV Start':>11} {'OHLCV End':>11} {'Micro Start':>11} {'Micro End':>11}  Flags")
    print(f"  {'─'*8} {'─'*18} {'─'*11} {'─'*11} {'─'*11} {'─'*11}  {'─'*30}")

    for r in sorted(flagged, key=lambda x: (x['sector'], x['ticker'])):
        o_s = r['ohlcv_start'].strftime('%Y-%m-%d') if r['ohlcv_start'] else '—'
        o_e = r['ohlcv_end'].strftime('%Y-%m-%d') if r['ohlcv_end'] else '—'
        m_s = r['micro_start'].strftime('%Y-%m-%d') if r['micro_start'] else '—'
        m_e = r['micro_end'].strftime('%Y-%m-%d') if r['micro_end'] else '—'
        sec_label = SECTORS.get(r['sector'], {}).get('label', r['sector'])
        flag_str = '  '.join(r['flags'])
        print(f"  {r['ticker']:<8} {sec_label:<18} {o_s:>11} {o_e:>11} {m_s:>11} {m_e:>11}  {flag_str}")

# ==============================================================================
# TABLE 3: TICKERS WITH NO MASTER FILE
# ==============================================================================
if no_master:
    print(f"\n{'=' * W}")
    print(f"  TABLE 3: NO MASTER FILE ({len(no_master)} tickers)")
    print(f"{'=' * W}")
    by_sec = {}
    for t in no_master:
        sec = ticker_to_sector[t]
        if sec not in by_sec:
            by_sec[sec] = []
        by_sec[sec].append(t)
    for sec in sorted(by_sec.keys()):
        label = SECTORS[sec]['label']
        print(f"  {label:<20} {', '.join(sorted(by_sec[sec]))}")

# ==============================================================================
# TABLE 4: CLEAN TICKERS (full coverage, no flags)
# ==============================================================================
clean = [r for r in results if r['has_master'] and not r['flags']]
if clean:
    print(f"\n{'=' * W}")
    print(f"  TABLE 4: FULL COVERAGE — NO GAPS ({len(clean)} tickers)")
    print(f"{'=' * W}")
    by_sec = {}
    for r in clean:
        sec = r['sector']
        if sec not in by_sec:
            by_sec[sec] = []
        by_sec[sec].append(r['ticker'])
    for sec in sorted(by_sec.keys()):
        label = SECTORS[sec]['label']
        print(f"  {label:<20} ({len(by_sec[sec]):>2}) {', '.join(sorted(by_sec[sec]))}")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print(f"\n{'=' * W}")
print(f"  FINAL SUMMARY")
print(f"{'=' * W}")
print(f"  Total candidates:     {total_all}")
print(f"  Have master file:     {has_master_all}")
print(f"  Full coverage (OK):   {full_ok_all}")
print(f"  Micro starts late:    {micro_late_all}")
print(f"  No micro at all:      {no_micro_all}")
print(f"  No master file:       {no_master_all}")
print()
print(f"  Training range: {TRAIN_START.strftime('%Y-%m-%d')} → {LAST_FOLD_CUTOFF.strftime('%Y-%m-%d')}")
print(f"  'Full coverage' = OHLCV starts ≤ {TRAIN_START.strftime('%Y-%m-%d')}, micro starts ≤ {TRAIN_START.strftime('%Y-%m-%d')},")
print(f"                    OHLCV ends ≥ {LAST_FOLD_CUTOFF.strftime('%Y-%m-%d')}, micro ends within 7d of OHLCV end")
print("=" * W)
