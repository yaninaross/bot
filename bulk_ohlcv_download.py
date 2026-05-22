"""
bulk_ohlcv_download.py — Bulk OHLCV Parquet Downloader
======================================================
Downloads 1-minute OHLCV bars from Polygon for ALL assets needed by
all experiments. Saves each ticker as:
  gs://{BUCKET}/Data/Inference/{clean_ticker}_1m_full_historical_data.parquet

Uploads via GCS API (not FUSE mount) for reliable persistence.

Usage:
  python bulk_ohlcv_download.py                  # all tickers
  python bulk_ohlcv_download.py VST CEG CCJ      # specific tickers only
"""

import sys
import io
import requests
import pandas as pd
from datetime import datetime
import time
from google.cloud import storage as gcs_storage

try:
    import google.colab
    if hasattr(google.colab, 'auth'):
        google.colab.auth.authenticate_user()
except ImportError:
    pass

# ==============================================================================
# CONFIGURATION
# ==============================================================================
API_KEY = '2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs'

BUCKET_NAME = "time-series-bucket-ys"
GCS_DATA_DIR = "Data/Inference"

gcs_client = gcs_storage.Client()
bucket = gcs_client.bucket(BUCKET_NAME)

START_DATE = '2022-01-01'

# ==============================================================================
# ASSET LISTS
# ==============================================================================
TARGET_TICKERS = [
    # --- Energy / AI Power cluster (EXP-150/151 core) ---
    'VST', 'CEG', 'GEV', 'BE', 'CCJ', 'NEE', 'DUK',
    # --- Space / Dual-Use (EXP-190) ---
    'PL', 'RDW', 'BKSY', 'MNTS', 'VOYG',
    # --- Defense Primes ---
    'GD', 'LHX', 'LMT', 'NOC', 'RTX', 'GE',
    # --- Crypto crossovers (EXP-151 teachers + EXP-152) ---
    'CIFR', 'IREN', 'HUT', 'MARA', 'WULF', 'APLD', 'RIOT', 'CLSK',
    'KEEL', 'MSTR', 'COIN',
    # --- Defense crossovers (EXP-151 teachers + EXP-152) ---
    'RCAT', 'BBAI', 'LUNR', 'ASTS', 'ACHR', 'MVIS', 'RKLB',
    # --- Energy expansion pool (EXP-151 scoring candidates) ---
    'VRT', 'SMCI', 'PLUG', 'ENPH', 'FCEL', 'FCX', 'CLF', 'MP',
    'XOM', 'CVX', 'COP', 'ETN', 'CAT', 'BHP', 'LIN', 'EQIX', 'ALAB',
    'TLN', 'NRG', 'PEG', 'BEPC', 'BWXT', 'OKLO', 'SMR', 'NXE',
    'POWL', 'MOD', 'HUBB', 'WMB', 'EQT', 'LNG', 'KMI',
    'FLNC', 'FSLR', 'NXT', 'TPL',
]

CONTEXT_ASSETS = [
    # --- OHLCV-only (no microstructure) ---
    'XLE',
    # ITA, SMH, MOD, VIX, TLT, IWM, BTCUSD, SPY — already downloaded
]

ALL_ASSETS = list(dict.fromkeys(TARGET_TICKERS + CONTEXT_ASSETS))


def clean_ticker(ticker):
    """Strip Polygon prefixes for filenames."""
    return ticker.replace("I:", "").replace("X:", "")


MAX_RETRIES = 3

def download_ohlcv(ticker, start_date, end_date):
    """Download all 1-minute OHLCV bars for a single ticker via Polygon pagination."""
    url = (
        f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{start_date}/{end_date}'
        f'?adjusted=true&sort=asc&limit=50000&apiKey={API_KEY}'
    )

    all_frames = []
    page = 0

    while url:
        retries = 0
        while retries <= MAX_RETRIES:
            try:
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                break
            except requests.exceptions.HTTPError:
                if resp.status_code == 429:
                    wait = 60 * (retries + 1)
                    print(f"      Rate limited — waiting {wait}s...")
                    time.sleep(wait)
                    retries += 1
                    continue
                else:
                    print(f"      HTTP {resp.status_code}: {resp.text[:200]}")
                    return pd.concat(all_frames, ignore_index=True) if all_frames else None
            except (ValueError, requests.exceptions.JSONDecodeError) as e:
                retries += 1
                if retries <= MAX_RETRIES:
                    wait = 15 * retries
                    print(f"      JSON decode error — retry {retries}/{MAX_RETRIES} in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"      JSON decode failed after {MAX_RETRIES} retries: {e}")
                    return pd.concat(all_frames, ignore_index=True) if all_frames else None
            except requests.exceptions.RequestException as e:
                retries += 1
                if retries <= MAX_RETRIES:
                    wait = 15 * retries
                    print(f"      Network error — retry {retries}/{MAX_RETRIES} in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"      Network error after {MAX_RETRIES} retries: {e}")
                    return pd.concat(all_frames, ignore_index=True) if all_frames else None
        else:
            return pd.concat(all_frames, ignore_index=True) if all_frames else None

        if 'results' in data and data.get('resultsCount', 0) > 0:
            df = pd.DataFrame(data['results'])
            all_frames.append(df)
            page += 1
            total = sum(len(d) for d in all_frames)
            print(f"      Page {page}: +{len(df)} rows  (total: {total:,})")

            if 'next_url' in data:
                url = data['next_url'] + f'&apiKey={API_KEY}'
            else:
                url = None
        else:
            if page == 0:
                print(f"      No data returned from Polygon")
            url = None

    if not all_frames:
        return None

    final_df = pd.concat(all_frames, ignore_index=True)
    final_df = final_df.rename(columns={
        't': 'Timestamp', 'o': 'Open', 'h': 'High',
        'l': 'Low', 'c': 'Close', 'v': 'Volume'
    })
    final_df['Timestamp'] = final_df['Timestamp'] / 1000
    final_df['Datetime'] = pd.to_datetime(final_df['Timestamp'], unit='s')
    final_df.set_index('Datetime', inplace=True)
    final_df.sort_index(inplace=True)
    final_df = final_df[['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]

    return final_df


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == '__main__':
    end_date = datetime.now().strftime('%Y-%m-%d')

    cli_args = [a.upper() for a in sys.argv[1:]
                if not a.startswith('-') and not a.endswith('.json')]
    if cli_args:
        assets = cli_args
    else:
        assets = list(ALL_ASSETS)

    total = len(assets)
    fresh, appended, up_to_date, failed = [], [], [], []

    print(f"{'='*60}")
    print(f"  BULK OHLCV DOWNLOAD — {total} assets")
    print(f"  Range: {START_DATE} → {end_date}")
    print(f"  Output: gs://{BUCKET_NAME}/{GCS_DATA_DIR}/")
    print(f"{'='*60}\n")

    for i, ticker in enumerate(assets, 1):
        ct = clean_ticker(ticker)
        gcs_path = f"{GCS_DATA_DIR}/{ct}_1m_full_historical_data.parquet"
        blob = bucket.blob(gcs_path)

        df_existing = None
        fetch_start = START_DATE

        if blob.exists():
            blob.reload()
            size_mb = blob.size / (1024 * 1024)
            print(f"[{i}/{total}] {ticker} — existing file ({size_mb:.1f} MB), checking freshness...")
            try:
                df_existing = pd.read_parquet(
                    io.BytesIO(blob.download_as_bytes()))
                last_ts = df_existing.index.max()
                last_date = pd.Timestamp(last_ts).strftime('%Y-%m-%d')
                days_stale = (pd.Timestamp(end_date) - pd.Timestamp(last_date)).days
                print(f"      Last data: {last_date}  ({days_stale} days ago)")
                if days_stale <= 1:
                    print(f"      Up to date — nothing to fetch\n")
                    up_to_date.append(ct)
                    continue
                fetch_start = (pd.Timestamp(last_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                print(f"      Fetching {fetch_start} → {end_date}")
            except Exception as e:
                print(f"      Could not read existing file: {e}")
                print(f"      Re-downloading from {START_DATE}")
                df_existing = None
        else:
            print(f"[{i}/{total}] {ticker} → new download")

        df_new = download_ohlcv(ticker, fetch_start, end_date)

        if df_new is not None and not df_new.empty:
            if df_existing is not None:
                before = len(df_existing)
                if df_existing.index.tz is not None and df_new.index.tz is None:
                    df_new.index = df_new.index.tz_localize(df_existing.index.tz)
                elif df_existing.index.tz is None and df_new.index.tz is not None:
                    df_existing.index = df_existing.index.tz_localize(df_new.index.tz)
                df_combined = pd.concat([df_existing, df_new])
                df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
                df_combined.sort_index(inplace=True)
                added = len(df_combined) - before
                print(f"      Appended {added:,} new rows (total: {len(df_combined):,})")
                df_final = df_combined
                appended.append(ct)
            else:
                df_final = df_new
                print(f"      Downloaded {len(df_final):,} rows")
                fresh.append(ct)

            buf = io.BytesIO()
            df_final.to_parquet(buf, engine='pyarrow', compression='snappy')
            buf.seek(0)
            blob.upload_from_file(buf, content_type='application/octet-stream')
            print(f"      Saved to GCS\n")
            del df_final
        else:
            if df_existing is not None:
                print(f"      No new data from Polygon — keeping existing file\n")
                up_to_date.append(ct)
            else:
                print(f"      FAILED — no data returned\n")
                failed.append(ct)

        if df_existing is not None:
            del df_existing
        if i < total:
            time.sleep(2)

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print(f"\n{'='*60}")
    print(f"  COMPLETE:")
    print(f"    Fresh downloads: {len(fresh)}")
    print(f"    Appended:        {len(appended)}")
    print(f"    Already current: {len(up_to_date)}")
    if failed:
        print(f"    FAILED:          {', '.join(failed)}")
    print(f"{'='*60}")
