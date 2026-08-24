# %% ── Build BTC/ETH Master Inference Files ─────────────────────────────────────
#
# Downloads 1-min OHLCV data for BTC-USD and ETH-USD from Polygon.io,
# builds master_inference parquet files matching the format used by
# existing equity tickers, and uploads to:
#   gs://time-series-bucket-ys/Data/Inference_v2/BTCUSD_master_inference.parquet
#   gs://time-series-bucket-ys/Data/Inference_v2/ETHUSD_master_inference.parquet
#
# Crypto trades 24/7 — ~1440 bars/day, ~525k bars/year.
# Full rebuild from 2022-01-01 to today.
#
# Polygon free tier: 5 requests/minute. We chunk by month to stay well within
# the 50k-bar-per-request limit (~44k bars/month for 24/7 crypto).
#
# Colab-ready. No GPU needed.

import os, sys, time
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

try:
    from polygon import RESTClient
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "polygon-api-client", "--quiet"])
    from polygon import RESTClient

from google.cloud import storage

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

W = 80


def log(msg, level='INFO'):
    ts = datetime.now(timezone.utc).strftime('%H:%M:%S')
    print(f"  [{ts}] {level:>5}  {msg}", flush=True)


# ==============================================================================
# CONFIG
# ==============================================================================
POLYGON_API_KEY = "2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs"

GCS_BUCKET    = 'time-series-bucket-ys'
GCS_PREFIX    = 'Data/Inference_v2'
LOCAL_DIR     = '/tmp/crypto_master_inference'
os.makedirs(LOCAL_DIR, exist_ok=True)

ASSETS = {
    'BTCUSD': 'X:BTCUSD',
    'ETHUSD': 'X:ETHUSD',
}

START_DATE = '2022-01-01'
END_DATE   = datetime.now(timezone.utc).strftime('%Y-%m-%d')

REQUESTS_PER_MINUTE = 5
REQUEST_DELAY       = 60.0 / REQUESTS_PER_MINUTE + 0.5  # ~12.5s between requests


# ==============================================================================
# HELPERS
# ==============================================================================
def month_ranges(start_str, end_str):
    """Generate (start_date, end_date) tuples for each calendar month."""
    start = datetime.strptime(start_str, '%Y-%m-%d')
    end   = datetime.strptime(end_str, '%Y-%m-%d')
    ranges = []
    current = start.replace(day=1)
    while current <= end:
        month_start = max(current, start)
        if current.month == 12:
            next_month = current.replace(year=current.year + 1, month=1)
        else:
            next_month = current.replace(month=current.month + 1)
        month_end = min(next_month - timedelta(days=1), end)
        ranges.append((month_start.strftime('%Y-%m-%d'), month_end.strftime('%Y-%m-%d')))
        current = next_month
    return ranges


def download_crypto_bars(poly_client, polygon_ticker, start_date, end_date):
    """Download all 1-min bars for a crypto pair from Polygon."""
    months = month_ranges(start_date, end_date)
    total_months = len(months)
    all_bars = []
    total_bars = 0
    failed_months = []

    log(f"Downloading {polygon_ticker}: {start_date} to {end_date} ({total_months} months)")

    for i, (m_start, m_end) in enumerate(months):
        attempt = 0
        max_retries = 3
        success = False

        while attempt < max_retries and not success:
            attempt += 1
            try:
                bars = list(poly_client.get_aggs(
                    polygon_ticker, 1, 'minute',
                    m_start, m_end,
                    limit=50000,
                ))
                n_bars = len(bars)
                total_bars += n_bars

                if n_bars > 0:
                    first_ts = datetime.fromtimestamp(bars[0].timestamp / 1000, tz=timezone.utc)
                    last_ts  = datetime.fromtimestamp(bars[-1].timestamp / 1000, tz=timezone.utc)
                    all_bars.extend(bars)
                    log(
                        f"[{i+1}/{total_months}] {m_start} to {m_end}: "
                        f"{n_bars:,} bars  ({first_ts.strftime('%Y-%m-%d %H:%M')} -> "
                        f"{last_ts.strftime('%Y-%m-%d %H:%M')})  "
                        f"running total: {total_bars:,}"
                    )
                else:
                    log(f"[{i+1}/{total_months}] {m_start} to {m_end}: 0 bars", 'WARN')

                success = True

            except Exception as e:
                log(
                    f"[{i+1}/{total_months}] {m_start} to {m_end}: "
                    f"RETRY {attempt}/{max_retries} — {e}", 'WARN'
                )
                time.sleep(REQUEST_DELAY * 2)

        if not success:
            failed_months.append((m_start, m_end))
            log(f"[{i+1}/{total_months}] {m_start} to {m_end}: FAILED after {max_retries} retries", 'ERROR')

        time.sleep(REQUEST_DELAY)

    log(f"Download complete: {total_bars:,} total bars, {len(failed_months)} failed months")
    if failed_months:
        log(f"Failed months: {failed_months}", 'ERROR')

    return all_bars, failed_months


def bars_to_dataframe(bars):
    """Convert Polygon bar objects to DataFrame matching master_inference format."""
    records = []
    for bar in bars:
        records.append({
            'timestamp': pd.Timestamp(bar.timestamp, unit='ms', tz='UTC'),
            'Open':   float(bar.open),
            'High':   float(bar.high),
            'Low':    float(bar.low),
            'Close':  float(bar.close),
            'Volume': float(bar.volume),
        })

    df = pd.DataFrame(records)
    df = df.set_index('timestamp')
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]

    return df


def validate_dataframe(df, ticker, start_date, end_date):
    """Run validation checks and print results."""
    log(f"Validation for {ticker}:")
    log(f"  Shape:        {df.shape}")
    log(f"  Date range:   {df.index.min()} to {df.index.max()}")
    log(f"  Columns:      {list(df.columns)}")
    log(f"  Index name:   {df.index.name}")
    log(f"  Index dtype:  {df.index.dtype}")

    nan_counts = df.isna().sum()
    if nan_counts.sum() > 0:
        log(f"  NaN counts:   {nan_counts.to_dict()}", 'WARN')
    else:
        log(f"  NaN counts:   None")

    log(f"  Close range:  {df['Close'].min():.2f} to {df['Close'].max():.2f}")
    log(f"  Volume range: {df['Volume'].min():.4f} to {df['Volume'].max():.4f}")

    df_years = df.index.year.value_counts().sort_index()
    for year, count in df_years.items():
        expected = 1440 * 365
        coverage = count / expected * 100
        log(f"  {year}: {count:>8,} bars ({coverage:.1f}% of theoretical max)")

    target_end = pd.Timestamp(end_date, tz='UTC')
    actual_end = df.index.max()
    gap_days = (target_end - actual_end).total_seconds() / 86400
    if gap_days > 2:
        log(f"  Gap to target end: {gap_days:.1f} days (target: {end_date})", 'WARN')
    else:
        log(f"  Gap to target end: {gap_days:.1f} days — OK")

    if ticker == 'BTCUSD':
        try:
            old_df = pd.read_parquet(f'gs://{GCS_BUCKET}/Data/Inference/BTCUSD_master_inference.parquet')
            overlap_end = min(df.index.max(), old_df.index.max())
            overlap_start = max(df.index.min(), old_df.index.min())
            old_overlap = old_df[(old_df.index >= overlap_start) & (old_df.index <= overlap_end)]
            new_overlap = df[(df.index >= overlap_start) & (df.index <= overlap_end)]
            common_idx = old_overlap.index.intersection(new_overlap.index)
            if len(common_idx) > 0:
                close_diff = (old_overlap.loc[common_idx, 'Close'] - new_overlap.loc[common_idx, 'Close']).abs()
                log(f"  Cross-check vs old file: {len(common_idx):,} overlapping bars")
                log(f"  Close price diff: mean={close_diff.mean():.4f}, max={close_diff.max():.4f}")
            del old_df
        except Exception as e:
            log(f"  (Could not cross-check: {e})")

    return True


def upload_to_gcs(local_path, gcs_path):
    """Upload file to GCS."""
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    size_mb = os.path.getsize(local_path) / 1e6
    log(f"Uploaded: gs://{GCS_BUCKET}/{gcs_path} ({size_mb:.1f} MB)")


# ==============================================================================
# MAIN
# ==============================================================================
print(f"\n{'=' * W}", flush=True)
print(f"  BUILD CRYPTO MASTER INFERENCE FILES (BTC + ETH)", flush=True)
print(f"{'=' * W}", flush=True)
print(f"  Source:     Polygon.io (1-min bars)", flush=True)
print(f"  Assets:     {list(ASSETS.keys())}", flush=True)
print(f"  Date range: {START_DATE} to {END_DATE}", flush=True)
print(f"  Target:     gs://{GCS_BUCKET}/{GCS_PREFIX}/", flush=True)
print(f"  Rate limit: {REQUESTS_PER_MINUTE} req/min ({REQUEST_DELAY:.1f}s delay)", flush=True)
print(f"{'=' * W}", flush=True)

poly_client = RESTClient(POLYGON_API_KEY)

all_failed = {}
summary = {}

for ticker, polygon_ticker in ASSETS.items():
    print(f"\n{'─' * W}", flush=True)
    print(f"  Processing: {ticker} ({polygon_ticker})", flush=True)
    print(f"{'─' * W}", flush=True)

    t0 = time.time()

    # Download
    bars, failed = download_crypto_bars(poly_client, polygon_ticker, START_DATE, END_DATE)
    if failed:
        all_failed[ticker] = failed

    if not bars:
        log(f"{ticker}: NO DATA DOWNLOADED — skipping", 'ERROR')
        continue

    # Convert to DataFrame
    log(f"Converting {len(bars):,} bars to DataFrame...")
    df = bars_to_dataframe(bars)
    del bars

    # Validate
    validate_dataframe(df, ticker, START_DATE, END_DATE)

    # Save locally
    local_path = os.path.join(LOCAL_DIR, f'{ticker}_master_inference.parquet')
    df.to_parquet(local_path, engine='pyarrow', compression='snappy')
    local_size = os.path.getsize(local_path) / 1e6
    log(f"Saved locally: {local_path} ({local_size:.1f} MB)")

    # Upload to GCS (Inference_v2)
    gcs_path = f'{GCS_PREFIX}/{ticker}_master_inference.parquet'
    upload_to_gcs(local_path, gcs_path)

    elapsed = time.time() - t0
    summary[ticker] = {
        'rows': len(df),
        'date_min': str(df.index.min()),
        'date_max': str(df.index.max()),
        'size_mb': local_size,
        'elapsed_s': elapsed,
        'failed_months': len(failed),
    }

    log(f"{ticker} complete: {len(df):,} rows, {elapsed:.0f}s")
    del df


# ==============================================================================
# VERIFICATION — read back from GCS and confirm
# ==============================================================================
print(f"\n{'=' * W}", flush=True)
print(f"  VERIFICATION — Reading back from GCS", flush=True)
print(f"{'=' * W}", flush=True)

for ticker in ASSETS:
    gcs_uri = f'gs://{GCS_BUCKET}/{GCS_PREFIX}/{ticker}_master_inference.parquet'
    try:
        df = pd.read_parquet(gcs_uri)
        log(f"{ticker}:")
        log(f"  GCS URI:    {gcs_uri}")
        log(f"  Shape:      {df.shape}")
        log(f"  Date range: {df.index.min()} to {df.index.max()}")
        log(f"  Columns:    {list(df.columns)}")
        log(f"  Index:      name={df.index.name}, dtype={df.index.dtype}")

        log(f"  Last 5 rows:")
        for idx, row in df.tail(5).iterrows():
            log(f"    {idx}  O={row['Open']:.2f}  H={row['High']:.2f}  "
                f"L={row['Low']:.2f}  C={row['Close']:.2f}  V={row['Volume']:.4f}")

        log(f"  Load test: simulating xasset data_prep load_lead_lag_asset()...")
        df_test = df.reset_index()
        dt_cols = [c for c in df_test.columns if pd.api.types.is_datetime64_any_dtype(df_test[c])]
        if dt_cols:
            log(f"  Datetime column detected: {dt_cols[0]}")
        ohlcv_ok = all(c in df_test.columns for c in ['Open', 'High', 'Low', 'Close', 'Volume'])
        log(f"  OHLCV columns present: {ohlcv_ok}")

        del df, df_test
    except Exception as e:
        log(f"{ticker}: VERIFICATION FAILED — {e}", 'ERROR')


# ==============================================================================
# SUMMARY
# ==============================================================================
print(f"\n{'=' * W}", flush=True)
print(f"  BUILD COMPLETE", flush=True)
print(f"{'=' * W}", flush=True)

for ticker, s in summary.items():
    print(f"  {ticker}:", flush=True)
    print(f"    Rows:          {s['rows']:,}", flush=True)
    print(f"    Date range:    {s['date_min']} to {s['date_max']}", flush=True)
    print(f"    File size:     {s['size_mb']:.1f} MB", flush=True)
    print(f"    Download time: {s['elapsed_s']:.0f}s ({s['elapsed_s']/60:.1f} min)", flush=True)
    print(f"    Failed months: {s['failed_months']}", flush=True)

if all_failed:
    print(f"\n  FAILED MONTHS (may have gaps):", flush=True)
    for ticker, months in all_failed.items():
        print(f"    {ticker}: {months}", flush=True)
else:
    print(f"\n  All months downloaded successfully — no gaps.", flush=True)

print(f"\n  Files in GCS:", flush=True)
print(f"    gs://{GCS_BUCKET}/{GCS_PREFIX}/BTCUSD_master_inference.parquet", flush=True)
print(f"    gs://{GCS_BUCKET}/{GCS_PREFIX}/ETHUSD_master_inference.parquet", flush=True)
print(f"\n  Next: run ra_moe_short_L5_2_xasset_data_prep.py", flush=True)
print(f"{'=' * W}", flush=True)
