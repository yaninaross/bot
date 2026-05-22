"""
list_tq_coverage.py — T&Q Data Coverage Scanner
=================================================
Lists all tickers with raw trades/quotes data in GCS,
shows date range and file counts, and flags which are
currently scored in daily_score_combined.py.

Run in Colab:  python list_tq_coverage.py
"""

from google.cloud import storage as gcs_storage
from collections import defaultdict

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

BUCKET_NAME = "time-series-bucket-ys"
TQ_PREFIX = "Data/TradesQuotes/"

SCORED_TICKERS = {
    'RIOT', 'MIGI', 'HOOD',
    'CLSK', 'APLD',
    'IREN', 'VRT',
    'KTOS', 'LUNR', 'RKLB',
    'JOBY', 'ASTS',
    'RCAT', 'BBAI', 'CIFR', 'HUT', 'KEEL', 'WULF', 'BTBT', 'ACHR',
}

client = gcs_storage.Client()
bucket = client.bucket(BUCKET_NAME)

ticker_trades = defaultdict(set)
ticker_quotes = defaultdict(set)

print("Scanning GCS blobs (may take a minute)...")
for blob in bucket.list_blobs(prefix=TQ_PREFIX):
    parts = blob.name.replace(TQ_PREFIX, "").split("/")
    if len(parts) != 2:
        continue
    ticker = parts[0]
    fname = parts[1]
    if not fname.endswith(".parquet"):
        continue
    if "_trades_" in fname:
        date_str = fname.split("_trades_")[1].replace(".parquet", "")
        ticker_trades[ticker].add(date_str)
    elif "_quotes_" in fname:
        date_str = fname.split("_quotes_")[1].replace(".parquet", "")
        ticker_quotes[ticker].add(date_str)

all_tickers = sorted(set(ticker_trades.keys()) | set(ticker_quotes.keys()))

W = 100
print(f"\n{'='*W}")
print(f"  T&Q DATA COVERAGE — {len(all_tickers)} tickers in GCS")
print(f"{'='*W}")
print(f"  {'Ticker':<8} {'Trades':>7} {'Quotes':>7} {'Both':>6} "
      f"{'First':>12} {'Last':>12}  {'Scored':>6}")
print(f"  {'─'*8} {'─'*7} {'─'*7} {'─'*6} {'─'*12} {'─'*12}  {'─'*6}")

scored_count = 0
not_scored = []

for ticker in all_tickers:
    t_dates = ticker_trades[ticker]
    q_dates = ticker_quotes[ticker]
    both = t_dates & q_dates
    all_dates = sorted(t_dates | q_dates)
    first = all_dates[0] if all_dates else "—"
    last = all_dates[-1] if all_dates else "—"
    is_scored = ticker in SCORED_TICKERS
    tag = "YES" if is_scored else ""
    if is_scored:
        scored_count += 1
    else:
        not_scored.append((ticker, len(both), first, last))

    print(f"  {ticker:<8} {len(t_dates):>7} {len(q_dates):>7} {len(both):>6} "
          f"{first:>12} {last:>12}  {tag:>6}")

print(f"\n  Total: {len(all_tickers)} tickers, {scored_count} currently scored")

print(f"\n{'='*W}")
print(f"  NOT SCORED ({len(not_scored)} tickers with T&Q data but no daily scoring)")
print(f"{'='*W}")
for ticker, n_days, first, last in sorted(not_scored, key=lambda x: -x[1]):
    print(f"  {ticker:<8}  {n_days:>5} days  ({first} → {last})")

print(f"{'='*W}")
