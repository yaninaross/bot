# %% ── Download Polygon News & Sentiment Data ────────────────────────────────
#
# Downloads ALL news articles from Polygon.io for deploy tickers + context assets.
# Stores raw article-level data with timestamps and pre-computed sentiment.
#
# Each article row contains:
#   - published_utc      : exact publication timestamp (leakage-safe boundary)
#   - ticker             : the ticker this insight applies to
#   - sentiment          : positive / neutral / negative (Polygon pre-computed)
#   - sentiment_reasoning: full text explanation
#   - title              : article title
#   - description        : article summary
#   - publisher          : source name (Zacks, Motley Fool, Benzinga, etc.)
#   - article_url        : link to full article
#   - keywords           : comma-separated keyword list
#   - all_tickers        : all tickers mentioned in the article
#   - article_id         : Polygon article ID (dedup key)
#
# One article may produce MULTIPLE rows if it mentions multiple tickers
# (each row has that ticker's specific sentiment insight).
# Articles WITHOUT sentiment insights are stored with sentiment='none'.
#
# Writes: gs://time-series-bucket-ys/Data/Sentiment/{ticker}_news.parquet
#
# Incremental: detects existing output, fetches only articles newer than
# the latest stored published_utc per ticker.
#
# Colab-ready, config vars at top.

import os, sys, time, gc, ctypes, warnings
import numpy as np
import pandas as pd
import pytz
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

try:
    from polygon import RESTClient
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "polygon-api-client", "--quiet"])
    from polygon import RESTClient

try:
    from google.cloud import storage as gcs_storage
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "google-cloud-storage", "pyarrow", "--quiet"])
    from google.cloud import storage as gcs_storage

try:
    from google.colab import auth
    auth.authenticate_user()
    _IN_COLAB = True
except (ImportError, Exception):
    _IN_COLAB = False

import matplotlib
if not _IN_COLAB:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ==============================================================================
# CONFIG
# ==============================================================================
POLYGON_API_KEY = "YOUR_POLYGON_API_KEY"
BUCKET_NAME     = "time-series-bucket-ys"
GCS_OUT_DIR     = "Data/Sentiment"

DATA_START = '2024-01-01'

DEPLOY_TICKERS_BATCH1 = [
    'ACHR', 'ACMR', 'AEHR', 'ALAB', 'AMD', 'APLD', 'ARM', 'ASTS', 'AVAV', 'AXTI',
    'BBAI', 'BE', 'BEAM', 'BKSY', 'BLZE', 'BTBT', 'CAT', 'CEG', 'CIFR', 'CLSK',
    'COIN', 'CORZ', 'CRDO', 'CVX', 'ETN', 'FCEL', 'FLNC', 'GEV', 'HOOD', 'HUT',
    'IREN', 'JOBY', 'KEEL', 'KTOS', 'LUNR',
]

DEPLOY_TICKERS_BATCH2 = [
    'MARA', 'MNTS', 'MRNA', 'MSTR', 'MU', 'MVIS', 'MXL', 'NBIS', 'NEE', 'NNE',
    'NTLA', 'OKLO', 'ONDS', 'PL', 'PLTR', 'PLUG', 'RCAT', 'RDW', 'RIOT', 'RKLB',
    'RXT', 'SLS', 'SMCI', 'SMR', 'SNDK', 'SOUN', 'TSLA', 'VOYG', 'VRT', 'VST',
    'WOLF', 'WULF', 'XOM',
]

CONTEXT_TICKERS = ['SPY', 'VIX', 'TLT', 'IWM', 'ITA']

SECTOR_PROXY_TICKERS = ['MARA', 'MSTR']

ALL_TICKERS = sorted(set(
    DEPLOY_TICKERS_BATCH1 + DEPLOY_TICKERS_BATCH2 +
    CONTEXT_TICKERS + SECTOR_PROXY_TICKERS
))

TICKER_RENAMES = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

ARTICLES_PER_REQUEST = 1000
REQUEST_DELAY_S = 0.1

poly_client = RESTClient(POLYGON_API_KEY)
gcs_client = gcs_storage.Client()
bucket = gcs_client.bucket(BUCKET_NAME)


def _release_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


# ==============================================================================
# GCS I/O
# ==============================================================================
def _read_existing(ticker):
    blob_path = f"{GCS_OUT_DIR}/{ticker}_news.parquet"
    local = f"/tmp/_news_{ticker}.parquet"
    try:
        bucket.blob(blob_path).download_to_filename(local)
        df = pd.read_parquet(local, engine='pyarrow')
        os.remove(local)
        return df
    except Exception:
        if os.path.exists(local):
            os.remove(local)
        return None


def _save_parquet(df, ticker):
    blob_path = f"{GCS_OUT_DIR}/{ticker}_news.parquet"
    local = f"/tmp/_news_{ticker}.parquet"
    df.to_parquet(local, engine='pyarrow', index=False)
    bucket.blob(blob_path).upload_from_filename(local)
    size_mb = os.path.getsize(local) / 1e6
    os.remove(local)
    return size_mb


# ==============================================================================
# DOWNLOAD ONE TICKER
# ==============================================================================
def download_ticker_news(ticker, since_utc=None):
    """
    Download all news articles mentioning `ticker` from Polygon.
    If since_utc is set, only fetch articles published after that timestamp.

    Returns a list of dicts, one per (article, ticker_insight) pair.
    Articles without insights get a single row with sentiment='none'.
    """
    rows = []
    seen_ids = set()

    kwargs = {
        'ticker': ticker,
        'limit': ARTICLES_PER_REQUEST,
        'order': 'desc',
        'sort': 'published_utc',
    }
    if since_utc:
        kwargs['published_utc_gte'] = since_utc

    try:
        for article in poly_client.list_ticker_news(**kwargs):
            aid = article.id
            if aid in seen_ids:
                continue
            seen_ids.add(aid)

            base = {
                'article_id': aid,
                'published_utc': article.published_utc,
                'title': article.title or '',
                'description': (article.description or '')[:2000],
                'publisher': article.publisher.name if article.publisher else '',
                'article_url': article.article_url or '',
                'keywords': ','.join(article.keywords) if article.keywords else '',
                'all_tickers': ','.join(article.tickers) if article.tickers else '',
            }

            if article.insights:
                for ins in article.insights:
                    if ins.ticker == ticker:
                        rows.append({
                            **base,
                            'ticker': ticker,
                            'sentiment': ins.sentiment or 'none',
                            'sentiment_reasoning': (ins.sentiment_reasoning or '')[:2000],
                        })
                        break
                else:
                    rows.append({
                        **base,
                        'ticker': ticker,
                        'sentiment': 'none',
                        'sentiment_reasoning': '',
                    })
            else:
                rows.append({
                    **base,
                    'ticker': ticker,
                    'sentiment': 'none',
                    'sentiment_reasoning': '',
                })

            if len(rows) % 500 == 0:
                time.sleep(REQUEST_DELAY_S)

    except Exception as e:
        print(f"    API error for {ticker}: {e}")

    return rows


# ==============================================================================
# PROCESS ONE TICKER (download + merge with existing + save)
# ==============================================================================
def process_ticker(ticker):
    """Download news for one ticker, merge with existing data, save to GCS."""
    t0 = time.time()

    existing = _read_existing(ticker)
    since_utc = None
    existing_count = 0

    if existing is not None and not existing.empty:
        existing_count = len(existing)
        latest = pd.to_datetime(existing['published_utc']).max()
        since_utc = latest.strftime('%Y-%m-%dT%H:%M:%SZ')

    new_rows = download_ticker_news(ticker, since_utc=since_utc)

    if not new_rows and existing is not None:
        elapsed = time.time() - t0
        return existing_count, 0, elapsed

    new_df = pd.DataFrame(new_rows)

    if existing is not None and not existing.empty:
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['article_id'], keep='last')
    else:
        combined = new_df

    if combined.empty:
        return 0, 0, time.time() - t0

    combined['published_utc'] = pd.to_datetime(combined['published_utc'], utc=True)

    if DATA_START:
        cutoff = pd.Timestamp(DATA_START, tz='UTC')
        combined = combined[combined['published_utc'] >= cutoff]

    combined = combined.sort_values('published_utc', ascending=False).reset_index(drop=True)

    size_mb = _save_parquet(combined, ticker)
    elapsed = time.time() - t0
    new_count = len(combined) - existing_count

    return len(combined), max(new_count, 0), elapsed


# ==============================================================================
# MAIN
# ==============================================================================
t_total = time.time()

print(f"{'=' * 80}")
print(f"  DOWNLOAD POLYGON NEWS & SENTIMENT DATA")
print(f"  Tickers: {len(ALL_TICKERS)} ({len(DEPLOY_TICKERS_BATCH1)+len(DEPLOY_TICKERS_BATCH2)} deploy + "
      f"{len(CONTEXT_TICKERS)} context)")
print(f"  Data start: {DATA_START}")
print(f"  Output: gs://{BUCKET_NAME}/{GCS_OUT_DIR}/")
print(f"{'=' * 80}")

stats = []
total_new = 0
total_articles = 0

for i, ticker in enumerate(ALL_TICKERS):
    print(f"\n  [{i+1}/{len(ALL_TICKERS)}] {ticker}...", end=' ')

    try:
        total, new, elapsed = process_ticker(ticker)
        total_articles += total
        total_new += new

        n_sent = '?'
        existing = _read_existing(ticker)
        if existing is not None and 'sentiment' in existing.columns:
            n_sent = (existing['sentiment'] != 'none').sum()

        stats.append({
            'ticker': ticker,
            'total': total,
            'new': new,
            'with_sentiment': n_sent,
            'time_s': elapsed,
        })

        if new > 0:
            print(f"{total} total (+{new} new), {elapsed:.1f}s")
        else:
            print(f"{total} total (up to date), {elapsed:.1f}s")

    except Exception as e:
        print(f"ERROR: {e}")
        stats.append({
            'ticker': ticker, 'total': 0, 'new': 0,
            'with_sentiment': 0, 'time_s': 0,
        })

    if (i + 1) % 10 == 0:
        _release_memory()

total_elapsed = time.time() - t_total


# ==============================================================================
# SUMMARY
# ==============================================================================
stats_df = pd.DataFrame(stats)
stats_df = stats_df.sort_values('total', ascending=False)

print(f"\n\n{'=' * 80}")
print(f"  DOWNLOAD COMPLETE")
print(f"{'=' * 80}")
print(f"  Tickers processed: {len(ALL_TICKERS)}")
print(f"  Total articles:    {total_articles:,}")
print(f"  New articles:      {total_new:,}")
print(f"  Total time:        {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
print(f"  Output:            gs://{BUCKET_NAME}/{GCS_OUT_DIR}/")

print(f"\n  {'Ticker':<8} {'Total':>7} {'New':>6} {'w/Sent':>7} {'Time':>6}")
print(f"  {'─' * 40}")
for _, row in stats_df.iterrows():
    print(f"  {row['ticker']:<8} {row['total']:>7} {row['new']:>6} "
          f"{row['with_sentiment']:>7} {row['time_s']:>5.1f}s")


# ==============================================================================
# COVERAGE CHART
# ==============================================================================
if len(stats_df) > 0 and stats_df['total'].sum() > 0:
    fig, axes = plt.subplots(1, 3, figsize=(20, max(6, len(stats_df) * 0.3)))

    # [0] Total articles per ticker
    ax = axes[0]
    colors = ['#4a90d9' if t not in CONTEXT_TICKERS else '#e8a838'
              for t in stats_df['ticker']]
    ax.barh(stats_df['ticker'], stats_df['total'], color=colors)
    ax.set_xlabel('Total Articles')
    ax.set_title('News Coverage per Ticker\n(blue=deploy, orange=context)')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # [1] Articles with sentiment
    ax = axes[1]
    sent_vals = pd.to_numeric(stats_df['with_sentiment'], errors='coerce').fillna(0)
    ax.barh(stats_df['ticker'], sent_vals, color='#50b848')
    ax.set_xlabel('Articles with Sentiment')
    ax.set_title('Sentiment Coverage')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # [2] Sentiment coverage ratio
    ax = axes[2]
    ratio = (sent_vals / stats_df['total'].clip(lower=1) * 100)
    bar_colors = ['#2E7D32' if r >= 50 else '#FFA000' if r >= 20 else '#C62828'
                  for r in ratio]
    ax.barh(stats_df['ticker'], ratio, color=bar_colors)
    ax.set_xlabel('% with Sentiment')
    ax.set_title('Sentiment Ratio\n(green>=50%, yellow>=20%, red<20%)')
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    chart_path = '/tmp/polygon_news_coverage.png'
    plt.savefig(chart_path, dpi=120, bbox_inches='tight')
    print(f"\n  Chart saved: {chart_path}")
    if _IN_COLAB:
        plt.show()
    plt.close()

    # === Second chart: temporal coverage ===
    print(f"\n  Building temporal coverage chart...")
    fig2, ax2 = plt.subplots(figsize=(16, max(6, len(ALL_TICKERS) * 0.25)))

    for yi, ticker in enumerate(sorted(ALL_TICKERS)):
        existing = _read_existing(ticker)
        if existing is None or existing.empty:
            continue
        dates = pd.to_datetime(existing['published_utc'])
        if dates.empty:
            continue
        monthly = dates.dt.to_period('M').value_counts().sort_index()
        for period, count in monthly.items():
            start = period.start_time
            intensity = min(count / 20.0, 1.0)
            ax2.barh(yi, 30, left=start, height=0.8,
                     color=plt.cm.Blues(0.3 + 0.7 * intensity),
                     edgecolor='none')

    ax2.set_yticks(range(len(sorted(ALL_TICKERS))))
    ax2.set_yticklabels(sorted(ALL_TICKERS), fontsize=7)
    ax2.set_xlabel('Date')
    ax2.set_title('News Coverage Timeline (darker = more articles/month)')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    chart2_path = '/tmp/polygon_news_timeline.png'
    plt.savefig(chart2_path, dpi=120, bbox_inches='tight')
    print(f"  Timeline chart saved: {chart2_path}")
    if _IN_COLAB:
        plt.show()
    plt.close()


# ==============================================================================
# CLEANUP
# ==============================================================================
_release_memory()

print(f"\n{'=' * 80}")
print(f"  NEWS DOWNLOAD COMPLETE")
print(f"{'=' * 80}")
