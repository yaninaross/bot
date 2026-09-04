# %% ── LLM-Based News Sentiment Scoring ─────────────────────────────────────
#
# Scores each news article using Gemini Flash to extract 12 nuanced features
# designed for intraday spike prediction.
#
# Features per article:
#   1. sentiment_score    — continuous [-1.0, +1.0] directional sentiment
#   2. price_impact       — [0-10] expected magnitude of price move
#   3. event_type         — categorical event classification
#   4. is_catalyst        — bool: will this cause price action within 24h?
#   5. surprise_factor    — [0-10] how unexpected is this news?
#   6. information_quality— [0-10] substantive reporting vs clickbait
#   7. retail_fomo_score  — [0-10] likelihood of retail buying/selling panic
#   8. volatility_catalyst— [0-10] will this widen intraday range?
#   9. urgency            — [0-10] breaking news vs old recap
#  10. forward_looking    — [0-10] future catalysts vs past events
#  11. overreaction_risk  — [0-10] headline more sensational than substance?
#  12. premarket_driver   — [0-10] will this drive pre-market/early session volume?
#
# Reads:  gs://time-series-bucket-ys/Data/Sentiment/{ticker}_news.parquet
# Writes: gs://time-series-bucket-ys/Data/Sentiment/{ticker}_llm_scores.parquet
#
# Incremental: skips articles already scored.
# Colab-ready, config vars at top.

import os, sys, time, gc, json, re, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

try:
    from google.cloud import storage as gcs_storage
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "google-cloud-storage", "pyarrow", "--quiet"])
    from google.cloud import storage as gcs_storage

try:
    from google import genai
    from google.genai import types
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "google-genai", "--quiet"])
    from google import genai
    from google.genai import types

try:
    from google.colab import auth
    try:
        auth.authenticate_user()
    except Exception:
        pass
    _IN_COLAB = True
except ImportError:
    _IN_COLAB = False

import matplotlib
if not _IN_COLAB:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ==============================================================================
# CONFIG
# ==============================================================================
GCP_PROJECT     = "yanina-everything"
GCP_LOCATION    = "us-central1"
GEMINI_MODEL    = "gemini-2.5-flash"

# --- Auth mode ---
# Option A: Vertex AI (needs `gcloud services enable aiplatform.googleapis.com`)
# Option B: Google AI Studio API key (free, get at aistudio.google.com/apikey)
GOOGLE_AI_API_KEY = ""   # <-- paste API key here for Option B; leave empty for Vertex

BUCKET_NAME     = "time-series-bucket-ys"
SENTIMENT_DIR   = "Data/Sentiment"

ANALYSIS_TICKERS = [
    'TSLA', 'AMD', 'PLTR', 'MU', 'CVX', 'RKLB',
]

MAX_WORKERS     = 10
BATCH_SAVE_EVERY = 200
REQUEST_DELAY_S  = 0.15
MAX_RETRIES     = 3

VALID_EVENT_TYPES = [
    'earnings', 'analyst_rating', 'fda_regulatory', 'contract_deal',
    'offering_dilution', 'insider_activity', 'short_interest',
    'macro_sector', 'legal', 'partnership', 'product_launch',
    'restructuring', 'crypto_market', 'government_policy',
    'routine_mention', 'other',
]

gcs_client = gcs_storage.Client()
bucket = gcs_client.bucket(BUCKET_NAME)

# --- Enable Vertex AI API if needed ---
if not GOOGLE_AI_API_KEY:
    import subprocess
    print("  Enabling Vertex AI API...")
    try:
        subprocess.check_call(
            ["gcloud", "services", "enable", "aiplatform.googleapis.com",
             f"--project={GCP_PROJECT}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print("  Vertex AI API enabled.")
    except Exception as e:
        print(f"  Warning: could not enable API automatically: {e}")

# --- Initialize GenAI client ---
if GOOGLE_AI_API_KEY:
    genai_client = genai.Client(api_key=GOOGLE_AI_API_KEY)
    print(f"  Using Google AI Studio (API key)")
else:
    genai_client = genai.Client(
        vertexai=True,
        project=GCP_PROJECT,
        location=GCP_LOCATION,
    )
    print(f"  Using Vertex AI ({GCP_PROJECT}/{GCP_LOCATION})")


# ==============================================================================
# PROMPT
# ==============================================================================
SYSTEM_PROMPT = """You are a financial news analyst scoring articles for their impact on INTRADAY stock price movement (specifically the first 1-2 hours of US market trading).

You will receive a news article's title, description, and ticker. Return a JSON object with exactly these fields:

{
  "sentiment_score": <float -1.0 to 1.0, where -1.0 is extremely bearish, 0 is neutral, +1.0 is extremely bullish for THIS SPECIFIC TICKER>,
  "price_impact": <int 0-10, expected magnitude of price move. 0=no impact, 3=minor, 5=moderate, 7=significant, 10=massive stock-moving event>,
  "event_type": <string, one of: earnings, analyst_rating, fda_regulatory, contract_deal, offering_dilution, insider_activity, short_interest, macro_sector, legal, partnership, product_launch, restructuring, crypto_market, government_policy, routine_mention, other>,
  "is_catalyst": <bool, true if this will likely cause visible price action within 24 hours>,
  "surprise_factor": <int 0-10, how unexpected is this? 0=fully priced in, 5=somewhat surprising, 10=total shock>,
  "information_quality": <int 0-10, 0=clickbait/speculation/listicle, 5=standard reporting, 10=primary source with hard data>,
  "retail_fomo_score": <int 0-10, likelihood this triggers retail buying/selling panic. Consider: meme-stock energy, viral potential, emotional language>,
  "volatility_catalyst": <int 0-10, will this widen intraday trading range regardless of direction? High for binary events (FDA, earnings), low for routine>,
  "urgency": <int 0-10, 0=recap of old news, 5=same-day reporting, 10=breaking/developing right now>,
  "forward_looking": <int 0-10, 0=purely about past events, 5=mixed, 10=entirely about upcoming catalysts>,
  "overreaction_risk": <int 0-10, is the headline more sensational than the substance warrants? High=likely to mean-revert after initial reaction>,
  "premarket_driver": <int 0-10, will this specifically drive pre-market or early-session (first 30 min) trading volume? Consider timing and newsworthiness>
}

IMPORTANT:
- Score for the SPECIFIC TICKER mentioned, not the market overall
- sentiment_score must be a float (e.g. 0.3, -0.7), all others are integers except is_catalyst (bool)
- event_type must be exactly one of the listed values
- Return ONLY the JSON object, no other text"""


def build_article_prompt(ticker, title, description, sentiment_reasoning=''):
    parts = [f"TICKER: {ticker}", f"TITLE: {title}"]
    if description and len(description.strip()) > 10:
        parts.append(f"DESCRIPTION: {description[:1500]}")
    if sentiment_reasoning and len(sentiment_reasoning.strip()) > 10:
        parts.append(f"EXISTING ANALYSIS: {sentiment_reasoning[:500]}")
    return '\n'.join(parts)


# ==============================================================================
# GCS I/O
# ==============================================================================
def _download_parquet(blob_path):
    local = f"/tmp/_llm_{os.path.basename(blob_path)}"
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
    blob_path = f"{SENTIMENT_DIR}/{ticker}_llm_scores.parquet"
    local = f"/tmp/_llm_{ticker}_scores.parquet"
    df.to_parquet(local, engine='pyarrow', index=False)
    bucket.blob(blob_path).upload_from_filename(local)
    size_mb = os.path.getsize(local) / 1e6
    os.remove(local)
    return size_mb


# ==============================================================================
# LLM SCORING
# ==============================================================================
def _extract_text(response):
    """Extract text from Gemini response, handling None and candidate structures."""
    if response.text is not None:
        return response.text
    try:
        return response.candidates[0].content.parts[0].text
    except (IndexError, AttributeError, TypeError):
        pass
    return None


def parse_llm_response(text):
    if text is None:
        return None
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```\s*$', '', text)
    text = text.strip()
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        text = json_match.group()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    parsed = {}
    try:
        parsed['sentiment_score'] = float(np.clip(data.get('sentiment_score', 0), -1.0, 1.0))
    except (ValueError, TypeError):
        parsed['sentiment_score'] = 0.0

    for field in ['price_impact', 'surprise_factor', 'information_quality',
                  'retail_fomo_score', 'volatility_catalyst', 'urgency',
                  'forward_looking', 'overreaction_risk', 'premarket_driver']:
        try:
            parsed[field] = int(np.clip(int(data.get(field, 0)), 0, 10))
        except (ValueError, TypeError):
            parsed[field] = 0

    et = str(data.get('event_type', 'other')).lower().strip()
    parsed['event_type'] = et if et in VALID_EVENT_TYPES else 'other'

    try:
        parsed['is_catalyst'] = bool(data.get('is_catalyst', False))
    except (ValueError, TypeError):
        parsed['is_catalyst'] = False

    return parsed


def score_one_article(article_id, ticker, title, description, sentiment_reasoning):
    prompt = build_article_prompt(ticker, title, description, sentiment_reasoning)
    last_err = 'unknown'

    for attempt in range(MAX_RETRIES):
        try:
            config_kwargs = dict(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.1,
                max_output_tokens=8192,
                response_mime_type='application/json',
            )

            response = genai_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            text = _extract_text(response)
            parsed = parse_llm_response(text)
            if parsed:
                parsed['article_id'] = article_id
                return parsed
            last_err = f"parse_failed|text={str(text)[:80]}"
        except Exception as e:
            err_str = str(e)
            last_err = err_str[:200]
            if '429' in err_str or 'RESOURCE_EXHAUSTED' in err_str:
                wait = (attempt + 1) * 10
                time.sleep(wait)
            elif attempt < MAX_RETRIES - 1:
                time.sleep(1)
            else:
                return {'article_id': article_id, 'error': last_err}

    return {'article_id': article_id, 'error': last_err}


# ==============================================================================
# PROCESS ONE TICKER
# ==============================================================================
def _worker_score(article_id, ticker, title, description, sentiment_reasoning):
    result = score_one_article(article_id, ticker, title, description, sentiment_reasoning)
    time.sleep(REQUEST_DELAY_S)
    return result


def process_ticker(ticker):
    t0 = time.time()

    news_df = _download_parquet(f"{SENTIMENT_DIR}/{ticker}_news.parquet")
    if news_df is None or news_df.empty:
        return 0, 0, 0

    existing_scores = _download_parquet(f"{SENTIMENT_DIR}/{ticker}_llm_scores.parquet")
    scored_ids = set()
    if existing_scores is not None and not existing_scores.empty:
        scored_ids = set(existing_scores['article_id'].values)

    to_score = news_df[~news_df['article_id'].isin(scored_ids)].copy()
    if to_score.empty:
        elapsed = time.time() - t0
        return len(news_df), 0, elapsed

    print(f"    {len(to_score)} articles to score ({len(scored_ids)} already done)")

    results = []
    errors = 0
    completed = 0

    rows_list = list(to_score.iterrows())

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for _, row in rows_list:
            fut = executor.submit(
                _worker_score,
                article_id=row['article_id'],
                ticker=ticker,
                title=row.get('title', ''),
                description=row.get('description', ''),
                sentiment_reasoning=row.get('sentiment_reasoning', ''),
            )
            futures[fut] = row['article_id']

        error_samples = {}
        for fut in as_completed(futures):
            result = fut.result()
            if 'error' in result:
                errors += 1
                err_msg = result['error']
                err_key = err_msg[:60]
                if err_key not in error_samples:
                    error_samples[err_key] = 0
                error_samples[err_key] += 1
                if errors <= 5 or errors % 50 == 0:
                    print(f"        ERR #{errors}: {err_msg[:120]}")
            results.append(result)
            completed += 1

            if completed % 50 == 0:
                elapsed_so_far = time.time() - t0
                rate = completed / elapsed_so_far if elapsed_so_far > 0 else 0
                print(f"      [{completed}/{len(to_score)}] scored, {errors} errors, "
                      f"{rate:.1f} art/s")

            if completed % BATCH_SAVE_EVERY == 0:
                _interim_save(ticker, results, existing_scores)

    if error_samples:
        print(f"    Error breakdown ({errors} total):")
        for msg, cnt in sorted(error_samples.items(), key=lambda x: -x[1]):
            print(f"      {cnt:>4}× {msg}")

    new_scores_df = pd.DataFrame([r for r in results if 'error' not in r])

    if new_scores_df.empty:
        elapsed = time.time() - t0
        return len(news_df), 0, elapsed

    if existing_scores is not None and not existing_scores.empty:
        combined = pd.concat([existing_scores, new_scores_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['article_id'], keep='last')
    else:
        combined = new_scores_df

    _save_parquet(combined, ticker)
    elapsed = time.time() - t0

    return len(news_df), len(new_scores_df), elapsed


def _interim_save(ticker, results, existing_scores):
    new_df = pd.DataFrame([r for r in results if 'error' not in r])
    if new_df.empty:
        return
    if existing_scores is not None and not existing_scores.empty:
        combined = pd.concat([existing_scores, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['article_id'], keep='last')
    else:
        combined = new_df
    _save_parquet(combined, ticker)


# ==============================================================================
# MAIN
# ==============================================================================
W = 100
t_total = time.time()

print(f"\n{'='*W}")
print(f"  LLM NEWS SCORING — Gemini Flash")
print(f"  Model: {GEMINI_MODEL}")
print(f"  Tickers: {len(ANALYSIS_TICKERS)} — {', '.join(ANALYSIS_TICKERS)}")
print(f"  Output: gs://{BUCKET_NAME}/{SENTIMENT_DIR}/{{ticker}}_llm_scores.parquet")
print(f"{'='*W}")

# --- Quick test ---
print(f"\n  Testing Gemini access...", end=' ')
_gemini_ok = False
try:
    test_resp = genai_client.models.generate_content(
        model=GEMINI_MODEL,
        contents="Return the JSON: {\"test\": true}",
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=256,
            response_mime_type='application/json',
        ),
    )
    test_text = _extract_text(test_resp)
    if test_text:
        print(f"OK ({test_text.strip()})")
        _gemini_ok = True
    else:
        print(f"WARNING: model returned empty response, trying without response_mime_type...")
        test_resp2 = genai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents="Return ONLY this JSON, no other text: {\"test\": true}",
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=256,
            ),
        )
        test_text2 = _extract_text(test_resp2)
        if test_text2:
            print(f"OK without mime_type ({test_text2.strip()[:50]})")
            _gemini_ok = True
        else:
            print(f"ERROR: model returns None for all attempts")
except Exception as e:
    print(f"\n  ERROR: Cannot access {GEMINI_MODEL}: {e}")

if not _gemini_ok:
    print(f"  Trying fallback model gemini-2.5-flash-lite...")
    try:
        GEMINI_MODEL = "gemini-2.5-flash-lite"
        test_resp3 = genai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents="Return ONLY: {\"test\": true}",
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=256),
        )
        test_text3 = _extract_text(test_resp3)
        if test_text3:
            print(f"  Fallback OK with {GEMINI_MODEL}: {test_text3.strip()[:50]}")
            _gemini_ok = True
    except Exception as e2:
        print(f"  Fallback also failed: {e2}")

if not _gemini_ok:
    raise RuntimeError(f"Cannot access any Gemini model on project {GCP_PROJECT}")

stats = []
total_scored = 0
total_articles = 0

for ti, ticker in enumerate(ANALYSIS_TICKERS):
    print(f"\n  [{ti+1}/{len(ANALYSIS_TICKERS)}] {ticker}")
    try:
        n_articles, n_scored, elapsed = process_ticker(ticker)
        total_articles += n_articles
        total_scored += n_scored
        stats.append({
            'ticker': ticker, 'articles': n_articles,
            'scored': n_scored, 'time_s': elapsed,
        })
        print(f"    Done: {n_scored} scored in {elapsed:.0f}s "
              f"({n_scored/elapsed:.1f} art/s)" if elapsed > 0 and n_scored > 0
              else f"    Done: {n_scored} new scores")
    except Exception as e:
        print(f"    ERROR: {e}")
        stats.append({'ticker': ticker, 'articles': 0, 'scored': 0, 'time_s': 0})

total_elapsed = time.time() - t_total


# ==============================================================================
# SUMMARY
# ==============================================================================
stats_df = pd.DataFrame(stats)

print(f"\n\n{'='*W}")
print(f"  LLM SCORING COMPLETE")
print(f"{'='*W}")
print(f"  Tickers:    {len(ANALYSIS_TICKERS)}")
print(f"  Articles:   {total_articles:,}")
print(f"  New scores: {total_scored:,}")
print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
print(f"  Output:     gs://{BUCKET_NAME}/{SENTIMENT_DIR}/{{ticker}}_llm_scores.parquet")

print(f"\n  {'Ticker':<8} {'Articles':>9} {'Scored':>8} {'Time':>7}")
print(f"  {'─'*35}")
for _, row in stats_df.iterrows():
    print(f"  {row['ticker']:<8} {row['articles']:>9} {row['scored']:>8} {row['time_s']:>6.0f}s")


# ==============================================================================
# QUICK VALIDATION — sample scored articles
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SAMPLE SCORED ARTICLES")
print(f"{'='*W}")

for ticker in ANALYSIS_TICKERS[:3]:
    scores_df = _download_parquet(f"{SENTIMENT_DIR}/{ticker}_llm_scores.parquet")
    news_df = _download_parquet(f"{SENTIMENT_DIR}/{ticker}_news.parquet")

    if scores_df is None or scores_df.empty or news_df is None:
        continue

    merged = news_df.merge(scores_df, on='article_id', how='inner')
    if merged.empty:
        continue

    high_impact = merged.nlargest(3, 'price_impact')
    print(f"\n  {ticker} — Top 3 by price_impact:")
    for _, row in high_impact.iterrows():
        print(f"    [{row.get('price_impact',0)}/10] sent={row.get('sentiment_score',0):+.2f} "
              f"type={row.get('event_type','?')} catalyst={row.get('is_catalyst',False)} "
              f"retail={row.get('retail_fomo_score',0)} surprise={row.get('surprise_factor',0)}")
        print(f"      {str(row.get('title',''))[:100]}")


# ==============================================================================
# SCORE DISTRIBUTION CHARTS
# ==============================================================================
all_scores = []
for ticker in ANALYSIS_TICKERS:
    s = _download_parquet(f"{SENTIMENT_DIR}/{ticker}_llm_scores.parquet")
    if s is not None and not s.empty:
        s['ticker'] = ticker
        all_scores.append(s)

if all_scores:
    all_df = pd.concat(all_scores, ignore_index=True)

    fig, axes = plt.subplots(3, 4, figsize=(22, 14))
    fig.suptitle(f'LLM Score Distributions — {len(all_df):,} articles across '
                 f'{len(ANALYSIS_TICKERS)} tickers', fontsize=13, fontweight='bold')

    score_cols = [
        ('sentiment_score', 'Sentiment Score [-1,+1]'),
        ('price_impact', 'Price Impact [0-10]'),
        ('surprise_factor', 'Surprise Factor [0-10]'),
        ('information_quality', 'Info Quality [0-10]'),
        ('retail_fomo_score', 'Retail FOMO [0-10]'),
        ('volatility_catalyst', 'Volatility Catalyst [0-10]'),
        ('urgency', 'Urgency [0-10]'),
        ('forward_looking', 'Forward Looking [0-10]'),
        ('overreaction_risk', 'Overreaction Risk [0-10]'),
        ('premarket_driver', 'Premarket Driver [0-10]'),
    ]

    for idx, (col, label) in enumerate(score_cols):
        ax = axes[idx // 4, idx % 4]
        if col in all_df.columns:
            vals = all_df[col].dropna()
            if col == 'sentiment_score':
                ax.hist(vals, bins=40, color='#1565C0', alpha=0.7, edgecolor='white')
            else:
                ax.hist(vals, bins=range(12), color='#2E7D32', alpha=0.7,
                        edgecolor='white', align='left')
            ax.set_title(label, fontsize=9)
            ax.set_ylabel('Count')
            mean_val = vals.mean()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5,
                       label=f'mean={mean_val:.2f}')
            ax.legend(fontsize=7)
        ax.grid(axis='y', alpha=0.3)

    # Event type distribution
    ax = axes[2, 2]
    if 'event_type' in all_df.columns:
        et_counts = all_df['event_type'].value_counts()
        ax.barh(et_counts.index, et_counts.values, color='#FF6F00', alpha=0.8)
        ax.set_xlabel('Count')
        ax.set_title('Event Type Distribution', fontsize=9)
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(axis='x', alpha=0.3)

    # Catalyst ratio per ticker
    ax = axes[2, 3]
    if 'is_catalyst' in all_df.columns:
        cat_rate = all_df.groupby('ticker')['is_catalyst'].mean().sort_values()
        ax.barh(cat_rate.index, cat_rate.values * 100, color='#C62828', alpha=0.8)
        ax.set_xlabel('% Articles Flagged as Catalyst')
        ax.set_title('Catalyst Rate per Ticker', fontsize=9)
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    chart_path = '/tmp/llm_score_distributions.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"\n  Chart saved: {chart_path}")
    if _IN_COLAB:
        plt.show()
    plt.close()

gc.collect()
print(f"\n{'='*W}")
print(f"  LLM SCORING COMPLETE")
print(f"{'='*W}")
