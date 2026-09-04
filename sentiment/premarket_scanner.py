# %% ── Pre-Market News Scanner Agent ─────────────────────────────────────────
#
# Scans Google Search via Gemini grounding for overnight/pre-market news
# across deploy tickers. Two-stage pipeline:
#
#   Stage 0: Market context (futures, macro, overnight theme)
#   Stage 1: Broad scan — Gemini 2.5 Flash + Google Search grounding
#            Quick catalyst assessment per ticker (0-10). Parallel.
#   Stage 2: Deep analysis — Claude Sonnet via Vertex AI
#            Detailed trade evaluation for top candidates. Sequential.
#   Output:  Top 3-5 LONG + 3-5 SHORT candidates with reasoning
#
# Uses:
#   - Gemini 2.5 Flash (Google Search grounding) for fast news retrieval
#   - Claude Sonnet (Vertex AI) for deep reasoning on trade candidates
#
# Colab-ready, config vars at top.

import os, sys, time, gc, json, re, warnings
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pytz

try:
    from google.cloud import bigquery
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "google-cloud-bigquery", "--quiet"])
    from google.cloud import bigquery

warnings.filterwarnings('ignore')

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "google-genai", "--quiet"])
    from google import genai
    from google.genai import types as genai_types

try:
    from anthropic import AnthropicVertex
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "anthropic[vertex]", "--quiet"])
    from anthropic import AnthropicVertex

try:
    from google.cloud import storage as gcs_storage
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "google-cloud-storage", "pyarrow", "--quiet"])
    from google.cloud import storage as gcs_storage

try:
    from google.colab import auth
    try:
        auth.authenticate_user()
    except Exception:
        pass
    _IN_COLAB = True
except (ImportError, Exception):
    _IN_COLAB = False



# ==============================================================================
# CONFIG
# ==============================================================================
GCP_PROJECT  = "yanina-everything"
GCP_LOCATION = "us-central1"
GEMINI_MODEL = "gemini-2.5-flash"

CLAUDE_REGION  = "us-east5"
CLAUDE_MODEL   = "claude-sonnet-4-6"

BUCKET_NAME = "time-series-bucket-ys"
OUTPUT_DIR  = "Data/Scanner"
BQ_SCANNER_TABLE = "ys-time-series.scoring_logs.premarket_scanner"

PST = pytz.timezone('America/Los_Angeles')
NOW_PT = datetime.now(PST)
TODAY_STR = NOW_PT.strftime('%Y-%m-%d')
TIME_STR = NOW_PT.strftime('%H:%M')

# --- Test mode: small ticker list for quick testing ---
TEST_MODE = False

TEST_TICKERS = [
    'TSLA', 'FCEL', 'HUT', 'OKLO', 'HOOD',
]

DEPLOY_TICKERS = [
    'ACHR', 'ACMR', 'AEHR', 'ALAB', 'AMD', 'APLD', 'ARM', 'ASTS',
    'AVAV', 'AXTI', 'BBAI', 'BE', 'BEAM', 'BKSY', 'BLZE', 'BTBT',
    'CAT', 'CEG', 'CIFR', 'CLSK', 'COIN', 'CORZ', 'CRDO', 'CVX',
    'ETN', 'FCEL', 'FLNC', 'GEV', 'HOOD', 'HUT', 'IREN', 'JOBY',
    'KEEL', 'KTOS', 'LUNR', 'MARA', 'MNTS', 'MRNA', 'MSTR', 'MU',
    'MVIS', 'MXL', 'NBIS', 'NEE', 'NNE', 'NTLA', 'OKLO', 'ONDS',
    'PL', 'PLTR', 'PLUG', 'RCAT', 'RDW', 'RIOT', 'RKLB', 'RXT',
    'SLS', 'SMCI', 'SMR', 'SNDK', 'SOUN', 'TSLA', 'VOYG', 'VRT',
    'VST', 'WOLF', 'WULF', 'XOM',
]

SCAN_TICKERS = TEST_TICKERS if TEST_MODE else DEPLOY_TICKERS

MAX_WORKERS_SCAN   = 5
TOP_N_DEEP         = 15
MAX_NEWS_AGE_HOURS = 16
SCANNER_VERSION    = 'V1'
FINAL_LONG         = 5
FINAL_SHORT        = 5
GEMINI_DELAY_S     = 0.3
MAX_RETRIES        = 3

W = 100


# ==============================================================================
# INITIALIZE CLIENTS
# ==============================================================================
gemini_client = genai.Client(
    vertexai=True,
    project=GCP_PROJECT,
    location=GCP_LOCATION,
)

gcs_client = gcs_storage.Client()
bucket = gcs_client.bucket(BUCKET_NAME)


# ==============================================================================
# PROMPTS — STAGE 1 (Gemini + Google Search)
# ==============================================================================
SCAN_SYSTEM = """You are a stock news analyst. Your job is to find the MOST RECENT news about a stock that could cause significant price movement.

Search for the most recent news about the given ticker. Prioritize the FRESHEST news first — something from 2 hours ago is far more valuable than something from 12 hours ago. Include news from the last 16 hours only. Do not include catalysts older than 16 hours.

Return ONLY a JSON object with these fields:
{
  "catalyst_score": <int 0-10, how likely this causes a significant move. 0=no news, 3=routine, 5=notable, 7=significant, 10=massive>,
  "direction": <"LONG" or "SHORT" or "NEUTRAL">,
  "headline": <most impactful headline found, or "No significant news">,
  "catalyst_type": <"earnings"|"analyst"|"regulatory"|"deal"|"offering"|"insider"|"macro"|"breaking"|"premarket_move"|"none">,
  "summary": <1-2 sentence summary of what's happening>,
  "news_time": <REQUIRED — the ACTUAL publish time of the key article/news, format: "YYYY-MM-DD H:MM AM/PM ET". Look at the article's publish timestamp or dateline. Examples: "2026-09-03 11:30 AM ET", "2026-09-03 2:45 PM ET", "2026-09-02 4:15 PM ET". Do NOT default to generic times like "5:00 AM" or "4:00 PM" — find the real publish time from the source article. If the article shows a date but no time, look at the article metadata or other articles about the same event to find the actual time.>,
  "key_articles": [
    {"title": "<article headline>", "source": "<publisher name>", "time": "<ACTUAL publish time from the article>"}
  ]
}

CATALYST WEIGHT SCALE — use this to calibrate catalyst_score:
  TIER 1 (score 8-10): earnings surprise (beat/miss), FDA approval/rejection,
         massive contract win (>$100M), bankruptcy/fraud, stock halt/resume,
         M&A deal where THIS TICKER IS THE TARGET (being acquired at a premium)
  TIER 2 (score 6-8):  analyst upgrade/downgrade with NEW price target, secondary offering,
         significant regulatory action, sector-moving earnings from peer companies,
         M&A deal where THIS TICKER IS THE ACQUIRER and deal size > 10% of market cap
  TIER 3 (score 4-6):  pre-market price move >3%, new partnership/product announcement,
         CEO change, activist investor involvement,
         M&A deal where THIS TICKER IS THE ACQUIRER and deal size < 10% of market cap
  TIER 4 (score 2-4):  routine analyst reiteration, insider buying/selling (Form 4/144),
         conference participation, routine SEC filings, index inclusion/exclusion rumors,
         APPROVAL/VOTE/CLOSE of an already-announced acquisition (procedural, priced in)
  TIER 5 (score 0-2):  no news, aggregator articles restating old info, law firm
         "investigation" announcements (ambulance-chasing), price-reporting articles

M&A/ACQUISITION RULES:
- If the ticker is the ACQUIRER (buyer): acquisitions are often NEUTRAL or NEGATIVE
  for the buyer. All-stock deals DILUTE existing shareholders. Cap score at 5 unless
  the deal is transformative (>20% of acquirer market cap).
- If the ticker is the TARGET (being acquired): bullish, premium over market price.
- "Shareholder vote approved" / "regulatory clearance" / "expected to close" for an
  already-announced deal is a FORMALITY — it is NOT new information. Score <= 3.
- Revenue guidance or synergies cited from an old announcement are ALREADY PRICED IN.

IMPORTANT:
- Use the weight scale above — insider trading (Form 4/144) is TIER 4, max score 4
- Only score >= 5 if there's a clear, specific catalyst (not routine mentions)
- "premarket_move" means the stock is already moving significantly pre-market
- Include up to 3 most relevant articles in key_articles
- Return ONLY the JSON, no other text"""

MARKET_SYSTEM = """You are a market analyst preparing the pre-market overview.
Search for the latest US stock market conditions as of right now.

Return ONLY a JSON object:
{
  "futures": "<S&P 500 / Nasdaq / Dow futures direction and %>",
  "overnight_theme": "<1-2 sentence summary of overnight market theme>",
  "key_sectors": "<which sectors are moving and why>",
  "macro_events": "<any scheduled economic data releases or Fed events today>",
  "risk_sentiment": "<risk-on|risk-off|mixed>",
  "vix_level": "<VIX level if found, or 'not found'>",
  "top_premarket_movers": "<any stocks moving 3%+ in pre-market>"
}"""


# ==============================================================================
# PROMPTS — STAGE 2 (Claude deep analysis)
# ==============================================================================
DEEP_SYSTEM = """You are a senior pre-market trading analyst preparing the morning briefing.
Your traders enter positions at 6:25 AM Pacific (market opens 6:30 AM PT) and hold for 30-60 minutes, targeting 2-4% intraday spikes.

CRITICAL CONCEPT — REMAINING MOVE vs TOTAL IMPACT:
Your job is NOT to assess whether news is bullish or bearish. Your job is to predict
what happens AFTER the trader enters at 6:25 AM PT. If a stock already dropped 13%
pre-market on bad earnings, the bearish move is OVER. The remaining move at open is
likely a BOUNCE (LONG), not further selling. You must separate:
  - total_catalyst_impact_pct: the full magnitude of the catalyst (e.g. -15% for terrible earnings)
  - move_already_absorbed_pct: how much has already moved in pre-market/after-hours
  - remaining_move_pct: what's LEFT for the trader at entry (this is what matters)

If |move_already_absorbed| >= |total_catalyst_impact|, the move is DONE. The direction
field should reflect the REMAINING move (often a reversal/bounce), NOT the catalyst direction.

NOISE FILTERS — discount these patterns:
- Law firm "investigations" announced AFTER a stock drops are ambulance-chasing, not new catalysts
- Routine analyst reiterations without price target changes
- News articles that merely REPORT a price move ("stock down 5% today") without a new catalyst
- Pre-market price quotes from aggregator sites that just restate known information

You will receive:
1. A ticker symbol
2. Initial scan results (headline, catalyst type, summary, source articles)
3. Market context (futures, macro, risk sentiment)
4. CURRENT PRICE DATA: pre-market % change from previous close (when available)

Based on this information, provide a deep analysis for intraday trading.

Return ONLY a JSON object:
{
  "ticker": "<TICKER>",
  "direction": "LONG" or "SHORT" <-- direction of the REMAINING move at entry, NOT the catalyst direction,
  "confidence": <int 1-10, how confident in the direction of the REMAINING move>,
  "total_catalyst_impact_pct": <float, total expected impact of the catalyst on the stock, e.g. -15.0>,
  "move_already_absorbed_pct": <float, how much has already moved pre-market/after-hours, e.g. -13.0>,
  "remaining_move_pct": <float, expected REMAINING % move after 6:25 AM PT entry, e.g. +2.0 for a bounce>,
  "expected_move_pct": <float, same as remaining_move_pct — this is what the trader captures>,
  "move_is_exhausted": <bool, true if the catalyst move is already fully priced in>,
  "reasoning": "<3-5 sentences explaining: (1) what the catalyst is, (2) how much the stock has ALREADY moved on it, (3) whether there is remaining edge at entry, (4) if the remaining move is continuation or reversal>",
  "catalyst": "<what's driving this, be specific — cite actual numbers, analyst targets, deal values>",
  "catalyst_type": "<earnings|analyst|regulatory|deal|offering|insider|macro|breaking|premarket_move>",
  "acquirer_or_target": "<'acquirer'|'target'|'n/a' — REQUIRED when catalyst_type is 'deal'. Is this ticker the buyer or the company being acquired?>",
  "deal_materiality": "<'immaterial'|'notable'|'transformative'|'n/a' — deal size vs market cap: <5% immaterial, 5-20% notable, >20% transformative>",
  "catalyst_is_new": <bool, true if this is a NEW announcement, false if it's a procedural update on an existing deal/event>,
  "news_timeline": [
    {"time": "<when this happened, e.g. '2026-09-02 4:15 PM ET'>", "event": "<what happened>"}
  ],
  "timing": "<breaking|overnight|pre_earnings|post_earnings|developing|scheduled>",
  "bull_case": "<1-2 sentences: why go long>",
  "bear_case": "<1-2 sentences: why go short>",
  "risks": "<what could go wrong with this trade>",
  "pre_market_action": "<any pre-market price/volume info from the scan>",
  "related_tickers": ["<other tickers affected by same catalyst>"]
}

CATALYST CONFIDENCE SCALE — calibrate confidence based on catalyst tier:
  TIER 1 (conf 7-10): earnings surprise, FDA action, massive contract,
         bankruptcy/fraud — these move stocks 5-20%.
         M&A where THIS TICKER IS THE TARGET (being acquired at premium).
  TIER 2 (conf 5-8):  analyst upgrade/downgrade with NEW target, offering, regulatory action,
         sector-moving peer earnings read-through.
         M&A where THIS TICKER IS THE ACQUIRER and deal > 10% of market cap.
  TIER 3 (conf 3-6):  pre-market price move, new partnership/product, CEO change.
         M&A where THIS TICKER IS THE ACQUIRER and deal < 10% of market cap.
  TIER 4 (conf 1-3):  insider Form 4/144 filings, conference attendance, routine filings.
         VOTE APPROVAL / REGULATORY CLEARANCE / EXPECTED CLOSE of already-announced deal.
  TIER 5 (conf 0-1):  no real catalyst, aggregator noise, law firm investigations

ACQUISITION / M&A ANALYSIS — CRITICAL:
When the catalyst involves an acquisition or merger, you MUST determine:
1. Is this ticker the ACQUIRER (buyer) or the TARGET (being acquired)?
2. Is this a NEW deal announcement, or a procedural update on an EXISTING deal?
3. What is the deal size relative to the acquirer's market cap?

KEY RULES:
- ACQUIRER stocks do NOT spike on deal news. Acquisitions are typically NEUTRAL or
  NEGATIVE for the buyer because: (a) all-stock deals dilute existing shareholders,
  (b) the acquirer pays a premium, (c) integration risk is borne by the acquirer.
- TARGET stocks spike because they receive a premium over market price.
- "Shareholder vote approved", "regulatory clearance granted", "deal expected to close"
  are PROCEDURAL steps — they are NOT new information. The market priced this in when
  the deal was first announced. Confidence should be 1-3 max for procedural updates.
- Revenue guidance or synergy projections from old announcements are ALREADY PRICED IN.
  Do not treat restated forward guidance as a new catalyst.
- Deal size matters: a $43M acquisition by a $4B company is <1% of market cap —
  this CANNOT justify a 5%+ expected move. Scale expected_move_pct to deal materiality.
- If the catalyst is "deal" and this ticker is the acquirer, and the deal was already
  announced, set confidence <= 3 and expected_move_pct <= 1%.

EARNINGS MISS/BEAT — DIRECTION RULES:
- An earnings MISS is a SHORT catalyst. The default direction is SHORT unless the stock has
  ALREADY fallen so much that a technical bounce is imminent AND the miss was minor.
- A massive earnings miss (revenue miss >10%, or guidance cut) that is STILL FRESH (<12h old)
  should remain SHORT — the selloff often continues at market open even after a large AH drop.
- Only flip to LONG (bounce) if: (a) the stock already dropped >15%, (b) the miss was minor
  or guidance was reaffirmed, AND (c) pre-market shows stabilization with buying volume.
- An earnings BEAT is a LONG catalyst. Only flip to SHORT (fade) if the stock already surged
  >15% and the beat was modest relative to expectations.
- When in doubt about direction after earnings: FOLLOW THE CATALYST DIRECTION, not the bounce thesis.

IMPORTANT:
- direction and expected_move_pct must reflect the REMAINING move, not the catalyst direction
- If a stock already crashed 13% pre-market, the remaining move MAY be LONG (bounce) — but only
  if the catalyst is minor or priced in. For major catalysts (big earnings miss, FDA rejection),
  the selloff often CONTINUES at open. Don't assume bounce.
- If a stock already surged 10%+ on STRONG earnings, the catalyst is REAL — don't auto-short it.
  A stock up 50% on great earnings may keep running. Only suggest SHORT (fade) if the surge is
  clearly irrational relative to the catalyst magnitude (e.g., +50% on a minor beat).
- Insider trading (Form 4/144) alone should NEVER get confidence above 3
- Be specific and factual — cite actual news, numbers, analyst targets
- expected_move_pct should be realistic (most stocks move 1-3%, only major events cause 5%+)
- The reasoning field is the most important — explain the full logic chain
- news_timeline should list key events in chronological order with timestamps
- Return ONLY the JSON"""


# ==============================================================================
# HELPERS
# ==============================================================================
def _extract_gemini_text(response):
    if response.text is not None:
        return response.text
    try:
        return response.candidates[0].content.parts[0].text
    except (IndexError, AttributeError, TypeError):
        pass
    return None


def _parse_json(text):
    if text is None:
        return None
    text = text.strip()
    text = re.sub(r'```(?:json)?', '', text).strip()

    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        text = json_match.group()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for suffix in ['"}', ']"}', '"}]', '}', '"]}'  ]:
        try:
            return json.loads(text + suffix)
        except json.JSONDecodeError:
            continue

    pairs = {}
    for m in re.finditer(r'"(\w+)"\s*:\s*("(?:[^"\\]|\\.)*"|\d+(?:\.\d+)?|true|false|null)', text):
        key, val = m.group(1), m.group(2)
        try:
            pairs[key] = json.loads(val)
        except json.JSONDecodeError:
            pairs[key] = val.strip('"')
    return pairs if pairs else None


def _extract_grounding_sources(response):
    sources = []
    try:
        gm = response.candidates[0].grounding_metadata
        if gm and hasattr(gm, 'grounding_chunks') and gm.grounding_chunks:
            for chunk in gm.grounding_chunks:
                if hasattr(chunk, 'web') and chunk.web:
                    sources.append({
                        'url': chunk.web.uri,
                        'title': chunk.web.title or '',
                    })
    except (IndexError, AttributeError):
        pass
    return sources


PRICE_SYSTEM = """Return ONLY a JSON object with the current/pre-market price data for this stock.
Search for the most recent quote. Return:
{
  "current_price": <float, current or most recent price>,
  "prev_close": <float, previous day's closing price>,
  "change_pct": <float, percent change from previous close, e.g. -13.2>,
  "premarket_price": <float or null, pre-market price if available>,
  "source": "<where you found the price data>"
}
Return ONLY the JSON, no other text."""


def fetch_price_data(ticker):
    try:
        result = gemini_search(
            f"What is the current stock price and pre-market price for {ticker}? "
            f"What was yesterday's closing price? What is the % change?",
            PRICE_SYSTEM,
        )
        if 'error' not in result:
            return result
    except Exception:
        pass
    return None


EARNINGS_VERIFY_SYSTEM = """You are verifying whether a company has ALREADY released its earnings report.
Search for the company's most recent earnings release date and the NEXT scheduled earnings date.

Return ONLY a JSON object:
{
  "last_earnings_date": "<date of most recent earnings release, e.g. 2026-08-04>",
  "last_earnings_quarter": "<which quarter, e.g. Q2 2026>",
  "next_earnings_date": "<date of next scheduled earnings, e.g. 2026-11-05, or 'unknown'>",
  "next_earnings_quarter": "<which quarter, e.g. Q3 2026>",
  "earnings_released_today": <bool, true ONLY if earnings were released today BEFORE market open>,
  "earnings_scheduled_today": <bool, true if earnings are scheduled for today but NOT YET released>,
  "source": "<where you found this information>"
}

IMPORTANT:
- "earnings_released_today" means the actual financial results are OUT and publicly available NOW
- If earnings are scheduled for AFTER market close today, set earnings_released_today=false and earnings_scheduled_today=true
- If the most recent earnings date is weeks/months ago, the current catalyst is NOT a fresh earnings report
- Return ONLY the JSON, no other text"""


def verify_earnings(ticker, today_str):
    try:
        result = gemini_search(
            f"When did {ticker} last report earnings? When is the next earnings date? "
            f"Today is {today_str}. Has {ticker} released earnings results today? "
            f"Search for {ticker} earnings date schedule.",
            EARNINGS_VERIFY_SYSTEM,
        )
        if 'error' not in result:
            return result
    except Exception:
        pass
    return None


def gemini_search(prompt, system_prompt, retries=MAX_RETRIES):
    for attempt in range(retries):
        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())],
                    temperature=0.2,
                    max_output_tokens=4096,
                ),
            )
            text = _extract_gemini_text(response)
            parsed = _parse_json(text)
            sources = _extract_grounding_sources(response)
            if parsed:
                parsed['_sources'] = sources
                return parsed
            return {'error': f'parse_failed: {str(text)[:100]}', 'raw': text,
                    '_sources': sources}
        except Exception as e:
            err = str(e)
            if '429' in err or 'RESOURCE_EXHAUSTED' in err:
                time.sleep((attempt + 1) * 5)
            elif attempt < retries - 1:
                time.sleep(1)
            else:
                return {'error': err[:200]}
    return {'error': 'max_retries'}


def claude_analyze(prompt, system_prompt, claude_client, retries=MAX_RETRIES):
    for attempt in range(retries):
        try:
            response = claude_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            parsed = _parse_json(text)
            if parsed:
                return parsed
            return {'error': f'parse_failed: {str(text)[:100]}', 'raw': text}
        except Exception as e:
            err = str(e)
            if '429' in err or 'overloaded' in err.lower():
                time.sleep((attempt + 1) * 5)
            elif attempt < retries - 1:
                time.sleep(2)
            else:
                return {'error': err[:200]}
    return {'error': 'max_retries'}


# ==============================================================================
# STAGE 0: MARKET CONTEXT
# ==============================================================================
t_total = time.time()

print(f"\n{'='*W}")
print(f"  PRE-MARKET NEWS SCANNER")
print(f"  {TODAY_STR} {TIME_STR} PT")
print(f"  Tickers: {len(SCAN_TICKERS)} {'(TEST MODE)' if TEST_MODE else ''}")
print(f"  Stage 1: {GEMINI_MODEL} + Google Search")
print(f"  Stage 2: Claude Sonnet via Vertex AI")
print(f"{'='*W}")

print(f"\n  Stage 0: Market context...", end=' ')
t0 = time.time()

market_ctx = gemini_search(
    f"What are the US stock market conditions right now on {TODAY_STR}? "
    f"Include S&P/Nasdaq/Dow futures, overnight developments, pre-market movers, "
    f"any scheduled economic data today, and VIX level.",
    MARKET_SYSTEM,
)

if 'error' not in market_ctx:
    print(f"done ({time.time()-t0:.1f}s)")
    print(f"\n  Futures:    {market_ctx.get('futures', 'N/A')}")
    print(f"  Theme:      {market_ctx.get('overnight_theme', 'N/A')}")
    print(f"  Sectors:    {market_ctx.get('key_sectors', 'N/A')}")
    print(f"  Macro:      {market_ctx.get('macro_events', 'N/A')}")
    print(f"  Sentiment:  {market_ctx.get('risk_sentiment', 'N/A')}")
    print(f"  VIX:        {market_ctx.get('vix_level', 'N/A')}")
    print(f"  Movers:     {market_ctx.get('top_premarket_movers', 'N/A')}")
else:
    print(f"error: {market_ctx.get('error', '')[:80]}")
    market_ctx = {}


# ==============================================================================
# STAGE 1: BROAD SCAN — Gemini + Google Search (parallel)
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  Stage 1: Broad scan — {len(SCAN_TICKERS)} tickers via Gemini + Google Search")
print(f"{'='*W}")

t1 = time.time()
scan_results = {}
scan_errors = 0
scan_completed = 0


def _scan_one_ticker(ticker):
    prompt = (
        f"Search for the latest news about {ticker} stock today {TODAY_STR}. "
        f"What happened in the LAST FEW HOURS? Include any price action today, "
        f"analyst notes, earnings, regulatory news, or breaking developments. "
        f"Prioritize the most RECENT news first."
    )
    result = gemini_search(prompt, SCAN_SYSTEM)
    time.sleep(GEMINI_DELAY_S)
    return ticker, result


with ThreadPoolExecutor(max_workers=MAX_WORKERS_SCAN) as executor:
    futures = {executor.submit(_scan_one_ticker, t): t for t in SCAN_TICKERS}

    for fut in as_completed(futures):
        ticker, result = fut.result()
        scan_completed += 1

        if 'error' in result:
            scan_errors += 1
            score = 0
            print(f"  [{scan_completed:>2}/{len(SCAN_TICKERS)}] {ticker:<6} "
                  f"ERR: {result['error'][:60]}")
        else:
            score = int(result.get('catalyst_score', 0))
            direction = result.get('direction', 'NEUTRAL')
            cat_type = result.get('catalyst_type', 'none')
            headline = str(result.get('headline', ''))[:70]

            news_time = str(result.get('news_time', ''))[:30]
            marker = ' *** ' if score >= 7 else ' **  ' if score >= 5 else '     '
            print(f"  [{scan_completed:>2}/{len(SCAN_TICKERS)}] {ticker:<6} "
                  f"score={score:>2}/10 {direction:<8} [{cat_type:<14}]{marker}")
            if score >= 5:
                print(f"  {'':>36} {headline}")
                if news_time and news_time != 'unknown':
                    print(f"  {'':>36} News broke: {news_time}")

        scan_results[ticker] = {**result, 'catalyst_score': score}

elapsed_s1 = time.time() - t1
print(f"\n  Stage 1 complete: {elapsed_s1:.0f}s, {scan_errors} errors")

# --- Rank by catalyst score ---
ranked = sorted(
    [(t, r) for t, r in scan_results.items() if 'error' not in r],
    key=lambda x: x[1].get('catalyst_score', 0),
    reverse=True,
)

n_with_catalyst = sum(1 for _, r in ranked if r.get('catalyst_score', 0) >= 5)
print(f"  Catalysts found (score >= 5): {n_with_catalyst}/{len(ranked)}")

print(f"\n  Top 10 by catalyst score:")
for i, (ticker, r) in enumerate(ranked[:10]):
    score = r.get('catalyst_score', 0)
    direction = r.get('direction', '?')
    summary = str(r.get('summary', ''))[:80]
    print(f"    {i+1:>2}. {ticker:<6} {score}/10 {direction:<8} {summary}")


# ==============================================================================
# STAGE 2: DEEP ANALYSIS — Claude Sonnet via Vertex AI
# ==============================================================================
deep_candidates = [t for t, r in ranked[:TOP_N_DEEP]
                   if r.get('catalyst_score', 0) >= 3]

if deep_candidates:
    print(f"\n\n{'='*W}")
    print(f"  Stage 2: Deep analysis — {len(deep_candidates)} candidates via Claude Sonnet")
    print(f"{'='*W}")

    # Initialize Claude client
    print(f"\n  Connecting to Claude via Vertex AI...", end=' ')
    claude_ok = False
    try:
        claude_client = AnthropicVertex(
            project_id=GCP_PROJECT,
            region=CLAUDE_REGION,
        )
        test_resp = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=64,
            messages=[{"role": "user", "content": "Return only: {\"test\": true}"}],
        )
        print(f"OK ({CLAUDE_MODEL})")
        claude_ok = True
    except Exception as e:
        print(f"\n  Claude not available: {e}")
        print(f"  Falling back to Gemini for Stage 2")

    t2 = time.time()
    deep_results = {}

    market_context_str = ""
    if market_ctx:
        market_context_str = (
            f"\nMARKET CONTEXT:\n"
            f"- Futures: {market_ctx.get('futures', 'unknown')}\n"
            f"- Theme: {market_ctx.get('overnight_theme', 'unknown')}\n"
            f"- Risk sentiment: {market_ctx.get('risk_sentiment', 'unknown')}\n"
            f"- Macro events: {market_ctx.get('macro_events', 'none')}\n"
            f"- VIX: {market_ctx.get('vix_level', 'unknown')}"
        )

    # Stage 1.5: Fetch current price data for deep candidates (parallel)
    print(f"\n  Fetching price data for {len(deep_candidates)} candidates...", end=' ')
    price_data = {}
    tp = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_SCAN) as executor:
        price_futures = {executor.submit(fetch_price_data, t): t
                         for t in deep_candidates}
        for fut in as_completed(price_futures):
            ticker_p = price_futures[fut]
            try:
                price_data[ticker_p] = fut.result()
            except Exception:
                price_data[ticker_p] = None
    n_prices = sum(1 for v in price_data.values() if v and 'change_pct' in v)
    print(f"done ({time.time()-tp:.0f}s, {n_prices}/{len(deep_candidates)} found)")

    for i, ticker in enumerate(deep_candidates):
        scan = scan_results[ticker]
        pd_info = price_data.get(ticker)
        print(f"\n  [{i+1}/{len(deep_candidates)}] {ticker} "
              f"(scan: {scan.get('catalyst_score',0)}/10, "
              f"{scan.get('direction','?')})", end='')
        if pd_info and 'change_pct' in pd_info:
            print(f"  [price: {pd_info.get('change_pct', 0):+.1f}% from prev close]...")
        else:
            print(f"...")

        # Build source articles section
        articles_str = ""
        scan_articles = scan.get('key_articles', [])
        scan_sources = scan.get('_sources', [])
        if scan_articles:
            articles_str = "\n\nSOURCE ARTICLES:\n"
            for j, art in enumerate(scan_articles[:5]):
                t = art.get('title', '') if isinstance(art, dict) else str(art)
                s = art.get('source', '') if isinstance(art, dict) else ''
                tm = art.get('time', '') if isinstance(art, dict) else ''
                url = scan_sources[j]['url'] if j < len(scan_sources) else ''
                articles_str += f"  {j+1}. [{s}] {t}"
                if tm:
                    articles_str += f" ({tm})"
                if url:
                    articles_str += f"\n     URL: {url}"
                articles_str += "\n"
        elif scan_sources:
            articles_str = "\n\nSOURCE ARTICLES:\n"
            for j, src in enumerate(scan_sources[:5]):
                articles_str += f"  {j+1}. {src.get('title', 'unknown')}: {src.get('url', '')}\n"

        # Build price context section
        price_str = ""
        if pd_info and 'change_pct' in pd_info:
            price_str = (
                f"\n\nCURRENT PRICE DATA:\n"
                f"- Current/pre-market price: ${pd_info.get('current_price', 'N/A')}\n"
                f"- Previous close: ${pd_info.get('prev_close', 'N/A')}\n"
                f"- Change from prev close: {pd_info.get('change_pct', 0):+.1f}%\n"
                f"- THIS IS CRITICAL: The stock has ALREADY moved {pd_info.get('change_pct', 0):+.1f}% "
                f"before the trader enters. Factor this into your remaining_move_pct estimate."
            )
        else:
            price_str = "\n\nCURRENT PRICE DATA: Not available — estimate from news reports."

        # Earnings verification: confirm if earnings have actually been released
        earnings_str = ""
        scan_cat = str(scan.get('catalyst_type', '')).lower()
        if scan_cat in ('earnings', 'post_earnings', 'pre_earnings'):
            print(f"    Verifying earnings date for {ticker}...", end=' ')
            ev = verify_earnings(ticker, TODAY_STR)
            if ev and 'error' not in ev:
                released_today = ev.get('earnings_released_today', False)
                scheduled_today = ev.get('earnings_scheduled_today', False)
                last_date = ev.get('last_earnings_date', 'unknown')
                next_date = ev.get('next_earnings_date', 'unknown')
                if scheduled_today and not released_today:
                    print(f"⚠ NOT YET RELEASED (scheduled today, after close)")
                    earnings_str = (
                        f"\n\n⚠ EARNINGS VERIFICATION:\n"
                        f"- Last earnings: {last_date} ({ev.get('last_earnings_quarter', '?')})\n"
                        f"- Next earnings: {next_date} ({ev.get('next_earnings_quarter', '?')})\n"
                        f"- STATUS: EARNINGS NOT YET RELEASED — scheduled for today but NOT OUT YET\n"
                        f"- WARNING: Any earnings results in the scan are SPECULATION, not fact.\n"
                        f"  Set catalyst_type to 'pre_earnings' and cap confidence at 4.\n"
                        f"  The trade is about pre-earnings positioning, NOT reaction to results."
                    )
                elif released_today:
                    print(f"✓ confirmed released today")
                    earnings_str = (
                        f"\n\nEARNINGS VERIFICATION:\n"
                        f"- Earnings CONFIRMED released today ({ev.get('last_earnings_quarter', '?')})\n"
                        f"- This is a real post-earnings catalyst."
                    )
                else:
                    print(f"last: {last_date}, next: {next_date}")
                    earnings_str = (
                        f"\n\nEARNINGS VERIFICATION:\n"
                        f"- Last earnings: {last_date} ({ev.get('last_earnings_quarter', '?')})\n"
                        f"- Next earnings: {next_date} ({ev.get('next_earnings_quarter', '?')})\n"
                        f"- Earnings were NOT released today. If the scan headline references "
                        f"earnings results, it may be citing OLD results from {last_date} or "
                        f"SPECULATING about upcoming results. Adjust confidence accordingly."
                    )
            else:
                print(f"lookup failed, proceeding without verification")

        news_time = scan.get('news_time', 'unknown')
        prompt = (
            f"Analyze {ticker} for intraday trading today ({TODAY_STR}).\n\n"
            f"SCAN RESULTS:\n"
            f"- Headline: {scan.get('headline', 'N/A')}\n"
            f"- Catalyst type: {scan.get('catalyst_type', 'unknown')}\n"
            f"- Summary: {scan.get('summary', 'N/A')}\n"
            f"- Initial direction: {scan.get('direction', 'NEUTRAL')}\n"
            f"- Catalyst score: {scan.get('catalyst_score', 0)}/10\n"
            f"- News broke: {news_time}\n"
            f"{articles_str}"
            f"{earnings_str}"
            f"{price_str}"
            f"{market_context_str}\n\n"
            f"Provide your deep analysis. Remember: direction and expected_move_pct "
            f"must reflect the REMAINING move at 6:25 AM PT entry, not the catalyst "
            f"direction. If the stock already moved significantly, the remaining move "
            f"may be a reversal."
        )

        if claude_ok:
            result = claude_analyze(prompt, DEEP_SYSTEM, claude_client)
        else:
            result = gemini_search(prompt, DEEP_SYSTEM)

        if 'error' not in result:
            direction = result.get('direction', '?')
            confidence = result.get('confidence', 0)
            remaining = float(result.get('remaining_move_pct',
                               result.get('expected_move_pct', 0)))
            absorbed = result.get('move_already_absorbed_pct', '')
            exhausted = result.get('move_is_exhausted', False)
            catalyst = str(result.get('catalyst', ''))[:70]
            exh_tag = ' [EXHAUSTED]' if exhausted else ''
            abs_str = f" (absorbed: {absorbed:+.1f}%)" if isinstance(absorbed, (int, float)) else ''
            print(f"    → {direction} conf={confidence}/10 "
                  f"remaining={remaining:+.1f}%{abs_str}{exh_tag}  |  {catalyst}")
        else:
            print(f"    → ERROR: {result['error'][:80]}")

        deep_results[ticker] = result
        time.sleep(0.5)

    elapsed_s2 = time.time() - t2
    print(f"\n  Stage 2 complete: {elapsed_s2:.0f}s")
else:
    deep_results = {}
    print(f"\n  No candidates above threshold for deep analysis.")


# ==============================================================================
# FINAL OUTPUT: RANKED SHORTLIST
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  ╔══════════════════════════════════════════════════════════════════╗")
print(f"  ║  PRE-MARKET SCANNER RESULTS — {TODAY_STR} {TIME_STR} PT{'':>16}║")
print(f"  ╚══════════════════════════════════════════════════════════════════╝")
print(f"{'='*W}")

# Market context summary
if market_ctx:
    print(f"\n  MARKET CONTEXT:")
    print(f"    Futures:   {market_ctx.get('futures', 'N/A')}")
    print(f"    Theme:     {market_ctx.get('overnight_theme', 'N/A')}")
    print(f"    Risk:      {market_ctx.get('risk_sentiment', 'N/A')}")

def _extract_timeline_time(deep_result):
    timeline = deep_result.get('news_timeline', [])
    if not timeline:
        return ''
    for ev in reversed(timeline):
        if isinstance(ev, dict):
            t = ev.get('time', '')
            if t and t not in ('unknown', 'N/A', '?'):
                return str(t)
    return ''

# Separate LONG and SHORT candidates
longs = []
shorts = []
for ticker, r in deep_results.items():
    if 'error' in r:
        continue
    scan_data = scan_results.get(ticker, {})
    _dir = r.get('direction', 'NEUTRAL')
    _remaining = float(r.get('remaining_move_pct', r.get('expected_move_pct', 0)))
    _signed_move = -abs(_remaining) if _dir == 'SHORT' else abs(_remaining)
    _exhausted = bool(r.get('move_is_exhausted', False))
    _absorbed = float(r.get('move_already_absorbed_pct', 0)) if r.get('move_already_absorbed_pct') is not None else 0.0
    _total_impact = float(r.get('total_catalyst_impact_pct', r.get('expected_move_pct', 0))) if r.get('total_catalyst_impact_pct') is not None else 0.0
    entry = {
        'ticker': ticker,
        'direction': _dir,
        'confidence': int(r.get('confidence', 0)),
        'expected_move_pct': _signed_move,
        'remaining_move_pct': _remaining,
        'move_already_absorbed_pct': _absorbed,
        'total_catalyst_impact_pct': _total_impact,
        'move_is_exhausted': _exhausted,
        'reasoning': r.get('reasoning', ''),
        'catalyst': r.get('catalyst', ''),
        'catalyst_type': r.get('catalyst_type', ''),
        'timing': r.get('timing', ''),
        'news_timeline': r.get('news_timeline', []),
        'bull_case': r.get('bull_case', ''),
        'bear_case': r.get('bear_case', ''),
        'risks': r.get('risks', ''),
        'pre_market': r.get('pre_market_action', ''),
        'related': r.get('related_tickers', []),
        'scan_score': scan_data.get('catalyst_score', 0),
        'scan_headline': scan_data.get('headline', ''),
        'scan_summary': scan_data.get('summary', ''),
        'news_time': scan_data.get('news_time', '') or _extract_timeline_time(r),
        'sources': scan_data.get('_sources', []),
        'key_articles': scan_data.get('key_articles', []),
        'composite': int(r.get('confidence', 0)) * abs(_remaining),
    }
    if entry['direction'] == 'LONG':
        longs.append(entry)
    elif entry['direction'] == 'SHORT':
        shorts.append(entry)

longs.sort(key=lambda x: x['composite'], reverse=True)
shorts.sort(key=lambda x: x['composite'], reverse=True)

now_pst = NOW_PT


# --- Parse news_time into datetime for time delta calculation ---
def _parse_news_time(news_time_str, scan_time=None):
    if not news_time_str or news_time_str in ('unknown', 'N/A', ''):
        return None
    s = str(news_time_str).strip()
    _ET = pytz.timezone('America/New_York')
    original_lower = news_time_str.lower()

    is_pt = any(tz in original_lower for tz in [' pt', ' pst', ' pdt', ' pacific'])
    tz_to_use = PST if is_pt else _ET

    for tz_label in ['ET', 'EST', 'EDT', 'PT', 'PST', 'PDT']:
        s = s.replace(f' {tz_label}', '')

    s_clean = re.sub(r'\s*(pre-market|premarket|after-hours|after hours|overnight|morning|evening|during market hours)\s*', ' ', s, flags=re.IGNORECASE).strip()

    fmts_with_time = [
        '%Y-%m-%d %I:%M %p',
        '%Y-%m-%d %I:%M%p',
        '%Y-%m-%d %H:%M',
    ]
    matched_has_time = False
    dt = None
    for fmt in fmts_with_time:
        try:
            dt = datetime.strptime(s_clean, fmt)
            dt = tz_to_use.localize(dt)
            matched_has_time = True
            break
        except ValueError:
            continue

    if dt is None:
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', s)
        if not date_match:
            return None
        try:
            dt = datetime.strptime(date_match.group(1), '%Y-%m-%d')
        except ValueError:
            return None
        time_match = re.search(r'(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?', s)
        if time_match:
            h, m = int(time_match.group(1)), int(time_match.group(2))
            ampm = (time_match.group(3) or '').upper()
            if ampm == 'PM' and h < 12:
                h += 12
            elif ampm == 'AM' and h == 12:
                h = 0
            dt = dt.replace(hour=h, minute=m)
        elif 'after' in original_lower or 'after-hours' in original_lower:
            dt = dt.replace(hour=16, minute=30)
        elif 'pre' in original_lower or 'morning' in original_lower:
            dt = dt.replace(hour=5, minute=0)
        elif 'during market' in original_lower:
            dt = dt.replace(hour=10, minute=0)
        elif 'overnight' in original_lower:
            dt = dt.replace(hour=23, minute=0)
        else:
            dt = dt.replace(hour=16, minute=0)
        dt = tz_to_use.localize(dt)

    ref_time = scan_time or NOW_PT
    dt_pst = dt.astimezone(PST)
    if dt_pst > ref_time and (dt_pst - ref_time).total_seconds() > 3600:
        dt_pst_prev = dt_pst - timedelta(days=1)
        if ref_time > dt_pst_prev:
            dt = (dt - timedelta(days=1))

    return dt


def _format_delta(td):
    total_hours = td.total_seconds() / 3600
    if total_hours < 1:
        return f"{int(td.total_seconds() / 60)}m ago"
    elif total_hours < 24:
        h = int(total_hours)
        m = int((total_hours - h) * 60)
        return f"{h}h{m:02d}m ago" if m else f"{h}h ago"
    else:
        d = int(total_hours / 24)
        h = int(total_hours - d * 24)
        return f"{d}d {h}h ago"


def _wrap_text(text, width):
    words = text.split()
    lines, current = [], ''
    for w in words:
        if len(current) + len(w) + 1 > width:
            lines.append(current)
            current = w
        else:
            current = f"{current} {w}" if current else w
    if current:
        lines.append(current)
    return lines


_CATEGORY_MAP = {
    'earnings': 'EARN', 'analyst': 'ANLST', 'regulatory': 'REG',
    'deal': 'DEAL', 'offering': 'OFFR', 'insider': 'INSD',
    'macro': 'MACRO', 'breaking': 'BREAK', 'premarket_move': 'PMKT',
    'none': '—',
}


def _get_event_time_and_age(e):
    news_time_raw = str(e.get('news_time', ''))
    news_dt = _parse_news_time(news_time_raw)
    if news_dt:
        news_dt_pst = news_dt.astimezone(PST)
        ts_str = news_dt_pst.strftime('%m/%d %I:%M%p PT').lstrip('0')
        delta = now_pst - news_dt_pst
        age = _format_delta(delta) if delta.total_seconds() > 0 else 'now'
        return ts_str, age
    if news_time_raw and news_time_raw not in ('', 'unknown', 'N/A', 'None'):
        return news_time_raw[:25], '?'
    return '—', '—'


def _freshness_multiplier(entry):
    news_time_raw = str(entry.get('news_time', ''))
    news_dt = _parse_news_time(news_time_raw)
    if not news_dt:
        return 0.2
    age_hours = (now_pst - news_dt.astimezone(PST)).total_seconds() / 3600
    if age_hours < 0:
        return 1.5
    if age_hours <= 2:
        return 1.5
    if age_hours <= 4:
        return 1.2
    if age_hours <= 6:
        return 1.0
    if age_hours <= 8:
        return 0.5
    if age_hours <= 12:
        return 0.2
    if age_hours <= 16:
        return 0.08
    return 0.02


def _print_summary_table(entries, direction_label):
    entries_sorted = sorted(entries, key=lambda x: x['composite'], reverse=True)
    n = len(entries_sorted)
    arrow = '▲' if direction_label == 'LONG' else '▼'
    print(f"\n  {arrow} {direction_label} CANDIDATES ({n})")
    if not entries_sorted:
        print(f"    No {direction_label} candidates found.\n")
        return
    print(f"    {'#':>2} {'Ticker':<6} {'Score':>5} {'Conf':>4} {'Remain':>7} "
          f"{'Absorb':>7} {'Type':<5} {'Event Time':<18} {'Age':>8}  Catalyst Summary")
    print(f"    {'─'*110}")
    for i, e in enumerate(entries_sorted, 1):
        cat = _CATEGORY_MAP.get(str(e.get('catalyst_type', 'none')).lower(), '?')
        ts_str, age = _get_event_time_and_age(e)
        absorbed = e.get('move_already_absorbed_pct', 0)
        abs_str = f"{absorbed:+.1f}%" if absorbed else '—'
        exh = ' EXH' if e.get('move_is_exhausted') else ''
        catalyst_short = str(e.get('catalyst', '') or e.get('scan_summary', '') or e.get('scan_headline', '') or e.get('reasoning', ''))[:80]
        print(f"    {i:>2} {e['ticker']:<6} {e['composite']:>5.0f} "
              f"{e['confidence']:>3}/10 {e['expected_move_pct']:>+6.1f}% "
              f"{abs_str:>7} {cat:<5} {ts_str:<18} {age:>8}{exh}  {catalyst_short}")
    print()


def _print_detail_cards(entries, direction_label):
    if not entries:
        return
    entries_sorted = sorted(entries, key=lambda x: x['composite'], reverse=True)
    arrow = '▲' if direction_label == 'LONG' else '▼'
    print(f"\n  {'='*100}")
    print(f"  {arrow} {direction_label} — DETAILED ANALYSIS")
    print(f"  {'='*100}")
    for i, e in enumerate(entries_sorted, 1):
        ts_str, age = _get_event_time_and_age(e)
        cat = _CATEGORY_MAP.get(str(e.get('catalyst_type', 'none')).lower(), '?')
        absorbed = e.get('move_already_absorbed_pct', 0)
        print(f"\n  ┌─ {i}. {e['ticker']}  {arrow} {direction_label}  "
              f"conf={e['confidence']}/10  remain={e['expected_move_pct']:+.1f}%  "
              f"absorbed={absorbed:+.1f}%  [{cat}]  score={e['composite']:.0f}")
        print(f"  │  Event: {ts_str}  ({age})")
        catalyst = str(e.get('catalyst', ''))
        if catalyst:
            print(f"  │")
            print(f"  │  Catalyst: ", end='')
            for li, line in enumerate(_wrap_text(catalyst, 90)):
                if li == 0:
                    print(line)
                else:
                    print(f"  │             {line}")
        reasoning = str(e.get('reasoning', ''))
        if reasoning:
            print(f"  │")
            print(f"  │  Why:      ", end='')
            for li, line in enumerate(_wrap_text(reasoning, 90)):
                if li == 0:
                    print(line)
                else:
                    print(f"  │             {line}")
        timeline = e.get('news_timeline', [])
        if timeline:
            print(f"  │")
            for ev in timeline[:4]:
                if isinstance(ev, dict):
                    print(f"  │    {ev.get('time', '?')}: {ev.get('event', '')[:80]}")
        if direction_label == 'LONG' and e.get('bull_case'):
            print(f"  │  Bull: {str(e['bull_case'])[:95]}")
        elif direction_label == 'SHORT' and e.get('bear_case'):
            print(f"  │  Bear: {str(e['bear_case'])[:95]}")
        if e.get('risks'):
            print(f"  │  Risk: {str(e['risks'])[:95]}")
        sources = e.get('sources', [])
        articles = e.get('key_articles', [])
        if articles:
            print(f"  │  Sources:")
            for j, art in enumerate(articles[:3]):
                if isinstance(art, dict):
                    src = art.get('source', '')
                    title = art.get('title', '')
                    url = sources[j]['url'] if j < len(sources) else ''
                    print(f"  │    {j+1}. [{src}] {title[:70]}")
                    if url:
                        print(f"  │       {url[:85]}")
        elif sources:
            print(f"  │  Sources:")
            for j, src in enumerate(sources[:3]):
                print(f"  │    {j+1}. {src.get('title', '')[:60]} — {src.get('url', '')[:70]}")
        print(f"  └{'─'*99}")


# ── Filter stale news (>MAX_NEWS_AGE_HOURS) ──

def _is_fresh(entry):
    news_time_raw = str(entry.get('news_time', ''))
    news_dt = _parse_news_time(news_time_raw)
    if not news_dt:
        return True
    age_hours = (now_pst - news_dt.astimezone(PST)).total_seconds() / 3600
    return age_hours <= MAX_NEWS_AGE_HOURS

n_before = len(longs) + len(shorts)
longs = [e for e in longs if _is_fresh(e)]
shorts = [e for e in shorts if _is_fresh(e)]
n_stale = n_before - len(longs) - len(shorts)
if n_stale:
    print(f"\n  Filtered {n_stale} candidates with news older than {MAX_NEWS_AGE_HOURS}h")

# ── Apply freshness multiplier to composite scores ──
for e in longs + shorts:
    fm = _freshness_multiplier(e)
    e['composite'] = e['composite'] * fm
    e['freshness_mult'] = fm
longs.sort(key=lambda x: x['composite'], reverse=True)
shorts.sort(key=lambda x: x['composite'], reverse=True)


# ── OUTPUT: Summary tables first, then detailed cards below ──

_print_summary_table(longs, 'LONG')
_print_summary_table(shorts, 'SHORT')

_print_detail_cards(longs, 'LONG')
_print_detail_cards(shorts, 'SHORT')


# ==============================================================================
# SAVE RESULTS
# ==============================================================================
output_data = {
    'scan_date': TODAY_STR,
    'scan_time_pt': TIME_STR,
    'test_mode': TEST_MODE,
    'n_tickers_scanned': len(SCAN_TICKERS),
    'market_context': market_ctx,
    'scan_results': {t: {k: v for k, v in r.items() if k != 'raw'}
                     for t, r in scan_results.items()},
    'deep_results': deep_results,
    'shortlist_long': longs[:FINAL_LONG],
    'shortlist_short': shorts[:FINAL_SHORT],
}

local_path = f"/tmp/premarket_scan_{TODAY_STR}.json"
with open(local_path, 'w') as f:
    json.dump(output_data, f, indent=2, default=str)

gcs_path = f"{OUTPUT_DIR}/scan_{TODAY_STR}.json"
try:
    bucket.blob(gcs_path).upload_from_filename(local_path)
    print(f"\n  Saved: gs://{BUCKET_NAME}/{gcs_path}")
except Exception as e:
    print(f"\n  GCS save failed: {e}")
print(f"  Local: {local_path}")


# ==============================================================================
# LOG TO BIGQUERY — one row per scored candidate for backtesting
# ==============================================================================
def log_scanner_to_bq(longs, shorts, scan_results, market_ctx, run_ts_str, time_str):
    bq_client = bigquery.Client()

    bq_client.query(f"""
    CREATE TABLE IF NOT EXISTS `{BQ_SCANNER_TABLE}` (
        scan_timestamp             TIMESTAMP,
        scan_date                  DATE,
        scan_time_pt               STRING,
        ticker                     STRING,
        direction                  STRING,
        rank_in_direction          INT64,
        selected                   BOOL,
        confidence                 INT64,
        expected_move_pct          FLOAT64,
        remaining_move_pct         FLOAT64,
        move_already_absorbed_pct  FLOAT64,
        total_catalyst_impact_pct  FLOAT64,
        move_is_exhausted          BOOL,
        composite_score            FLOAT64,
        scan_score                 INT64,
        catalyst_type              STRING,
        catalyst                   STRING,
        reasoning                  STRING,
        headline                   STRING,
        news_time                  STRING,
        timing                     STRING,
        bull_case                  STRING,
        bear_case                  STRING,
        risks                      STRING,
        pre_market_action          STRING,
        stage2_model               STRING,
        market_futures             STRING,
        market_risk                STRING,
        vix_level                  STRING
    )""").result()

    for col_name, col_type in [
        ('remaining_move_pct', 'FLOAT64'),
        ('move_already_absorbed_pct', 'FLOAT64'),
        ('total_catalyst_impact_pct', 'FLOAT64'),
        ('move_is_exhausted', 'BOOL'),
        ('acquirer_or_target', 'STRING'),
        ('deal_materiality', 'STRING'),
        ('catalyst_is_new', 'BOOL'),
        ('news_timeline_json', 'STRING'),
        ('freshness_multiplier', 'FLOAT64'),
        ('scanner_version', 'STRING'),
    ]:
        try:
            bq_client.query(f"ALTER TABLE `{BQ_SCANNER_TABLE}` ADD COLUMN {col_name} {col_type}").result()
        except Exception:
            pass

    rows = []
    now_utc = datetime.now(pytz.utc).isoformat()
    futures = str(market_ctx.get('futures', ''))[:500] if market_ctx else ''
    risk = str(market_ctx.get('risk_sentiment', ''))[:200] if market_ctx else ''
    vix = str(market_ctx.get('vix_level', ''))[:50] if market_ctx else ''
    stage2_model = 'gemini-2.5-flash'

    for direction_label, entries in [('LONG', longs), ('SHORT', shorts)]:
        for rank, e in enumerate(entries, 1):
            rows.append({
                'scan_timestamp': now_utc,
                'scan_date': run_ts_str,
                'scan_time_pt': time_str,
                'ticker': e['ticker'],
                'direction': direction_label,
                'rank_in_direction': rank,
                'selected': rank <= (FINAL_LONG if direction_label == 'LONG' else FINAL_SHORT),
                'confidence': e['confidence'],
                'expected_move_pct': e['expected_move_pct'],
                'remaining_move_pct': e.get('remaining_move_pct', e['expected_move_pct']),
                'move_already_absorbed_pct': e.get('move_already_absorbed_pct', 0.0),
                'total_catalyst_impact_pct': e.get('total_catalyst_impact_pct', 0.0),
                'move_is_exhausted': e.get('move_is_exhausted', False),
                'composite_score': e['composite'],
                'scan_score': e['scan_score'],
                'catalyst_type': str(e.get('catalyst_type', ''))[:100],
                'catalyst': str(e.get('catalyst', ''))[:500],
                'reasoning': str(e.get('reasoning', ''))[:2000],
                'headline': str(e.get('scan_headline', ''))[:500],
                'news_time': str(e.get('news_time', ''))[:100],
                'timing': str(e.get('timing', ''))[:50],
                'bull_case': str(e.get('bull_case', ''))[:500],
                'bear_case': str(e.get('bear_case', ''))[:500],
                'risks': str(e.get('risks', ''))[:500],
                'pre_market_action': str(e.get('pre_market', ''))[:500],
                'stage2_model': stage2_model,
                'market_futures': futures,
                'market_risk': risk,
                'vix_level': vix,
                'acquirer_or_target': str(e.get('acquirer_or_target', 'n/a'))[:20],
                'deal_materiality': str(e.get('deal_materiality', 'n/a'))[:20],
                'catalyst_is_new': e.get('catalyst_is_new', True),
                'news_timeline_json': json.dumps(e.get('news_timeline', []), default=str)[:2000],
                'freshness_multiplier': e.get('freshness_mult', 1.0),
                'scanner_version': SCANNER_VERSION,
            })

    # Also log scan-only tickers (no deep analysis) for coverage tracking
    deep_tickers = set(e['ticker'] for e in longs + shorts)
    for ticker, sr in scan_results.items():
        if ticker in deep_tickers or 'error' in sr:
            continue
        rows.append({
            'scan_timestamp': now_utc,
            'scan_date': run_ts_str,
            'scan_time_pt': time_str,
            'ticker': ticker,
            'direction': sr.get('direction', 'NEUTRAL'),
            'rank_in_direction': 0,
            'selected': False,
            'confidence': 0,
            'expected_move_pct': 0.0,
            'remaining_move_pct': 0.0,
            'move_already_absorbed_pct': 0.0,
            'total_catalyst_impact_pct': 0.0,
            'move_is_exhausted': False,
            'composite_score': 0.0,
            'scan_score': int(sr.get('catalyst_score', 0)),
            'catalyst_type': str(sr.get('catalyst_type', 'none'))[:100],
            'catalyst': '',
            'reasoning': '',
            'headline': str(sr.get('headline', ''))[:500],
            'news_time': str(sr.get('news_time', ''))[:100],
            'timing': '',
            'bull_case': '',
            'bear_case': '',
            'risks': '',
            'pre_market_action': '',
            'stage2_model': '',
            'market_futures': futures,
            'market_risk': risk,
            'vix_level': vix,
            'scanner_version': SCANNER_VERSION,
        })

    if rows:
        try:
            errors = bq_client.insert_rows_json(BQ_SCANNER_TABLE, rows)
            if errors:
                print(f"\n  BQ insert errors: {errors[:3]}")
            else:
                deep_n = len([r for r in rows if r['confidence'] > 0])
                scan_n = len(rows) - deep_n
                print(f"  BQ logged: {deep_n} deep + {scan_n} scan-only = {len(rows)} rows → {BQ_SCANNER_TABLE}")
        except Exception as e:
            print(f"  BQ logging failed: {e}")


log_scanner_to_bq(longs, shorts, scan_results, market_ctx, TODAY_STR, TIME_STR)


# ==============================================================================
# CLEANUP
# ==============================================================================
total_elapsed = time.time() - t_total
gc.collect()

print(f"\n{'='*W}")
print(f"  PRE-MARKET SCAN COMPLETE")
print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
print(f"  Results: {len(longs)} LONG + {len(shorts)} SHORT candidates")
print(f"{'='*W}")
