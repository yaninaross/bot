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
import logging
logging.getLogger('urllib3.connectionpool').setLevel(logging.ERROR)

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
    import yfinance
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "yfinance", "--quiet"])
    import yfinance

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
BQ_PATTERNS_TABLE = "ys-time-series.scoring_logs.scanner_patterns"

PST = pytz.timezone('America/Los_Angeles')
_ET = pytz.timezone('America/New_York')
NOW_PT = datetime.now(PST)
NOW_ET = NOW_PT.astimezone(_ET)
TODAY_STR = NOW_PT.strftime('%Y-%m-%d')
YESTERDAY_STR = (NOW_PT - timedelta(days=1)).strftime('%Y-%m-%d')
TIME_STR = NOW_PT.strftime('%H:%M')
TIME_ET_STR = NOW_ET.strftime('%I:%M %p ET')

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
SCANNER_VERSION    = 'V3.5'
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
CUTOFF_4H = (NOW_ET - timedelta(hours=4)).strftime('%Y-%m-%d %I:%M %p ET')
CUTOFF_16H = (NOW_ET - timedelta(hours=16)).strftime('%Y-%m-%d %I:%M %p ET')

SCAN_SYSTEM = f"""You are a stock news analyst. Your job is to find the MOST RECENT news about a stock that could cause significant price movement.

CURRENT TIME: {NOW_ET.strftime('%Y-%m-%d %I:%M %p ET')} ({NOW_PT.strftime('%I:%M %p PT')})
PRIORITY WINDOW: News published AFTER {CUTOFF_4H} (last 4 hours) is highest priority.
HARD CUTOFF: Do NOT include any news published BEFORE {CUTOFF_16H} (older than 16 hours).

FRESHNESS SCORING:
  - Published within last 2 hours: MOST VALUABLE — always include, boost catalyst_score by +2
  - Published 2-4 hours ago: HIGH VALUE — include with normal scoring
  - Published 4-8 hours ago: MODERATE — include only if catalyst_score >= 5
  - Published 8-16 hours ago: LOW VALUE — include only if catalyst_score >= 7
  - Published >16 hours ago: EXCLUDE — do not return

Return ONLY a JSON object with these fields:
{{
  "catalyst_score": <int 0-10, how likely this causes a significant move. 0=no news, 3=routine, 5=notable, 7=significant, 10=massive>,
  "direction": <"LONG" or "SHORT" or "NEUTRAL">,
  "headline": <most impactful headline found, or "No significant news">,
  "catalyst_type": <"earnings"|"analyst"|"regulatory"|"deal"|"offering"|"insider"|"macro"|"breaking"|"premarket_move"|"none">,
  "summary": <1-2 sentence summary of what's happening>,
  "news_time": <REQUIRED — the ACTUAL publish time of the key article/news, format: "YYYY-MM-DD H:MM AM/PM ET". Look at the article's publish timestamp or dateline. Examples: "2026-09-03 11:30 AM ET", "2026-09-03 2:45 PM ET", "2026-09-02 4:15 PM ET". Do NOT default to generic times like "5:00 AM" or "4:00 PM" — find the real publish time from the source article. If the article shows a date but no time, look at the article metadata or other articles about the same event to find the actual time.>,
  "key_articles": [
    {{"title": "<article headline>", "source": "<publisher name>", "time": "<ACTUAL publish time from the article>"}}
  ]
}}

CATALYST WEIGHT SCALE — use this to calibrate catalyst_score:
  TIER 1 (score 8-10): earnings surprise (beat/miss >5%), FULL FDA APPROVAL of a drug,
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
         APPROVAL/VOTE/CLOSE of an already-announced acquisition (procedural, priced in),
         FDA PROCEDURAL STEPS (BLA/NDA/IND acceptance, Priority Review granted, PDUFA date set,
         AdCom scheduled) — these are NOT the same as FDA approval, they are routine milestones
  TIER 5 (score 0-2):  no news, aggregator articles restating old info, law firm
         "investigation" announcements (ambulance-chasing), price-reporting articles,
         Goldman/analyst sector notes with no new data, Forbes/editorial opinion pieces

CATALYST EXPECTED IMPACT TABLE — use this to cap total_catalyst_impact:
  Earnings beat >10% revenue surprise:        +8% to +20% (small cap), +3% to +8% (mega cap >$100B)
  Earnings beat 2-10% revenue surprise:       +3% to +10% (small cap), +1% to +4% (mega cap)
  Earnings beat on NON-RECURRING items only:   +1% to +3% max — market fades quickly
  Full FDA drug APPROVAL:                     +5% to +30% (depends on drug revenue potential)
  FDA BLA/NDA acceptance or Priority Review:  +2% to +5% max — procedural, NOT approval
  FDA AdCom positive vote:                    +10% to +40%
  FDA rejection / CRL:                        -15% to -50%
  Contract win >$100M:                        +3% to +10%
  Analyst upgrade with NEW target:            +2% to +5%
  Analyst sector note / reiteration:          +0% to +1% — NOT a real catalyst
  Product/platform announcement (no revenue): +1% to +3% (mega cap), +2% to +6% (small cap)
  Partnership with no financial terms:        +0% to +2% — noise unless transformative
  Editorial/opinion article:                  +0% — NOT a catalyst, do not trade

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
  "spy_futures_pct": <float, S&P 500 futures % change, e.g. 0.45 or -0.8. MUST be a number>,
  "overnight_theme": "<1-2 sentence summary of overnight market theme>",
  "key_sectors": "<which sectors are moving and why>",
  "macro_events": "<any scheduled economic data releases or Fed events today>",
  "risk_sentiment": "<risk-on|risk-off|mixed>",
  "vix_level": "<VIX level if found, or 'not found'>",
  "vix_level_float": <float, VIX numeric value, e.g. 14.5 or 22.3. MUST be a number>,
  "top_premarket_movers": "<any stocks moving 3%+ in pre-market>"
}"""


# ==============================================================================
# PROMPTS — STAGE 2 (Claude deep analysis)
# ==============================================================================
DEEP_SYSTEM = """You are a senior pre-market trading analyst preparing the morning briefing.
Your traders enter positions at 6:25 AM Pacific (market opens 6:30 AM PT) and hold for 30-60 minutes, targeting 2-4% intraday spikes.

CRITICAL CONCEPT — REMAINING MOVE vs TOTAL IMPACT:
Your job is to predict what happens AFTER the trader enters at 6:25 AM PT. You must separate:
  - total_catalyst_impact_pct: the full magnitude of the catalyst (SIGNED: positive for bullish, negative for bearish)
  - move_already_absorbed_pct: how much has already moved in pre-market/after-hours
  - remaining_move_pct: what's LEFT for the trader at entry (this is what matters)

IMPORTANT: total_catalyst_impact_pct must match the CATALYST direction, not your trade thesis.
If earnings guidance was cut and the stock dropped, total_catalyst_impact_pct should be NEGATIVE
(e.g., -25%), reflecting the bearish catalyst. Do NOT set it positive for a bounce prediction.

For EARNINGS catalysts: the after-hours/pre-market drop is Wave 1. Wave 2 (continued selling at
open) is the norm, not the exception — institutions and retail sell at open after reading the
full report. Do NOT assume a bounce just because the stock already dropped a lot.

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
5. HISTORICAL PATTERN EXAMPLES: similar past trades with actual outcomes (when available)

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
  TIER 1 (conf 7-10): earnings surprise >5%, FULL FDA APPROVAL, massive contract >$100M,
         bankruptcy/fraud — these move stocks 5-20%.
         M&A where THIS TICKER IS THE TARGET (being acquired at premium).
  TIER 2 (conf 5-8):  analyst upgrade/downgrade with NEW target, offering, regulatory action,
         sector-moving peer earnings read-through.
         M&A where THIS TICKER IS THE ACQUIRER and deal > 10% of market cap.
  TIER 3 (conf 3-6):  pre-market price move, new partnership/product, CEO change.
         M&A where THIS TICKER IS THE ACQUIRER and deal < 10% of market cap.
  TIER 4 (conf 1-3):  insider Form 4/144 filings, conference attendance, routine filings.
         VOTE APPROVAL / REGULATORY CLEARANCE / EXPECTED CLOSE of already-announced deal.
         FDA PROCEDURAL STEPS: BLA/NDA acceptance, Priority Review, PDUFA date set, AdCom
         scheduled — these are NOT the same as full FDA approval. Max conf 4.
  TIER 5 (conf 0-1):  no real catalyst, aggregator noise, law firm investigations,
         analyst sector notes without new data, editorial/opinion articles.

CATALYST EXPECTED IMPACT TABLE — use this to cap remaining_move_pct:
  Full FDA drug APPROVAL:                     +5% to +30% (depends on drug TAM)
  FDA BLA/NDA acceptance or Priority Review:  +2% to +5% max — procedural, NOT approval
  FDA AdCom positive vote:                    +10% to +40%
  FDA rejection / CRL:                        -15% to -50%
  Earnings beat >10% revenue surprise:        +8% to +20% (small cap), +3% to +8% (mega cap >$100B)
  Earnings beat on NON-RECURRING items only:   +1% to +3% max — fades fast
  Contract win >$100M:                        +3% to +10%
  Product/platform announcement (no revenue): +1% to +3% (mega cap >$100B)
  Partnership with no financial terms:        +0% to +2% — noise unless transformative
  Analyst sector note / reiteration:          +0% to +1%
  Editorial/opinion article:                  +0% — NOT a catalyst

CRITICAL — FDA procedural milestones (BLA acceptance, Priority Review, PDUFA date set)
are NOT the same as FDA APPROVAL. They are routine regulatory checkboxes. Do NOT assign
confidence > 4 or remaining_move > 5% for these events. The premarket gap on FDA procedural
news is typically the ENTIRE move — it fades at open.

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

EARNINGS — GUIDANCE IS EVERYTHING:
The market is forward-looking. Historical results (Q2 beat) do NOT matter if forward guidance
is cut. The stock is repriced to reflect FUTURE earning power, not past performance.

EARNINGS + GUIDANCE CUT = TWO-WAVE SELLOFF (NEVER predict a bounce):
  IF earnings beat or meet on historical numbers
  BUT forward guidance is lowered (revenue, EPS, or margin outlook)
  THEN the stock WILL continue dropping at open. This is NOT a bounce opportunity.
  Wave 1: After-hours crash (algos + fast money react to headlines)
  Wave 2: Pre-market/open continuation (institutions + retail read the full report and sell)
  Direction: SHORT. Do NOT predict LONG bounce on guidance cuts regardless of AH drop size.
  Real example: 2026-09-09 TTAN beat Q2 EPS ($0.40 vs $0.35) and revenue ($292.8M vs $285.9M),
    but cut Q3 guidance by $1-3M citing agriculture slowdown. AH: -20.3%. Open: another -12.4%.
    Scanner predicted LONG bounce → actual result: -17.2% from entry. MASSIVE FP.

EARNINGS MISS — DIRECTION RULES:
- An earnings MISS is a SHORT catalyst. Default SHORT unless miss was trivial (<2% EPS miss)
  AND stock shows clear pre-market stabilization (flat for >2 hours).
- A massive earnings miss (revenue miss >10%, guidance cut) ALWAYS remains SHORT at open.
  After-hours liquidity is limited — institutional sellers dump at open. Never call a bounce.
- An earnings BEAT with guidance RAISED is LONG. Only flip to SHORT if stock surged >20%
  and the beat was modest.
- An earnings BEAT with guidance UNCHANGED is cautious LONG (the beat is real, but limited upside).
- An earnings BEAT with guidance CUT is SHORT — the guidance cut dominates the beat.
- When in doubt: FOLLOW THE GUIDANCE DIRECTION, not the historical results.

PROVEN TRADE OUTCOME PATTERNS — learn from actual past results:

PARTNERSHIP ASYMMETRY:
  When a mega-cap ($100B+) announces a partnership with a smaller company ($10-50B):
  → Trade the SMALLER partner LONG (expect +3-8% intraday)
  → Do NOT trade the larger partner — move is <1% and often negative
  Real example: 2026-09-08 PLTR ($174B) named NBIS ($50B) as preferred sovereign AI partner.
    NBIS: gap +3.7%, MFE +6.2%, EOD +6.2% → STRONG TP
    PLTR: gap -0.7%, EOD -1.0% → FP (partnership immaterial to mega-cap revenue)
  Why: the deal represents a much larger % of the smaller company's TAM/validation.

NON-RECURRING EARNINGS:
  When earnings beat is driven by one-time gains (derivative gains, mark-to-market investments,
  asset sales, legal settlements) rather than organic revenue growth:
  → Cap confidence at 4, cap remaining_move at +2%
  → Market discounts non-recurring items quickly — the spike fades at open
  Real example: 2026-09-08 GME Q2 "beat" from $166M derivative gain + $72M unrealized eBay gain.
    GME: gap +1.4%, ret_60m -2.9%, EOD -2.9% → FP

PRE-EARNINGS SPECULATION:
  When earnings have NOT YET been reported (scheduled for after close or later this week):
  → There is NO catalyst to trade. Confidence should be ≤2.
  → Options-implied moves are volatility estimates, NOT directional signals.
  Real example: 2026-09-08 BRZE earnings after close. Scanner: conf=3, remain=+18%.
    BRZE: ret_60m -5.5% → FP (no catalyst = no edge = random walk)

CLINICAL TRIAL DATA TRAP:
  When a biotech/pharma stock spikes on Phase 1/2/3 clinical trial data with words like
  "positive results", "outperforms standard of care", "doubles response rate":
  → Do NOT go LONG. Institutions SELL clinical data events. Cap confidence at 3.
  → Four reasons this pattern fails: (1) Valuation exhaustion — biotech runs up into scheduled
    readouts, data release removes uncertainty → immediate profit-taking. (2) "Positive" Phase 2
    does NOT guarantee Phase 3 success, especially with small sample sizes (N<100). (3) Trial
    expansion announcements = delayed timeline + higher cash burn. (4) P/S ratios >100x with
    negative EPS mean the data is already priced in.
  Real example: 2026-09-08 INBX Phase 2 HexAgon: INBRX-106 + Keytruda in HNSCC.
    Headlines: "48.3% ORR vs 26.5% Keytruda alone", "nearly doubles response rate"
    Scanner prediction: LONG conf=7, expected +36.1%
    INBX: entry $117.22, ret_30m -14.11%, ret_60m -14.55%, EOD -5.49% → MASSIVE FP
    Institutions sold the pop. Phase 2 expansion (not Phase 3) announced. N=63 too small.

MOMENTUM vs. MEAN REVERSION (GAP RULES) — CRITICAL:
1. THE "GAP AND GO": If a stock is gapping up >3% pre-market on a real catalyst, momentum
   is established. DO NOT fade (SHORT) purely because it "overshot" the catalyst's fair value.
   Retail FOMO, algorithmic momentum, and short covering frequently drive gap-ups HIGHER in
   the first 60 minutes. Evaluate for a continuation LONG.
2. SHORT SQUEEZE DANGER: If Short Interest is >10% AND the stock is gapping up >3%, heavily
   bias toward LONG. Short sellers will be forced to cover at the open, amplifying the move.
   A stock with 20%+ SI gapping up is almost NEVER a SHORT.
3. RECOVERY BOUNCES: If the stock is >10% below its 20-day high and gapping up, it's recovering
   lost ground — not overshooting. Default LONG, not SHORT.
4. WHEN TO ACTUALLY SHORT A GAP-UP ("GAP AND CRAP"): ONLY call SHORT on a gap-up if:
   (a) The catalyst is demonstrably deceptive (fake partnership, pump-and-dump), OR
   (b) The stock is a micro-cap (<$500M) with a promotional/dilutive catalyst, OR
   (c) The stock is AT or ABOVE its 20-day high AND the catalyst is weak (editorial, procedural)
   Even then, confidence for shorting a gap-up should rarely exceed 4.
5. EXHAUSTED ≠ SHORT: When move_is_exhausted is True, it means the gap has absorbed the
   catalyst's estimated impact. The correct default is NEUTRAL (skip), NOT SHORT (fade).
   Only call SHORT if you have specific evidence beyond fair-value math.
   Real data: On 2026-09-08, ALL 12 stocks flagged as exhausted SHORTs continued UP from open.
   KEEL +8.6%, AXTI +7.9%, BE +5.2%, CLSK +5.3%, FLNC +4.8%, IREN +4.6%, AMD +4.0%.
   Fading gap-ups is a proven losing strategy in the first 60 minutes.

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
- If HISTORICAL PATTERN EXAMPLES are provided below the prompt, use them to calibrate your
  confidence and expected move. Patterns with confirmed outcomes are more reliable than theory.
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
    # Primary: Polygon (verified exchange data)
    if _POLYGON_OK:
        try:
            prev = get_prev_session_close(ticker)
            if prev and prev['close'] > 0:
                prev_close = prev['close']
                bars_1m = list(_polygon_client.get_aggs(
                    ticker=ticker, multiplier=1, timespan="minute",
                    from_=TODAY_STR, to=TODAY_STR, limit=500,
                ))
                premarket_volume = 0
                if bars_1m:
                    current_price = float(bars_1m[-1].close)
                    premarket_volume = sum(int(getattr(b, 'volume', 0) or 0) for b in bars_1m)
                else:
                    bars_5m = list(_polygon_client.get_aggs(
                        ticker=ticker, multiplier=5, timespan="minute",
                        from_=prev['date'], to=TODAY_STR, limit=500,
                    ))
                    current_price = float(bars_5m[-1].close) if bars_5m else prev_close
                change_pct = ((current_price - prev_close) / prev_close) * 100
                avg_daily_volume = 0
                try:
                    _yesterday = (NOW_PT - timedelta(days=1)).strftime('%Y-%m-%d')
                    _from_20d = (NOW_PT - timedelta(days=25)).strftime('%Y-%m-%d')
                    daily_bars = list(_polygon_client.get_aggs(
                        ticker=ticker, multiplier=1, timespan="day",
                        from_=_from_20d, to=_yesterday, limit=10,
                    ))
                    if daily_bars:
                        avg_daily_volume = int(sum(
                            int(getattr(b, 'volume', 0) or 0) for b in daily_bars
                        ) / len(daily_bars))
                except Exception:
                    pass
                return {
                    'current_price': round(current_price, 2),
                    'prev_close': round(prev_close, 2),
                    'change_pct': round(change_pct, 2),
                    'premarket_price': round(current_price, 2),
                    'premarket_volume': premarket_volume,
                    'avg_daily_volume': avg_daily_volume,
                    'source': 'polygon',
                }
        except Exception as e:
            print(f"    [PRICE] Polygon failed for {ticker}: {e}")
    # Fallback: yfinance (real market data, never an LLM)
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        hist = t.history(period='2d', prepost=True)
        if len(hist) >= 2:
            last_date = hist.index[-1].tz_convert(PST).date()
            today_date = NOW_PT.date()
            if last_date == today_date:
                prev_close = round(float(hist['Close'].iloc[-2]), 2)
                current_price = round(float(hist['Close'].iloc[-1]), 2)
                premarket_volume = int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0
            else:
                current_price = round(float(hist['Close'].iloc[-1]), 2)
                prev_close = current_price
                premarket_volume = 0
            change_pct = round(((current_price - prev_close) / prev_close) * 100, 2) if prev_close > 0 else 0.0
            return {
                'current_price': current_price,
                'prev_close': prev_close,
                'change_pct': change_pct,
                'premarket_price': current_price,
                'premarket_volume': premarket_volume,
                'avg_daily_volume': 0,
                'source': 'yfinance',
            }
        elif len(hist) == 1:
            current_price = round(float(hist['Close'].iloc[-1]), 2)
            return {
                'current_price': current_price,
                'prev_close': current_price,
                'change_pct': 0.0,
                'premarket_price': current_price,
                'premarket_volume': 0,
                'avg_daily_volume': 0,
                'source': 'yfinance (single day)',
            }
    except Exception as e:
        print(f"    [PRICE] yfinance failed for {ticker}: {e}")
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


# ==============================================================================
# PATTERN LIBRARY — BQ-backed few-shot retrieval (batch prefetch)
# ==============================================================================
_PATTERN_CACHE_DF = None

CAT_MAP = {
    'earnings': 'earnings', 'analyst': 'analyst', 'regulatory': 'regulatory',
    'deal': 'deal', 'offering': 'offering', 'insider': 'deal',
    'breaking': 'breaking', 'premarket_move': 'deal',
}


def prefetch_all_patterns():
    global _PATTERN_CACHE_DF
    try:
        bq_client = bigquery.Client()
        q = f"""
        SELECT pattern_id, event_date, pattern_type, catalyst_type, tickers,
               primary_ticker, secondary_ticker, catalyst_summary,
               primary_direction, primary_gap_pct, primary_ret_60m, primary_mfe_pct, primary_eod_pct, primary_outcome,
               secondary_direction, secondary_gap_pct, secondary_ret_60m, secondary_eod_pct, secondary_outcome,
               lesson, pattern_rule
        FROM `{BQ_PATTERNS_TABLE}`
        ORDER BY event_date DESC
        LIMIT 50
        """
        _PATTERN_CACHE_DF = bq_client.query(q).to_dataframe()
        print(f"  Pattern library: loaded {len(_PATTERN_CACHE_DF)} patterns from BQ")
    except Exception as e:
        print(f"  Pattern library: failed to load ({e})")
        _PATTERN_CACHE_DF = pd.DataFrame()


def fetch_similar_patterns(catalyst_type, ticker=None, max_results=3):
    global _PATTERN_CACHE_DF
    if _PATTERN_CACHE_DF is None:
        prefetch_all_patterns()
    df = _PATTERN_CACHE_DF
    if df is None or df.empty:
        return ""
    try:
        mask = pd.Series(False, index=df.index)
        if catalyst_type:
            mapped = CAT_MAP.get(catalyst_type, catalyst_type)
            mask = mask | (df['catalyst_type'] == mapped)
        if ticker:
            mask = mask | df['tickers'].str.contains(ticker, na=False)
            mask = mask | (df['primary_ticker'] == ticker)
            mask = mask | (df['secondary_ticker'] == ticker)
        matched = df[mask].head(max_results)
        if matched.empty:
            return ""
        lines = ["\n\nHISTORICAL PATTERN EXAMPLES (from verified past trades):"]
        for _, p in matched.iterrows():
            lines.append(f"\n  PATTERN: {p['pattern_type']} ({p['event_date']})")
            lines.append(f"  Tickers: {p['tickers']} | Catalyst: {p['catalyst_type']}")
            lines.append(f"  {str(p['catalyst_summary'])[:200]}")
            lines.append(f"  → {p['primary_ticker']} ({p['primary_direction']}): "
                         f"gap {p['primary_gap_pct']:+.1f}%, 60m {p['primary_ret_60m']:+.1f}%, "
                         f"MFE {p['primary_mfe_pct']:+.1f}%, EOD {p['primary_eod_pct']:+.1f}% → {p['primary_outcome']}")
            lines.append(f"  → {p['secondary_ticker']} ({p['secondary_direction']}): "
                         f"gap {p['secondary_gap_pct']:+.1f}%, 60m {p['secondary_ret_60m']:+.1f}%, "
                         f"EOD {p['secondary_eod_pct']:+.1f}% → {p['secondary_outcome']}")
            lines.append(f"  LESSON: {str(p['lesson'])[:200]}")
            if p.get('pattern_rule'):
                lines.append(f"  RULE: {str(p['pattern_rule'])[:200]}")
        return "\n".join(lines)
    except Exception as e:
        print(f"    [PATTERNS] Failed to filter: {e}")
        return ""


# ==============================================================================
# POLYGON-BASED EVENT TIMING & ABSORPTION TOOLS
# ==============================================================================
POLYGON_API_KEY = "2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs"

try:
    from polygon import RESTClient as _PolygonClient
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "polygon-api-client", "--quiet"])
    from polygon import RESTClient as _PolygonClient

try:
    _polygon_client = _PolygonClient(POLYGON_API_KEY)
    _POLYGON_OK = True
except Exception:
    _POLYGON_OK = False
    _polygon_client = None


def get_prev_session_close(ticker, ref_date_str=None):
    if not _POLYGON_OK:
        return None
    ref = ref_date_str or TODAY_STR
    try:
        from_date = (datetime.strptime(ref, '%Y-%m-%d') - timedelta(days=5)).strftime('%Y-%m-%d')
        bars = list(_polygon_client.get_aggs(
            ticker=ticker, multiplier=1, timespan="day",
            from_=from_date, to=ref, limit=10,
        ))
        if not bars:
            return None
        ref_dt = datetime.strptime(ref, '%Y-%m-%d')
        prev_bars = [b for b in bars
                     if pd.Timestamp(b.timestamp, unit='ms', tz='UTC').date()
                     < ref_dt.date()]
        if prev_bars:
            last = prev_bars[-1]
            return {
                'close': float(last.close),
                'date': pd.Timestamp(last.timestamp, unit='ms', tz='UTC').strftime('%Y-%m-%d'),
            }
        return None
    except Exception:
        return None


def fetch_20d_high(ticker, ref_date_str=None):
    if not _POLYGON_OK:
        return None
    ref = ref_date_str or TODAY_STR
    try:
        ref_dt = datetime.strptime(ref, '%Y-%m-%d')
        from_date = (ref_dt - timedelta(days=30)).strftime('%Y-%m-%d')
        to_date = (ref_dt - timedelta(days=1)).strftime('%Y-%m-%d')
        bars = list(_polygon_client.get_aggs(
            ticker=ticker, multiplier=1, timespan="day",
            from_=from_date, to=to_date, limit=30,
        ))
        if not bars:
            return None
        bars_sorted = sorted(bars, key=lambda b: b.timestamp)
        recent_20 = bars_sorted[-20:] if len(bars_sorted) >= 20 else bars_sorted
        high_20d = max(float(b.high) for b in recent_20)
        prev_close = float(bars_sorted[-1].close)
        drawdown_pct = ((prev_close - high_20d) / high_20d) * 100
        return {
            'high_20d': round(high_20d, 2),
            'prev_close': round(prev_close, 2),
            'drawdown_pct': round(drawdown_pct, 1),
            'n_bars': len(recent_20),
        }
    except Exception:
        return None


def get_actual_event_time(ticker, ref_date_str=None, move_threshold_pct=3.0):
    if not _POLYGON_OK:
        return None
    ref = ref_date_str or TODAY_STR
    try:
        ref_dt = datetime.strptime(ref, '%Y-%m-%d')
        from_date = (ref_dt - timedelta(days=7)).strftime('%Y-%m-%d')
        daily_bars = list(_polygon_client.get_aggs(
            ticker=ticker, multiplier=1, timespan="day",
            from_=from_date, to=ref, limit=10,
        ))
        if len(daily_bars) < 2:
            return None
        prev_bars = [b for b in daily_bars
                     if pd.Timestamp(b.timestamp, unit='ms', tz='UTC').date()
                     < ref_dt.date()]
        if not prev_bars:
            return None
        t_minus_1 = prev_bars[-1]
        prev_close = float(t_minus_1.close)
        t_minus_1_date = pd.Timestamp(t_minus_1.timestamp, unit='ms', tz='UTC').date()
        if len(prev_bars) >= 2:
            prev_close_before = float(prev_bars[-2].close)
        else:
            prev_close_before = prev_close
        t_minus_1_from = t_minus_1_date.strftime('%Y-%m-%d')
        minute_bars = list(_polygon_client.get_aggs(
            ticker=ticker, multiplier=5, timespan="minute",
            from_=t_minus_1_from, to=ref, limit=500,
        ))
        if not minute_bars:
            return None
        _ET_tz = pytz.timezone('America/New_York')
        threshold = move_threshold_pct / 100.0
        event_time = None
        event_price = None
        peak_price = prev_close
        peak_time = None
        trough_price = prev_close
        trough_time = None
        current_price = prev_close
        ref_close = prev_close_before
        for bar in minute_bars:
            ts = pd.Timestamp(bar.timestamp, unit='ms', tz='UTC').tz_convert(_ET_tz)
            bar_date = ts.date()
            if bar_date == t_minus_1_date:
                ref_close = prev_close_before
            else:
                ref_close = prev_close
            close_val = float(bar.close)
            high_val = float(bar.high)
            low_val = float(bar.low)
            pct_change = abs(close_val - ref_close) / ref_close if ref_close > 0 else 0
            if pct_change >= threshold and event_time is None:
                event_time = ts
                event_price = close_val
            if close_val > peak_price:
                peak_price = close_val
                peak_time = ts
            if high_val > peak_price:
                peak_price = high_val
                peak_time = ts
            if close_val < trough_price:
                trough_price = close_val
                trough_time = ts
            if low_val < trough_price:
                trough_price = low_val
                trough_time = ts
            current_price = close_val
        if event_time is None:
            return None
        return {
            'event_time_et': event_time.strftime('%Y-%m-%d %I:%M %p ET'),
            'event_time_dt': event_time,
            'prev_close': prev_close,
            'prev_close_before': prev_close_before,
            'event_price': event_price,
            'peak_price': peak_price,
            'peak_time_et': peak_time.strftime('%Y-%m-%d %I:%M %p ET') if peak_time else None,
            'trough_price': trough_price,
            'trough_time_et': trough_time.strftime('%Y-%m-%d %I:%M %p ET') if trough_time else None,
            'current_price': current_price,
            'ref_close_for_day': ref_close,
        }
    except Exception as e:
        print(f"      [EVENT_TIME] Error for {ticker}: {e}")
        return None


def calculate_absorption(prev_close, peak_price, current_price, trough_price=None):
    if trough_price is None:
        trough_price = prev_close
    if prev_close <= 0:
        return {'total_move_pct': 0, 'remaining_pct': 0, 'absorbed_pct': 0, 'is_exhausted': False}
    up_move = (peak_price - prev_close) / prev_close
    down_move = (trough_price - prev_close) / prev_close
    total_move = up_move if abs(up_move) >= abs(down_move) else down_move
    remaining = (current_price - prev_close) / prev_close
    if abs(total_move) < 0.001:
        return {'total_move_pct': 0, 'remaining_pct': 0, 'absorbed_pct': 0, 'is_exhausted': False}
    absorbed_frac = 1.0 - (remaining / total_move)
    absorbed_frac = max(0.0, min(absorbed_frac, 2.0))
    is_exhausted = (absorbed_frac > 0.80 and abs(remaining) < 0.03)
    if total_move > 0 and remaining <= 0:
        is_exhausted = True
    if total_move < 0 and remaining >= 0:
        is_exhausted = True
    return {
        'total_move_pct': round(total_move * 100, 2),
        'remaining_pct': round(remaining * 100, 2),
        'absorbed_pct': round(absorbed_frac * 100, 1),
        'is_exhausted': is_exhausted,
    }


def validate_ticker_polygon(ticker):
    """Validate that a ticker is a real US-listed stock via Polygon reference API.
    Returns (True, ticker) if valid US stock, (False, None) if not."""
    import requests
    try:
        url = f"https://api.polygon.io/v3/reference/tickers/{ticker}"
        resp = requests.get(url, params={'apiKey': POLYGON_API_KEY}, timeout=5)
        if resp.status_code != 200:
            return (False, None)
        data = resp.json()
        result = data.get('results', {})
        if not result:
            return (False, None)
        primary_exchange = result.get('primary_exchange', '')
        market = result.get('market', '')
        locale = result.get('locale', '')
        us_exchanges = {'XNYS', 'XNAS', 'XASE', 'BATS', 'ARCX'}
        if primary_exchange in us_exchanges:
            return (True, ticker)
        if market == 'stocks' and locale == 'us':
            return (True, ticker)
        return (False, None)
    except Exception:
        # On network error, assume valid to avoid blocking real tickers
        return (True, ticker)


def fetch_sec_filing_time(ticker, filing_date=None):
    """Look up the SEC 8-K filing acceptance time for a ticker on a given date.
    Uses SEC EDGAR Submissions API — structured data, not LLM.
    Returns ET datetime string (e.g. '2026-09-09 4:34 PM ET') or None."""
    import requests as _req
    ref_date = filing_date or TODAY_STR
    _headers = {'User-Agent': 'ScannerBot admin@example.com'}

    try:
        pg_url = f"https://api.polygon.io/v3/reference/tickers/{ticker}"
        pg_resp = _req.get(pg_url, params={'apiKey': POLYGON_API_KEY}, timeout=5)
        if pg_resp.status_code != 200:
            return None
        cik = pg_resp.json().get('results', {}).get('cik', '')
        if not cik:
            return None
        cik_padded = cik.zfill(10)
    except Exception:
        return None

    try:
        sec_url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
        sec_resp = _req.get(sec_url, headers=_headers, timeout=10)
        if sec_resp.status_code != 200:
            return None
        data = sec_resp.json()
        recent = data.get('filings', {}).get('recent', {})
        forms = recent.get('form', [])
        dates = recent.get('filingDate', [])
        acceptance = recent.get('acceptanceDateTime', [])
    except Exception:
        return None

    for form, fdate, acc_dt in zip(forms, dates, acceptance):
        if form == '8-K' and fdate == ref_date and acc_dt:
            try:
                from datetime import datetime as _dt
                utc_dt = _dt.strptime(acc_dt[:19], '%Y-%m-%dT%H:%M:%S')
                utc_dt = pytz.utc.localize(utc_dt)
                et_dt = utc_dt.astimezone(_ET)
                return et_dt.strftime('%Y-%m-%d %I:%M %p ET')
            except Exception:
                return None

    return None


def verify_and_correct_event_timing(entry):
    ticker = entry.get('ticker', '')
    if not ticker or not _POLYGON_OK:
        return

    cat_type = str(entry.get('catalyst_type', '')).lower()

    sec_types = {'earnings', 'deal', 'regulatory', 'offering'}
    if cat_type in sec_types:
        print(f"    [{ticker}] SEC 8-K lookup for {cat_type}...", end=' ')
        sec_time = fetch_sec_filing_time(ticker, TODAY_STR)
        if sec_time:
            entry['news_time_original'] = entry.get('news_time', '')
            entry['news_time'] = sec_time
            entry['news_time_source'] = 'sec_edgar'
            print(f"found: {sec_time}")
        else:
            sec_time_prev = fetch_sec_filing_time(ticker, YESTERDAY_STR)
            if sec_time_prev:
                entry['news_time_original'] = entry.get('news_time', '')
                entry['news_time'] = sec_time_prev
                entry['news_time_source'] = 'sec_edgar'
                print(f"found (yesterday): {sec_time_prev}")
            else:
                print("no 8-K found")
    else:
        print(f"    [{ticker}] Verifying event timing via Polygon...", end=' ')

    event_info = get_actual_event_time(ticker, TODAY_STR)
    if not event_info:
        if cat_type not in sec_types:
            print("no spike found")
        return

    event_dt = event_info['event_time_dt']
    event_age_hours = (NOW_ET - event_dt).total_seconds() / 3600

    if entry.get('news_time_source') not in ('sec_edgar',):
        news_time_raw = str(entry.get('news_time', ''))
        article_dt = _parse_news_time(news_time_raw)
        article_age_hours = None
        if article_dt:
            article_age_hours = (NOW_ET - article_dt.astimezone(_ET)).total_seconds() / 3600

        if article_age_hours is not None and event_age_hours > article_age_hours + 1.0:
            _today_et = NOW_ET.date()
            _event_is_overnight_ah = (
                event_dt.date() < _today_et
                and event_dt.hour >= 16
                and event_age_hours <= 20
            )
            if not _event_is_overnight_ah:
                entry['news_time_original'] = entry.get('news_time', '')
                entry['news_time'] = event_info['event_time_et']
                entry['news_time_source'] = 'polygon_spike'

    prev_close = event_info['prev_close']
    absorption = calculate_absorption(prev_close, event_info['peak_price'],
                                       event_info['current_price'],
                                       event_info.get('trough_price'))
    entry['polygon_prev_close'] = prev_close
    entry['polygon_peak_price'] = event_info['peak_price']
    entry['polygon_trough_price'] = event_info.get('trough_price', prev_close)
    entry['polygon_current_price'] = event_info['current_price']
    entry['polygon_total_move_pct'] = absorption['total_move_pct']
    entry['polygon_remaining_pct'] = absorption['remaining_pct']
    entry['polygon_absorbed_frac'] = absorption['absorbed_pct']
    entry['polygon_event_time'] = event_info['event_time_et']

    if absorption['is_exhausted']:
        entry['move_is_exhausted'] = True

    src = entry.get('news_time_source', 'llm')
    print(f"    [{ticker}] timing={entry.get('news_time','?')[:25]} (src={src}), "
          f"absorbed={absorption['absorbed_pct']:.0f}%"
          f"{' EXHAUSTED' if absorption['is_exhausted'] else ''}")


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


DEEP_ANALYSIS_TOOL = {
    "name": "record_analysis",
    "description": "Record the deep analysis result for a stock ticker",
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string"},
            "catalyst": {"type": "string", "description": "Specific catalyst with numbers, targets, deal values"},
            "catalyst_type": {"type": "string", "enum": ["earnings", "analyst", "regulatory", "deal", "offering", "insider", "macro", "breaking", "premarket_move"]},
            "catalyst_is_new": {"type": "boolean"},
            "news_timeline": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "time": {"type": "string"},
                        "event": {"type": "string"},
                    },
                    "required": ["time", "event"],
                },
            },
            "timing": {"type": "string", "enum": ["breaking", "overnight", "pre_earnings", "post_earnings", "developing", "scheduled"]},
            "reasoning": {"type": "string", "description": "3-5 sentences: catalyst, absorption, remaining edge, continuation vs reversal"},
            "bull_case": {"type": "string"},
            "bear_case": {"type": "string"},
            "risks": {"type": "string"},
            "pre_market_action": {"type": "string"},
            "acquirer_or_target": {"type": "string", "enum": ["acquirer", "target", "n/a"]},
            "deal_materiality": {"type": "string", "enum": ["immaterial", "notable", "transformative", "n/a"]},
            "related_tickers": {"type": "array", "items": {"type": "string"}},
            "total_catalyst_impact_pct": {"type": "number", "description": "Total expected % impact of the catalyst"},
            "move_already_absorbed_pct": {"type": "number", "description": "How much % has already moved pre-market/AH"},
            "remaining_move_pct": {"type": "number", "description": "Expected remaining % move after entry"},
            "expected_move_pct": {"type": "number", "description": "Same as remaining_move_pct"},
            "move_is_exhausted": {"type": "boolean", "description": "True if catalyst move is fully priced in"},
            "direction": {"type": "string", "enum": ["LONG", "SHORT"]},
            "confidence": {"type": "integer", "description": "1-10 confidence in direction of remaining move"},
        },
        "required": [
            "ticker", "catalyst", "catalyst_type", "catalyst_is_new",
            "news_timeline", "timing", "reasoning",
            "bull_case", "bear_case", "risks",
            "total_catalyst_impact_pct", "move_already_absorbed_pct",
            "remaining_move_pct", "expected_move_pct", "move_is_exhausted",
            "direction", "confidence",
        ],
    },
}


def claude_analyze(prompt, system_prompt, claude_client, retries=MAX_RETRIES):
    for attempt in range(retries):
        try:
            response = claude_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=4096,
                system=system_prompt,
                tools=[DEEP_ANALYSIS_TOOL],
                tool_choice={"type": "tool", "name": "record_analysis"},
                messages=[{"role": "user", "content": prompt}],
            )
            for block in response.content:
                if block.type == "tool_use":
                    return block.input
            text = response.content[0].text if response.content else ''
            parsed = _parse_json(text)
            if parsed:
                return parsed
            return {'error': f'no_tool_use: {str(text)[:100]}'}
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
        f"Search for the latest news about {ticker} stock as of {TIME_ET_STR} on {TODAY_STR}. "
        f"What happened in the LAST FEW HOURS? Include any price action, "
        f"analyst notes, earnings, regulatory news, or breaking developments. "
        f"Prioritize the most RECENT news first — news from the last 4 hours is most valuable."
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

# --- Validate tickers via Polygon (reject non-US, invalid, company names) ---
_pre_deep = [(t, r) for t, r in ranked[:TOP_N_DEEP]
             if r.get('catalyst_score', 0) >= 3]
if _pre_deep:
    print(f"\n  Validating {len(_pre_deep)} Stage 2 candidates via Polygon...")
    _valid_tickers = set()
    _filtered = []
    for _t, _r in _pre_deep:
        if '.' in _t:
            _filtered.append((_t, 'non-US exchange (dot in symbol)'))
            continue
        _ok, _ = validate_ticker_polygon(_t)
        if _ok:
            _valid_tickers.add(_t)
        else:
            _filtered.append((_t, 'not found on US exchange'))
    if _filtered:
        print(f"  Filtered {len(_filtered)} invalid tickers:")
        for _ft, _reason in _filtered:
            print(f"    - {_ft}: {_reason}")
    else:
        print(f"  All {len(_pre_deep)} tickers validated OK")
else:
    _valid_tickers = set()

# ==============================================================================
# STAGE 2: DEEP ANALYSIS — Claude Sonnet via Vertex AI
# ==============================================================================
deep_candidates = [t for t, r in ranked[:TOP_N_DEEP]
                   if r.get('catalyst_score', 0) >= 3
                   and t in _valid_tickers]

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

    # Stage 1.5: Fetch price data + float/short interest for deep candidates
    def _fetch_price_and_fundamentals(ticker):
        pd_info = fetch_price_data(ticker)
        fund = {}
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info
            if info.get('floatShares'):
                fund['float_shares'] = info['floatShares']
            if info.get('shortPercentOfFloat'):
                fund['short_pct_float'] = round(info['shortPercentOfFloat'] * 100, 1)
            if info.get('marketCap'):
                fund['market_cap'] = info['marketCap']
        except Exception:
            pass
        return pd_info, fund

    prefetch_all_patterns()

    print(f"\n  Fetching price data + fundamentals for {len(deep_candidates)} candidates...", end=' ')
    price_data = {}
    fund_data = {}
    tp = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_SCAN) as executor:
        price_futures = {executor.submit(_fetch_price_and_fundamentals, t): t
                         for t in deep_candidates}
        for fut in as_completed(price_futures):
            ticker_p = price_futures[fut]
            try:
                pd_info, fund = fut.result()
                price_data[ticker_p] = pd_info
                fund_data[ticker_p] = fund
            except Exception:
                price_data[ticker_p] = None
                fund_data[ticker_p] = {}
    n_prices = sum(1 for v in price_data.values() if v and 'change_pct' in v)
    n_funds = sum(1 for v in fund_data.values() if v.get('float_shares'))
    print(f"done ({time.time()-tp:.0f}s, {n_prices}/{len(deep_candidates)} prices, {n_funds} with float data)")

    # ── GAP SCREENER: fetch 20d context for gap-up tickers ───────────────────
    GAP_MIN_PCT = 3.0
    gap_context = {}
    gap_tickers = [t for t in deep_candidates
                   if price_data.get(t) and price_data[t].get('change_pct', 0) >= GAP_MIN_PCT]
    if gap_tickers:
        print(f"\n  Gap screener: checking {len(gap_tickers)} tickers with gap >= {GAP_MIN_PCT}%...")
        for t in gap_tickers:
            hist = fetch_20d_high(t)
            if hist:
                gap_pct = price_data[t]['change_pct']
                dd = hist['drawdown_pct']
                if dd <= -20:
                    zone = "DEEP_RECOVERY"
                elif dd <= -10:
                    zone = "MODERATE_RECOVERY"
                elif dd <= -5:
                    zone = "MILD_DIP"
                else:
                    zone = "NEAR_HIGHS"
                recovery_target = hist['prev_close'] + (hist['high_20d'] - hist['prev_close']) * 0.5
                gap_context[t] = {
                    'gap_pct': gap_pct,
                    'high_20d': hist['high_20d'],
                    'prev_close': hist['prev_close'],
                    'drawdown_pct': dd,
                    'premarket_price': price_data[t].get('premarket_price', 0),
                    'recovery_target': round(recovery_target, 2),
                    'zone': zone,
                }
                print(f"    {'✓' if 'RECOVERY' in zone else '○'} {t}: gap +{gap_pct:.1f}%, "
                      f"20d-high ${hist['high_20d']}, drawdown {dd:.0f}% → {zone}")
        if not gap_context:
            print(f"    No gap tickers with 20d history available")

    # ── Phase A: Build all prompts (sequential — earnings verification + progress) ──
    stage2_prompts = {}
    for i, ticker in enumerate(deep_candidates):
        scan = scan_results[ticker]
        pd_info = price_data.get(ticker)
        print(f"\n  [{i+1}/{len(deep_candidates)}] {ticker} "
              f"(scan: {scan.get('catalyst_score',0)}/10, "
              f"{scan.get('direction','?')})", end='')
        if pd_info and 'change_pct' in pd_info:
            print(f"  [price: {pd_info.get('change_pct', 0):+.1f}% from prev close]")
        else:
            print()

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

        price_str = ""
        if pd_info and 'change_pct' in pd_info:
            price_str = (
                f"\n\nCURRENT PRICE DATA (from Polygon, verified):\n"
                f"- Current/pre-market price: ${pd_info.get('current_price', 'N/A')}\n"
                f"- Previous close: ${pd_info.get('prev_close', 'N/A')}\n"
                f"- Change from prev close: {pd_info.get('change_pct', 0):+.1f}%\n"
                f"- THIS IS CRITICAL: The stock has ALREADY moved {pd_info.get('change_pct', 0):+.1f}% "
                f"before the trader enters. Factor this into your remaining_move_pct estimate."
            )
        else:
            price_str = "\n\nCURRENT PRICE DATA: Not available — estimate from news reports."

        fd = fund_data.get(ticker, {})
        if fd:
            fund_str = "\n\nFUNDAMENTAL DATA (from Yahoo Finance):"
            if 'float_shares' in fd:
                float_m = fd['float_shares'] / 1e6
                fund_str += f"\n- Float: {float_m:.1f}M shares"
            if 'short_pct_float' in fd:
                fund_str += f"\n- Short interest: {fd['short_pct_float']:.1f}% of float"
                if fd['short_pct_float'] > 20:
                    fund_str += " (HIGH — short squeeze potential)"
            if 'market_cap' in fd:
                mcap_b = fd['market_cap'] / 1e9
                fund_str += f"\n- Market cap: ${mcap_b:.1f}B"
            price_str += fund_str

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
        scan_cat = str(scan.get('catalyst_type', '')).lower()
        pattern_str = fetch_similar_patterns(scan_cat, ticker)

        bounce_str = ""
        if ticker in gap_context:
            _gc = gap_context[ticker]
            _zone = _gc['zone']
            bounce_str = (
                f"\n\n20-DAY TECHNICAL CONTEXT (all prices from Polygon API, verified):\n"
                f"- 20-day high: ${_gc['high_20d']}\n"
                f"- Previous close: ${_gc['prev_close']}\n"
                f"- Drawdown from 20d high: {_gc['drawdown_pct']:.0f}%\n"
                f"- Current premarket gap: +{_gc['gap_pct']:.1f}%\n"
                f"- Recovery target (50% retracement): ${_gc['recovery_target']}\n"
                f"- Zone: {_zone}\n"
            )
            _fd = fund_data.get(ticker, {})
            _si = _fd.get('short_pct_float', 0)
            if _si > 10:
                bounce_str += (
                    f"- SHORT SQUEEZE WARNING: {_si:.1f}% short interest. With a +{_gc['gap_pct']:.1f}% "
                    f"gap, short covering will amplify the move. Do NOT go SHORT.\n"
                )
            if 'RECOVERY' in _zone or _zone == 'MILD_DIP':
                bounce_str += (
                    f"- MOMENTUM BIAS: Stock is {abs(_gc['drawdown_pct']):.0f}% below its 20d high. "
                    f"This gap-up is recovery, NOT overextension. Continuation LONG is more "
                    f"likely than fade. Do NOT default to SHORT."
                )
            else:
                bounce_str += (
                    f"- Stock is near its 20-day highs ({_gc['drawdown_pct']:.0f}% from peak). "
                    f"Gap-up momentum can still continue, but if the catalyst is weak "
                    f"(editorial, procedural), a fade is more plausible at these levels."
                )

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
            f"{bounce_str}"
            f"{market_context_str}"
            f"{pattern_str}\n\n"
            f"Provide your deep analysis. Remember: direction and expected_move_pct "
            f"must reflect the REMAINING move at 6:25 AM PT entry, not the catalyst "
            f"direction. If the stock already moved significantly, the remaining move "
            f"may be a reversal."
        )
        stage2_prompts[ticker] = prompt

    # ── Phase B: Fire all API calls in parallel ──────────────────────────────
    MAX_WORKERS_DEEP = 3
    print(f"\n  Submitting {len(stage2_prompts)} deep analyses in parallel "
          f"(max {MAX_WORKERS_DEEP} workers)...")

    def _deep_analyze_one(ticker_prompt_pair):
        _ticker, _prompt = ticker_prompt_pair
        if claude_ok:
            return _ticker, claude_analyze(_prompt, DEEP_SYSTEM, claude_client)
        else:
            return _ticker, gemini_search(_prompt, DEEP_SYSTEM)

    raw_results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_DEEP) as executor:
        deep_futures = {executor.submit(_deep_analyze_one, (t, p)): t
                        for t, p in stage2_prompts.items()}
        for fut in as_completed(deep_futures):
            _tk = deep_futures[fut]
            try:
                _, _res = fut.result()
                raw_results[_tk] = _res
                print(f"    ✓ {_tk} done")
            except Exception as e:
                raw_results[_tk] = {'error': str(e)[:200]}
                print(f"    ✗ {_tk} failed: {str(e)[:60]}")

    # ── Phase C: Post-process results (sequential — clean output) ────────────
    for ticker in deep_candidates:
        result = raw_results.get(ticker)
        if result is None:
            continue
        pd_info = price_data.get(ticker)

        if 'error' not in result:
            claude_total = float(result.get('total_catalyst_impact_pct', 0))
            polygon_absorbed = float(pd_info.get('change_pct', 0)) if pd_info and 'change_pct' in pd_info else None
            if polygon_absorbed is not None and abs(claude_total) > 0.1:
                same_direction = (claude_total > 0 and polygon_absorbed > 0) or \
                                 (claude_total < 0 and polygon_absorbed < 0)
                if same_direction:
                    computed_remaining = claude_total - polygon_absorbed
                else:
                    computed_remaining = claude_total
                computed_remaining = max(min(computed_remaining, abs(claude_total)),
                                         -abs(claude_total))
                result['move_already_absorbed_pct'] = polygon_absorbed
                result['remaining_move_pct'] = round(computed_remaining, 2)
                result['expected_move_pct'] = round(computed_remaining, 2)
                result['_remaining_source'] = 'python_math'
                if same_direction and abs(computed_remaining) < 3.0 and (
                    abs(polygon_absorbed / claude_total) > 0.80 or
                    (claude_total > 0 and polygon_absorbed > claude_total) or
                    (claude_total < 0 and polygon_absorbed < claude_total)
                ):
                    result['move_is_exhausted'] = True
                elif not same_direction:
                    result['move_is_exhausted'] = True

            direction = result.get('direction', '?')
            confidence = result.get('confidence', 0)
            remaining = float(result.get('remaining_move_pct',
                               result.get('expected_move_pct', 0)))
            absorbed = result.get('move_already_absorbed_pct', '')
            exhausted = result.get('move_is_exhausted', False)
            catalyst = str(result.get('catalyst', ''))[:70]
            exh_tag = ' [EXHAUSTED]' if exhausted else ''
            abs_str = f" (absorbed: {absorbed:+.1f}%)" if isinstance(absorbed, (int, float)) else ''
            print(f"    {ticker}: {direction} conf={confidence}/10 "
                  f"remaining={remaining:+.1f}%{abs_str}{exh_tag}  |  {catalyst}")
        else:
            print(f"    {ticker}: ERROR: {result['error'][:80]}")

        deep_results[ticker] = result

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
        '_price_data': price_data.get(ticker),
    }
    if entry['direction'] == 'LONG':
        longs.append(entry)
    elif entry['direction'] == 'SHORT':
        shorts.append(entry)

# ==============================================================================
# POST-LLM QUALITY GATES — override LLM output when reasoning contradicts signal
# ==============================================================================
SELF_CONTRADICTION_PHRASES = [
    'untradeable', 'un-tradeable', 'zero remaining edge', 'zero edge',
    'no remaining edge',
    'not a new actionable catalyst', 'no new catalyst', 'no new information',
    'very low-conviction', 'very low conviction', 'essentially noise',
    'not yet occurred', 'not yet released', 'has not yet materialized',
    'working against the catalyst', 'contradicts the',
    'immaterial', 'noise catalyst',
]
MIN_CONFIDENCE_FOR_SELECTION = 4

all_candidates = longs + shorts
_gate_log = []
for entry in all_candidates:
    ticker = entry['ticker']
    reasoning_lower = entry.get('reasoning', '').lower()
    risks_lower = entry.get('risks', '').lower()
    bear_lower = entry.get('bear_case', '').lower()
    combined_text = reasoning_lower + ' ' + risks_lower

    # Gate 1: Self-consistency — reasoning says "don't trade" but LLM traded
    contradictions = [p for p in SELF_CONTRADICTION_PHRASES if p in combined_text]
    if contradictions and entry['confidence'] <= 4:
        _gate_log.append(f"  GATE: {ticker} {entry['direction']} conf={entry['confidence']} "
                         f"→ KILLED (self-contradiction: '{contradictions[0]}')")
        entry['confidence'] = 0
        entry['_gate_reason'] = f"self_contradiction: {contradictions[0]}"

    # Gate 2: Premarket direction mismatch — LONG but stock clearly red, SHORT but stock clearly green
    # Use hard Polygon/yfinance data, not LLM's estimate
    _pd = entry.get('_price_data', {})
    _absorbed = float(_pd.get('change_pct', 0)) if _pd else float(entry.get('move_already_absorbed_pct', 0))
    if entry['direction'] == 'LONG' and isinstance(_absorbed, (int, float)) and _absorbed < -2.0:
        if entry['confidence'] <= 5:
            _gate_log.append(f"  GATE: {ticker} LONG conf={entry['confidence']} "
                             f"→ KILLED (premarket is {_absorbed:+.1f}%, opposing LONG)")
            entry['confidence'] = 0
            entry['_gate_reason'] = f"premarket_direction_mismatch: {_absorbed:+.1f}%"
    elif entry['direction'] == 'SHORT' and isinstance(_absorbed, (int, float)) and _absorbed > 2.0:
        if entry['confidence'] <= 5:
            _gate_log.append(f"  GATE: {ticker} SHORT conf={entry['confidence']} "
                             f"→ KILLED (premarket is {_absorbed:+.1f}%, opposing SHORT)")
            entry['confidence'] = 0
            entry['_gate_reason'] = f"premarket_direction_mismatch: {_absorbed:+.1f}%"

    # Gate 3: Momentum flip — exhausted SHORTs on gap-ups flip to LONG in risk-on regime
    if (entry['direction'] == 'SHORT' and entry.get('move_is_exhausted')
            and isinstance(_absorbed, (int, float)) and _absorbed > 3.0
            and not entry.get('_gate_reason')):
        _spy_pct = 0.0
        _vix = 20.0
        try:
            _spy_pct = float(market_ctx.get('spy_futures_pct', 0))
        except (TypeError, ValueError):
            _futures_str = str(market_ctx.get('futures', ''))
            _spy_match = re.search(r'S&P\s*500?\s*([+-]?\d+\.?\d*)%', _futures_str)
            _spy_pct = float(_spy_match.group(1)) if _spy_match else 0.0
        try:
            _vix = float(market_ctx.get('vix_level_float', 20))
        except (TypeError, ValueError):
            _vix_str = str(market_ctx.get('vix_level', '20'))
            _vix_match = re.search(r'(\d+\.?\d*)', _vix_str)
            _vix = float(_vix_match.group(1)) if _vix_match else 20.0
        _is_risk_on = _spy_pct > -0.5 and _vix < 25

        _gc = gap_context.get(ticker, {})
        _drawdown = _gc.get('drawdown_pct', 0)
        _original_conf = entry['confidence']

        if _is_risk_on:
            if _drawdown <= -15:
                _new_conf = _original_conf
            elif _drawdown <= -5:
                _new_conf = max(4, _original_conf - 1)
            else:
                _new_conf = min(4, _original_conf)

            entry['direction'] = 'LONG'
            entry['confidence'] = _new_conf
            entry['expected_move_pct'] = abs(entry.get('remaining_move_pct', 0))
            entry['_gate_reason'] = (f"momentum_flip_risk_on: gap +{_absorbed:.1f}%, "
                                     f"drawdown {_drawdown:.0f}%, SPY {_spy_pct:+.1f}%")
            entry['_flipped'] = True
            _gate_log.append(
                f"  GATE: {ticker} SHORT conf={_original_conf} "
                f"→ FLIPPED to LONG conf={_new_conf} (momentum: gap +{_absorbed:.1f}%, "
                f"{_drawdown:.0f}% below 20d high, SPY {_spy_pct:+.1f}%, VIX {_vix:.0f})")
        else:
            entry['confidence'] = 0
            entry['_gate_reason'] = (f"exhausted_short_killed_risk_off: gap +{_absorbed:.1f}%, "
                                     f"SPY {_spy_pct:+.1f}%, VIX {_vix:.0f}")
            _gate_log.append(
                f"  GATE: {ticker} SHORT conf={_original_conf} "
                f"→ KILLED (risk-off: SPY {_spy_pct:+.1f}%, VIX {_vix:.0f})")

    # Gate 4: Earnings reaction — market already priced in the catalyst
    # Kill earnings signals where the stock already moved >5% premarket.
    # Exception: earnings beat + positive gap (LONG direction) may still have continuation.
    _cat_type = str(entry.get('catalyst_type', '')).lower()
    if _cat_type == 'earnings' and _pd and not entry.get('_gate_reason'):
        _earn_change = float(_pd.get('change_pct', 0))
        _earn_move = abs(_earn_change)
        _earn_pm_vol = int(_pd.get('premarket_volume', 0))
        _is_gap_down = _earn_change < -5.0
        _is_gap_up_long = _earn_change > 5.0 and entry['direction'] == 'LONG'
        if _is_gap_down:
            entry['confidence'] = 0
            entry['_gate_reason'] = (f"earnings_gap_down: moved {_earn_change:+.1f}% "
                                     f"on earnings, PM vol: {_earn_pm_vol:,}")
            _gate_log.append(
                f"  GATE: {ticker} {entry['direction']} "
                f"→ KILLED (earnings gap down {_earn_change:+.1f}%)")
        elif _earn_move > 5.0 and not _is_gap_up_long:
            entry['confidence'] = 0
            entry['_gate_reason'] = (f"earnings_priced_in: moved {_earn_change:+.1f}% "
                                     f"on earnings, PM vol: {_earn_pm_vol:,}")
            _gate_log.append(
                f"  GATE: {ticker} {entry['direction']} "
                f"→ KILLED (earnings priced in {_earn_change:+.1f}%)")

    # Gate 5: Clinical trial data trap — biotech "positive data" events get sold by institutions
    CLINICAL_KEYWORDS = [
        'phase 1', 'phase 2', 'phase 3', 'phase i', 'phase ii', 'phase iii',
        'clinical trial', 'clinical data', 'trial results', 'trial data',
        'objective response rate', 'overall response rate', ' orr ',
        'complete response', 'progression-free survival', 'overall survival',
        'primary endpoint', 'secondary endpoint', 'pivotal trial',
        'fda approval', 'fda accepted', 'bla ', 'nda ', 'pdufa',
    ]
    catalyst_lower = entry.get('catalyst', '').lower()
    cat_type_lower = entry.get('catalyst_type', '').lower()
    is_clinical = any(kw in catalyst_lower or kw in reasoning_lower for kw in CLINICAL_KEYWORDS)
    if (is_clinical and entry['direction'] == 'LONG' and cat_type_lower in ('regulatory', 'breaking')
            and not entry.get('_gate_reason')):
        _gate_log.append(f"  GATE: {ticker} LONG conf={entry['confidence']} "
                         f"→ CAPPED at 3 (clinical trial data trap: institutions sell these events)")
        entry['confidence'] = min(entry['confidence'], 3)
        entry['_gate_reason'] = f"clinical_data_trap"

    # Recalculate composite after gates
    entry['composite'] = entry['confidence'] * abs(entry.get('remaining_move_pct', 0))

# Move momentum-flipped entries from shorts to longs
_flipped = [e for e in shorts if e.get('_flipped')]
for e in _flipped:
    shorts.remove(e)
    longs.append(e)

if _gate_log:
    print(f"\n  POST-LLM QUALITY GATES:")
    for msg in _gate_log:
        print(msg)

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


def _is_selected(entry, rank, direction_label):
    top_n = FINAL_LONG if direction_label == 'LONG' else FINAL_SHORT
    return (rank <= top_n
            and entry['confidence'] >= MIN_CONFIDENCE_FOR_SELECTION
            and not entry.get('_gate_reason'))


def _print_summary_row(i, e):
    cat = _CATEGORY_MAP.get(str(e.get('catalyst_type', 'none')).lower(), '?')
    ts_str, age = _get_event_time_and_age(e)
    absorbed = e.get('move_already_absorbed_pct', 0)
    abs_str = f"{absorbed:+.1f}%" if absorbed else '—'
    exh = ' EXH' if e.get('move_is_exhausted') else ''
    gate = f"  [{e['_gate_reason'][:30]}]" if e.get('_gate_reason') else ''
    catalyst_short = str(e.get('catalyst', '') or e.get('scan_summary', '') or e.get('scan_headline', '') or e.get('reasoning', ''))[:55]
    print(f"    {i:>2} {e['ticker']:<6} {e['composite']:>5.0f} "
          f"{e['confidence']:>3}/10 {e['expected_move_pct']:>+6.1f}% "
          f"{abs_str:>7} {cat:<5} {ts_str:<18} {age:>8}{exh}  {catalyst_short}{gate}")


def _print_summary_table(entries, direction_label):
    entries_sorted = sorted(entries, key=lambda x: x['composite'], reverse=True)
    n = len(entries_sorted)
    arrow = '▲' if direction_label == 'LONG' else '▼'
    print(f"\n  {arrow} {direction_label} CANDIDATES ({n})")
    if not entries_sorted:
        print(f"    No {direction_label} candidates found.\n")
        return

    for e_idx, e in enumerate(entries_sorted, 1):
        e['_is_selected'] = _is_selected(e, e_idx, direction_label)

    selected = [e for e in entries_sorted if e['_is_selected']]
    not_selected = [e for e in entries_sorted if not e['_is_selected']]

    header = (f"    {'#':>2} {'Ticker':<6} {'Score':>5} {'Conf':>4} {'Remain':>7} "
              f"{'Absorb':>7} {'Type':<5} {'Event Time':<18} {'Age':>8}  Catalyst Summary")

    if selected:
        print(f"\n    ═══ SELECTED ({len(selected)}) ═══")
        print(header)
        print(f"    {'─'*110}")
        for i, e in enumerate(selected, 1):
            _print_summary_row(i, e)

    if not_selected:
        print(f"\n    ─── NOT SELECTED ({len(not_selected)}) ───")
        print(header)
        print(f"    {'─'*110}")
        for i, e in enumerate(not_selected, 1):
            _print_summary_row(i, e)

    if not selected:
        print(f"\n    ═══ NO SIGNALS SELECTED ═══")

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
        sel_tag = ' ✅ SELECTED' if e.get('_is_selected') else ' ⛔ NOT SELECTED'
        print(f"\n  ┌─ {i}. {e['ticker']}{sel_tag}  {arrow} {direction_label}  "
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


# ── Verify event timing via Polygon for candidates with large absorbed moves ──
if _POLYGON_OK:
    print(f"\n  Verifying event timing via Polygon...")
    for e in longs + shorts:
        verify_and_correct_event_timing(e)
else:
    print(f"\n  Polygon not available — skipping event timing verification")

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

# ── Apply catalyst-type weight to composite scores (Bayesian-shrunk) ──
# Empirical weights from Sep 3-9 2026 backtest, shrunk toward 1.0 via
# w = 1 + (w_emp - 1) * n / (n + k), k=30. Small samples pull toward neutral.
BAYESIAN_K = 30
_CATALYST_RAW = {
    'macro':          (1.2, 3),
    'breaking':       (1.2, 12),
    'insider':        (1.2, 5),
    'offering':       (1.0, 8),
    'analyst':        (1.0, 15),
    'squeeze':        (1.0, 0),
    'halt':           (1.0, 0),
    'premarket_move': (0.5, 9),
    'earnings':       (0.7, 24),
    'deal':           (0.6, 25),
    'regulatory':     (0.4, 10),
    'bankruptcy':     (0.5, 0),
}
CATALYST_TYPE_WEIGHTS = {
    k: round(1.0 + (w - 1.0) * n / (n + BAYESIAN_K), 3)
    for k, (w, n) in _CATALYST_RAW.items()
}
for e in longs + shorts:
    cat = str(e.get('catalyst_type', '')).lower()
    cw = CATALYST_TYPE_WEIGHTS.get(cat, 1.0)
    e['composite'] = e['composite'] * cw
    e['catalyst_weight'] = cw
longs.sort(key=lambda x: x['composite'], reverse=True)
shorts.sort(key=lambda x: x['composite'], reverse=True)

# ── Mismatch Index: price move vs LLM catalyst assessment ──
# Ratio of actual gap to LLM confidence. High mismatch = retail FOMO
# (big price move on weak catalyst). Boosts SHORT on high mismatch, LONG on low mismatch.
for e in longs + shorts:
    _pd = e.get('_price_data') or {}
    gap_pct = abs(float(_pd.get('change_pct', 0)))
    _conf = max(int(e.get('confidence', 5)), 1)
    mismatch = gap_pct / _conf
    if mismatch > 2.0 and e['direction'] == 'LONG':
        e['mismatch_mult'] = 0.5
    elif mismatch > 2.0 and e['direction'] == 'SHORT':
        e['mismatch_mult'] = 1.5
    elif mismatch < 0.5 and e['direction'] == 'LONG':
        e['mismatch_mult'] = 1.3
    else:
        e['mismatch_mult'] = 1.0
    e['composite'] = e['composite'] * e['mismatch_mult']
    e['mismatch_index'] = round(mismatch, 2)
longs.sort(key=lambda x: x['composite'], reverse=True)
shorts.sort(key=lambda x: x['composite'], reverse=True)

# ── Volume ratio multiplier ──
for e in longs + shorts:
    _pd = e.get('_price_data') or {}
    pm_vol = float(_pd.get('premarket_volume', 0))
    avg_vol = float(_pd.get('avg_daily_volume', 0))
    if avg_vol > 0 and pm_vol > 0:
        vol_ratio = pm_vol / avg_vol
        if vol_ratio >= 3.0:
            vol_mult = 1.3
        elif vol_ratio >= 1.0:
            vol_mult = 1.1
        elif vol_ratio < 0.3:
            vol_mult = 0.7
        else:
            vol_mult = 1.0
    else:
        vol_ratio = 0.0
        vol_mult = 1.0
    e['composite'] = e['composite'] * vol_mult
    e['volume_ratio'] = round(vol_ratio, 2)
    e['volume_mult'] = vol_mult
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
        ('gate_reason', 'STRING'),
        ('catalyst_weight', 'FLOAT64'),
        ('mismatch_index', 'FLOAT64'),
        ('volume_ratio', 'FLOAT64'),
        ('volume_mult', 'FLOAT64'),
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
                'selected': rank <= (FINAL_LONG if direction_label == 'LONG' else FINAL_SHORT) and e['confidence'] >= MIN_CONFIDENCE_FOR_SELECTION and not e.get('_gate_reason'),
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
                'catalyst_weight': e.get('catalyst_weight', 1.0),
                'mismatch_index': e.get('mismatch_index', 0.0),
                'volume_ratio': e.get('volume_ratio', 0.0),
                'volume_mult': e.get('volume_mult', 1.0),
                'scanner_version': SCANNER_VERSION,
                'gate_reason': str(e.get('_gate_reason', ''))[:200],
            })

    # Also log scan-only tickers (no deep analysis) for coverage tracking
    # Direction set to NEUTRAL — Stage 1 direction is a rough classification, not a signal
    deep_tickers = set(e['ticker'] for e in longs + shorts)
    for ticker, sr in scan_results.items():
        if ticker in deep_tickers or 'error' in sr:
            continue
        rows.append({
            'scan_timestamp': now_utc,
            'scan_date': run_ts_str,
            'scan_time_pt': time_str,
            'ticker': ticker,
            'direction': 'NEUTRAL',
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
