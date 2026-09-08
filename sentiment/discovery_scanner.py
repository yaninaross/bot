# %% ── Discovery Scanner — Find Biggest Movers Outside Deploy Universe ────────
#
# Thematic search agent that discovers stocks making massive news overnight.
# NOT tied to the deploy ticker list — finds opportunities you're not tracking.
#
#   Stage 0: Market context (same as premarket_scanner)
#   Stage 1: Thematic searches — 12 queries covering earnings, FDA, M&A,
#            halts, analyst actions, insider, short squeeze, etc.
#            Gemini extracts ticker + expected move from search results.
#   Stage 2: Deep analysis — score each discovered ticker for tradability
#   Output:  Top 5 LONG + 5 SHORT with >5% expected move, ranked by
#            move_score = expected_move_pct × catalyst_strength × freshness
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
BQ_TABLE    = "ys-time-series.scoring_logs.discovery_scanner"
BQ_PATTERNS_TABLE = "ys-time-series.scoring_logs.scanner_patterns"

PST = pytz.timezone('America/Los_Angeles')
NOW_PT = datetime.now(PST)
TODAY_STR = NOW_PT.strftime('%Y-%m-%d')
TIME_STR = NOW_PT.strftime('%H:%M')

MIN_MOVE_PCT       = 5.0
MAX_NEWS_AGE_HOURS = 16
SCANNER_VERSION    = 'V3'
FINAL_LONG         = 5
FINAL_SHORT        = 5
MAX_WORKERS        = 4
GEMINI_DELAY_S     = 0.5
MAX_RETRIES        = 3

DEPLOY_TICKERS = {
    'ACHR', 'ACMR', 'AEHR', 'ALAB', 'AMD', 'APLD', 'ARM', 'ASTS',
    'AVAV', 'AXTI', 'BBAI', 'BE', 'BEAM', 'BKSY', 'BLZE', 'BTBT',
    'CAT', 'CEG', 'CIFR', 'CLSK', 'COIN', 'CORZ', 'CRDO', 'CVX',
    'ETN', 'FCEL', 'FLNC', 'GEV', 'HOOD', 'HUT', 'IREN', 'JOBY',
    'KEEL', 'KTOS', 'LUNR', 'MARA', 'MNTS', 'MRNA', 'MSTR', 'MU',
    'MVIS', 'MXL', 'NBIS', 'NEE', 'NNE', 'NTLA', 'OKLO', 'ONDS',
    'PL', 'PLTR', 'PLUG', 'RCAT', 'RDW', 'RIOT', 'RKLB', 'RXT',
    'SLS', 'SMCI', 'SMR', 'SNDK', 'SOUN', 'TSLA', 'VOYG', 'VRT',
    'VST', 'WOLF', 'WULF', 'XOM',
}

W = 100


# ==============================================================================
# THEMATIC SEARCH QUERIES
# ==============================================================================
_ET = pytz.timezone('America/New_York')
NOW_ET = NOW_PT.astimezone(_ET)
TIME_ET_STR = NOW_ET.strftime('%I:%M %p ET')

THEMATIC_QUERIES = [
    {
        'id': 'stocks_moving_now',
        'query': (f'stocks moving right now {TODAY_STR} as of {TIME_ET_STR} '
                  f'biggest gainers losers surging crashing'),
        'direction_bias': None,
    },
    {
        'id': 'afterhours_movers',
        'query': (f'stocks after hours movers tonight {TODAY_STR} '
                  f'earnings reaction surging crashing after close'),
        'direction_bias': None,
    },
    {
        'id': 'earnings_today',
        'query': (f'earnings results today {TODAY_STR} '
                  f'EPS revenue beat miss guidance stock reaction'),
        'direction_bias': None,
    },
    {
        'id': 'biggest_gainers',
        'query': (f'biggest stock gainers today {TODAY_STR} '
                  f'stocks up more than 5 percent as of {TIME_ET_STR}'),
        'direction_bias': 'LONG',
    },
    {
        'id': 'biggest_losers',
        'query': (f'biggest stock losers today {TODAY_STR} '
                  f'stocks down more than 5 percent as of {TIME_ET_STR}'),
        'direction_bias': 'SHORT',
    },
    {
        'id': 'fda_regulatory',
        'query': (f'FDA approval rejection drug stock {TODAY_STR} '
                  f'biotech pharma halted clinical trial'),
        'direction_bias': None,
    },
    {
        'id': 'mergers_acquisitions',
        'query': (f'stock acquisition merger takeover announced {TODAY_STR} '
                  f'premium buyout offer deal'),
        'direction_bias': 'LONG',
    },
    {
        'id': 'analyst_actions',
        'query': (f'stock analyst upgrade downgrade price target raised lowered '
                  f'{TODAY_STR} biggest moves as of {TIME_ET_STR}'),
        'direction_bias': None,
    },
    {
        'id': 'halt_resume',
        'query': (f'stock trading halted resumed {TODAY_STR} '
                  f'NYSE NASDAQ halt pending news'),
        'direction_bias': None,
    },
    {
        'id': 'short_squeeze',
        'query': (f'short squeeze stocks today {TODAY_STR} '
                  f'high short interest rallying gamma squeeze'),
        'direction_bias': 'LONG',
    },
    {
        'id': 'offerings_dilution',
        'query': (f'stock offering dilution secondary share sale announced '
                  f'{TODAY_STR} stock plunge'),
        'direction_bias': 'SHORT',
    },
    {
        'id': 'contracts_deals',
        'query': (f'stock major contract deal partnership billion {TODAY_STR} '
                  f'government defense AI'),
        'direction_bias': 'LONG',
    },
    {
        'id': 'bankruptcy_fraud',
        'query': (f'stock fraud investigation bankruptcy SEC lawsuit {TODAY_STR} '
                  f'crash plunge'),
        'direction_bias': 'SHORT',
    },
    {
        'id': 'insider_activity',
        'query': (f'insider buying selling stock large stake {TODAY_STR} '
                  f'CEO CFO director'),
        'direction_bias': None,
    },
]


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
# PROMPTS
# ==============================================================================
CUTOFF_4H = (NOW_ET - timedelta(hours=4)).strftime('%Y-%m-%d %I:%M %p ET')
CUTOFF_16H = (NOW_ET - timedelta(hours=16)).strftime('%Y-%m-%d %I:%M %p ET')

DISCOVER_SYSTEM = f"""You are a stock screener. Your job is to extract specific stock tickers and their expected moves from search results about today's biggest movers.

CURRENT TIME: {NOW_ET.strftime('%Y-%m-%d %I:%M %p ET')} ({NOW_PT.strftime('%I:%M %p PT')})
PRIORITY WINDOW: News published AFTER {CUTOFF_4H} (last 4 hours) is highest priority.
HARD CUTOFF: Do NOT include any news published BEFORE {CUTOFF_16H} (older than 16 hours).

FRESHNESS SCORING:
  - Published within last 2 hours: MOST VALUABLE — always include, boost catalyst_strength by +2
  - Published 2-4 hours ago: HIGH VALUE — include with normal scoring
  - Published 4-8 hours ago: MODERATE — include only if catalyst_strength >= 5
  - Published 8-16 hours ago: LOW VALUE — include only if catalyst_strength >= 7
  - Published >16 hours ago: EXCLUDE entirely

From the search results, identify individual stocks that are likely to move MORE THAN 5%.

Return ONLY a JSON object:
{{
  "discovered": [
    {{
      "ticker": "<stock ticker symbol, e.g. AAPL>",
      "company": "<company name>",
      "direction": "LONG" or "SHORT",
      "expected_move_pct": <float, expected % move, e.g. 8.5>,
      "catalyst_strength": <int 1-10, how strong/certain is this catalyst>,
      "catalyst": "<specific event driving the move — cite numbers, %s, deal values>",
      "catalyst_type": "<earnings|analyst|regulatory|deal|offering|insider|halt|squeeze|bankruptcy|macro>",
      "news_time": "<REQUIRED — ACTUAL publish time from the source article, format: 'YYYY-MM-DD H:MM AM/PM ET'. Look at the article dateline/timestamp. Do NOT default to generic times like '5:00 AM' or '4:00 PM'. Find the real time.>",
      "source_headline": "<actual headline from the search result>",
      "premarket_move": "<pre-market % change if mentioned, or 'unknown'>",
      "market_cap_tier": "<mega|large|mid|small|micro|unknown>"
    }}
  ]
}}

M&A/ACQUISITION RULES:
- For M&A news, you MUST identify whether each ticker is the ACQUIRER (buyer) or TARGET (being acquired)
- TARGET stocks get premium → expected 15-40% moves, catalyst_strength 8-10
- ACQUIRER stocks are typically NEUTRAL or NEGATIVE (dilution from issuing shares,
  integration risk, paying premium). Do NOT include acquirer stocks unless the deal is
  truly transformative (>20% of acquirer market cap)
- "Shareholder vote approved" / "regulatory clearance" / "expected to close" for an
  already-announced deal is a PROCEDURAL update — NOT a new catalyst. Do NOT include these.
- Always note in the catalyst field whether the ticker is "ACQUIRER" or "TARGET"

IMPORTANT:
- Only include stocks with CLEAR catalysts likely to cause >5% moves
- Be specific: cite actual numbers (EPS miss by $0.30, FDA approved drug X, acquired for $Y/share at Z% premium)
- expected_move_pct should reflect the magnitude of the catalyst — earnings miss = 5-15%, FDA rejection = 10-30%, M&A target = 15-40%, M&A acquirer = usually <3%
- catalyst_strength: 10 = confirmed event (earnings released, deal signed), 5 = developing (investigation, rumor), 1 = speculative
- Include ALL qualifying stocks you find, even if there are many
- Return ONLY the JSON"""

DEEP_SYSTEM = """You are a senior pre-market trading analyst evaluating a stock for a potential 5%+ intraday trade.
Your traders enter at market open and target large intraday moves (5-15%+).

CRITICAL CONCEPT — REMAINING MOVE vs TOTAL IMPACT:
Your job is to predict what happens AFTER the trader enters at market open. If a stock already
dropped 13% pre-market on bad earnings, the bearish move may be OVER. The remaining move at
open could be a BOUNCE (LONG), not further selling. You must separate:
  - expected_move_pct: the full magnitude of the catalyst
  - move_already_priced_pct: how much has already moved in pre-market/after-hours
  - remaining_move_pct: what's LEFT for the trader at entry (this is what matters)

If |move_already_priced| >= |expected_move|, the move is DONE. The direction field should
reflect the REMAINING move (often a reversal/bounce), NOT the catalyst direction.

Analyze whether this stock will actually deliver a 5%+ move today based on the catalyst described. Be realistic and quantitative.

Return ONLY a JSON object:
{
  "ticker": "<TICKER>",
  "direction": "LONG" or "SHORT" <-- direction of the REMAINING move at entry, NOT the catalyst direction,
  "confidence": <int 1-10>,
  "expected_move_pct": <float, total expected impact of the catalyst>,
  "move_already_priced_pct": <float, how much has already moved in pre-market/after-hours, e.g. 3.0 means 3% already moved>,
  "remaining_move_pct": <float, expected REMAINING % move after entry, what's left to capture>,
  "move_is_exhausted": <bool, true if the catalyst move is already fully priced in>,
  "catalyst_strength": <int 1-10, 10=confirmed hard catalyst, 1=rumor>,
  "freshness": <int 1-10, 10=broke in last hour, 5=overnight, 1=yesterday morning>,
  "liquidity_risk": <int 1-10, 10=very liquid mega-cap, 1=illiquid micro-cap>,
  "reasoning": "<3-5 sentences: WHY this specific catalyst should cause this move, what comparable events caused in the past, whether the move is already priced in, and what could go wrong>",
  "catalyst": "<specific catalyst with numbers>",
  "catalyst_type": "<earnings|analyst|regulatory|deal|offering|insider|halt|squeeze|bankruptcy|macro>",
  "acquirer_or_target": "<'acquirer'|'target'|'n/a' — REQUIRED when catalyst_type is 'deal'>",
  "deal_materiality": "<'immaterial'|'notable'|'transformative'|'n/a' — deal size vs market cap>",
  "catalyst_is_new": <bool, true if NEW announcement, false if procedural update on existing deal/event>,
  "news_timeline": [
    {"time": "<timestamp>", "event": "<what happened>"}
  ],
  "bull_case": "<why go long>",
  "bear_case": "<why go short>",
  "risks": "<what could go wrong>",
  "comparable_events": "<similar past events and their outcomes, e.g. 'XYZ missed by similar margin in Q2, dropped 12%'>",
  "pre_market_action": "<pre-market price/volume details>"
}

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
  are PROCEDURAL steps — they are NOT new information. Confidence should be 1-3 max.
- Revenue guidance or synergy projections from old announcements are ALREADY PRICED IN.
- Deal size matters: a $43M acquisition by a $4B company is <1% of market cap —
  this CANNOT justify a 5%+ expected move. Scale expected_move_pct to deal materiality.

CATALYST IMPACT REFERENCE TABLE — calibrate expected_move_pct against these empirical ranges:
  Category             | Typical Intraday Impact   | Notes
  ---------------------|---------------------------|------
  FDA full approval    | +5% to +30%               | Depends on drug revenue potential
  FDA BLA/NDA accepted | +2% to +5% at most        | Procedural, NOT approval
  FDA Priority Review  | +1% to +3%                | Procedural milestone
  PDUFA date set       | +1% to +2%                | Calendar event, not outcome
  Earnings beat (big)  | +5% to +20%               | >15% EPS beat + guidance raise
  Earnings beat (small)| +2% to +5%                | Modest beat, inline guidance
  Non-recurring profit | +0% to +2%                | One-time gains (derivative, M2M); market discounts
  Analyst upgrade      | +2% to +5%                | With >10% PT raise
  M&A target           | +10% to +40%              | Premium to pre-deal price
  M&A acquirer         | -2% to +2%                | Typically FLAT or NEGATIVE
  Secondary offering   | -5% to -15%               | Dilution + discount
  Clinical enrollment  | +1% to +5%                | Milestone, not data readout
  Design win / contract| +3% to +10%               | Depends on revenue materiality
  Stock split          | +0% to +2%                | Cosmetic, no fundamental change
  Editorial / opinion  | 0%                         | NOT a catalyst

WARNING — FDA PROCEDURAL EVENTS:
FDA procedural milestones (BLA acceptance, Priority Review designation, PDUFA date set) are NOT
the same as FDA APPROVAL. Do NOT assign confidence > 4 or remaining_move > 5% for these events.
Only full FDA approval or complete response letters justify large expected moves.

WARNING — NON-RECURRING EARNINGS:
One-time gains (derivative assets, mark-to-market investments, asset sales, legal settlements)
are NOT organic business improvement. Markets routinely discount non-recurring items.
Confidence should be capped at 4 for earnings beats driven primarily by one-time gains.

WARNING — PRE-EARNINGS SPECULATION:
If a company has NOT yet reported earnings (report expected later today or this week),
there is NO catalyst to trade. Do NOT recommend a direction based on options-implied moves
or analyst expectations. Set confidence <= 2 and note "no catalyst has occurred yet."

NOISE FILTERS — discount these patterns:
- Law firm "investigations" announced AFTER a stock drops are ambulance-chasing, not new catalysts
- Routine analyst reiterations without price target changes
- News articles that merely REPORT a price move without a new catalyst
- Pre-market price quotes from aggregator sites that just restate known information

EARNINGS MISS/BEAT — DIRECTION RULES:
- An earnings MISS is a SHORT catalyst. Default direction is SHORT unless the stock has
  ALREADY fallen >15%, the miss was minor, AND pre-market shows stabilization.
- A massive earnings miss (revenue miss >10%, guidance cut) that is STILL FRESH (<12h old)
  should remain SHORT — the selloff often continues at market open even after a large AH drop.
- Only flip to LONG (bounce) if the miss was minor AND the stock is showing clear stabilization.
- An earnings BEAT is a LONG catalyst. Only flip to SHORT (fade) if the stock already surged
  >15% and the beat was modest.
- When in doubt: FOLLOW THE CATALYST DIRECTION, not the bounce thesis.

PROVEN TRADE OUTCOME PATTERNS — learn from actual past results:

PARTNERSHIP ASYMMETRY:
  When a mega-cap ($100B+) announces a partnership with a smaller company ($10-50B):
  → Trade the SMALLER partner LONG (expect +3-8% intraday)
  → Do NOT trade the larger partner — move is <1% and often negative
  Real example: 2026-09-08 PLTR ($174B) named NBIS ($50B) as preferred sovereign AI partner.
    NBIS: gap +3.7%, MFE +6.2%, EOD +6.2% → STRONG TP
    PLTR: gap -0.7%, EOD -1.0% → FP (partnership immaterial to mega-cap revenue)

NON-RECURRING EARNINGS:
  When earnings beat is driven by one-time gains (derivative, M2M, asset sales):
  → Cap confidence at 4, cap remaining_move at +2%
  Real example: 2026-09-08 GME Q2 "beat" from $166M derivative gain → ret_60m -2.9%, FP

PRE-EARNINGS SPECULATION:
  When earnings have NOT YET been reported:
  → Confidence ≤2. Options-implied moves are NOT directional signals.
  Real example: 2026-09-08 BRZE earnings after close → ret_60m -5.5%, FP

PENNY STOCK LONG TRAP:
  Sub-$3 stocks that gap up 20%+ pre-market on news almost always fade at open.
  → Do NOT go LONG on sub-$3 stocks. Gap-and-crap reversal is the dominant pattern.
  Real example: 2026-09-08 ARBE ($0.83) +33% pre-market on design win → ret_60m -1.5%, FP
  Real example: 2026-09-08 BJDX ($1.02) +24% pre-market on enrollment → ret_60m -4.6%, FP

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
- direction and remaining_move_pct must reflect the REMAINING move, not the catalyst direction
- If a stock already crashed 13% pre-market, the remaining move MAY be LONG (bounce) — but only
  if the catalyst is minor or priced in. For major catalysts (big earnings miss, FDA rejection),
  the selloff often CONTINUES at open. Don't assume bounce.
- If a stock already surged 10%+ on STRONG earnings, the catalyst is REAL — don't auto-short it.
  A stock up 52% on great earnings may keep running. Only suggest SHORT (fade) if the surge is
  clearly irrational relative to the catalyst magnitude (e.g., +50% on a minor beat).
- remaining_move_pct = expected_move_pct - move_already_priced_pct (approximately)
- catalyst_strength × freshness × remaining_move_pct is the key trading signal
- Be honest about liquidity — micro-caps with <$1M daily volume are untradeable
- If HISTORICAL PATTERN EXAMPLES are provided below the prompt, use them to calibrate your
  confidence and expected move. Patterns with confirmed outcomes are more reliable than theory.
- Return ONLY the JSON"""

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
  "top_premarket_movers": "<any stocks moving 5%+ in pre-market>"
}"""


# ==============================================================================
# HELPERS (shared with premarket_scanner)
# ==============================================================================
def _extract_gemini_text(response):
    if response.text is not None:
        return response.text
    try:
        return response.candidates[0].content.parts[0].text
    except (IndexError, AttributeError, TypeError):
        pass
    return None


def _safe_int(val, default=5):
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str):
        import re as _re
        m = _re.search(r'\d+', val)
        return int(m.group()) if m else default
    return default

def _safe_float(val, default=0.0):
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        import re as _re
        m = _re.search(r'[+-]?\d+\.?\d*', val)
        return float(m.group()) if m else default
    return default


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
    for suffix in ['"}', ']}', ']"}', '"}]', '}', '"]}']:
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
                return {
                    'current_price': round(current_price, 2),
                    'prev_close': round(prev_close, 2),
                    'change_pct': round(change_pct, 2),
                    'premarket_price': round(current_price, 2),
                    'premarket_volume': premarket_volume,
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
                'source': 'yfinance (single day)',
            }
    except Exception as e:
        print(f"    [PRICE] yfinance failed for {ticker}: {e}")
    return None


# ==============================================================================
# PATTERN LIBRARY — BQ-backed few-shot retrieval (batch prefetch)
_PATTERN_CACHE_DF = None

CAT_MAP = {
    'earnings': 'earnings', 'analyst': 'deal', 'regulatory': 'regulatory',
    'deal': 'deal', 'offering': 'offering', 'insider': 'deal',
    'breaking': 'deal', 'premarket_move': 'deal', 'halt': 'deal',
    'squeeze': 'deal', 'bankruptcy': 'deal', 'macro': 'deal',
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
            'current_price': current_price,
            'ref_close_for_day': ref_close,
        }
    except Exception as e:
        print(f"      [EVENT_TIME] Error for {ticker}: {e}")
        return None


def calculate_absorption(prev_close, peak_price, current_price):
    total_move = (peak_price - prev_close) / prev_close if prev_close > 0 else 0
    remaining = (current_price - prev_close) / prev_close if prev_close > 0 else 0
    if abs(total_move) < 0.001:
        return {'total_move_pct': 0, 'remaining_pct': 0, 'absorbed_pct': 0, 'is_exhausted': False}
    absorbed_frac = 1.0 - (remaining / total_move) if total_move != 0 else 0
    return {
        'total_move_pct': round(total_move * 100, 2),
        'remaining_pct': round(remaining * 100, 2),
        'absorbed_pct': round(absorbed_frac * 100, 1),
        'is_exhausted': absorbed_frac > 0.80 or (total_move > 0 and remaining <= 0),
    }


def verify_and_correct_event_timing(entry):
    ticker = entry.get('ticker', '')
    if not ticker or not _POLYGON_OK:
        return

    print(f"    [{ticker}] Verifying event timing via Polygon...", end=' ')
    event_info = get_actual_event_time(ticker, TODAY_STR)
    if not event_info:
        print("no spike found")
        return

    event_dt = event_info['event_time_dt']
    event_age_hours = (NOW_ET - event_dt).total_seconds() / 3600
    news_time_raw = str(entry.get('news_time', ''))
    article_dt = _parse_news_time(news_time_raw)
    article_age_hours = None
    if article_dt:
        article_age_hours = (NOW_ET - article_dt.astimezone(_ET)).total_seconds() / 3600

    corrected = False
    if article_age_hours is not None and event_age_hours > article_age_hours + 1.0:
        entry['news_time_original'] = entry.get('news_time', '')
        entry['news_time'] = event_info['event_time_et']
        corrected = True

    prev_close = event_info['prev_close']
    absorption = calculate_absorption(prev_close, event_info['peak_price'],
                                       event_info['current_price'])
    entry['polygon_prev_close'] = prev_close
    entry['polygon_peak_price'] = event_info['peak_price']
    entry['polygon_current_price'] = event_info['current_price']
    entry['polygon_total_move_pct'] = absorption['total_move_pct']
    entry['polygon_remaining_pct'] = absorption['remaining_pct']
    entry['polygon_absorbed_frac'] = absorption['absorbed_pct']
    entry['polygon_event_time'] = event_info['event_time_et']

    if absorption['is_exhausted']:
        entry['move_is_exhausted'] = True

    if corrected:
        print(f"CORRECTED: event at {event_info['event_time_et']} "
              f"({event_age_hours:.1f}h ago, was {article_age_hours:.1f}h), "
              f"absorbed={absorption['absorbed_pct']:.0f}%"
              f"{' EXHAUSTED' if absorption['is_exhausted'] else ''}")
    else:
        print(f"OK: event ~{event_age_hours:.1f}h ago, "
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
                    max_output_tokens=8192,
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
            "direction": {"type": "string", "enum": ["LONG", "SHORT"]},
            "confidence": {"type": "integer", "description": "1-10 confidence in direction of remaining move"},
            "total_catalyst_impact_pct": {"type": "number", "description": "Total expected % impact of the catalyst"},
            "move_already_absorbed_pct": {"type": "number", "description": "How much % has already moved pre-market/AH"},
            "remaining_move_pct": {"type": "number", "description": "Expected remaining % move after entry"},
            "expected_move_pct": {"type": "number", "description": "Same as remaining_move_pct"},
            "move_is_exhausted": {"type": "boolean", "description": "True if catalyst move is fully priced in"},
            "reasoning": {"type": "string", "description": "3-5 sentences: catalyst, absorption, remaining edge, continuation vs reversal"},
            "catalyst": {"type": "string", "description": "Specific catalyst with numbers, targets, deal values"},
            "catalyst_type": {"type": "string", "enum": ["earnings", "analyst", "regulatory", "deal", "offering", "insider", "macro", "breaking", "premarket_move"]},
            "acquirer_or_target": {"type": "string", "enum": ["acquirer", "target", "n/a"]},
            "deal_materiality": {"type": "string", "enum": ["immaterial", "notable", "transformative", "n/a"]},
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
            "bull_case": {"type": "string"},
            "bear_case": {"type": "string"},
            "risks": {"type": "string"},
            "pre_market_action": {"type": "string"},
            "related_tickers": {"type": "array", "items": {"type": "string"}},
            "liquidity_risk": {"type": "integer"},
            "comparable_events": {"type": "string"},
            "market_cap_tier": {"type": "string"},
        },
        "required": [
            "ticker", "direction", "confidence", "total_catalyst_impact_pct",
            "move_already_absorbed_pct", "remaining_move_pct", "expected_move_pct",
            "move_is_exhausted", "reasoning", "catalyst", "catalyst_type",
            "catalyst_is_new", "news_timeline", "timing",
            "bull_case", "bear_case", "risks",
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


# ==============================================================================
# STAGE 0: MARKET CONTEXT
# ==============================================================================
t_total = time.time()

print(f"\n{'='*W}")
print(f"  DISCOVERY SCANNER — Find Biggest Movers")
print(f"  {TODAY_STR} {TIME_STR} PT")
print(f"  Thematic searches: {len(THEMATIC_QUERIES)}")
print(f"  Target: stocks with >5% expected move at open")
print(f"  Stage 1: {GEMINI_MODEL} + Google Search (thematic)")
print(f"  Stage 2: Deep analysis (Gemini / Claude)")
print(f"{'='*W}")

print(f"\n  Stage 0: Market context...", end=' ')
t0 = time.time()

market_ctx = gemini_search(
    f"What are the US stock market conditions right now on {TODAY_STR}? "
    f"Include S&P/Nasdaq/Dow futures, overnight developments, pre-market movers "
    f"moving more than 5%, any scheduled economic data today, and VIX level.",
    MARKET_SYSTEM,
)

if 'error' not in market_ctx:
    print(f"done ({time.time()-t0:.1f}s)")
    print(f"\n  Futures:    {market_ctx.get('futures', 'N/A')}")
    print(f"  Theme:      {market_ctx.get('overnight_theme', 'N/A')}")
    print(f"  Risk:       {market_ctx.get('risk_sentiment', 'N/A')}")
    print(f"  VIX:        {market_ctx.get('vix_level', 'N/A')}")
    print(f"  Big movers: {market_ctx.get('top_premarket_movers', 'N/A')}")
else:
    print(f"error: {market_ctx.get('error', '')[:80]}")
    market_ctx = {}


# ==============================================================================
# STAGE 1: THEMATIC DISCOVERY — parallel Gemini searches
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  Stage 1: Thematic discovery — {len(THEMATIC_QUERIES)} searches")
print(f"{'='*W}")

t1 = time.time()
all_discovered = {}  # ticker -> best discovery entry
theme_results = {}


def _run_thematic_search(theme):
    result = gemini_search(theme['query'], DISCOVER_SYSTEM)
    time.sleep(GEMINI_DELAY_S)
    return theme['id'], theme.get('direction_bias'), result


with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futs = {executor.submit(_run_thematic_search, t): t for t in THEMATIC_QUERIES}

    for fut in as_completed(futs):
        theme_id, direction_bias, result = fut.result()

        if 'error' in result:
            print(f"  [{theme_id:<20}] ERROR: {result['error'][:60]}")
            theme_results[theme_id] = {'error': result['error']}
            continue

        discovered = result.get('discovered', [])
        if not isinstance(discovered, list):
            discovered = []

        sources = result.get('_sources', [])
        n_new = 0
        for d in discovered:
            if not isinstance(d, dict):
                continue
            ticker = str(d.get('ticker', '')).upper().strip()
            if not ticker or len(ticker) > 6:
                continue

            move = abs(float(d.get('expected_move_pct', 0)))
            strength = int(d.get('catalyst_strength', 0))
            d['_theme'] = theme_id
            d['_sources'] = sources
            d['in_deploy'] = ticker in DEPLOY_TICKERS

            if ticker not in all_discovered:
                d['_all_themes'] = [theme_id]
                d['_all_catalysts'] = [str(d.get('catalyst', ''))]
                all_discovered[ticker] = d
                n_new += 1
            else:
                existing = all_discovered[ticker]
                existing.setdefault('_all_themes', []).append(theme_id)
                existing.setdefault('_all_catalysts', []).append(str(d.get('catalyst', '')))
                if move > abs(float(existing.get('expected_move_pct', 0))):
                    d['_all_themes'] = existing['_all_themes']
                    d['_all_catalysts'] = existing['_all_catalysts']
                    all_discovered[ticker] = d
                    n_new += 1

        print(f"  [{theme_id:<20}] found {len(discovered):>2} stocks, "
              f"{n_new} new/upgraded  "
              f"({', '.join(str(d.get('ticker','?')) for d in discovered[:5])})")

        theme_results[theme_id] = {
            'n_discovered': len(discovered),
            'tickers': [d.get('ticker') for d in discovered],
        }

elapsed_s1 = time.time() - t1
print(f"\n  Stage 1 complete: {elapsed_s1:.0f}s")
print(f"  Total unique tickers discovered: {len(all_discovered)}")

# --- Filter and rank ---
candidates = []
for ticker, d in all_discovered.items():
    move = abs(float(d.get('expected_move_pct', 0)))
    strength = int(d.get('catalyst_strength', 0))
    # move_score: quantitative ranking metric
    move_score = move * strength / 10.0
    d['move_score'] = round(move_score, 2)
    d['expected_move_pct'] = round(move, 1)
    candidates.append(d)

candidates.sort(key=lambda x: x['move_score'], reverse=True)

# Separate by direction
longs_raw = [c for c in candidates if c.get('direction') == 'LONG']
shorts_raw = [c for c in candidates if c.get('direction') == 'SHORT']

n_above_thresh = sum(1 for c in candidates if c['expected_move_pct'] >= MIN_MOVE_PCT)
print(f"  Above {MIN_MOVE_PCT}% threshold: {n_above_thresh}")
print(f"  LONG candidates: {len(longs_raw)}, SHORT candidates: {len(shorts_raw)}")

print(f"\n  Top 15 by move_score (expected_move × catalyst_strength):")
for i, c in enumerate(candidates[:15]):
    deploy_flag = ' [DEPLOY]' if c.get('in_deploy') else ''
    print(f"    {i+1:>2}. {c.get('ticker','?'):<6} {c.get('direction','?'):<6} "
          f"move={c['expected_move_pct']:>+5.1f}%  str={c.get('catalyst_strength',0):>2}/10  "
          f"score={c['move_score']:>5.1f}  [{c.get('catalyst_type','?'):<10}]{deploy_flag}")
    print(f"        {str(c.get('catalyst',''))[:80]}")


# ==============================================================================
# STAGE 2: DEEP ANALYSIS — top candidates
# ==============================================================================
# Take top candidates by move_score, prioritize non-deploy tickers
deep_pool = sorted(candidates, key=lambda x: (-int(not x.get('in_deploy', False)),
                                                -x['move_score']))
deep_candidates = deep_pool[:20]

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

price_data = {}
fund_data = {}
if deep_candidates:
    print(f"\n  Fetching price data + fundamentals for {len(deep_candidates)} candidates...", end=' ')
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        price_futures = {executor.submit(_fetch_price_and_fundamentals, c['ticker']): c['ticker']
                         for c in deep_candidates}
        for fut in as_completed(price_futures):
            t = price_futures[fut]
            try:
                pd_info, fund = fut.result()
                price_data[t] = pd_info
                fund_data[t] = fund
            except Exception:
                price_data[t] = None
                fund_data[t] = {}
    n_prices = sum(1 for v in price_data.values() if v and 'change_pct' in v)
    n_funds = sum(1 for v in fund_data.values() if v.get('float_shares'))
    print(f"got prices for {n_prices}/{len(deep_candidates)}, {n_funds} with float data")

    # ── GAP SCREENER: fetch 20d context for gap-up tickers ───────────────────
    GAP_MIN_PCT = 3.0
    gap_context = {}
    gap_tickers = [c['ticker'] for c in deep_candidates
                   if price_data.get(c['ticker']) and price_data[c['ticker']].get('change_pct', 0) >= GAP_MIN_PCT]
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
else:
    gap_context = {}

if deep_candidates:
    print(f"\n\n{'='*W}")
    print(f"  Stage 2: Deep analysis — {len(deep_candidates)} candidates")
    print(f"{'='*W}")

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
        print(f"\n  Claude not available — falling back to Gemini")

    t2 = time.time()
    deep_results = {}

    market_context_str = ""
    if market_ctx:
        market_context_str = (
            f"\nMARKET CONTEXT:\n"
            f"- Futures: {market_ctx.get('futures', 'unknown')}\n"
            f"- Theme: {market_ctx.get('overnight_theme', 'unknown')}\n"
            f"- Risk: {market_ctx.get('risk_sentiment', 'unknown')}\n"
            f"- VIX: {market_ctx.get('vix_level', 'unknown')}"
        )

    # ── Phase A: Build all prompts (sequential — progress output) ──────────
    stage2_prompts = {}
    cand_map = {}
    for i, cand in enumerate(deep_candidates):
        ticker = cand.get('ticker', '?')
        cand_map[ticker] = cand
        print(f"\n  [{i+1}/{len(deep_candidates)}] {ticker} "
              f"(move={cand['expected_move_pct']:+.1f}%, "
              f"str={cand.get('catalyst_strength',0)}/10, "
              f"{cand.get('direction','?')})")

        sources_str = ""
        sources = cand.get('_sources', [])
        if sources:
            sources_str = "\n\nSOURCE ARTICLES:\n"
            for j, src in enumerate(sources[:5]):
                sources_str += f"  {j+1}. {src.get('title', '')} — {src.get('url', '')}\n"

        pd_info = price_data.get(ticker)
        price_str = ""
        if pd_info and pd_info.get('change_pct') is not None:
            price_str = (
                f"\nACTUAL PRICE DATA (from Polygon, verified):\n"
                f"- Current/pre-market price: ${pd_info.get('current_price', 'N/A')}\n"
                f"- Previous close: ${pd_info.get('prev_close', 'N/A')}\n"
                f"- Change from prev close: {float(pd_info['change_pct']):+.1f}%\n"
                f"USE THIS ACTUAL % CHANGE as the move_already_priced_pct. "
                f"Do NOT guess the priced-in amount from news articles.\n"
            )

        fd = fund_data.get(ticker, {})
        if fd:
            fund_str = "\nFUNDAMENTAL DATA (from Yahoo Finance):"
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

        all_catalysts = cand.get('_all_catalysts', [])
        all_themes = cand.get('_all_themes', [])
        multi_theme_str = ""
        if len(all_themes) > 1:
            multi_theme_str = f"\n\nMULTIPLE DISCOVERY THEMES ({len(all_themes)}):\n"
            for th, cat in zip(all_themes, all_catalysts):
                multi_theme_str += f"  - [{th}]: {cat[:120]}\n"

        cand_cat = str(cand.get('catalyst_type', '')).lower()
        pattern_str = fetch_similar_patterns(cand_cat, ticker)

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
            f"Analyze {ticker} ({cand.get('company', 'unknown')}) for a potential "
            f"5%+ intraday trade today ({TODAY_STR}).\n\n"
            f"DISCOVERY SCAN RESULTS:\n"
            f"- Catalyst: {cand.get('catalyst', 'N/A')}\n"
            f"- Type: {cand.get('catalyst_type', 'unknown')}\n"
            f"- Direction: {cand.get('direction', '?')}\n"
            f"- Expected move: {cand['expected_move_pct']:+.1f}%\n"
            f"- Catalyst strength: {cand.get('catalyst_strength', 0)}/10\n"
            f"- News time: {cand.get('news_time', 'unknown')}\n"
            f"- Pre-market: {cand.get('premarket_move', 'unknown')}\n"
            f"- Headline: {cand.get('source_headline', 'N/A')}\n"
            f"- Market cap: {cand.get('market_cap_tier', 'unknown')}\n"
            f"{multi_theme_str}"
            f"{price_str}"
            f"{bounce_str}"
            f"{sources_str}"
            f"{market_context_str}"
            f"{pattern_str}\n\n"
            f"Evaluate whether this stock will actually move 5%+ today. "
            f"Be specific about how much is already priced in pre-market "
            f"and how much remaining move is left to capture at open."
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
    for cand in deep_candidates:
        ticker = cand.get('ticker', '?')
        result = raw_results.get(ticker)
        if result is None:
            continue
        pd_info = price_data.get(ticker)

        if 'error' not in result:
            claude_total = float(result.get('total_catalyst_impact_pct', 0))
            polygon_absorbed = float(pd_info.get('change_pct', 0)) if pd_info and 'change_pct' in pd_info else None
            if polygon_absorbed is not None and abs(claude_total) > 0.1:
                computed_remaining = claude_total - polygon_absorbed
                result['move_already_absorbed_pct'] = polygon_absorbed
                result['remaining_move_pct'] = round(computed_remaining, 2)
                result['expected_move_pct'] = round(computed_remaining, 2)
                result['_remaining_source'] = 'python_math'
                if abs(polygon_absorbed / claude_total) > 0.80 or (
                    (claude_total > 0 and polygon_absorbed > claude_total) or
                    (claude_total < 0 and polygon_absorbed < claude_total)
                ):
                    result['move_is_exhausted'] = True

            direction = result.get('direction', '?')
            confidence = result.get('confidence', 0)
            remaining = float(result.get('remaining_move_pct', 0))
            priced_in = float(result.get('move_already_absorbed_pct',
                               result.get('move_already_priced_pct', 0)))
            strength = result.get('catalyst_strength', confidence)
            freshness = result.get('freshness', 5)
            tradability = abs(remaining) * int(strength) * int(freshness) / 100.0
            result['tradability_score'] = round(tradability, 2)
            result['_discovery'] = cand
            catalyst = str(result.get('catalyst', ''))[:70]
            exhausted = result.get('move_is_exhausted', False)
            exh_tag = ' [EXHAUSTED]' if exhausted else ''
            print(f"    {ticker}: {direction} conf={confidence}/10  "
                  f"remaining={remaining:+.1f}%  "
                  f"(absorbed={priced_in:+.1f}%)  "
                  f"tradability={tradability:.1f}{exh_tag}")
        else:
            print(f"    {ticker}: ERROR: {result['error'][:60]}")
            result['tradability_score'] = 0
            result['_discovery'] = cand

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
print(f"  ║  DISCOVERY SCANNER RESULTS — {TODAY_STR} {TIME_STR} PT{'':>17}║")
print(f"  ╚══════════════════════════════════════════════════════════════════╝")
print(f"{'='*W}")

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

# Build final ranked lists
longs = []
shorts = []
for ticker, r in deep_results.items():
    if 'error' in r:
        continue
    disc = r.get('_discovery', {})
    _dir = r.get('direction', 'NEUTRAL')
    _remaining = float(r.get('remaining_move_pct', 0))
    _exhausted = bool(r.get('move_is_exhausted', False))
    _priced = float(r.get('move_already_priced_pct', 0))
    entry = {
        'ticker': ticker,
        'company': disc.get('company', ''),
        'direction': _dir,
        'confidence': _safe_int(r.get('confidence', 0), 0),
        'expected_move_pct': _safe_float(r.get('expected_move_pct', 0)),
        'move_already_priced_pct': _priced,
        'remaining_move_pct': _remaining,
        'move_is_exhausted': _exhausted,
        'catalyst_strength': _safe_int(r.get('catalyst_strength', 0)),
        'freshness': _safe_int(r.get('freshness', 5)),
        'liquidity_risk': _safe_int(r.get('liquidity_risk', 5)),
        'tradability_score': _safe_float(r.get('tradability_score', 0)),
        'reasoning': r.get('reasoning', ''),
        'catalyst': r.get('catalyst', ''),
        'catalyst_type': r.get('catalyst_type', disc.get('catalyst_type', '')),
        'acquirer_or_target': r.get('acquirer_or_target', 'n/a'),
        'deal_materiality': r.get('deal_materiality', 'n/a'),
        'catalyst_is_new': r.get('catalyst_is_new', True),
        'news_timeline': r.get('news_timeline', []),
        'bull_case': r.get('bull_case', ''),
        'bear_case': r.get('bear_case', ''),
        'risks': r.get('risks', ''),
        'comparable_events': r.get('comparable_events', ''),
        'pre_market': r.get('pre_market_action', ''),
        'market_cap_tier': disc.get('market_cap_tier', 'unknown'),
        'sources': disc.get('_sources', []),
        'news_time': disc.get('news_time', '') or _extract_timeline_time(r),
        'theme': disc.get('_theme', ''),
        'in_deploy': disc.get('in_deploy', False),
        'scan_move_score': disc.get('move_score', 0),
        'composite': int(r.get('confidence', 0)) * abs(_remaining),
        '_price_data': price_data.get(ticker),
    }
    if entry['direction'] == 'LONG':
        longs.append(entry)
    elif entry['direction'] == 'SHORT':
        shorts.append(entry)

longs.sort(key=lambda x: x['tradability_score'], reverse=True)
shorts.sort(key=lambda x: x['tradability_score'], reverse=True)

now_pst = NOW_PT


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

    s_clean = re.sub(
        r'\s*(pre-market|premarket|after-hours|after hours|overnight|morning|evening|during market hours)\s*',
        ' ', s, flags=re.IGNORECASE).strip()

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


_CATEGORY_MAP = {
    'earnings': 'EARN', 'analyst': 'ANLST', 'regulatory': 'REG',
    'deal': 'DEAL', 'offering': 'OFFR', 'insider': 'INSD',
    'macro': 'MACRO', 'breaking': 'BREAK', 'premarket_move': 'PMKT',
    'halt': 'HALT', 'squeeze': 'SQZE', 'bankruptcy': 'BNKR',
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
    entries_sorted = sorted(entries, key=lambda x: x['tradability_score'], reverse=True)
    n = len(entries_sorted)
    arrow = '▲' if direction_label == 'LONG' else '▼'
    print(f"\n  {arrow} {direction_label} DISCOVERIES ({n})")
    if not entries_sorted:
        print(f"    No {direction_label} discoveries found.\n")
        return
    print(f"    {'#':>2} {'Ticker':<6} {'Trad':>5} {'Conf':>4} {'Remain':>7} "
          f"{'Priced':>7} {'Type':<5} {'Event Time':<18} {'Age':>8}  {'Dply':>4}  Catalyst Summary")
    print(f"    {'─'*115}")
    for i, e in enumerate(entries_sorted, 1):
        cat = _CATEGORY_MAP.get(str(e.get('catalyst_type', 'none')).lower(), '?')
        ts_str, age = _get_event_time_and_age(e)
        priced = e.get('move_already_priced_pct', 0)
        priced_str = f"{priced:+.1f}%" if priced else '—'
        exh = ' EXH' if e.get('move_is_exhausted') else ''
        gated = ' GATED' if e.get('gate_reason') else ''
        deploy_flag = '  ✓' if e.get('in_deploy') else ''
        catalyst_short = str(e.get('catalyst', '') or e.get('reasoning', ''))[:60]
        print(f"    {i:>2} {e['ticker']:<6} {e['tradability_score']:>5.1f} "
              f"{e['confidence']:>3}/10 {e['remaining_move_pct']:>+6.1f}% "
              f"{priced_str:>7} {cat:<5} {ts_str:<18} {age:>8}{exh}{gated}{deploy_flag:>5}  {catalyst_short}")
    print()


def _print_detail_cards(entries, direction_label):
    if not entries:
        return
    entries_sorted = sorted(entries, key=lambda x: x['tradability_score'], reverse=True)
    arrow = '▲' if direction_label == 'LONG' else '▼'
    print(f"\n  {'='*100}")
    print(f"  {arrow} {direction_label} — DETAILED ANALYSIS")
    print(f"  {'='*100}")
    for i, e in enumerate(entries_sorted, 1):
        ts_str, age = _get_event_time_and_age(e)
        cat = _CATEGORY_MAP.get(str(e.get('catalyst_type', 'none')).lower(), '?')
        priced = e.get('move_already_priced_pct', 0)
        deploy_tag = ' [DEPLOY]' if e.get('in_deploy') else ''
        acq_tag = ''
        if str(e.get('acquirer_or_target', 'n/a')).lower() in ('acquirer', 'target'):
            acq_tag = f"  [{e['acquirer_or_target'].upper()}]"
        print(f"\n  ┌─ {i}. {e['ticker']}  ({e.get('company', '')[:30]}){deploy_tag}  "
              f"{arrow} {direction_label}  "
              f"conf={e['confidence']}/10  remain={e['remaining_move_pct']:+.1f}%  "
              f"priced={priced:+.1f}%  [{cat}]{acq_tag}  trad={e['tradability_score']:.1f}")
        print(f"  │  Event: {ts_str}  ({age})  |  "
              f"strength={e['catalyst_strength']}/10  fresh={e['freshness']}/10  "
              f"liq={e['liquidity_risk']}/10  cap={e.get('market_cap_tier', '?')}")
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
        comp = str(e.get('comparable_events', ''))
        if comp:
            print(f"  │  Compare:  {comp[:95]}")
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
        if e.get('pre_market'):
            print(f"  │  Pre-mkt: {str(e['pre_market'])[:95]}")
        sources = e.get('sources', [])
        if sources:
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

# ── Apply freshness multiplier to tradability scores ──
for e in longs + shorts:
    fm = _freshness_multiplier(e)
    e['tradability_score'] = e['tradability_score'] * fm
    e['composite'] = e['composite'] * fm
    e['freshness_mult'] = fm
longs.sort(key=lambda x: x['tradability_score'], reverse=True)
shorts.sort(key=lambda x: x['tradability_score'], reverse=True)


# ── V3 POST-LLM QUALITY GATES ──────────────────────────────────────────────────
SELF_CONTRADICTION_PHRASES = [
    'untradeable', 'un-tradeable', 'zero remaining edge', 'zero edge',
    'not a new actionable catalyst', 'no new catalyst', 'direction unknown',
    'direction is unknown', 'not yet occurred', 'not yet released',
    'no intraday catalyst', 'no earnings catalyst', 'pure gamma play',
    'virtually untradeable', 'essentially untradeable', 'no remaining upside',
    'zero remaining', 'not yet reported', 'very low-conviction',
    'very low conviction', 'tier 4', 'tier 5', 'tier 4/5',
    'contradicts the', 'immaterial', 'no remaining edge',
]
PRE_EARNINGS_KILL_PHRASES = [
    'expected to report', 'not yet reported', 'after market close',
    'after the close', 'after close today', 'earnings not yet',
    'has not yet reported', 'report expected after', 'reports after',
    'expected after close', 'expected after the close',
]
MIN_CONFIDENCE_FOR_SELECTION = 4
MIN_PRICE_LONG = 3.00
MIN_PRICE_SHORT = 0.25
MIN_PREMARKET_VOLUME = 20000

n_gated = 0
for e in longs + shorts:
    gate_reasons = []
    reasoning_lower = str(e.get('reasoning', '')).lower()
    catalyst_lower = str(e.get('catalyst', '')).lower()
    direction = e.get('direction', '')
    _pd = e.get('_price_data', {})
    current_price = float(_pd.get('current_price', 0)) if _pd else 0

    # Gate 1: Minimum price (direction-dependent)
    if current_price > 0:
        min_p = MIN_PRICE_LONG if direction == 'LONG' else MIN_PRICE_SHORT
        if current_price < min_p:
            gate_reasons.append(f'PENNY_STOCK (${current_price:.2f} < ${min_p:.2f})')

    # Gate 1b: Pre-market volume — illiquid micro-caps cause massive slippage
    _pm_vol = int(_pd.get('premarket_volume', 0)) if _pd else 0
    if current_price > 0 and 0 < _pm_vol < MIN_PREMARKET_VOLUME:
        gate_reasons.append(f'LOW_LIQUIDITY (vol: {_pm_vol:,})')

    # Gate 2: Self-consistency — reasoning contradicts selection
    for phrase in SELF_CONTRADICTION_PHRASES:
        if phrase in reasoning_lower:
            gate_reasons.append(f'SELF_CONTRADICTION')
            break

    # Gate 3: Pre-earnings kill — no catalyst has occurred yet
    for phrase in PRE_EARNINGS_KILL_PHRASES:
        if phrase in catalyst_lower or phrase in reasoning_lower:
            gate_reasons.append(f'PRE_EARNINGS')
            break

    # Gate 4: Minimum confidence
    if e['confidence'] < MIN_CONFIDENCE_FOR_SELECTION:
        gate_reasons.append(f'LOW_CONF ({e["confidence"]})')

    # Gate 5: Momentum flip — exhausted SHORTs on gap-ups flip to LONG in risk-on regime
    _absorbed = float(_pd.get('change_pct', 0)) if _pd else 0.0
    if (direction == 'SHORT' and e.get('move_is_exhausted')
            and _absorbed > 3.0 and not gate_reasons):
        import re as _re5
        _spy_pct = 0.0
        _vix = 20.0
        try:
            _spy_pct = float(market_ctx.get('spy_futures_pct', 0))
        except (TypeError, ValueError):
            _futures_str = str(market_ctx.get('futures', ''))
            _spy_match = _re5.search(r'S&P\s*500?\s*([+-]?\d+\.?\d*)%', _futures_str)
            _spy_pct = float(_spy_match.group(1)) if _spy_match else 0.0
        try:
            _vix = float(market_ctx.get('vix_level_float', 20))
        except (TypeError, ValueError):
            _vix_str = str(market_ctx.get('vix_level', '20'))
            _vix_match = _re5.search(r'(\d+\.?\d*)', _vix_str)
            _vix = float(_vix_match.group(1)) if _vix_match else 20.0
        _is_risk_on = _spy_pct > -0.5 and _vix < 25

        _gc = gap_context.get(e['ticker'], {})
        _drawdown = _gc.get('drawdown_pct', 0)
        _original_conf = e['confidence']

        if _is_risk_on:
            if _drawdown <= -15:
                _new_conf = _original_conf
            elif _drawdown <= -5:
                _new_conf = max(4, _original_conf - 1)
            else:
                _new_conf = min(4, _original_conf)
            e['direction'] = 'LONG'
            e['confidence'] = _new_conf
            e['expected_move_pct'] = abs(e.get('remaining_move_pct', 0))
            e['_flipped'] = True
            gate_reasons.append(
                f'MOMENTUM_FLIP_RISK_ON (gap +{_absorbed:.1f}%, '
                f'drawdown {_drawdown:.0f}%, SPY {_spy_pct:+.1f}%)')
        else:
            gate_reasons.append(
                f'EXHAUSTED_SHORT_KILLED_RISK_OFF (gap +{_absorbed:.1f}%, '
                f'SPY {_spy_pct:+.1f}%, VIX {_vix:.0f})')

    # Gate 6: Clinical trial data trap — biotech "positive data" events get sold by institutions
    CLINICAL_KEYWORDS = [
        'phase 1', 'phase 2', 'phase 3', 'phase i', 'phase ii', 'phase iii',
        'clinical trial', 'clinical data', 'trial results', 'trial data',
        'objective response rate', 'overall response rate', ' orr ',
        'complete response', 'progression-free survival', 'overall survival',
        'primary endpoint', 'secondary endpoint', 'pivotal trial',
    ]
    cat_type = str(e.get('catalyst_type', '')).lower()
    is_clinical = any(kw in catalyst_lower or kw in reasoning_lower for kw in CLINICAL_KEYWORDS)
    if is_clinical and direction == 'LONG' and not gate_reasons:
        gate_reasons.append(f'CLINICAL_DATA_TRAP')

    if gate_reasons:
        e['gate_reason'] = '; '.join(gate_reasons)
        if not e.get('_flipped'):
            e['confidence'] = min(e['confidence'], MIN_CONFIDENCE_FOR_SELECTION - 1)
        n_gated += 1
    else:
        e['gate_reason'] = ''

# Move momentum-flipped entries from shorts to longs
_flipped = [e for e in shorts if e.get('_flipped')]
for e in _flipped:
    shorts.remove(e)
    longs.append(e)

if n_gated:
    print(f"\n  V3 quality gates: {n_gated} candidates gated")
    for e in longs + shorts:
        if e.get('gate_reason'):
            print(f"    {e['ticker']} {e['direction']}: {e['gate_reason']}")

longs.sort(key=lambda x: x.get('tradability_score', 0), reverse=True)
shorts.sort(key=lambda x: x.get('tradability_score', 0), reverse=True)


# ── OUTPUT: Summary tables first, then detailed cards below ──

_print_summary_table(longs, 'LONG')
_print_summary_table(shorts, 'SHORT')

_print_detail_cards(longs[:FINAL_LONG], 'LONG')
_print_detail_cards(shorts[:FINAL_SHORT], 'SHORT')


# ==============================================================================
# SAVE RESULTS
# ==============================================================================
output_data = {
    'scan_date': TODAY_STR,
    'scan_time_pt': TIME_STR,
    'n_themes_searched': len(THEMATIC_QUERIES),
    'n_discovered': len(all_discovered),
    'market_context': market_ctx,
    'theme_results': theme_results,
    'deep_results': {t: {k: v for k, v in r.items() if k not in ('raw', '_sources')}
                     for t, r in deep_results.items()},
    'shortlist_long': longs[:FINAL_LONG],
    'shortlist_short': shorts[:FINAL_SHORT],
}

local_path = f"/tmp/discovery_scan_{TODAY_STR}.json"
with open(local_path, 'w') as f:
    json.dump(output_data, f, indent=2, default=str)

gcs_path = f"{OUTPUT_DIR}/discovery_{TODAY_STR}.json"
try:
    bucket.blob(gcs_path).upload_from_filename(local_path)
    print(f"\n  Saved: gs://{BUCKET_NAME}/{gcs_path}")
except Exception as e:
    print(f"\n  GCS save failed: {e}")
print(f"  Local: {local_path}")


# ==============================================================================
# LOG TO BIGQUERY
# ==============================================================================
def log_discovery_to_bq(longs, shorts, all_discovered, market_ctx):
    bq_client = bigquery.Client()

    bq_client.query(f"""
    CREATE TABLE IF NOT EXISTS `{BQ_TABLE}` (
        scan_timestamp          TIMESTAMP,
        scan_date               DATE,
        scan_time_pt            STRING,
        ticker                  STRING,
        company                 STRING,
        direction               STRING,
        rank_in_direction       INT64,
        selected                BOOL,
        in_deploy_universe      BOOL,
        confidence              INT64,
        expected_move_pct       FLOAT64,
        move_already_priced_pct FLOAT64,
        remaining_move_pct      FLOAT64,
        catalyst_strength       INT64,
        freshness               INT64,
        liquidity_risk          INT64,
        tradability_score       FLOAT64,
        scan_move_score         FLOAT64,
        catalyst_type           STRING,
        catalyst                STRING,
        reasoning               STRING,
        comparable_events       STRING,
        headline                STRING,
        news_time               STRING,
        theme                   STRING,
        market_cap_tier         STRING,
        bull_case               STRING,
        bear_case               STRING,
        risks                   STRING,
        pre_market_action       STRING,
        stage2_model            STRING,
        market_futures          STRING,
        market_risk             STRING,
        vix_level               STRING
    )""").result()

    for col_name, col_type in [
        ('move_is_exhausted', 'BOOL'),
        ('acquirer_or_target', 'STRING'),
        ('deal_materiality', 'STRING'),
        ('catalyst_is_new', 'BOOL'),
        ('news_timeline_json', 'STRING'),
        ('freshness_multiplier', 'FLOAT64'),
        ('scanner_version', 'STRING'),
        ('gate_reason', 'STRING'),
    ]:
        try:
            bq_client.query(f"ALTER TABLE `{BQ_TABLE}` ADD COLUMN {col_name} {col_type}").result()
        except Exception:
            pass

    rows = []
    now_utc = datetime.now(pytz.utc).isoformat()
    futures = str(market_ctx.get('futures', ''))[:500] if market_ctx else ''
    risk = str(market_ctx.get('risk_sentiment', ''))[:200] if market_ctx else ''
    vix = str(market_ctx.get('vix_level', ''))[:50] if market_ctx else ''

    for direction_label, entries in [('LONG', longs), ('SHORT', shorts)]:
        for rank, e in enumerate(entries, 1):
            rows.append({
                'scan_timestamp': now_utc,
                'scan_date': TODAY_STR,
                'scan_time_pt': TIME_STR,
                'ticker': e['ticker'],
                'company': str(e.get('company', ''))[:200],
                'direction': direction_label,
                'rank_in_direction': rank,
                'selected': rank <= (FINAL_LONG if direction_label == 'LONG' else FINAL_SHORT) and e['confidence'] >= MIN_CONFIDENCE_FOR_SELECTION and not e.get('gate_reason'),
                'in_deploy_universe': e.get('in_deploy', False),
                'confidence': e['confidence'],
                'expected_move_pct': e['expected_move_pct'],
                'move_already_priced_pct': e.get('move_already_priced_pct', 0),
                'remaining_move_pct': e.get('remaining_move_pct', 0),
                'catalyst_strength': e.get('catalyst_strength', 0),
                'freshness': e.get('freshness', 0),
                'liquidity_risk': e.get('liquidity_risk', 0),
                'tradability_score': e.get('tradability_score', 0),
                'scan_move_score': e.get('scan_move_score', 0),
                'catalyst_type': str(e.get('catalyst_type', ''))[:100],
                'catalyst': str(e.get('catalyst', ''))[:500],
                'reasoning': str(e.get('reasoning', ''))[:2000],
                'comparable_events': str(e.get('comparable_events', ''))[:500],
                'headline': '',
                'news_time': str(e.get('news_time', ''))[:100],
                'theme': str(e.get('theme', ''))[:50],
                'market_cap_tier': str(e.get('market_cap_tier', ''))[:20],
                'bull_case': str(e.get('bull_case', ''))[:500],
                'bear_case': str(e.get('bear_case', ''))[:500],
                'risks': str(e.get('risks', ''))[:500],
                'pre_market_action': str(e.get('pre_market', ''))[:500],
                'stage2_model': 'gemini-2.5-flash',
                'market_futures': futures,
                'market_risk': risk,
                'vix_level': vix,
                'move_is_exhausted': e.get('move_is_exhausted', False),
                'acquirer_or_target': str(e.get('acquirer_or_target', 'n/a'))[:20],
                'deal_materiality': str(e.get('deal_materiality', 'n/a'))[:20],
                'catalyst_is_new': e.get('catalyst_is_new', True),
                'news_timeline_json': json.dumps(e.get('news_timeline', []), default=str)[:2000],
                'freshness_multiplier': e.get('freshness_mult', 1.0),
                'scanner_version': SCANNER_VERSION,
                'gate_reason': str(e.get('gate_reason', ''))[:500],
            })

    # Also log scan-only discoveries (not deep-analyzed)
    # Direction set to NEUTRAL — Stage 1 direction is a rough classification, not a signal
    deep_tickers = set(e['ticker'] for e in longs + shorts)
    for ticker, d in all_discovered.items():
        if ticker in deep_tickers:
            continue
        rows.append({
            'scan_timestamp': now_utc,
            'scan_date': TODAY_STR,
            'scan_time_pt': TIME_STR,
            'ticker': ticker,
            'company': str(d.get('company', ''))[:200],
            'direction': 'NEUTRAL',
            'rank_in_direction': 0,
            'selected': False,
            'in_deploy_universe': d.get('in_deploy', False),
            'confidence': 0,
            'expected_move_pct': float(d.get('expected_move_pct', 0)),
            'move_already_priced_pct': 0.0,
            'remaining_move_pct': 0.0,
            'catalyst_strength': _safe_int(d.get('catalyst_strength', 0), 0),
            'freshness': 0,
            'liquidity_risk': 0,
            'tradability_score': 0.0,
            'scan_move_score': float(d.get('move_score', 0)),
            'catalyst_type': str(d.get('catalyst_type', ''))[:100],
            'catalyst': str(d.get('catalyst', ''))[:500],
            'reasoning': '',
            'comparable_events': '',
            'headline': str(d.get('source_headline', ''))[:500],
            'news_time': str(d.get('news_time', ''))[:100],
            'theme': str(d.get('_theme', ''))[:50],
            'market_cap_tier': str(d.get('market_cap_tier', ''))[:20],
            'bull_case': '',
            'bear_case': '',
            'risks': '',
            'pre_market_action': str(d.get('premarket_move', ''))[:200],
            'stage2_model': '',
            'market_futures': futures,
            'market_risk': risk,
            'vix_level': vix,
            'scanner_version': SCANNER_VERSION,
        })

    if rows:
        try:
            errors = bq_client.insert_rows_json(BQ_TABLE, rows)
            if errors:
                print(f"\n  BQ insert errors: {errors[:3]}")
            else:
                deep_n = len([r for r in rows if r['confidence'] > 0])
                scan_n = len(rows) - deep_n
                print(f"  BQ logged: {deep_n} deep + {scan_n} scan-only = {len(rows)} rows -> {BQ_TABLE}")
        except Exception as e:
            print(f"  BQ logging failed: {e}")


log_discovery_to_bq(longs, shorts, all_discovered, market_ctx)



# ==============================================================================
# CLEANUP
# ==============================================================================
total_elapsed = time.time() - t_total
gc.collect()

print(f"\n{'='*W}")
print(f"  DISCOVERY SCAN COMPLETE")
print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
print(f"  Discovered: {len(all_discovered)} tickers across {len(THEMATIC_QUERIES)} themes")
print(f"  Results: {len(longs)} LONG + {len(shorts)} SHORT")
print(f"{'='*W}")
