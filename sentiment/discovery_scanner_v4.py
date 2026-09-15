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

from scanner_dedup import (
    ensure_table as _dedup_ensure_table,
    load_state as _dedup_load_state,
    check_ticker as _dedup_check,
    record_selection as _dedup_record,
    update_last_run as _dedup_update_last_run,
    get_state_summary as _dedup_summary,
)

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
YESTERDAY_STR = (NOW_PT - timedelta(days=1)).strftime('%Y-%m-%d')
TIME_STR = NOW_PT.strftime('%H:%M')

MIN_MOVE_PCT       = 5.0
MAX_NEWS_AGE_HOURS = 16
SCANNER_VERSION    = 'V4.1'
STRONG_CATALYST_SCORE = 7
STRONG_CATALYST_MOVE  = 8.0
FINAL_LONG         = 3
FINAL_SHORT        = 3
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
        'query': (f'stocks after hours movers last night {YESTERDAY_STR} '
                  f'earnings reaction premarket {TODAY_STR} surging dropping'),
        'direction_bias': None,
    },
    {
        'id': 'earnings_today',
        'query': (f'earnings results today {TODAY_STR} '
                  f'EPS revenue beat miss guidance stock reaction'),
        'direction_bias': None,
    },
    {
        'id': 'earnings_surprise',
        'query': (f'earnings surprise today {TODAY_STR} stock surging plunging '
                  f'quarterly results beat miss small cap mid cap premarket'),
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
Your job is to predict what happens AFTER the trader enters at market open. You must separate:
  - total_catalyst_impact_pct: the full magnitude of the catalyst (SIGNED: positive for bullish, negative for bearish)
  - move_already_priced_pct: how much has already moved in pre-market/after-hours
  - remaining_move_pct: what's LEFT for the trader at entry (this is what matters)

IMPORTANT: total_catalyst_impact_pct must match the CATALYST direction, not your trade thesis.
If earnings guidance was cut and the stock dropped, total_catalyst_impact_pct should be NEGATIVE
(e.g., -25%), reflecting the bearish catalyst. Do NOT set it positive for a bounce prediction.

For EARNINGS catalysts: the after-hours/pre-market drop is Wave 1. Wave 2 (continued selling at
open) is the norm, not the exception — institutions and retail sell at open after reading the
full report. Do NOT assume a bounce just because the stock already dropped a lot.

Analyze whether this stock will actually deliver a 5%+ move today based on the catalyst described. Be realistic and quantitative.

Return ONLY a JSON object:
{
  "ticker": "<TICKER>",
  "direction": "LONG" or "SHORT" or "NEUTRAL" <-- direction of the REMAINING move at entry. Use NEUTRAL when there is no clear tradable edge — this should be the MOST COMMON answer,
  "confidence": <int 1-10 (for logging -- selection uses residual-based EV)>,
  "expected_move_low_pct": <float, conservative end of expected move range (%)>,
  "expected_move_high_pct": <float, aggressive end of expected move range (%)>,
  "catalyst_direction": "LONG" or "SHORT" <-- direction implied by the catalyst itself, independent of price action,
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

EXPECTED MOVE INTERVAL:
  Instead of a single expected_move_pct, provide a LOW and HIGH estimate:
  - expected_move_low_pct: conservative end (what happens if market absorbs quickly)
  - expected_move_high_pct: aggressive end (what happens with full continuation/momentum)
  The interval width reflects your uncertainty. A Goldman Buy initiation on a $50B company
  might be [+1%, +4%] -- narrow, one-sided. Mixed earnings with EPS beat / revenue miss
  might be [-3%, +2%] -- wide, crosses zero.

  Use the CATALYST IMPACT REFERENCE TABLE to calibrate these ranges.

CATALYST DIRECTION:
  This is the direction the catalyst itself implies, REGARDLESS of current price action.
  - Buy initiation -> catalyst_direction = "LONG"
  - Offering -> catalyst_direction = "SHORT"
  - Earnings beat + guidance raise -> catalyst_direction = "LONG"
  - Earnings miss + guidance cut -> catalyst_direction = "SHORT"
  This may differ from your "direction" field (which considers price action).

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
5. NEUTRAL IS THE DEFAULT: Most stocks do NOT have a clear tradable edge. If you are unsure
   about the direction, or the catalyst is unclear/weak/already priced in, return NEUTRAL.
   NEUTRAL should be your most common answer (expect 30-60% of tickers to be NEUTRAL).
   Specifically use NEUTRAL when:
   - The catalyst is unknown, unclear, or unverifiable
   - move_is_exhausted is True and you have no specific evidence for continuation or reversal
   - Confidence would be 3 or below
   - The catalyst type is editorial, procedural, or promotional without hard numbers
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
                avg_daily_volume = 0
                daily_return_std = 0.0
                try:
                    _from_20d = (NOW_PT - timedelta(days=25)).strftime('%Y-%m-%d')
                    daily_bars = list(_polygon_client.get_aggs(
                        ticker=ticker, multiplier=1, timespan="day",
                        from_=_from_20d, to=YESTERDAY_STR, limit=10,
                    ))
                    if daily_bars:
                        avg_daily_volume = int(sum(
                            int(getattr(b, 'volume', 0) or 0) for b in daily_bars
                        ) / len(daily_bars))
                        if len(daily_bars) >= 3:
                            closes = [float(b.close) for b in daily_bars if float(b.close) > 0]
                            if len(closes) >= 3:
                                rets = [(closes[i] - closes[i-1]) / closes[i-1] * 100
                                        for i in range(1, len(closes))]
                                daily_return_std = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0
                except Exception:
                    pass
                return {
                    'current_price': round(current_price, 2),
                    'prev_close': round(prev_close, 2),
                    'change_pct': round(change_pct, 2),
                    'premarket_price': round(current_price, 2),
                    'premarket_volume': premarket_volume,
                    'avg_daily_volume': avg_daily_volume,
                    'daily_return_std': round(daily_return_std, 4),
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


# ==============================================================================
# PATTERN LIBRARY — BQ-backed few-shot retrieval (batch prefetch)
_PATTERN_CACHE_DF = None

CAT_MAP = {
    'earnings': 'earnings', 'analyst': 'analyst', 'regulatory': 'regulatory',
    'deal': 'deal', 'offering': 'offering', 'insider': 'deal',
    'breaking': 'breaking', 'premarket_move': 'deal', 'halt': 'deal',
    'squeeze': 'deal', 'bankruptcy': 'deal', 'macro': 'breaking',
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


def fetch_comparable_patterns(catalyst_type, n=5):
    try:
        q = f"""
        SELECT primary_ticker, catalyst_type, primary_direction, primary_gap_pct,
               primary_ret_60m, primary_mfe_pct, primary_eod_pct, primary_outcome,
               lesson, pattern_rule
        FROM `ys-time-series.scoring_logs.scanner_patterns`
        WHERE catalyst_type = '{catalyst_type}'
        ORDER BY event_date DESC
        LIMIT {n}
        """
        rows = list(bigquery.Client().query(q).result())
        if not rows:
            return ""
        lines = [f"\nHISTORICAL PATTERN EXAMPLES ({len(rows)} comparable {catalyst_type} trades):"]
        for r in rows:
            lines.append(f"  {r['primary_ticker']} {r['primary_direction']}: "
                        f"gap={r['primary_gap_pct']:+.1f}%, 60m={r['primary_ret_60m']:+.1f}%, "
                        f"MFE={r['primary_mfe_pct']:+.1f}%, EOD={r['primary_eod_pct']:+.1f}% "
                        f"→ {r['primary_outcome']}")
            if r.get('lesson'):
                lines.append(f"    Lesson: {r['lesson']}")
        return "\n".join(lines)
    except Exception:
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

    # Step 1: Get CIK from Polygon (already cached in ticker reference)
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

    # Step 2: Hit SEC EDGAR Submissions API
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

    # Step 3: Find 8-K filed on target date (earnings press releases are 8-K)
    for form, fdate, acc_dt in zip(forms, dates, acceptance):
        if form == '8-K' and fdate == ref_date and acc_dt:
            # acc_dt format: '2026-09-09T16:34:31.000Z' (UTC)
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

    # Step 1: For earnings/deals/offerings/regulatory, look up the actual
    # SEC 8-K filing time. This is authoritative — structured data from EDGAR,
    # not an LLM interpreting article publication dates.
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
            # No 8-K today — check yesterday (after-hours earnings from prior day)
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

    # Step 2: Always fetch Polygon absorption data.
    event_info = get_actual_event_time(ticker, TODAY_STR)
    if not event_info:
        if cat_type not in sec_types:
            print("no spike found")
        return

    event_dt = event_info['event_time_dt']
    event_age_hours = (NOW_ET - event_dt).total_seconds() / 3600

    # If SEC didn't have it and the LLM timestamp looks wrong, use Polygon
    # spike time as a last resort (approximate but better than hallucinated).
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
            "comparable_events": {"type": "string"},
            "related_tickers": {"type": "array", "items": {"type": "string"}},
            "liquidity_risk": {"type": "integer"},
            "market_cap_tier": {"type": "string"},
            "total_catalyst_impact_pct": {"type": "number", "description": "Total expected % impact of the catalyst"},
            "move_already_absorbed_pct": {"type": "number", "description": "How much % has already moved pre-market/AH"},
            "remaining_move_pct": {"type": "number", "description": "Expected remaining % move after entry"},
            "expected_move_pct": {"type": "number", "description": "Same as remaining_move_pct"},
            "move_is_exhausted": {"type": "boolean", "description": "True if catalyst move is fully priced in"},
            "direction": {"type": "string", "enum": ["LONG", "SHORT", "NEUTRAL"]},
            "confidence": {"type": "integer", "description": "1-10 confidence in direction of remaining move"},
            "expected_move_low_pct": {"type": "number", "description": "Conservative end of expected move range (%)"},
            "expected_move_high_pct": {"type": "number", "description": "Aggressive end of expected move range (%)"},
            "catalyst_direction": {"type": "string", "enum": ["LONG", "SHORT"], "description": "Direction implied by the catalyst itself, independent of price action"},
        },
        "required": [
            "ticker", "catalyst", "catalyst_type", "catalyst_is_new",
            "news_timeline", "timing", "reasoning",
            "bull_case", "bear_case", "risks",
            "total_catalyst_impact_pct", "move_already_absorbed_pct",
            "remaining_move_pct", "expected_move_pct", "move_is_exhausted",
            "direction", "confidence",
            "expected_move_low_pct", "expected_move_high_pct", "catalyst_direction",
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

# --- Validate tickers via Polygon (reject non-US, invalid, company names) ---
print(f"\n  Validating {len(candidates)} discovered tickers via Polygon...")
_validated_candidates = []
_filtered_tickers = []
for _c in candidates:
    _tk = _c.get('ticker', '')
    # Instant reject: dots in ticker name (non-US exchange, e.g. FUM.HE)
    if '.' in _tk:
        _filtered_tickers.append((_tk, 'non-US exchange (dot in symbol)'))
        continue
    _ok, _ = validate_ticker_polygon(_tk)
    if _ok:
        _validated_candidates.append(_c)
    else:
        _filtered_tickers.append((_tk, 'not found on US exchange'))
if _filtered_tickers:
    print(f"  Filtered {len(_filtered_tickers)} invalid tickers:")
    for _ft, _reason in _filtered_tickers:
        print(f"    - {_ft}: {_reason}")
else:
    print(f"  All {len(candidates)} tickers validated OK")
candidates = _validated_candidates

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
# Sort purely by move_score — deploy tickers compete fairly.
deep_pool = sorted(candidates, key=lambda x: -x['move_score'])
deep_candidates = deep_pool[:20]

# Strong catalyst override: force-include tickers outside top-20 that have
# catalyst_strength >= 7 OR expected_move >= 8%. Prevents high-conviction
# deploy tickers from being dropped by the top-N cutoff.
_deep_tickers = set(c.get('ticker') for c in deep_candidates)
_strong_overrides = []
for c in deep_pool[20:]:
    _tk = c.get('ticker', '')
    if _tk in _deep_tickers:
        continue
    _cs = int(c.get('catalyst_strength', 0))
    _em = abs(float(c.get('expected_move_pct', 0)))
    if _cs >= STRONG_CATALYST_SCORE or _em >= STRONG_CATALYST_MOVE:
        deep_candidates.append(c)
        _deep_tickers.add(_tk)
        _strong_overrides.append((_tk, _cs, _em))
if _strong_overrides:
    print(f"\n  Strong catalyst override: +{len(_strong_overrides)} tickers forced into Stage 2:")
    for _t, _cs, _em in _strong_overrides:
        print(f"    {_t}: catalyst_strength={_cs}/10, expected_move={_em:.1f}%")

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
        print(f"\n  Claude not available: {e}")
        print(f"  Falling back to Gemini for Stage 2")

    print(f"  Stage 2 primary model: {CLAUDE_MODEL if claude_ok else GEMINI_MODEL + ' (Claude unavailable)'}")

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
        comparable_str = fetch_comparable_patterns(cand_cat)

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
            f"{pattern_str}"
            f"{comparable_str}\n\n"
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
        _meta = {'claude_attempted': False, 'claude_error': '', 'model_used': ''}
        if claude_ok:
            _meta['claude_attempted'] = True
            _result = claude_analyze(_prompt, DEEP_SYSTEM, claude_client)
            if 'error' not in _result:
                _meta['model_used'] = CLAUDE_MODEL
                _result['_s2_meta'] = _meta
                return _ticker, _result
            _meta['claude_error'] = str(_result['error'])[:200]
            print(f"    ⟳ {_ticker} Claude failed ({_result['error'][:60]}), trying Gemini...")
            _result = gemini_search(_prompt, DEEP_SYSTEM)
            _meta['model_used'] = GEMINI_MODEL if 'error' not in _result else ''
            _result['_s2_meta'] = _meta
            return _ticker, _result
        else:
            _result = gemini_search(_prompt, DEEP_SYSTEM)
            _meta['model_used'] = GEMINI_MODEL if 'error' not in _result else ''
            _result['_s2_meta'] = _meta
            return _ticker, _result

    raw_results = {}
    _s2_errors = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_DEEP) as executor:
        deep_futures = {executor.submit(_deep_analyze_one, (t, p)): t
                        for t, p in stage2_prompts.items()}
        for fut in as_completed(deep_futures):
            _tk = deep_futures[fut]
            try:
                _, _res = fut.result()
                raw_results[_tk] = _res
                if 'error' in _res:
                    _s2_errors.append((_tk, _res['error'][:100]))
                    print(f"    ✗ {_tk} error: {_res['error'][:80]}")
                else:
                    print(f"    ✓ {_tk} done")
            except Exception as e:
                raw_results[_tk] = {'error': str(e)[:200]}
                _s2_errors.append((_tk, str(e)[:100]))
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

    _s2_ok = sum(1 for r in deep_results.values() if 'error' not in r)
    _s2_err = sum(1 for r in deep_results.values() if 'error' in r)
    _s2_missing = len(deep_candidates) - len(deep_results)
    _n_claude = sum(1 for r in deep_results.values()
                    if r.get('_s2_meta', {}).get('model_used') == CLAUDE_MODEL)
    _n_gemini_fb = sum(1 for r in deep_results.values()
                       if r.get('_s2_meta', {}).get('claude_error') and 'error' not in r)
    _n_gemini_direct = sum(1 for r in deep_results.values()
                           if not r.get('_s2_meta', {}).get('claude_attempted') and 'error' not in r)
    elapsed_s2 = time.time() - t2
    print(f"\n  Stage 2 complete: {elapsed_s2:.0f}s")
    print(f"    {_s2_ok}/{len(deep_candidates)} succeeded, "
          f"{_s2_err} errors, {_s2_missing} missing")
    print(f"    Models: {_n_claude} Claude, {_n_gemini_fb} Gemini-fallback, "
          f"{_n_gemini_direct} Gemini-direct, {_s2_err} failed-both")
    if _s2_errors:
        print(f"    Error details:")
        for _et, _ee in _s2_errors:
            print(f"      {_et}: {_ee}")
    _claude_errors = [(t, r.get('_s2_meta', {}).get('claude_error', ''))
                      for t, r in deep_results.items()
                      if r.get('_s2_meta', {}).get('claude_error')]
    if _claude_errors:
        print(f"    Claude errors (recovered via Gemini):")
        for _ct, _ce in _claude_errors:
            print(f"      {_ct}: {_ce[:80]}")
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
stage2_neutrals = []
for ticker, r in deep_results.items():
    if 'error' in r:
        continue
    disc = r.get('_discovery', {})
    _dir = r.get('direction', 'NEUTRAL')
    _remaining = float(r.get('remaining_move_pct', 0))
    _exhausted = bool(r.get('move_is_exhausted', False))
    _priced = float(r.get('move_already_absorbed_pct',
                         r.get('move_already_priced_pct', 0)))
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
        'expected_move_low_pct': _safe_float(r.get('expected_move_low_pct', 0)),
        'expected_move_high_pct': _safe_float(r.get('expected_move_high_pct', 0)),
        'catalyst_direction': str(r.get('catalyst_direction', '')),
    }
    if entry['direction'] == 'LONG':
        longs.append(entry)
    elif entry['direction'] == 'SHORT':
        shorts.append(entry)
    else:
        stage2_neutrals.append(entry)

if stage2_neutrals:
    print(f"\n  Stage 2 NEUTRAL (no tradable edge): {len(stage2_neutrals)}")
    for e in stage2_neutrals:
        print(f"    {e['ticker']}: {e.get('catalyst_type', '?')} — {e.get('reasoning', '')[:80]}")

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


MIN_EV_FOR_SELECTION = 0.5

def _is_selected(entry, rank, direction_label):
    top_n = FINAL_LONG if direction_label == 'LONG' else FINAL_SHORT
    return (rank <= top_n
            and entry.get('_ev', 0) >= MIN_EV_FOR_SELECTION
            and not entry.get('_hard_block'))


def _print_summary_row(i, e):
    cat = _CATEGORY_MAP.get(str(e.get('catalyst_type', 'none')).lower(), '?')
    ts_str, age = _get_event_time_and_age(e)
    ev = e.get('_ev', 0)
    res_dir = e.get('_residual_direction', '?')
    clarity = e.get('_residual_clarity', '?')
    deploy_flag = '  +' if e.get('in_deploy') else ''
    block = f"  [{e['_hard_block'][:30]}]" if e.get('_hard_block') else ''
    penalty = f" P:{e['_ev_penalty'][:20]}" if e.get('_ev_penalty') else ''
    catalyst_short = str(e.get('catalyst', '') or e.get('reasoning', ''))[:50]
    interval = e.get('_move_interval', '')
    gap = e.get('_gap_at_entry', 0)
    print(f"    {i:>2} {e['ticker']:<6} {ev:>5.1f}ev "
          f"{res_dir:<5} {clarity:<9} gap={gap:>+5.1f}% {interval:>16} "
          f"{cat:<5} {ts_str:<18} {age:>8}{deploy_flag:>3}  {catalyst_short}{block}{penalty}")


def _print_summary_table(entries, direction_label):
    entries_sorted = sorted(entries, key=lambda x: x.get('_ev', 0), reverse=True)
    n = len(entries_sorted)
    arrow = '▲' if direction_label == 'LONG' else '▼'
    print(f"\n  {arrow} {direction_label} DISCOVERIES ({n})")
    if not entries_sorted:
        print(f"    No {direction_label} discoveries found.\n")
        return

    for e_idx, e in enumerate(entries_sorted, 1):
        e['_is_selected'] = _is_selected(e, e_idx, direction_label)

    selected = [e for e in entries_sorted if e['_is_selected']]
    not_selected = [e for e in entries_sorted if not e['_is_selected']]

    header = (f"    {'#':>2} {'Ticker':<6} {'EV':>5}    "
              f"{'Dir':<5} {'Clarity':<9} {'Gap':>6}  {'Interval':>16} "
              f"{'Type':<5} {'Event Time':<18} {'Age':>8}     Catalyst Summary")

    if selected:
        print(f"\n    === SELECTED ({len(selected)}) ===")
        print(header)
        print(f"    {'─'*120}")
        for i, e in enumerate(selected, 1):
            _print_summary_row(i, e)

    if not_selected:
        print(f"\n    --- NOT SELECTED ({len(not_selected)}) ---")
        print(header)
        print(f"    {'─'*120}")
        for i, e in enumerate(not_selected, 1):
            _print_summary_row(i, e)

    if not selected:
        print(f"\n    === NO SIGNALS SELECTED ===")

    print()


def _print_detail_cards(entries, direction_label):
    if not entries:
        return
    entries_sorted = sorted(entries, key=lambda x: x.get('_ev', 0), reverse=True)
    arrow = '▲' if direction_label == 'LONG' else '▼'
    print(f"\n  {'='*100}")
    print(f"  {arrow} {direction_label} -- DETAILED ANALYSIS")
    print(f"  {'='*100}")
    for i, e in enumerate(entries_sorted, 1):
        ts_str, age = _get_event_time_and_age(e)
        cat = _CATEGORY_MAP.get(str(e.get('catalyst_type', 'none')).lower(), '?')
        deploy_tag = ' [DEPLOY]' if e.get('in_deploy') else ''
        sel_tag = ' SELECTED' if e.get('_is_selected') else ' NOT SELECTED'
        acq_tag = ''
        if str(e.get('acquirer_or_target', 'n/a')).lower() in ('acquirer', 'target'):
            acq_tag = f"  [{e['acquirer_or_target'].upper()}]"
        res_dir = e.get('_residual_direction', '?')
        clarity = e.get('_residual_clarity', '?')
        ev = e.get('_ev', 0)
        interval = e.get('_move_interval', '')
        gap = e.get('_gap_at_entry', 0)
        informed = e.get('_informed_score', 0)
        tape_disc = e.get('_tape_discount', 1.0)
        print(f"\n  +- {i}. {e['ticker']}  ({e.get('company', '')[:30]}){deploy_tag}{sel_tag}  "
              f"{arrow} {res_dir} ({clarity})  "
              f"EV={ev:.2f}  gap={gap:+.1f}%  interval={interval}  "
              f"[{cat}]{acq_tag}")
        print(f"  |  Event: {ts_str}  ({age})  |  "
              f"informed={informed:.2f}  tape_disc={tape_disc:.2f}  "
              f"liq={e['liquidity_risk']}/10  cap={e.get('market_cap_tier', '?')}  "
              f"(LLM: conf={e['confidence']}/10 dir={e.get('direction','?')})")
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

# ── Apply catalyst-type weight to tradability scores (Bayesian-shrunk) ──
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
    e['tradability_score'] = e['tradability_score'] * cw
    e['composite'] = e['composite'] * cw
    e['catalyst_weight'] = cw
longs.sort(key=lambda x: x['tradability_score'], reverse=True)
shorts.sort(key=lambda x: x['tradability_score'], reverse=True)

# ── Mismatch Index: price move vs LLM catalyst assessment ──
# Ratio of actual gap to LLM-assigned catalyst_strength. High mismatch = retail FOMO
# (big price move on weak catalyst). Boosts SHORT on high mismatch, LONG on low mismatch.
for e in longs + shorts:
    _pd = e.get('_price_data') or {}
    gap_pct = abs(float(_pd.get('change_pct', 0)))
    _strength = max(int(e.get('catalyst_strength', e.get('confidence', 5))), 1)
    mismatch = gap_pct / _strength
    if mismatch > 2.0 and e['direction'] == 'LONG':
        e['mismatch_mult'] = 0.5
    elif mismatch > 2.0 and e['direction'] == 'SHORT':
        e['mismatch_mult'] = 1.5
    elif mismatch < 0.5 and e['direction'] == 'LONG':
        e['mismatch_mult'] = 1.3
    else:
        e['mismatch_mult'] = 1.0
    e['tradability_score'] = e['tradability_score'] * e['mismatch_mult']
    e['composite'] = e['composite'] * e['mismatch_mult']
    e['mismatch_index'] = round(mismatch, 2)
longs.sort(key=lambda x: x['tradability_score'], reverse=True)
shorts.sort(key=lambda x: x['tradability_score'], reverse=True)

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
    e['tradability_score'] = e['tradability_score'] * vol_mult
    e['composite'] = e['composite'] * vol_mult
    e['volume_ratio'] = round(vol_ratio, 2)
    e['volume_mult'] = vol_mult
longs.sort(key=lambda x: x['tradability_score'], reverse=True)
shorts.sort(key=lambda x: x['tradability_score'], reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# V4: RESIDUAL-BASED SIGNAL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

# Cross-run, cross-scanner dedup — shared state table
_dedup_ensure_table()
_dedup_load_state(TODAY_STR)

# Hard blockers (data validity only -- everything else feeds EV)
MIN_PRICE_FLOOR = 3.00
STALE_ORIGIN_HOURS = 36
MIN_PREMARKET_VOLUME = 20000

for e in longs + shorts:
    ticker = e['ticker']
    _pd = e.get('_price_data', {})

    # --- Hard blockers ---
    e['_hard_block'] = None
    current_price = float(_pd.get('current_price', 0)) if _pd else 0
    if current_price > 0 and current_price < MIN_PRICE_FLOOR:
        e['_hard_block'] = f'PRICE_FLOOR (${current_price:.2f})'
        continue

    _pm_vol = int(_pd.get('premarket_volume', 0)) if _pd else 0
    if current_price > 0 and 0 < _pm_vol < MIN_PREMARKET_VOLUME:
        e['_hard_block'] = f'LOW_LIQUIDITY (vol: {_pm_vol:,})'
        continue

    # Check stale origin from news_timeline
    _timeline = e.get('news_timeline', [])
    if isinstance(_timeline, list) and _timeline:
        _earliest = None
        for _ev in _timeline:
            if isinstance(_ev, dict):
                _evt = str(_ev.get('time', ''))
                if _evt and (not _earliest or _evt < _earliest):
                    _earliest = _evt
        if _earliest:
            _orig_dt = _parse_news_time(_earliest)
            if _orig_dt:
                _orig_age = (NOW_ET - _orig_dt.astimezone(_ET)).total_seconds() / 3600
                if _orig_age > STALE_ORIGIN_HOURS:
                    e['_hard_block'] = f'STALE_ORIGIN ({_orig_age:.0f}h)'
                    continue

    # --- Compute residual ---
    gap_at_entry = float(_pd.get('change_pct', 0)) if _pd else 0

    # Get move interval from LLM (fall back to point estimate if interval not provided)
    move_low = float(e.get('expected_move_low_pct', e.get('total_catalyst_impact_pct', e.get('expected_move_pct', 0))))
    move_high = float(e.get('expected_move_high_pct', e.get('total_catalyst_impact_pct', e.get('expected_move_pct', 0))))
    if move_low == move_high:
        _mag = abs(move_low) * 0.3
        move_low = move_low - _mag if move_low > 0 else move_low + _mag
        move_high = move_high + _mag if move_high > 0 else move_high - _mag

    if move_low > move_high:
        move_low, move_high = move_high, move_low

    residual_low = move_low - gap_at_entry
    residual_high = move_high - gap_at_entry
    residual_mid = (residual_low + residual_high) / 2

    e['_residual_low'] = round(residual_low, 2)
    e['_residual_high'] = round(residual_high, 2)
    e['_residual_mid'] = round(residual_mid, 2)
    e['_gap_at_entry'] = round(gap_at_entry, 2)
    e['_move_interval'] = f'[{move_low:+.1f}%, {move_high:+.1f}%]'

    # --- Residual direction ---
    if residual_low > 0.5:
        e['_residual_direction'] = 'LONG'
        e['_residual_clarity'] = 'clear'
    elif residual_high < -0.5:
        e['_residual_direction'] = 'SHORT'
        e['_residual_clarity'] = 'clear'
    elif residual_low > -0.5 and residual_high > 0.5:
        e['_residual_direction'] = 'LONG'
        e['_residual_clarity'] = 'lean'
    elif residual_high < 0.5 and residual_low < -0.5:
        e['_residual_direction'] = 'SHORT'
        e['_residual_clarity'] = 'lean'
    else:
        e['_residual_direction'] = 'NEUTRAL'
        e['_residual_clarity'] = 'ambiguous'

    # --- Informed score (Fable #3 fix) ---
    pm_vol = float(_pd.get('premarket_volume', 0)) if _pd else 0
    avg_vol = float(_pd.get('avg_daily_volume', 0)) if _pd else 0
    # Normalize against premarket baseline (~2% of daily vol) so ratio≈1 on normal day
    pm_baseline = avg_vol * 0.02 if avg_vol > 0 else 1
    vol_ratio = pm_vol / pm_baseline if pm_baseline > 0 else 0

    spy_pct = 0.0
    try:
        spy_pct = float(market_ctx.get('spy_futures_pct', 0))
    except (TypeError, ValueError):
        pass
    idio_move = abs(gap_at_entry - spy_pct)

    daily_ret_std = float(_pd.get('daily_return_std', 0)) if _pd else 0
    gap_z = abs(gap_at_entry) / daily_ret_std if daily_ret_std > 0.5 else abs(gap_at_entry) / 2.0

    vol_signal = min(1.0, np.log1p(vol_ratio) / np.log1p(10))
    gap_signal = min(1.0, gap_z / 3.0)
    informed_score = vol_signal * 0.6 + gap_signal * 0.4

    e['_vol_ratio'] = round(vol_ratio, 2)
    e['_idio_move'] = round(idio_move, 2)
    e['_gap_z'] = round(gap_z, 2)
    e['_informed_score'] = round(informed_score, 3)

    # --- Tape discount ---
    catalyst_dir = e.get('catalyst_direction', e.get('direction', ''))
    tape_opposes = (
        (catalyst_dir == 'LONG' and gap_at_entry < -1.0) or
        (catalyst_dir == 'SHORT' and gap_at_entry > 1.0)
    )

    if tape_opposes and informed_score > 0.3:
        tape_discount = max(0.0, 1.0 - (informed_score - 0.3) * 1.43)
    else:
        tape_discount = 1.0
    e['_tape_discount'] = round(tape_discount, 3)

    # --- EV computation ---
    clarity_mult = {'clear': 1.0, 'lean': 0.6, 'ambiguous': 0.2}.get(e['_residual_clarity'], 0.2)
    ev = abs(residual_mid) * tape_discount * clarity_mult
    e['_ev'] = round(ev, 3)

    # --- Soft penalties on EV ---
    reasoning_lower = str(e.get('reasoning', '')).lower()
    catalyst_lower = str(e.get('catalyst', '')).lower()
    combined_text = reasoning_lower + ' ' + catalyst_lower

    SELF_CONTRADICTION_PHRASES = [
        'untradeable', 'un-tradeable', 'zero remaining edge', 'zero edge',
        'not a new actionable catalyst', 'no new catalyst', 'direction unknown',
        'direction is unknown', 'not yet occurred', 'not yet released',
        'no intraday catalyst', 'no earnings catalyst', 'pure gamma play',
        'virtually untradeable', 'essentially untradeable', 'no remaining upside',
        'zero remaining', 'not yet reported', 'very low-conviction',
        'very low conviction', 'contradicts the', 'immaterial', 'no remaining edge',
    ]
    if any(p in combined_text for p in SELF_CONTRADICTION_PHRASES):
        e['_ev'] *= 0.3
        e['_ev_penalty'] = 'self_contradiction'

    # Direction-remaining contradiction
    llm_dir = e.get('direction', '')
    if llm_dir == 'LONG' and residual_mid < -1.0:
        e['_ev'] *= 0.5
        e['_ev_penalty'] = e.get('_ev_penalty', '') + ';dir_residual_conflict'
    elif llm_dir == 'SHORT' and residual_mid > 1.0:
        e['_ev'] *= 0.5
        e['_ev_penalty'] = e.get('_ev_penalty', '') + ';dir_residual_conflict'

    # Deal materiality
    if str(e.get('deal_materiality', '')).lower() in ('immaterial', 'low', 'minimal'):
        e['_ev'] *= 0.2
        e['_ev_penalty'] = e.get('_ev_penalty', '') + ';immaterial_deal'

    # Vague analyst
    cat_type = str(e.get('catalyst_type', '')).lower()
    if cat_type == 'analyst':
        _cat_text = str(e.get('catalyst', ''))
        _has_firm = bool(re.search(
            r'(?:Morgan Stanley|Goldman|JPMorgan|Citi|BofA|Barclays|UBS|'
            r'Deutsche|RBC|Wells Fargo|Jefferies|Piper|Cowen|Needham|'
            r'Wedbush|Stifel|Raymond James|Baird|Oppenheimer|Wolfe|'
            r'Truist|KeyBanc|DA Davidson|Canaccord|HC Wainwright|'
            r'Bernstein|BTIG|Mizuho|Evercore|William Blair|Cantor|'
            r'Loop Capital|B\. Riley|Craig-Hallum|Northland|Roth|'
            r'Benchmark|Ladenburg|Maxim|Lake Street)', _cat_text, re.IGNORECASE))
        if not _has_firm:
            _vague = ['multiple analysts', 'multiple upgrades', 'several analysts',
                      'analyst consensus', 'wall street']
            if any(vp in _cat_text.lower() for vp in _vague):
                e['_ev'] *= 0.3
                e['_ev_penalty'] = e.get('_ev_penalty', '') + ';vague_analyst'

    # Pre-earnings (no catalyst yet)
    PRE_EARNINGS_KILL_PHRASES = [
        'expected to report', 'not yet reported', 'after market close',
        'after the close', 'after close today', 'earnings not yet',
        'has not yet reported', 'report expected after', 'reports after',
    ]
    if any(p in catalyst_lower or p in reasoning_lower for p in PRE_EARNINGS_KILL_PHRASES):
        e['_ev'] *= 0.1
        e['_ev_penalty'] = e.get('_ev_penalty', '') + ';pre_earnings'


# --- Rebuild longs/shorts from residual direction ---
_all_entries = longs + shorts
longs = [e for e in _all_entries if e.get('_residual_direction') == 'LONG' and not e.get('_hard_block')]
shorts = [e for e in _all_entries if e.get('_residual_direction') == 'SHORT' and not e.get('_hard_block')]
blocked = [e for e in _all_entries if e.get('_hard_block')]
neutral = [e for e in _all_entries if e.get('_residual_direction') in ('NEUTRAL', None) and not e.get('_hard_block')]

longs.sort(key=lambda x: x.get('_ev', 0), reverse=True)
shorts.sort(key=lambda x: x.get('_ev', 0), reverse=True)

# --- Cross-run / cross-scanner dedup (runs AFTER EV is computed) ---
_dedup_blocked = []
for _direction_list, _dir_label in [(longs, 'LONG'), (shorts, 'SHORT')]:
    for e in _direction_list:
        if e.get('_hard_block'):
            continue
        _allowed, _reason = _dedup_check(
            e['ticker'], _dir_label, e.get('_ev', 0), 'discovery', TIME_STR)
        if not _allowed:
            e['_hard_block'] = _reason
            _dedup_blocked.append((e['ticker'], _dir_label, _reason))

if _dedup_blocked:
    print(f"\n  Cross-run dedup blocked {len(_dedup_blocked)} signals:")
    for _t, _d, _r in _dedup_blocked:
        print(f"    {_t} {_d}: {_r}")

# Rebuild after dedup blocking
_all_entries = longs + shorts + blocked + neutral
longs = [e for e in _all_entries if e.get('_residual_direction') == 'LONG' and not e.get('_hard_block')]
shorts = [e for e in _all_entries if e.get('_residual_direction') == 'SHORT' and not e.get('_hard_block')]
blocked = [e for e in _all_entries if e.get('_hard_block')]

longs.sort(key=lambda x: x.get('_ev', 0), reverse=True)
shorts.sort(key=lambda x: x.get('_ev', 0), reverse=True)

print(f"\n  V4 Residual-Based Evaluation:")
print(f"    LONG candidates:  {len(longs)} (from residual)")
print(f"    SHORT candidates: {len(shorts)} (from residual)")
print(f"    NEUTRAL (ambiguous residual): {len(neutral)}")
print(f"    NEUTRAL (Stage 2 — no edge): {len(stage2_neutrals)}")
print(f"    Hard-blocked: {len(blocked)}")
if blocked:
    for e in blocked:
        print(f"      BLOCKED {e['ticker']}: {e['_hard_block']}")
for e in longs + shorts:
    llm_dir = e.get('direction', '')
    res_dir = e.get('_residual_direction', '')
    if llm_dir and res_dir and llm_dir != res_dir:
        print(f"    FLIP {e['ticker']}: LLM said {llm_dir}, residual says {res_dir} "
              f"(gap={e.get('_gap_at_entry', 0):+.1f}%, interval={e.get('_move_interval', '?')}, "
              f"residual_mid={e.get('_residual_mid', 0):+.1f}%)")


# ── OUTPUT: Summary tables first, then detailed cards below ──

_print_summary_table(longs, 'LONG')
_print_summary_table(shorts, 'SHORT')

# --- Record selections to cross-run dedup state ---
for _dir_entries, _dir_label in [(longs, 'LONG'), (shorts, 'SHORT')]:
    for e in _dir_entries:
        if e.get('_is_selected'):
            _dedup_record(TODAY_STR, e['ticker'], _dir_label,
                          e.get('_ev', 0), 'discovery', TIME_STR)

_dedup_state = _dedup_summary()
if _dedup_state != "No active state":
    print(f"\n  Active ticker state after this run:")
    print(_dedup_state)

_print_detail_cards(longs, 'LONG')
_print_detail_cards(shorts, 'SHORT')


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
        ('catalyst_weight', 'FLOAT64'),
        ('mismatch_index', 'FLOAT64'),
        ('volume_ratio', 'FLOAT64'),
        ('volume_mult', 'FLOAT64'),
        ('stage2_attempted', 'BOOL'),
        ('stage2_error', 'STRING'),
        ('claude_error', 'STRING'),
        ('residual_low', 'FLOAT64'),
        ('residual_high', 'FLOAT64'),
        ('residual_mid', 'FLOAT64'),
        ('ev', 'FLOAT64'),
        ('residual_direction', 'STRING'),
        ('residual_clarity', 'STRING'),
        ('informed_score', 'FLOAT64'),
        ('gap_at_entry', 'FLOAT64'),
        ('expected_move_low_pct', 'FLOAT64'),
        ('expected_move_high_pct', 'FLOAT64'),
        ('catalyst_direction', 'STRING'),
        ('tape_discount', 'FLOAT64'),
        ('gap_z', 'FLOAT64'),
        ('ev_penalty', 'STRING'),
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
                'selected': rank <= (FINAL_LONG if direction_label == 'LONG' else FINAL_SHORT) and e.get('_ev', 0) >= MIN_EV_FOR_SELECTION and not e.get('_hard_block'),
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
                'stage2_model': str(deep_results.get(e['ticker'], {}).get('_s2_meta', {}).get('model_used', ''))[:50],
                'market_futures': futures,
                'market_risk': risk,
                'vix_level': vix,
                'move_is_exhausted': e.get('move_is_exhausted', False),
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
                'gate_reason': str(e.get('gate_reason', ''))[:500],
                'stage2_attempted': True,
                'stage2_error': '',
                'claude_error': str(deep_results.get(e['ticker'], {}).get('_s2_meta', {}).get('claude_error', ''))[:200],
                'residual_mid': e.get('_residual_mid', 0.0),
                'residual_low': e.get('_residual_low', 0.0),
                'residual_high': e.get('_residual_high', 0.0),
                'ev': e.get('_ev', 0.0),
                'residual_direction': e.get('_residual_direction', ''),
                'residual_clarity': e.get('_residual_clarity', ''),
                'informed_score': e.get('_informed_score', 0.0),
                'gap_at_entry': e.get('_gap_at_entry', 0.0),
                'expected_move_low_pct': float(e.get('expected_move_low_pct', 0)),
                'expected_move_high_pct': float(e.get('expected_move_high_pct', 0)),
                'catalyst_direction': str(e.get('catalyst_direction', '')),
                'tape_discount': e.get('_tape_discount', 1.0),
                'gap_z': e.get('_gap_z', 0.0),
                'ev_penalty': str(e.get('_ev_penalty', '')),
            })

    # Log Stage 2 NEUTRAL entries (LLM said no tradable edge)
    for e in stage2_neutrals:
        rows.append({
            'scan_timestamp': now_utc,
            'scan_date': TODAY_STR,
            'scan_time_pt': TIME_STR,
            'ticker': e['ticker'],
            'company': str(e.get('company', ''))[:200],
            'direction': 'NEUTRAL',
            'rank_in_direction': 0,
            'selected': False,
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
            'stage2_model': str(deep_results.get(e['ticker'], {}).get('_s2_meta', {}).get('model_used', ''))[:50],
            'market_futures': futures,
            'market_risk': risk,
            'vix_level': vix,
            'move_is_exhausted': e.get('move_is_exhausted', False),
            'scanner_version': SCANNER_VERSION,
            'gate_reason': 'STAGE2_NEUTRAL',
            'stage2_attempted': True,
            'stage2_error': '',
            'claude_error': '',
            'residual_mid': 0.0,
            'residual_low': 0.0,
            'residual_high': 0.0,
            'ev': 0.0,
            'residual_direction': 'NEUTRAL',
            'residual_clarity': '',
            'informed_score': 0.0,
            'gap_at_entry': 0.0,
            'expected_move_low_pct': float(e.get('expected_move_low_pct', 0)),
            'expected_move_high_pct': float(e.get('expected_move_high_pct', 0)),
            'catalyst_direction': str(e.get('catalyst_direction', '')),
            'tape_discount': 1.0,
            'gap_z': 0.0,
            'ev_penalty': '',
        })

    # Also log scan-only discoveries (not deep-analyzed)
    # Log actual Stage 1 direction (was force-set to NEUTRAL before Fix M)
    deep_tickers = set(e['ticker'] for e in longs + shorts + stage2_neutrals)
    _deep_cand_tickers = set(c.get('ticker') for c in deep_candidates) if 'deep_candidates' in dir() else set()
    for ticker, d in all_discovered.items():
        if ticker in deep_tickers:
            continue
        _was_attempted = ticker in _deep_cand_tickers
        _s2_err = ''
        _s2_claude_err = ''
        _s2_model = ''
        if _was_attempted and ticker in deep_results:
            _dr = deep_results[ticker]
            _dm = _dr.get('_s2_meta', {})
            if 'error' in _dr:
                _s2_err = str(_dr['error'])[:200]
            _s2_claude_err = str(_dm.get('claude_error', ''))[:200]
            _s2_model = str(_dm.get('model_used', ''))[:50]
        rows.append({
            'scan_timestamp': now_utc,
            'scan_date': TODAY_STR,
            'scan_time_pt': TIME_STR,
            'ticker': ticker,
            'company': str(d.get('company', ''))[:200],
            'direction': str(d.get('direction', 'NEUTRAL')).upper() or 'NEUTRAL',
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
            'stage2_model': _s2_model,
            'market_futures': futures,
            'market_risk': risk,
            'vix_level': vix,
            'scanner_version': SCANNER_VERSION,
            'stage2_attempted': _was_attempted,
            'stage2_error': _s2_err,
            'claude_error': _s2_claude_err,
            'residual_mid': 0.0,
            'residual_low': 0.0,
            'residual_high': 0.0,
            'ev': 0.0,
            'residual_direction': '',
            'residual_clarity': '',
            'informed_score': 0.0,
            'gap_at_entry': 0.0,
            'expected_move_low_pct': 0.0,
            'expected_move_high_pct': 0.0,
            'catalyst_direction': '',
            'tape_discount': 1.0,
            'ev_penalty': '',
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
