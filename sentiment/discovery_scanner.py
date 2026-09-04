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
CLAUDE_MODEL   = "claude-sonnet-4-5-20241022"

BUCKET_NAME = "time-series-bucket-ys"
OUTPUT_DIR  = "Data/Scanner"
BQ_TABLE    = "ys-time-series.scoring_logs.discovery_scanner"

PST = pytz.timezone('America/Los_Angeles')
NOW_PT = datetime.now(PST)
TODAY_STR = NOW_PT.strftime('%Y-%m-%d')
TIME_STR = NOW_PT.strftime('%H:%M')

MIN_MOVE_PCT       = 5.0
MAX_NEWS_AGE_HOURS = 24
SCANNER_VERSION    = 'V1'
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
THEMATIC_QUERIES = [
    {
        'id': 'premarket_gainers',
        'query': f'biggest stock pre-market gainers today {TODAY_STR} stocks up more than 5 percent',
        'direction_bias': 'LONG',
    },
    {
        'id': 'premarket_losers',
        'query': f'biggest stock pre-market losers today {TODAY_STR} stocks down more than 5 percent',
        'direction_bias': 'SHORT',
    },
    {
        'id': 'earnings_surprises',
        'query': f'stocks earnings surprise after hours {TODAY_STR} beat miss biggest reaction',
        'direction_bias': None,
    },
    {
        'id': 'fda_regulatory',
        'query': f'FDA approval rejection drug stock {TODAY_STR} biotech pharma halted',
        'direction_bias': None,
    },
    {
        'id': 'mergers_acquisitions',
        'query': f'stock acquisition merger takeover announced {TODAY_STR} premium buyout offer',
        'direction_bias': 'LONG',
    },
    {
        'id': 'analyst_actions',
        'query': f'stock analyst upgrade downgrade price target raised lowered {TODAY_STR} biggest moves',
        'direction_bias': None,
    },
    {
        'id': 'halt_resume',
        'query': f'stock trading halted resumed {TODAY_STR} NYSE NASDAQ halt pending news',
        'direction_bias': None,
    },
    {
        'id': 'short_squeeze',
        'query': f'short squeeze stocks today {TODAY_STR} high short interest rallying gamma squeeze',
        'direction_bias': 'LONG',
    },
    {
        'id': 'offerings_dilution',
        'query': f'stock offering dilution secondary share sale announced {TODAY_STR} stock plunge',
        'direction_bias': 'SHORT',
    },
    {
        'id': 'contracts_deals',
        'query': f'stock major contract deal partnership billion {TODAY_STR} government defense AI',
        'direction_bias': 'LONG',
    },
    {
        'id': 'bankruptcy_fraud',
        'query': f'stock fraud investigation bankruptcy SEC lawsuit {TODAY_STR} crash plunge',
        'direction_bias': 'SHORT',
    },
    {
        'id': 'insider_activity',
        'query': f'insider buying selling stock large stake {TODAY_STR} CEO CFO director',
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
DISCOVER_SYSTEM = """You are a pre-market stock screener. Your job is to extract specific stock tickers and their expected moves from search results about today's biggest movers.

From the search results, identify individual stocks that are likely to move MORE THAN 5% at today's market open. Focus on specific, actionable events from the LAST 12-24 HOURS ONLY (after-hours yesterday through pre-market today). Ignore older news — it is already priced in. Do not include catalysts older than 24 hours.

Return ONLY a JSON object:
{
  "discovered": [
    {
      "ticker": "<stock ticker symbol, e.S. AAPL>",
      "company": "<company name>",
      "direction": "LONG" or "SHORT",
      "expected_move_pct": <float, expected % move at open, e.g. 8.5>,
      "catalyst_strength": <int 1-10, how strong/certain is this catalyst>,
      "catalyst": "<specific event driving the move — cite numbers, %s, deal values>",
      "catalyst_type": "<earnings|analyst|regulatory|deal|offering|insider|halt|squeeze|bankruptcy|macro>",
      "news_time": "<REQUIRED — EXACT timestamp, format: 'YYYY-MM-DD H:MM AM/PM ET'. NEVER return 'unknown'. ESTIMATE if needed: earnings after close='4:05 PM ET', overnight='6:00 PM ET', pre-market='5:00 AM ET'.>",
      "source_headline": "<actual headline from the search result>",
      "premarket_move": "<pre-market % change if mentioned, or 'unknown'>",
      "market_cap_tier": "<mega|large|mid|small|micro|unknown>"
    }
  ]
}

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
- Return ONLY the JSON"""

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

            if ticker not in all_discovered or move > abs(float(all_discovered[ticker].get('expected_move_pct', 0))):
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

price_data = {}
if deep_candidates:
    print(f"\n  Fetching price data for {len(deep_candidates)} candidates...", end=' ')
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        price_futures = {executor.submit(fetch_price_data, c['ticker']): c['ticker']
                         for c in deep_candidates}
        for fut in as_completed(price_futures):
            t = price_futures[fut]
            try:
                price_data[t] = fut.result()
            except Exception:
                price_data[t] = None
    n_prices = sum(1 for v in price_data.values() if v and 'change_pct' in v)
    print(f"got prices for {n_prices}/{len(deep_candidates)}")

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

    for i, cand in enumerate(deep_candidates):
        ticker = cand.get('ticker', '?')
        print(f"\n  [{i+1}/{len(deep_candidates)}] {ticker} "
              f"(move={cand['expected_move_pct']:+.1f}%, "
              f"str={cand.get('catalyst_strength',0)}/10, "
              f"{cand.get('direction','?')})...")

        sources_str = ""
        sources = cand.get('_sources', [])
        if sources:
            sources_str = "\n\nSOURCE ARTICLES:\n"
            for j, src in enumerate(sources[:5]):
                sources_str += f"  {j+1}. {src.get('title', '')} — {src.get('url', '')}\n"

        pd_info = price_data.get(ticker)
        price_str = ""
        if pd_info and 'change_pct' in pd_info:
            price_str = (
                f"\nACTUAL PRICE DATA (verified):\n"
                f"- Current/pre-market price: ${pd_info.get('current_price', 'N/A')}\n"
                f"- Previous close: ${pd_info.get('prev_close', 'N/A')}\n"
                f"- Change from prev close: {pd_info.get('change_pct', 0):+.1f}%\n"
                f"USE THIS ACTUAL % CHANGE as the move_already_priced_pct. "
                f"Do NOT guess the priced-in amount from news articles.\n"
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
            f"{price_str}"
            f"{sources_str}"
            f"{market_context_str}\n\n"
            f"Evaluate whether this stock will actually move 5%+ today. "
            f"Be specific about how much is already priced in pre-market "
            f"and how much remaining move is left to capture at open."
        )

        if claude_ok:
            result = claude_analyze(prompt, DEEP_SYSTEM, claude_client)
        else:
            result = gemini_search(prompt, DEEP_SYSTEM)

        if 'error' not in result:
            direction = result.get('direction', '?')
            confidence = result.get('confidence', 0)
            remaining = result.get('remaining_move_pct', 0)
            priced_in = result.get('move_already_priced_pct', 0)
            strength = result.get('catalyst_strength', 0)
            freshness = result.get('freshness', 5)
            # Tradability score: what's left to capture × how strong × how fresh
            tradability = abs(float(remaining)) * int(strength) * int(freshness) / 100.0
            result['tradability_score'] = round(tradability, 2)
            result['_discovery'] = cand
            catalyst = str(result.get('catalyst', ''))[:70]
            print(f"    -> {direction} conf={confidence}/10  "
                  f"remaining={remaining:+.1f}%  "
                  f"(priced_in={priced_in:+.1f}%)  "
                  f"tradability={tradability:.1f}")
        else:
            print(f"    -> ERROR: {result['error'][:60]}")
            result['tradability_score'] = 0
            result['_discovery'] = cand

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
print(f"  ║  DISCOVERY SCANNER RESULTS — {TODAY_STR} {TIME_STR} PT{'':>17}║")
print(f"  ╚══════════════════════════════════════════════════════════════════╝")
print(f"{'='*W}")

if market_ctx:
    print(f"\n  MARKET CONTEXT:")
    print(f"    Futures:   {market_ctx.get('futures', 'N/A')}")
    print(f"    Theme:     {market_ctx.get('overnight_theme', 'N/A')}")
    print(f"    Risk:      {market_ctx.get('risk_sentiment', 'N/A')}")

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
        'confidence': int(r.get('confidence', 0)),
        'expected_move_pct': float(r.get('expected_move_pct', 0)),
        'move_already_priced_pct': _priced,
        'remaining_move_pct': _remaining,
        'move_is_exhausted': _exhausted,
        'catalyst_strength': int(r.get('catalyst_strength', 0)),
        'freshness': int(r.get('freshness', 5)),
        'liquidity_risk': int(r.get('liquidity_risk', 5)),
        'tradability_score': float(r.get('tradability_score', 0)),
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
        'news_time': disc.get('news_time', ''),
        'theme': disc.get('_theme', ''),
        'in_deploy': disc.get('in_deploy', False),
        'scan_move_score': disc.get('move_score', 0),
        'composite': int(r.get('confidence', 0)) * abs(_remaining),
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
        return 0.5
    age_hours = (now_pst - news_dt.astimezone(PST)).total_seconds() / 3600
    if age_hours < 0:
        return 1.0
    if age_hours <= 6:
        return 1.0
    if age_hours <= 12:
        return 0.75
    if age_hours <= 18:
        return 0.5
    return 0.25


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
        deploy_flag = '  ✓' if e.get('in_deploy') else ''
        catalyst_short = str(e.get('catalyst', '') or e.get('reasoning', ''))[:60]
        print(f"    {i:>2} {e['ticker']:<6} {e['tradability_score']:>5.1f} "
              f"{e['confidence']:>3}/10 {e['remaining_move_pct']:>+6.1f}% "
              f"{priced_str:>7} {cat:<5} {ts_str:<18} {age:>8}{exh}{deploy_flag:>5}  {catalyst_short}")
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
                'selected': rank <= (FINAL_LONG if direction_label == 'LONG' else FINAL_SHORT),
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
            })

    # Also log scan-only discoveries (not deep-analyzed)
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
            'direction': d.get('direction', 'NEUTRAL'),
            'rank_in_direction': 0,
            'selected': False,
            'in_deploy_universe': d.get('in_deploy', False),
            'confidence': 0,
            'expected_move_pct': float(d.get('expected_move_pct', 0)),
            'move_already_priced_pct': 0.0,
            'remaining_move_pct': 0.0,
            'catalyst_strength': int(d.get('catalyst_strength', 0)),
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
