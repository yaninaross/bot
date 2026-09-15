# %% Agent 2 — Self-Contained Discovery + T+2 Watchlist
#
# Standalone script: discovers moving stocks via Gemini thematic search,
# runs deep LLM analysis (Claude Sonnet via Vertex), applies T+2 gap-fade
# SHORT filters, selects top 2 candidates, outputs trading watchlist.
#
# Usage:
#   python agent2.py                     # live scan
#   python agent2.py scan --date 2026-09-08   # historical
#   python agent2.py scan --dry-run      # no DB writes
#   python agent2.py report              # performance report
#   python agent2.py fill                # fill T+2 outcomes
#   python agent2.py log_entry --ticker PDSB --entry_price 5.52 \
#                              --entry_time "09:37" --entry_type local_high
#
# Colab: set MODE/SCAN_DATE/DRY_RUN config vars and run as cell.

import os, sys, time, gc, json, re, warnings, logging, sqlite3, argparse
from datetime import datetime, timedelta, date
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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
    import requests
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "requests", "--quiet"])
    import requests

try:
    from tabulate import tabulate
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "tabulate", "--quiet"])
    from tabulate import tabulate

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
MODE = 'scan'
SCAN_DATE = None
DRY_RUN = False
REPORT_WEEKS = 4

GCP_PROJECT  = "yanina-everything"
GCP_LOCATION = "us-central1"
GEMINI_MODEL = "gemini-2.5-flash"

CLAUDE_REGION  = "us-east5"
CLAUDE_MODEL   = "claude-sonnet-4-6"

BUCKET_NAME = "time-series-bucket-ys"
OUTPUT_DIR  = "Data/Scanner"
BQ_TABLE    = "YOUR_GCP_PROJECT.scoring_logs.agent2_signal_log"
BQ_RUNS_TABLE = "YOUR_GCP_PROJECT.scoring_logs.agent2_daily_runs"
BQ_PATTERNS_TABLE = "YOUR_GCP_PROJECT.scoring_logs.scanner_patterns"

POLYGON_API_KEY = "YOUR_POLYGON_API_KEY"

PST = pytz.timezone('America/Los_Angeles')
ET = pytz.timezone('America/New_York')

MIN_MOVE_PCT       = 2.0
MAX_NEWS_AGE_HOURS = 16
SCANNER_VERSION    = 'AGENT2_V1'
STRONG_CATALYST_SCORE = 7
STRONG_CATALYST_MOVE  = 8.0
FINAL_LONG         = 0
FINAL_SHORT        = 2
MAX_WORKERS        = 4
GEMINI_DELAY_S     = 0.5
MAX_RETRIES        = 3

# T+2 gap-fade filters
GAP_MIN_T1T2    = 0.020
GAP_MIN_T3      = 0.015
PM_RETRACE_MAX  = 0.80
TARGET_TRADES   = 2

CATALYST_T1 = {
    'offering', 'secondary_offering',
    'regulatory', 'fda_approval',
    'government_contract',
}
CATALYST_EXCLUDE = {
    'deal', 'merger', 'squeeze', 'halt', 'bankruptcy',
}

MIN_CONFIDENCE_FOR_SELECTION = 4
MIN_PRICE_LONG = 3.00
MIN_PRICE_SHORT = 0.25
MIN_PREMARKET_VOLUME = 20000

DEPLOY_TICKERS = set()  # Agent 2 considers ALL tickers

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

DB_PATH = os.environ.get("DB_PATH", os.path.join(BASE_DIR, 'data', 'agent2.db'))
LOG_PATH = os.path.join(BASE_DIR, 'logs', 'agent2.log')

W = 100

_NYSE_HOLIDAYS = {
    '2026-01-01', '2026-01-19', '2026-02-16', '2026-04-03', '2026-05-25',
    '2026-06-19', '2026-07-03', '2026-09-07', '2026-09-12', '2026-11-26', '2026-12-25',
    '2027-01-01', '2027-01-18', '2027-02-15', '2027-03-26', '2027-05-31',
    '2027-06-18', '2027-07-05', '2027-09-06', '2027-11-25', '2027-12-24',
}


# ==============================================================================
# LOGGING
# ==============================================================================
def setup_logging():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    fmt = '%(asctime)s | %(levelname)s | %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for h in root.handlers[:]:
        root.removeHandler(h)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(ch)
    fh = logging.FileHandler(LOG_PATH)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(fh)
    return logging.getLogger('agent2')

log = setup_logging()


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
# POLYGON CLIENT
# ==============================================================================
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


# ==============================================================================
# TRADING CALENDAR
# ==============================================================================
def is_trading_day(d):
    ds = d if isinstance(d, str) else d.strftime('%Y-%m-%d')
    dt = datetime.strptime(ds, '%Y-%m-%d').date() if isinstance(ds, str) else ds
    return dt.weekday() < 5 and ds not in _NYSE_HOLIDAYS

def get_prior_trading_date(target_date):
    if isinstance(target_date, str):
        target_date = datetime.strptime(target_date, '%Y-%m-%d').date()
    d = target_date - timedelta(days=1)
    for _ in range(10):
        if is_trading_day(d):
            return d
        d -= timedelta(days=1)
    raise ValueError(f"No trading day found in 10 days before {target_date}")

def get_next_n_trading_dates(from_date, n=2):
    if isinstance(from_date, str):
        from_date = datetime.strptime(from_date, '%Y-%m-%d').date()
    results = []
    d = from_date + timedelta(days=1)
    for _ in range(30):
        if is_trading_day(d):
            results.append(d)
            if len(results) >= n:
                break
        d += timedelta(days=1)
    return results

def get_t1_t2_dates(scan_date_str):
    nxt = get_next_n_trading_dates(scan_date_str, 2)
    t1 = nxt[0].strftime('%Y-%m-%d') if len(nxt) >= 1 else None
    t2 = nxt[1].strftime('%Y-%m-%d') if len(nxt) >= 2 else None
    return t1, t2


# ==============================================================================
# DATABASE — SQLite local backup
# ==============================================================================
def init_db(db_path=DB_PATH):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS daily_runs (
        run_date          TEXT PRIMARY KEY,
        run_timestamp     TEXT,
        runtime_seconds   REAL,
        n_themes_searched INTEGER,
        n_discovered      INTEGER,
        n_deep_analyzed   INTEGER,
        n_qualified       INTEGER,
        n_selected        INTEGER,
        market_futures    TEXT,
        market_risk       TEXT,
        vix_level         TEXT,
        scanner_version   TEXT
    );

    CREATE TABLE IF NOT EXISTS candidates (
        candidate_id      TEXT PRIMARY KEY,
        run_date          TEXT,
        ticker            TEXT,
        company           TEXT,
        direction         TEXT,
        confidence        INTEGER,
        expected_move_pct REAL,
        remaining_move_pct REAL,
        move_already_priced_pct REAL,
        move_is_exhausted INTEGER DEFAULT 0,
        catalyst_strength INTEGER,
        freshness         INTEGER,
        liquidity_risk    INTEGER,
        tradability_score REAL,
        catalyst_type     TEXT,
        catalyst          TEXT,
        reasoning         TEXT,
        headline          TEXT,
        news_time         TEXT,
        theme             TEXT,
        market_cap_tier   TEXT,
        gap_pct           REAL,
        change_pct        REAL,
        prev_close        REAL,
        current_price     REAL,
        pm_high           REAL,
        pm_last           REAL,
        pm_retrace_pct    REAL,
        agent_tier        TEXT,
        passed_direction  INTEGER,
        passed_exhausted  INTEGER,
        passed_gap        INTEGER,
        passed_catalyst   INTEGER,
        passed_retrace    INTEGER,
        passed_all        INTEGER DEFAULT 0,
        filter_fail       TEXT,
        gate_reason       TEXT,
        selected          INTEGER DEFAULT 0,
        selection_rank    INTEGER,
        t1_date           TEXT,
        t2_date           TEXT,
        entry_price       REAL,
        entry_time        TEXT,
        entry_type        TEXT,
        t0_close          REAL,
        t1_close          REAL,
        t2_close          REAL,
        pnl_eod           REAL,
        pnl_t1            REAL,
        pnl_t2            REAL,
        scanner_version   TEXT,
        log_timestamp     TEXT
    );

    CREATE TABLE IF NOT EXISTS trades (
        trade_id          TEXT PRIMARY KEY,
        run_date          TEXT,
        ticker            TEXT,
        selection_rank    INTEGER,
        direction         TEXT,
        confidence        INTEGER,
        gap_pct           REAL,
        agent_tier        TEXT,
        catalyst_type     TEXT,
        catalyst          TEXT,
        tradability_score REAL,
        pm_retrace_pct    REAL,
        t1_date           TEXT,
        t2_date           TEXT,
        entry_price       REAL DEFAULT NULL,
        entry_time        TEXT DEFAULT NULL,
        entry_type        TEXT DEFAULT NULL,
        skip_reason       TEXT DEFAULT NULL,
        position_size_usd REAL DEFAULT NULL,
        t0_close          REAL DEFAULT NULL,
        t1_close          REAL DEFAULT NULL,
        t2_close          REAL DEFAULT NULL,
        pnl_eod           REAL DEFAULT NULL,
        pnl_t1            REAL DEFAULT NULL,
        pnl_t2            REAL DEFAULT NULL,
        actually_traded   INTEGER DEFAULT 0,
        actual_pnl        REAL DEFAULT NULL,
        notes             TEXT DEFAULT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_cand_date ON candidates(run_date);
    CREATE INDEX IF NOT EXISTS idx_cand_selected ON candidates(selected);
    CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(run_date);
    CREATE INDEX IF NOT EXISTS idx_trades_t2 ON trades(t2_date);
    """)
    conn.close()
    return db_path

def _get_conn():
    return sqlite3.connect(DB_PATH)


# ==============================================================================
# THEMATIC SEARCH QUERIES
# ==============================================================================
def build_thematic_queries(now_et, now_pt, today_str, yesterday_str):
    time_et_str = now_et.strftime('%I:%M %p ET')
    return [
        {
            'id': 'stocks_moving_now',
            'query': (f'stocks moving right now {today_str} as of {time_et_str} '
                      f'biggest gainers losers surging crashing'),
            'direction_bias': None,
        },
        {
            'id': 'afterhours_movers',
            'query': (f'stocks after hours movers last night {yesterday_str} '
                      f'earnings reaction premarket {today_str} surging dropping'),
            'direction_bias': None,
        },
        {
            'id': 'earnings_today',
            'query': (f'earnings results today {today_str} '
                      f'EPS revenue beat miss guidance stock reaction'),
            'direction_bias': None,
        },
        {
            'id': 'earnings_surprise',
            'query': (f'earnings surprise today {today_str} stock surging plunging '
                      f'quarterly results beat miss small cap mid cap premarket'),
            'direction_bias': None,
        },
        {
            'id': 'biggest_gainers',
            'query': (f'biggest stock gainers today {today_str} '
                      f'stocks up more than 5 percent as of {time_et_str}'),
            'direction_bias': 'LONG',
        },
        {
            'id': 'biggest_losers',
            'query': (f'biggest stock losers today {today_str} '
                      f'stocks down more than 5 percent as of {time_et_str}'),
            'direction_bias': 'SHORT',
        },
        {
            'id': 'fda_regulatory',
            'query': (f'FDA approval rejection drug stock {today_str} '
                      f'biotech pharma halted clinical trial'),
            'direction_bias': None,
        },
        {
            'id': 'mergers_acquisitions',
            'query': (f'stock acquisition merger takeover announced {today_str} '
                      f'premium buyout offer deal'),
            'direction_bias': 'LONG',
        },
        {
            'id': 'analyst_actions',
            'query': (f'stock analyst upgrade downgrade price target raised lowered '
                      f'{today_str} biggest moves as of {time_et_str}'),
            'direction_bias': None,
        },
        {
            'id': 'halt_resume',
            'query': (f'stock trading halted resumed {today_str} '
                      f'NYSE NASDAQ halt pending news'),
            'direction_bias': None,
        },
        {
            'id': 'short_squeeze',
            'query': (f'short squeeze stocks today {today_str} '
                      f'high short interest rallying gamma squeeze'),
            'direction_bias': 'LONG',
        },
        {
            'id': 'offerings_dilution',
            'query': (f'stock offering dilution secondary share sale announced '
                      f'{today_str} stock plunge'),
            'direction_bias': 'SHORT',
        },
        {
            'id': 'contracts_deals',
            'query': (f'stock major contract deal partnership billion {today_str} '
                      f'government defense AI'),
            'direction_bias': 'LONG',
        },
        {
            'id': 'bankruptcy_fraud',
            'query': (f'stock fraud investigation bankruptcy SEC lawsuit {today_str} '
                      f'crash plunge'),
            'direction_bias': 'SHORT',
        },
        {
            'id': 'insider_activity',
            'query': (f'insider buying selling stock large stake {today_str} '
                      f'CEO CFO director'),
            'direction_bias': None,
        },
    ]


# ==============================================================================
# PROMPTS
# ==============================================================================
def build_prompts(now_et, now_pt):
    cutoff_4h = (now_et - timedelta(hours=4)).strftime('%Y-%m-%d %I:%M %p ET')
    cutoff_16h = (now_et - timedelta(hours=16)).strftime('%Y-%m-%d %I:%M %p ET')

    discover_system = f"""You are a stock screener. Your job is to extract specific stock tickers and their expected moves from search results about today's biggest movers.

CURRENT TIME: {now_et.strftime('%Y-%m-%d %I:%M %p ET')} ({now_pt.strftime('%I:%M %p PT')})
PRIORITY WINDOW: News published AFTER {cutoff_4h} (last 4 hours) is highest priority.
HARD CUTOFF: Do NOT include any news published BEFORE {cutoff_16h} (older than 16 hours).

FRESHNESS SCORING:
  - Published within last 2 hours: MOST VALUABLE — always include, boost catalyst_strength by +2
  - Published 2-4 hours ago: HIGH VALUE — include with normal scoring
  - Published 4-8 hours ago: MODERATE — include only if catalyst_strength >= 5
  - Published 8-16 hours ago: LOW VALUE — include only if catalyst_strength >= 7
  - Published >16 hours ago: EXCLUDE entirely

From the search results, identify individual stocks that are likely to move MORE THAN {MIN_MOVE_PCT}%.

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
- ACQUIRER stocks are typically NEUTRAL or NEGATIVE. Do NOT include acquirer stocks unless the deal is truly transformative.
- "Shareholder vote approved" / "regulatory clearance" / "expected to close" for an already-announced deal is a PROCEDURAL update — NOT a new catalyst. Do NOT include these.

IMPORTANT:
- Only include stocks with CLEAR catalysts likely to cause >{MIN_MOVE_PCT}% moves
- Be specific: cite actual numbers
- Return ONLY the JSON"""

    deep_system = """You are a senior pre-market trading analyst evaluating a stock for a potential 5%+ intraday trade.
Your traders enter at market open and target large intraday moves (5-15%+).

CRITICAL CONCEPT — REMAINING MOVE vs TOTAL IMPACT:
Your job is to predict what happens AFTER the trader enters at market open. You must separate:
  - total_catalyst_impact_pct: the full magnitude of the catalyst (SIGNED: positive for bullish, negative for bearish)
  - move_already_priced_pct: how much has already moved in pre-market/after-hours
  - remaining_move_pct: what's LEFT for the trader at entry (this is what matters)

IMPORTANT: total_catalyst_impact_pct must match the CATALYST direction, not your trade thesis.

For EARNINGS catalysts: the after-hours/pre-market drop is Wave 1. Wave 2 (continued selling at
open) is the norm, not the exception. Do NOT assume a bounce just because the stock already dropped a lot.

EARNINGS + GUIDANCE CUT = TWO-WAVE SELLOFF (NEVER predict a bounce):
  IF earnings beat or meet on historical numbers BUT forward guidance is lowered
  THEN the stock WILL continue dropping at open. Direction: SHORT.

EARNINGS MISS — DIRECTION RULES:
- An earnings MISS is a SHORT catalyst. Default SHORT unless miss was trivial.
- An earnings BEAT with guidance CUT is SHORT — the guidance cut dominates.
- When in doubt: FOLLOW THE GUIDANCE DIRECTION.

GAP RULES (CRITICAL FOR DIRECTION):
- A stock gapping UP 5%+ with NO CLEAR CATALYST = SHORT (gap fades are high probability)
- A stock gapping UP 5%+ on OFFERING/DILUTION = SHORT OPPORTUNITY (offering gap fade)
- A stock gapping DOWN on EARNINGS MISS = SHORT (wave 2 selling)
- Only go LONG on gap-up when catalyst is STRONG + NEW + CONTINUING (e.g. FDA approval, M&A target)

CATALYST IMPACT REFERENCE TABLE:
  FDA full approval: +5% to +30%
  FDA BLA/NDA accepted: +2% to +5% at most (procedural)
  Earnings beat (big): +5% to +20%
  Analyst upgrade: +2% to +5%
  M&A target: +10% to +40%
  M&A acquirer: -2% to +2% (typically FLAT)
  Secondary offering: -5% to -15%

PROVEN GAP-FADE TRADE PATTERNS:
  1. OFFERING GAP FADE: Stock gaps up 10%+ on news, then offering announced at discount → SHORT at local high
  2. EARNINGS GAP FADE: Stock gaps up on beat but guidance lowered → SHORT at open
  3. CATALYST EXHAUSTION: Stock gaps up 15%+ but catalyst is one-time (contract, deal) → SHORT at local high
  4. CLINICAL TRIAL TRAP: Biotech gaps up on phase 1/2 data → OFTEN fades by 30-60% of gap within 2 days

Return ONLY a JSON object:
{
  "ticker": "<TICKER>",
  "direction": "LONG" or "SHORT",
  "confidence": <int 1-10>,
  "expected_move_pct": <float, total expected impact>,
  "move_already_priced_pct": <float>,
  "remaining_move_pct": <float, expected REMAINING % move after entry>,
  "move_is_exhausted": <bool>,
  "catalyst_strength": <int 1-10>,
  "freshness": <int 1-10>,
  "liquidity_risk": <int 1-10>,
  "reasoning": "<3-5 sentences>",
  "catalyst": "<specific catalyst with numbers>",
  "catalyst_type": "<earnings|analyst|regulatory|deal|offering|insider|macro|breaking|premarket_move>",
  "acquirer_or_target": "<'acquirer'|'target'|'n/a'>",
  "deal_materiality": "<'immaterial'|'notable'|'transformative'|'n/a'>",
  "catalyst_is_new": <bool>,
  "news_timeline": [{"time": "<timestamp>", "event": "<what happened>"}],
  "bull_case": "<why go long>",
  "bear_case": "<why go short>",
  "risks": "<what could go wrong>",
  "comparable_events": "<similar past events>",
  "pre_market_action": "<pre-market details>"
}

IMPORTANT:
- direction and remaining_move_pct must reflect the REMAINING move, not the catalyst direction
- Return ONLY the JSON"""

    market_system = """You are a market analyst preparing the pre-market overview.
Search for the latest US stock market conditions as of right now.

Return ONLY a JSON object:
{
  "futures": "<S&P 500 / Nasdaq / Dow futures direction and %>",
  "spy_futures_pct": <float, S&P 500 futures % change>,
  "overnight_theme": "<1-2 sentence summary>",
  "key_sectors": "<which sectors are moving>",
  "macro_events": "<any scheduled economic data releases>",
  "risk_sentiment": "<risk-on|risk-off|mixed>",
  "vix_level": "<VIX level if found>",
  "vix_level_float": <float, VIX numeric value>,
  "top_premarket_movers": "<stocks moving 5%+ in pre-market>"
}"""

    price_system = """Return ONLY a JSON object with price data for this stock.
{
  "current_price": <float>,
  "prev_close": <float>,
  "change_pct": <float>,
  "premarket_price": <float or null>,
  "source": "<source>"
}
Return ONLY the JSON."""

    return discover_system, deep_system, market_system, price_system


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

def _safe_int(val, default=5):
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str):
        m = re.search(r'\d+', val)
        return int(m.group()) if m else default
    return default

def _safe_float(val, default=0.0):
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        m = re.search(r'[+-]?\d+\.?\d*', val)
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
                    sources.append({'url': chunk.web.uri, 'title': chunk.web.title or ''})
    except (IndexError, AttributeError):
        pass
    return sources

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

def _parse_news_time(news_time_str, scan_time=None):
    if not news_time_str or news_time_str in ('unknown', 'N/A', ''):
        return None
    s = str(news_time_str).strip()
    original_lower = news_time_str.lower()
    is_pt = any(tz in original_lower for tz in [' pt', ' pst', ' pdt', ' pacific'])
    tz_to_use = PST if is_pt else ET
    for tz_label in ['ET', 'EST', 'EDT', 'PT', 'PST', 'PDT']:
        s = s.replace(f' {tz_label}', '')
    s_clean = re.sub(
        r'\s*(pre-market|premarket|after-hours|after hours|overnight|morning|evening|during market hours)\s*',
        ' ', s, flags=re.IGNORECASE).strip()
    fmts_with_time = ['%Y-%m-%d %I:%M %p', '%Y-%m-%d %I:%M%p', '%Y-%m-%d %H:%M']
    dt = None
    for fmt in fmts_with_time:
        try:
            dt = datetime.strptime(s_clean, fmt)
            dt = tz_to_use.localize(dt)
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
        elif 'after' in original_lower:
            dt = dt.replace(hour=16, minute=30)
        elif 'pre' in original_lower or 'morning' in original_lower:
            dt = dt.replace(hour=5, minute=0)
        else:
            dt = dt.replace(hour=16, minute=0)
        dt = tz_to_use.localize(dt)
    ref_time = scan_time or datetime.now(PST)
    dt_pst = dt.astimezone(PST)
    if dt_pst > ref_time and (dt_pst - ref_time).total_seconds() > 3600:
        dt = dt - timedelta(days=1)
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
    'halt': 'HALT', 'squeeze': 'SQZE', 'bankruptcy': 'BNKR', 'none': '—',
}

CAT_MAP = {
    'earnings': 'earnings', 'analyst': 'analyst', 'regulatory': 'regulatory',
    'deal': 'deal', 'offering': 'offering', 'insider': 'deal',
    'breaking': 'breaking', 'premarket_move': 'deal', 'halt': 'deal',
    'squeeze': 'deal', 'bankruptcy': 'deal', 'macro': 'breaking',
}


# ==============================================================================
# POLYGON FUNCTIONS
# ==============================================================================
def get_prev_session_close(ticker, ref_date_str=None):
    if not _POLYGON_OK:
        return None
    ref = ref_date_str or datetime.now(PST).strftime('%Y-%m-%d')
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
                     if pd.Timestamp(b.timestamp, unit='ms', tz='UTC').date() < ref_dt.date()]
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
    ref = ref_date_str or datetime.now(PST).strftime('%Y-%m-%d')
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
    ref = ref_date_str or datetime.now(PST).strftime('%Y-%m-%d')
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
                     if pd.Timestamp(b.timestamp, unit='ms', tz='UTC').date() < ref_dt.date()]
        if not prev_bars:
            return None
        t_minus_1 = prev_bars[-1]
        prev_close = float(t_minus_1.close)
        t_minus_1_date = pd.Timestamp(t_minus_1.timestamp, unit='ms', tz='UTC').date()
        prev_close_before = float(prev_bars[-2].close) if len(prev_bars) >= 2 else prev_close
        t_minus_1_from = t_minus_1_date.strftime('%Y-%m-%d')
        minute_bars = list(_polygon_client.get_aggs(
            ticker=ticker, multiplier=5, timespan="minute",
            from_=t_minus_1_from, to=ref, limit=500,
        ))
        if not minute_bars:
            return None
        threshold = move_threshold_pct / 100.0
        event_time = None
        event_price = None
        peak_price = prev_close
        peak_time = None
        trough_price = prev_close
        trough_time = None
        current_price = prev_close
        ref_close = prev_close
        for bar in minute_bars:
            ts = pd.Timestamp(bar.timestamp, unit='ms', tz='UTC').tz_convert(ET)
            bar_date = ts.date()
            ref_close = prev_close_before if bar_date == t_minus_1_date else prev_close
            close_val = float(bar.close)
            high_val = float(bar.high)
            low_val = float(bar.low)
            pct_change = abs(close_val - ref_close) / ref_close if ref_close > 0 else 0
            if pct_change >= threshold and event_time is None:
                event_time = ts
                event_price = close_val
            if high_val > peak_price:
                peak_price = high_val
                peak_time = ts
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
    except Exception:
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
        return (True, ticker)

def fetch_sec_filing_time(ticker, filing_date):
    _headers = {'User-Agent': 'ScannerBot admin@example.com'}
    try:
        pg_url = f"https://api.polygon.io/v3/reference/tickers/{ticker}"
        pg_resp = requests.get(pg_url, params={'apiKey': POLYGON_API_KEY}, timeout=5)
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
        sec_resp = requests.get(sec_url, headers=_headers, timeout=10)
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
        if form == '8-K' and fdate == filing_date and acc_dt:
            try:
                utc_dt = datetime.strptime(acc_dt[:19], '%Y-%m-%dT%H:%M:%S')
                utc_dt = pytz.utc.localize(utc_dt)
                et_dt = utc_dt.astimezone(ET)
                return et_dt.strftime('%Y-%m-%d %I:%M %p ET')
            except Exception:
                return None
    return None

def fetch_price_data(ticker, today_str=None):
    if today_str is None:
        today_str = datetime.now(PST).strftime('%Y-%m-%d')
    yesterday_str = (datetime.strptime(today_str, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
    if _POLYGON_OK:
        try:
            prev = get_prev_session_close(ticker, today_str)
            if prev and prev['close'] > 0:
                prev_close = prev['close']
                bars_1m = list(_polygon_client.get_aggs(
                    ticker=ticker, multiplier=1, timespan="minute",
                    from_=today_str, to=today_str, limit=500,
                ))
                premarket_volume = 0
                if bars_1m:
                    current_price = float(bars_1m[-1].close)
                    premarket_volume = sum(int(getattr(b, 'volume', 0) or 0) for b in bars_1m)
                else:
                    bars_5m = list(_polygon_client.get_aggs(
                        ticker=ticker, multiplier=5, timespan="minute",
                        from_=prev['date'], to=today_str, limit=500,
                    ))
                    current_price = float(bars_5m[-1].close) if bars_5m else prev_close
                change_pct = ((current_price - prev_close) / prev_close) * 100
                avg_daily_volume = 0
                try:
                    _from_20d = (datetime.strptime(today_str, '%Y-%m-%d') - timedelta(days=25)).strftime('%Y-%m-%d')
                    daily_bars = list(_polygon_client.get_aggs(
                        ticker=ticker, multiplier=1, timespan="day",
                        from_=_from_20d, to=yesterday_str, limit=10,
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
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        hist = t.history(period='2d', prepost=True)
        if len(hist) >= 2:
            current_price = round(float(hist['Close'].iloc[-1]), 2)
            prev_close = round(float(hist['Close'].iloc[-2]), 2)
            premarket_volume = int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0
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
    except Exception:
        pass
    return None


# ==============================================================================
# PATTERN LIBRARY
# ==============================================================================
_PATTERN_CACHE_DF = None

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
        matched = df[mask].head(max_results)
        if matched.empty:
            return ""
        lines = ["\n\nHISTORICAL PATTERN EXAMPLES (from verified past trades):"]
        for _, p in matched.iterrows():
            lines.append(f"\n  PATTERN: {p['pattern_type']} ({p['event_date']})")
            lines.append(f"  Tickers: {p['tickers']} | Catalyst: {p['catalyst_type']}")
            lines.append(f"  {str(p['catalyst_summary'])[:200]}")
            lines.append(f"  -> {p['primary_ticker']} ({p['primary_direction']}): "
                         f"gap {p['primary_gap_pct']:+.1f}%, 60m {p['primary_ret_60m']:+.1f}%, "
                         f"MFE {p['primary_mfe_pct']:+.1f}%, EOD {p['primary_eod_pct']:+.1f}% -> {p['primary_outcome']}")
            lines.append(f"  LESSON: {str(p['lesson'])[:200]}")
        return "\n".join(lines)
    except Exception:
        return ""


# ==============================================================================
# LLM API CALLS
# ==============================================================================
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
            return {'error': f'parse_failed: {str(text)[:100]}', 'raw': text, '_sources': sources}
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
            "catalyst": {"type": "string"},
            "catalyst_type": {"type": "string", "enum": ["earnings", "analyst", "regulatory", "deal", "offering", "insider", "macro", "breaking", "premarket_move"]},
            "catalyst_is_new": {"type": "boolean"},
            "news_timeline": {"type": "array", "items": {"type": "object", "properties": {"time": {"type": "string"}, "event": {"type": "string"}}, "required": ["time", "event"]}},
            "timing": {"type": "string", "enum": ["breaking", "overnight", "pre_earnings", "post_earnings", "developing", "scheduled"]},
            "reasoning": {"type": "string"},
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
            "total_catalyst_impact_pct": {"type": "number"},
            "move_already_absorbed_pct": {"type": "number"},
            "remaining_move_pct": {"type": "number"},
            "expected_move_pct": {"type": "number"},
            "move_is_exhausted": {"type": "boolean"},
            "direction": {"type": "string", "enum": ["LONG", "SHORT"]},
            "confidence": {"type": "integer"},
            "catalyst_strength": {"type": "integer"},
            "freshness": {"type": "integer"},
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
# EVENT TIMING VERIFICATION
# ==============================================================================
def verify_and_correct_event_timing(entry, today_str, now_et, yesterday_str):
    ticker = entry.get('ticker', '')
    if not ticker or not _POLYGON_OK:
        return
    cat_type = str(entry.get('catalyst_type', '')).lower()
    sec_types = {'earnings', 'deal', 'regulatory', 'offering'}
    if cat_type in sec_types:
        print(f"    [{ticker}] SEC 8-K lookup for {cat_type}...", end=' ')
        sec_time = fetch_sec_filing_time(ticker, today_str)
        if sec_time:
            entry['news_time_original'] = entry.get('news_time', '')
            entry['news_time'] = sec_time
            entry['news_time_source'] = 'sec_edgar'
            print(f"found: {sec_time}")
        else:
            sec_time_prev = fetch_sec_filing_time(ticker, yesterday_str)
            if sec_time_prev:
                entry['news_time_original'] = entry.get('news_time', '')
                entry['news_time'] = sec_time_prev
                entry['news_time_source'] = 'sec_edgar'
                print(f"found (yesterday): {sec_time_prev}")
            else:
                print("no 8-K found")
    else:
        print(f"    [{ticker}] Verifying event timing via Polygon...", end=' ')

    event_info = get_actual_event_time(ticker, today_str)
    if not event_info:
        if cat_type not in sec_types:
            print("no spike found")
        return

    event_dt = event_info['event_time_dt']
    event_age_hours = (now_et - event_dt).total_seconds() / 3600

    if entry.get('news_time_source') not in ('sec_edgar',):
        news_time_raw = str(entry.get('news_time', ''))
        article_dt = _parse_news_time(news_time_raw)
        article_age_hours = None
        if article_dt:
            article_age_hours = (now_et - article_dt.astimezone(ET)).total_seconds() / 3600
        if article_age_hours is not None and event_age_hours > article_age_hours + 1.0:
            _today_et = now_et.date()
            _event_is_overnight_ah = (
                event_dt.date() < _today_et and event_dt.hour >= 16 and event_age_hours <= 20
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


# ==============================================================================
# STAGE 3: T+2 FILTER + SELECTION
# ==============================================================================
def fetch_pm_bars_for_candidates(tickers, date_str):
    results = {}
    for ticker in tickers:
        try:
            bars = list(_polygon_client.get_aggs(
                ticker=ticker, multiplier=1, timespan="minute",
                from_=date_str, to=date_str, limit=500,
                adjusted=True, sort='asc',
            ))
            pm_bars = [
                b for b in bars
                if pd.Timestamp(b.timestamp, unit='ms', tz='UTC')
                   .tz_convert(ET).hour < 9
                or (pd.Timestamp(b.timestamp, unit='ms', tz='UTC')
                    .tz_convert(ET).hour == 9
                    and pd.Timestamp(b.timestamp, unit='ms', tz='UTC')
                    .tz_convert(ET).minute < 30)
            ]
            if not pm_bars:
                results[ticker] = {'pm_high': None, 'pm_last': None, 'pm_retrace_pct': None}
                continue
            pm_open = float(pm_bars[0].open)
            pm_high = max(float(b.high) for b in pm_bars)
            pm_last = float(pm_bars[-1].close)
            total_move = pm_high - pm_open
            pm_retrace = (pm_high - pm_last) / total_move if total_move > 0 else 0.0
            results[ticker] = {
                'pm_high': round(pm_high, 2),
                'pm_last': round(pm_last, 2),
                'pm_retrace_pct': round(pm_retrace, 3),
            }
        except Exception as e:
            print(f"  [PM BARS] {ticker}: {e}")
            results[ticker] = {'pm_high': None, 'pm_last': None, 'pm_retrace_pct': None}
    return results


def apply_t2_filters(candidates, pm_data):
    for c in candidates:
        c['passed_direction'] = (c.get('direction') == 'SHORT')
        if not c['passed_direction']:
            c['filter_fail'] = 'direction_not_short'
            c['passed_all'] = False
            c['passed_exhausted'] = None
            c['passed_gap'] = None
            c['passed_catalyst'] = None
            c['passed_retrace'] = None
            continue

        if c.get('move_is_exhausted'):
            c['passed_exhausted'] = False
            c['filter_fail'] = 'move_exhausted'
            c['passed_all'] = False
            c['passed_gap'] = None
            c['passed_catalyst'] = None
            c['passed_retrace'] = None
            continue
        c['passed_exhausted'] = True

        gap = abs(c.get('change_pct', 0) or 0) / 100.0
        c['gap_pct'] = gap
        c['passed_gap'] = gap >= GAP_MIN_T3
        if not c['passed_gap']:
            c['filter_fail'] = 'gap_too_small'
            c['passed_all'] = False
            c['passed_catalyst'] = None
            c['passed_retrace'] = None
            continue

        cat = str(c.get('catalyst_type', 'unknown')).lower()
        squeeze = c.get('squeeze_risk', False)
        c['passed_catalyst'] = (cat not in CATALYST_EXCLUDE and not squeeze)
        if not c['passed_catalyst']:
            c['filter_fail'] = 'catalyst_excluded' if cat in CATALYST_EXCLUDE else 'squeeze_risk'
            c['passed_all'] = False
            c['passed_retrace'] = None
            continue

        pm = pm_data.get(c['ticker'], {})
        retrace = pm.get('pm_retrace_pct')
        c['pm_retrace_pct'] = retrace
        c['pm_high'] = pm.get('pm_high')
        c['pm_last'] = pm.get('pm_last')
        c['passed_retrace'] = (retrace is None or retrace <= PM_RETRACE_MAX)
        if not c['passed_retrace']:
            c['filter_fail'] = 'pm_retrace_too_high'
            c['passed_all'] = False
            continue

        c['passed_all'] = True
        c['filter_fail'] = None

        if cat in CATALYST_T1 and retrace is not None and 0.20 <= retrace <= 0.80:
            c['agent_tier'] = 'T1'
        elif gap >= GAP_MIN_T1T2:
            c['agent_tier'] = 'T2'
        else:
            c['agent_tier'] = 'T3'

    return candidates


def select_top2(candidates):
    qualified = [c for c in candidates if c.get('passed_all')]
    t1 = sorted([c for c in qualified if c.get('agent_tier') == 'T1'],
                key=lambda x: x.get('gap_pct', 0), reverse=True)
    t2 = sorted([c for c in qualified if c.get('agent_tier') == 'T2'],
                key=lambda x: x.get('gap_pct', 0), reverse=True)
    t3 = sorted([c for c in qualified if c.get('agent_tier') == 'T3'],
                key=lambda x: x.get('gap_pct', 0), reverse=True)
    selected = (t1 + t2 + t3)[:TARGET_TRADES]
    for i, c in enumerate(selected):
        c['selected'] = True
        c['selection_rank'] = i + 1
    return selected


# ==============================================================================
# QUALITY GATES (from discovery scanner V3)
# ==============================================================================
SELF_CONTRADICTION_PHRASES = [
    'untradeable', 'zero remaining edge', 'zero edge',
    'not a new actionable catalyst', 'no new catalyst',
    'no intraday catalyst', 'pure gamma play',
    'virtually untradeable', 'no remaining upside',
    'zero remaining', 'not yet reported', 'very low-conviction',
    'contradicts the', 'immaterial', 'no remaining edge',
]
PRE_EARNINGS_KILL_PHRASES = [
    'expected to report', 'not yet reported', 'after market close',
    'after the close', 'after close today', 'earnings not yet',
    'has not yet reported', 'report expected after', 'reports after',
]

BAYESIAN_K = 30
_CATALYST_RAW = {
    'macro': (1.2, 3), 'breaking': (1.2, 12), 'insider': (1.2, 5),
    'offering': (1.0, 8), 'analyst': (1.0, 15), 'squeeze': (1.0, 0),
    'halt': (1.0, 0), 'premarket_move': (0.5, 9), 'earnings': (0.7, 24),
    'deal': (0.6, 25), 'regulatory': (0.4, 10), 'bankruptcy': (0.5, 0),
}
CATALYST_TYPE_WEIGHTS = {
    k: round(1.0 + (w - 1.0) * n / (n + BAYESIAN_K), 3)
    for k, (w, n) in _CATALYST_RAW.items()
}


def apply_quality_gates(entries, market_ctx, gap_context, now_et, now_pst):
    n_gated = 0
    for e in entries:
        gate_reasons = []
        reasoning_lower = str(e.get('reasoning', '')).lower()
        catalyst_lower = str(e.get('catalyst', '')).lower()
        direction = e.get('direction', '')
        _pd = e.get('_price_data', {})
        current_price = float(_pd.get('current_price', 0)) if _pd else 0

        if current_price > 0:
            min_p = MIN_PRICE_LONG if direction == 'LONG' else MIN_PRICE_SHORT
            if current_price < min_p:
                gate_reasons.append(f'PENNY_STOCK (${current_price:.2f} < ${min_p:.2f})')

        _pm_vol = int(_pd.get('premarket_volume', 0)) if _pd else 0
        if current_price > 0 and 0 < _pm_vol < MIN_PREMARKET_VOLUME:
            gate_reasons.append(f'LOW_LIQUIDITY (vol: {_pm_vol:,})')

        for phrase in SELF_CONTRADICTION_PHRASES:
            if phrase in reasoning_lower:
                gate_reasons.append('SELF_CONTRADICTION')
                break

        for phrase in PRE_EARNINGS_KILL_PHRASES:
            if phrase in catalyst_lower or phrase in reasoning_lower:
                gate_reasons.append('PRE_EARNINGS')
                break

        if e.get('confidence', 0) < MIN_CONFIDENCE_FOR_SELECTION:
            gate_reasons.append(f'LOW_CONF ({e.get("confidence", 0)})')

        _cat_type = str(e.get('catalyst_type', '')).lower()
        if _cat_type == 'deal' and not gate_reasons:
            _deal_mat = str(e.get('deal_materiality', '')).lower()
            if _deal_mat in ('immaterial', 'low', 'minimal'):
                gate_reasons.append(f'IMMATERIAL_DEAL (materiality={_deal_mat})')

        if _cat_type == 'analyst' and not gate_reasons:
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
                _vague_phrases = ['multiple analysts', 'multiple upgrades', 'several analysts',
                                  'analyst consensus', 'wall street', 'analysts']
                if any(vp in _cat_text.lower() for vp in _vague_phrases):
                    gate_reasons.append('VAGUE_ANALYST (no named firm)')

        _timeline = e.get('news_timeline', [])
        if isinstance(_timeline, list) and _timeline and not gate_reasons:
            _earliest_time_str = None
            for _ev in _timeline:
                if isinstance(_ev, dict):
                    _evt = str(_ev.get('time', ''))
                    if _evt and (not _earliest_time_str or _evt < _earliest_time_str):
                        _earliest_time_str = _evt
            if _earliest_time_str:
                _orig_dt = _parse_news_time(_earliest_time_str)
                if _orig_dt:
                    _orig_age_h = (now_et - _orig_dt.astimezone(ET)).total_seconds() / 3600
                    if _orig_age_h > 36:
                        gate_reasons.append(f'STALE_EVENT (original event {_orig_age_h:.0f}h ago)')

        if gate_reasons:
            e['gate_reason'] = '; '.join(gate_reasons)
            e['confidence'] = min(e.get('confidence', 0), MIN_CONFIDENCE_FOR_SELECTION - 1)
            n_gated += 1
        else:
            e['gate_reason'] = ''

    return entries, n_gated


def apply_score_adjustments(entries, now_pst, market_ctx):
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

    for e in entries:
        fm = _freshness_multiplier(e)
        e['tradability_score'] = e.get('tradability_score', 0) * fm
        e['freshness_mult'] = fm

        cat = str(e.get('catalyst_type', '')).lower()
        cw = CATALYST_TYPE_WEIGHTS.get(cat, 1.0)
        e['tradability_score'] = e['tradability_score'] * cw
        e['catalyst_weight'] = cw

        _pd = e.get('_price_data') or {}
        gap_pct = abs(float(_pd.get('change_pct', 0)))
        _strength = max(int(e.get('catalyst_strength', e.get('confidence', 5))), 1)
        mismatch = gap_pct / _strength
        if mismatch > 2.0 and e.get('direction') == 'SHORT':
            e['mismatch_mult'] = 1.5
        elif mismatch > 2.0 and e.get('direction') == 'LONG':
            e['mismatch_mult'] = 0.5
        elif mismatch < 0.5 and e.get('direction') == 'LONG':
            e['mismatch_mult'] = 1.3
        else:
            e['mismatch_mult'] = 1.0
        e['tradability_score'] = e['tradability_score'] * e['mismatch_mult']
        e['mismatch_index'] = round(mismatch, 2)

        pm_vol = float(_pd.get('premarket_volume', 0))
        avg_vol = float(_pd.get('avg_daily_volume', 0))
        if avg_vol > 0 and pm_vol > 0:
            vol_ratio = pm_vol / avg_vol
            vol_mult = 1.3 if vol_ratio >= 3.0 else (1.1 if vol_ratio >= 1.0 else (0.7 if vol_ratio < 0.3 else 1.0))
        else:
            vol_ratio = 0.0
            vol_mult = 1.0
        e['tradability_score'] = e['tradability_score'] * vol_mult
        e['volume_ratio'] = round(vol_ratio, 2)
        e['volume_mult'] = vol_mult

    return entries


# ==============================================================================
# DB WRITE
# ==============================================================================
def write_to_db(scan_date_str, all_entries, selected, run_meta):
    conn = _get_conn()
    now_utc = datetime.now(pytz.UTC).isoformat()
    t1, t2 = get_t1_t2_dates(scan_date_str)

    conn.execute("""INSERT OR REPLACE INTO daily_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (scan_date_str, now_utc, run_meta.get('runtime', 0),
         run_meta.get('n_themes', 0), run_meta.get('n_discovered', 0),
         run_meta.get('n_deep', 0), run_meta.get('n_qualified', 0),
         len(selected), run_meta.get('futures', ''),
         run_meta.get('risk', ''), run_meta.get('vix', ''),
         SCANNER_VERSION))

    for c in all_entries:
        cid = f"{scan_date_str}_{c.get('ticker', 'UNK')}"
        _pd = c.get('_price_data') or {}
        conn.execute("""INSERT OR REPLACE INTO candidates VALUES
            (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (cid, scan_date_str, c.get('ticker'), c.get('company', ''),
             c.get('direction'), c.get('confidence', 0),
             c.get('expected_move_pct', 0), c.get('remaining_move_pct', 0),
             c.get('move_already_priced_pct', 0),
             1 if c.get('move_is_exhausted') else 0,
             c.get('catalyst_strength', 0), c.get('freshness', 0),
             c.get('liquidity_risk', 0), c.get('tradability_score', 0),
             c.get('catalyst_type', ''), str(c.get('catalyst', ''))[:500],
             str(c.get('reasoning', ''))[:1000],
             str(c.get('headline', ''))[:500], c.get('news_time', ''),
             c.get('theme', ''), c.get('market_cap_tier', ''),
             c.get('gap_pct'), float(_pd.get('change_pct', 0)) if _pd else None,
             float(_pd.get('prev_close', 0)) if _pd else None,
             float(_pd.get('current_price', 0)) if _pd else None,
             c.get('pm_high'), c.get('pm_last'), c.get('pm_retrace_pct'),
             c.get('agent_tier'),
             1 if c.get('passed_direction') else (0 if c.get('passed_direction') is not None else None),
             1 if c.get('passed_exhausted') else (0 if c.get('passed_exhausted') is not None else None),
             1 if c.get('passed_gap') else (0 if c.get('passed_gap') is not None else None),
             1 if c.get('passed_catalyst') else (0 if c.get('passed_catalyst') is not None else None),
             1 if c.get('passed_retrace') else (0 if c.get('passed_retrace') is not None else None),
             1 if c.get('passed_all') else 0,
             c.get('filter_fail'), c.get('gate_reason', ''),
             1 if c.get('selected') else 0, c.get('selection_rank'),
             t1, t2,
             None, None, None,
             None, None, None,
             None, None, None,
             SCANNER_VERSION, now_utc))

    for s in selected:
        tid = f"{scan_date_str}_{s['ticker']}"
        conn.execute("""INSERT OR REPLACE INTO trades
            (trade_id, run_date, ticker, selection_rank, direction,
             confidence, gap_pct, agent_tier, catalyst_type, catalyst,
             tradability_score, pm_retrace_pct, t1_date, t2_date)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (tid, scan_date_str, s['ticker'], s.get('selection_rank'),
             s.get('direction'), s.get('confidence'),
             s.get('gap_pct'), s.get('agent_tier'),
             s.get('catalyst_type'), str(s.get('catalyst', ''))[:500],
             s.get('tradability_score'), s.get('pm_retrace_pct'),
             t1, t2))

    conn.commit()
    conn.close()
    log.info(f"DB write: {len(all_entries)} candidates, {len(selected)} trades")


# ==============================================================================
# BQ LOGGING
# ==============================================================================
def log_to_bq(all_entries, selected, market_ctx, scan_date_str, time_str):
    try:
        bq_client = bigquery.Client()
        bq_client.query(f"""
        CREATE TABLE IF NOT EXISTS `{BQ_TABLE}` (
            scan_timestamp TIMESTAMP, scan_date DATE, scan_time_pt STRING,
            ticker STRING, company STRING, direction STRING,
            rank_in_direction INT64, selected BOOL,
            confidence INT64, expected_move_pct FLOAT64,
            remaining_move_pct FLOAT64, move_is_exhausted BOOL,
            catalyst_strength INT64, tradability_score FLOAT64,
            catalyst_type STRING, catalyst STRING, reasoning STRING,
            news_time STRING, theme STRING, market_cap_tier STRING,
            gap_pct FLOAT64, agent_tier STRING,
            passed_all BOOL, filter_fail STRING, gate_reason STRING,
            pm_retrace_pct FLOAT64,
            market_futures STRING, market_risk STRING, vix_level STRING,
            scanner_version STRING
        )""").result()
    except Exception as e:
        log.warning(f"BQ table creation: {e}")

    rows = []
    now_utc = datetime.now(pytz.utc).isoformat()
    futures = str(market_ctx.get('futures', ''))[:500] if market_ctx else ''
    risk = str(market_ctx.get('risk_sentiment', ''))[:200] if market_ctx else ''
    vix = str(market_ctx.get('vix_level', ''))[:50] if market_ctx else ''
    selected_tickers = {s['ticker'] for s in selected}

    for e in all_entries:
        rows.append({
            'scan_timestamp': now_utc,
            'scan_date': scan_date_str,
            'scan_time_pt': time_str,
            'ticker': e.get('ticker', ''),
            'company': str(e.get('company', ''))[:200],
            'direction': e.get('direction', ''),
            'rank_in_direction': e.get('selection_rank', 0),
            'selected': e.get('ticker', '') in selected_tickers,
            'confidence': e.get('confidence', 0),
            'expected_move_pct': e.get('expected_move_pct', 0),
            'remaining_move_pct': e.get('remaining_move_pct', 0),
            'move_is_exhausted': e.get('move_is_exhausted', False),
            'catalyst_strength': e.get('catalyst_strength', 0),
            'tradability_score': e.get('tradability_score', 0),
            'catalyst_type': str(e.get('catalyst_type', ''))[:100],
            'catalyst': str(e.get('catalyst', ''))[:500],
            'reasoning': str(e.get('reasoning', ''))[:2000],
            'news_time': str(e.get('news_time', ''))[:100],
            'theme': str(e.get('theme', ''))[:50],
            'market_cap_tier': str(e.get('market_cap_tier', ''))[:20],
            'gap_pct': e.get('gap_pct'),
            'agent_tier': e.get('agent_tier'),
            'passed_all': bool(e.get('passed_all')),
            'filter_fail': str(e.get('filter_fail', ''))[:200],
            'gate_reason': str(e.get('gate_reason', ''))[:500],
            'pm_retrace_pct': e.get('pm_retrace_pct'),
            'market_futures': futures,
            'market_risk': risk,
            'vix_level': vix,
            'scanner_version': SCANNER_VERSION,
        })

    if rows:
        try:
            errors = bq_client.insert_rows_json(BQ_TABLE, rows)
            if errors:
                log.warning(f"BQ insert errors: {errors[:3]}")
            else:
                log.info(f"BQ logged: {len(rows)} rows -> {BQ_TABLE}")
        except Exception as e:
            log.warning(f"BQ logging failed: {e}")


# ==============================================================================
# FILL OUTCOMES
# ==============================================================================
def fill_outcomes(fill_date_str=None):
    if fill_date_str is None:
        fill_date_str = datetime.now(ET).strftime('%Y-%m-%d')
    log.info(f"FILL OUTCOMES — data available through {fill_date_str}")
    conn = _get_conn()

    trades = conn.execute("""
        SELECT t.trade_id, t.run_date, t.ticker, t.t1_date, t.t2_date,
               COALESCE(t.entry_price, c.current_price) as ref_price,
               t.t0_close, t.t1_close, t.t2_close
        FROM trades t
        LEFT JOIN candidates c ON t.run_date = c.run_date AND t.ticker = c.ticker
        WHERE t.t0_close IS NULL OR t.t1_close IS NULL OR t.t2_close IS NULL
    """).fetchall()

    open_candidates = conn.execute("""
        SELECT candidate_id, run_date, ticker, t1_date, t2_date,
               current_price, t0_close, t1_close, t2_close
        FROM candidates
        WHERE (t0_close IS NULL OR t1_close IS NULL OR t2_close IS NULL)
          AND passed_all = 1
    """).fetchall()

    all_items = []
    for t in trades:
        all_items.append(('trades', t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8]))
    for c in open_candidates:
        all_items.append(('candidates', c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8]))

    if not all_items:
        log.info("  No items need filling.")
        conn.close()
        return

    log.info(f"  {len(trades)} trades + {len(open_candidates)} candidates need outcome data")

    dates_needed = set()
    for item in all_items:
        _, _, scan_date, _, t1_date, t2_date, _, _, _, _ = item
        dates_needed.add(scan_date)
        if t1_date:
            dates_needed.add(t1_date)
        if t2_date:
            dates_needed.add(t2_date)

    daily_data = {}
    for dt in dates_needed:
        if dt > fill_date_str:
            continue
        try:
            url = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{dt}"
            resp = requests.get(url, params={'adjusted': 'true', 'apiKey': POLYGON_API_KEY}, timeout=30)
            results = resp.json().get('results', [])
            daily_data[dt] = {r['T']: {'close': float(r.get('c', 0))} for r in results if 'T' in r}
            log.info(f"  Grouped daily ({dt}): {len(daily_data[dt])} tickers")
            time.sleep(0.05)
        except Exception as e:
            log.warning(f"  Failed to fetch {dt}: {e}")

    n_filled = 0
    for item in all_items:
        table, item_id, scan_date, ticker, t1_date, t2_date, ref_price, t0c, t1c, t2c = item
        if ref_price is None or ref_price <= 0:
            continue
        updates = {}
        if t0c is None and scan_date in daily_data:
            close_val = daily_data[scan_date].get(ticker, {}).get('close', 0)
            if close_val and close_val > 0:
                updates['t0_close'] = close_val
                updates['pnl_eod'] = round((ref_price - close_val) / ref_price * 100, 4)
        if t1c is None and t1_date and t1_date in daily_data:
            close_val = daily_data[t1_date].get(ticker, {}).get('close', 0)
            if close_val and close_val > 0:
                updates['t1_close'] = close_val
                updates['pnl_t1'] = round((ref_price - close_val) / ref_price * 100, 4)
        if t2c is None and t2_date and t2_date in daily_data:
            close_val = daily_data[t2_date].get(ticker, {}).get('close', 0)
            if close_val and close_val > 0:
                updates['t2_close'] = close_val
                updates['pnl_t2'] = round((ref_price - close_val) / ref_price * 100, 4)
        if updates:
            set_clause = ', '.join(f"{k}=?" for k in updates.keys())
            id_col = 'trade_id' if table == 'trades' else 'candidate_id'
            conn.execute(f"UPDATE {table} SET {set_clause} WHERE {id_col}=?",
                         list(updates.values()) + [item_id])
            n_filled += 1

    conn.commit()
    conn.close()
    log.info(f"  Filled {n_filled} items")


# ==============================================================================
# REPORT
# ==============================================================================
def run_report(weeks=REPORT_WEEKS):
    conn = _get_conn()
    cutoff = (datetime.now() - timedelta(weeks=weeks)).strftime('%Y-%m-%d')

    runs = conn.execute("SELECT COUNT(*), AVG(runtime_seconds) FROM daily_runs WHERE run_date >= ?",
                        (cutoff,)).fetchone()
    n_runs = runs[0] or 0
    avg_runtime = runs[1] or 0

    cand_stats = conn.execute("""
        SELECT COUNT(*), SUM(passed_all), SUM(selected)
        FROM candidates WHERE run_date >= ?
    """, (cutoff,)).fetchone()

    trades = conn.execute("""
        SELECT ticker, run_date, direction, agent_tier, catalyst_type,
               entry_type, pnl_eod, pnl_t1, pnl_t2
        FROM trades WHERE run_date >= ?
        ORDER BY run_date
    """, (cutoff,)).fetchall()

    print(f"\n  {'='*66}")
    print(f"  |{'AGENT 2 — PERFORMANCE REPORT':^64}|")
    print(f"  {'='*66}")
    print(f"\n  Period: Last {weeks} weeks (cutoff: {cutoff})")
    print(f"  Total runs: {n_runs}  |  Avg runtime: {avg_runtime:.1f}s")

    if cand_stats and cand_stats[0]:
        total = cand_stats[0]
        qualified = cand_stats[1] or 0
        selected = cand_stats[2] or 0
        print(f"\n  === SIGNAL VOLUME ===")
        print(f"  Total signals processed:    {total}")
        print(f"  Passed T+2 filters:         {qualified}")
        print(f"  Selected for trade:         {selected}")

    pnl_t2 = [t[8] for t in trades if t[8] is not None]
    if pnl_t2:
        arr = np.array(pnl_t2)
        n = len(arr)
        avg = float(np.mean(arr))
        med = float(np.median(arr))
        wr = float((arr > 0).mean())
        std = float(np.std(arr, ddof=1)) if n > 1 else 0
        se = std / np.sqrt(n) if n > 0 else 0
        print(f"\n  === TRADE PERFORMANCE ===")
        print(f"  Trades with T+2:           {n}")
        print(f"  Avg PnL T+2:               {avg:+.2f}%")
        print(f"  Median PnL T+2:            {med:+.2f}%")
        print(f"  Win Rate:                   {wr*100:.0f}%")
        print(f"  CI 95%:                     [{avg-1.96*se:+.2f}%, {avg+1.96*se:+.2f}%]")
    else:
        print(f"\n  No T+2 outcomes yet.")

    bt_rows = conn.execute("""
        SELECT ticker, run_date, agent_tier, catalyst_type, gap_pct, pnl_t2
        FROM candidates
        WHERE run_date >= ? AND passed_all = 1 AND pnl_t2 IS NOT NULL
        ORDER BY run_date
    """, (cutoff,)).fetchall()

    if bt_rows:
        bt_pnl = np.array([r[5] for r in bt_rows])
        n = len(bt_pnl)
        avg = float(np.mean(bt_pnl))
        med = float(np.median(bt_pnl))
        wr = float((bt_pnl > 0).mean()) * 100
        std = float(np.std(bt_pnl, ddof=1)) if n > 1 else 0
        se = std / np.sqrt(n)
        trimmed = np.sort(bt_pnl)[max(1, n//10): n - max(1, n//10)]
        t_avg = float(np.mean(trimmed)) if len(trimmed) > 0 else avg
        print(f"\n  === CANDIDATE BACKTESTING (current_price -> T+2 short PnL) ===")
        print(f"  All qualified    n={n:<4d}  avg={avg:+.2f}%  med={med:+.2f}%  "
              f"trim={t_avg:+.2f}%  WR={wr:.0f}%  CI=[{avg-1.96*se:+.2f},{avg+1.96*se:+.2f}]")
        for tier in ['T1', 'T2', 'T3']:
            tier_arr = np.array([r[5] for r in bt_rows if r[2] == tier])
            if len(tier_arr) > 0:
                print(f"    {tier:4s}  n={len(tier_arr):<4d}  avg={np.mean(tier_arr):+.2f}%  "
                      f"WR={float((tier_arr>0).mean())*100:.0f}%")

    if trades:
        print(f"\n  === TRADE LOG ===")
        rows = []
        for t in trades:
            rows.append([t[1], t[0], t[3] or '?', t[4] or '—',
                         f"{t[6]:+.2f}%" if t[6] is not None else '—',
                         f"{t[8]:+.2f}%" if t[8] is not None else '—'])
        print()
        print(tabulate(rows, headers=['Date', 'Ticker', 'Tier', 'Catalyst', 'EOD', 'T+2'],
                       tablefmt='simple'))

    conn.close()
    print()


# ==============================================================================
# LOG ENTRY
# ==============================================================================
def log_entry(ticker, entry_price, entry_time=None, entry_type=None, size_usd=None, notes=None):
    conn = _get_conn()
    today = datetime.now(ET).strftime('%Y-%m-%d')
    tid = f"{today}_{ticker}"
    updates = {'entry_price': entry_price, 'actually_traded': 1}
    if entry_time:
        updates['entry_time'] = entry_time
    if entry_type:
        updates['entry_type'] = entry_type
    if size_usd:
        updates['position_size_usd'] = size_usd
    if notes:
        updates['notes'] = notes
    set_clause = ', '.join(f"{k}=?" for k in updates.keys())
    conn.execute(f"UPDATE trades SET {set_clause} WHERE trade_id=?",
                 list(updates.values()) + [tid])
    conn.commit()
    row = conn.execute("SELECT * FROM trades WHERE trade_id=?", (tid,)).fetchone()
    conn.close()
    if row:
        log.info(f"Logged entry: {ticker} @ ${entry_price} ({entry_type or 'unspecified'})")
    else:
        log.warning(f"Trade {tid} not found in database. Run scan first.")


# ==============================================================================
# WATCHLIST OUTPUT
# ==============================================================================
def print_watchlist(selected, all_entries, scan_date_str):
    t1, t2 = get_t1_t2_dates(scan_date_str)
    now_str = datetime.now(PST).strftime('%H:%M:%S')

    print(f"\n  {'='*60}")
    print(f"  |  AGENT 2 WATCHLIST — {scan_date_str:<38}|")
    print(f"  |  Run: {now_str} PT  |  T+2 Exit: {t2 or 'TBD'} close{' '*14}|")
    print(f"  {'='*60}")

    if not selected:
        print(f"\n  No qualifying SHORT trades found.\n")
    else:
        print(f"\n  TOP SHORT FADE CANDIDATES (T+2 hold)\n")
        rows = []
        for s in selected:
            cat = str(s.get('catalyst_type', ''))[:16]
            ret = f"{s.get('pm_retrace_pct', 0):.2f}" if s.get('pm_retrace_pct') is not None else 'n/a'
            gap = s.get('gap_pct', 0) * 100 if s.get('gap_pct') else abs(float((s.get('_price_data') or {}).get('change_pct', 0)))
            rows.append([
                f"#{s.get('selection_rank', '?')}",
                s['ticker'],
                f"+{gap:.1f}%",
                s.get('agent_tier', '?'),
                cat,
                ret,
                s.get('confidence', 0),
            ])
        print(tabulate(rows, headers=['Rank', 'Ticker', 'Gap%', 'Tier', 'Catalyst', 'PM Ret', 'Conf'],
                       tablefmt='simple'))

        print(f"\n  ENTRY RULES:")
        print(f"    -> Watch for local high (ThinkScript alert)")
        print(f"    -> Enter SHORT on confirmation")
        print(f"    -> Fallback: enter SHORT at 9:45 AM ET if no local high")
        print(f"    -> Skip if down >15% from open by 9:45 AM ET")
        print(f"    Hold: T+2 close ({t2 or 'TBD'}) — no stop loss\n")

        for s in selected:
            _pd = s.get('_price_data') or {}
            cat = s.get('catalyst_type', '—')
            catalyst = str(s.get('catalyst', ''))
            if len(catalyst) > 80:
                catalyst = catalyst[:78] + '..'
            print(f"  #{s.get('selection_rank', '?')} {s['ticker']}  ({s.get('agent_tier', '?')})")
            gap_pct_val = s.get('gap_pct', 0) * 100 if s.get('gap_pct') else abs(float(_pd.get('change_pct', 0)))
            print(f"     Gap: +{gap_pct_val:.1f}%  |  Conf: {s.get('confidence', 0)}/10")
            if s.get('pm_retrace_pct') is not None:
                ret = s['pm_retrace_pct']
                note = "held" if ret < 0.2 else ("sweet spot" if ret <= 0.5 else ("moderate" if ret <= 0.8 else "full"))
                print(f"     PM retrace: {ret:.2f} — {note}")
            print(f"     Catalyst: [{cat}] {catalyst}")
            if s.get('reasoning'):
                for line in _wrap_text(str(s['reasoning']), 80)[:3]:
                    print(f"     {line}")
            print()

    not_selected = [e for e in all_entries if not e.get('selected')]
    shorts_not_sel = [e for e in not_selected if e.get('direction') == 'SHORT']
    shorts_not_sel.sort(key=lambda x: x.get('tradability_score', 0), reverse=True)
    if shorts_not_sel[:10]:
        print(f"  ALL SHORT SIGNALS (not selected)")
        rows2 = []
        for e in shorts_not_sel[:10]:
            _pd = e.get('_price_data') or {}
            gap = abs(float(_pd.get('change_pct', 0)))
            rows2.append([
                e['ticker'], f"+{gap:.1f}%",
                e.get('agent_tier', '—'), e.get('filter_fail', e.get('gate_reason', ''))[:30],
            ])
        print(tabulate(rows2, headers=['Ticker', 'Gap%', 'Tier', 'Filtered by'], tablefmt='simple'))
    print()


# ==============================================================================
# MAIN SCAN
# ==============================================================================
def run_scan(scan_date_str=None, dry_run=False):
    t_start = time.time()

    now_pt = datetime.now(PST)
    now_et = now_pt.astimezone(ET)

    if scan_date_str is None:
        scan_date_str = now_pt.strftime('%Y-%m-%d')
    today_str = scan_date_str
    yesterday_str = (datetime.strptime(today_str, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
    time_str = now_pt.strftime('%H:%M')

    print(f"\n{'='*W}")
    print(f"  AGENT 2 — Discovery + T+2 Watchlist")
    print(f"  {today_str} {time_str} PT")
    print(f"  Target: top {TARGET_TRADES} SHORT candidates with >{MIN_MOVE_PCT}% expected move")
    print(f"  Stage 1: {GEMINI_MODEL} + Google Search (15 themes)")
    print(f"  Stage 2: {CLAUDE_MODEL} deep analysis")
    print(f"  Stage 3: T+2 gap-fade filters + selection")
    print(f"{'='*W}")

    THEMATIC_QUERIES = build_thematic_queries(now_et, now_pt, today_str, yesterday_str)
    DISCOVER_SYSTEM, DEEP_SYSTEM, MARKET_SYSTEM, PRICE_SYSTEM = build_prompts(now_et, now_pt)

    # Fill outcomes for prior open trades
    if not dry_run:
        try:
            fill_outcomes(fill_date_str=today_str)
        except Exception as e:
            log.warning(f"fill_outcomes skipped: {e}")

    # ── STAGE 0: Market context ──
    print(f"\n  Stage 0: Market context...", end=' ')
    t0 = time.time()
    market_ctx = gemini_search(
        f"What are the US stock market conditions right now on {today_str}? "
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
    else:
        print(f"error: {market_ctx.get('error', '')[:80]}")
        market_ctx = {}

    # ── STAGE 1: Thematic discovery ──
    print(f"\n\n{'='*W}")
    print(f"  Stage 1: Thematic discovery — {len(THEMATIC_QUERIES)} searches")
    print(f"{'='*W}")

    t1_start = time.time()
    all_discovered = {}

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
                d['_theme'] = theme_id
                d['_sources'] = sources
                d['in_deploy'] = False
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
            print(f"  [{theme_id:<20}] found {len(discovered):>2} stocks, {n_new} new/upgraded  "
                  f"({', '.join(str(d.get('ticker','?')) for d in discovered[:5])})")

    elapsed_s1 = time.time() - t1_start
    print(f"\n  Stage 1 complete: {elapsed_s1:.0f}s")
    print(f"  Total unique tickers discovered: {len(all_discovered)}")

    # Filter and rank
    candidates = []
    for ticker, d in all_discovered.items():
        move = abs(float(d.get('expected_move_pct', 0)))
        strength = int(d.get('catalyst_strength', 0))
        move_score = move * strength / 10.0
        d['move_score'] = round(move_score, 2)
        d['expected_move_pct'] = round(move, 1)
        candidates.append(d)
    candidates.sort(key=lambda x: x['move_score'], reverse=True)

    # Validate tickers
    print(f"\n  Validating {len(candidates)} tickers via Polygon...")
    validated = []
    for c in candidates:
        tk = c.get('ticker', '')
        if '.' in tk:
            continue
        ok, _ = validate_ticker_polygon(tk)
        if ok:
            validated.append(c)
    print(f"  {len(validated)}/{len(candidates)} validated")
    candidates = validated

    n_above = sum(1 for c in candidates if c['expected_move_pct'] >= MIN_MOVE_PCT)
    print(f"  Above {MIN_MOVE_PCT}% threshold: {n_above}")

    # ── STAGE 2: Deep analysis ──
    deep_pool = sorted(candidates, key=lambda x: -x['move_score'])
    deep_candidates = deep_pool[:20]

    _deep_tickers = set(c.get('ticker') for c in deep_candidates)
    for c in deep_pool[20:]:
        tk = c.get('ticker', '')
        if tk in _deep_tickers:
            continue
        cs = int(c.get('catalyst_strength', 0))
        em = abs(float(c.get('expected_move_pct', 0)))
        if cs >= STRONG_CATALYST_SCORE or em >= STRONG_CATALYST_MOVE:
            deep_candidates.append(c)
            _deep_tickers.add(tk)

    prefetch_all_patterns()

    # Fetch price data
    price_data = {}
    fund_data = {}
    def _fetch_price_and_fundamentals(ticker):
        pd_info = fetch_price_data(ticker, today_str)
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

    if deep_candidates:
        print(f"\n  Fetching price data for {len(deep_candidates)} candidates...", end=' ')
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
        print(f"got {n_prices}/{len(deep_candidates)}")

    # Gap context
    GAP_MIN_PCT = 3.0
    gap_context = {}
    gap_tickers = [c['ticker'] for c in deep_candidates
                   if price_data.get(c['ticker']) and
                   abs(float(price_data[c['ticker']].get('change_pct', 0))) >= GAP_MIN_PCT]
    for t in gap_tickers:
        hist = fetch_20d_high(t, today_str)
        if hist:
            dd = hist['drawdown_pct']
            zone = "DEEP_RECOVERY" if dd <= -20 else ("MODERATE_RECOVERY" if dd <= -10 else ("MILD_DIP" if dd <= -5 else "NEAR_HIGHS"))
            gap_context[t] = {
                'gap_pct': price_data[t]['change_pct'],
                'high_20d': hist['high_20d'],
                'prev_close': hist['prev_close'],
                'drawdown_pct': dd,
                'zone': zone,
            }

    # Stage 2: Deep analysis
    if deep_candidates:
        print(f"\n\n{'='*W}")
        print(f"  Stage 2: Deep analysis — {len(deep_candidates)} candidates")
        print(f"{'='*W}")

        print(f"\n  Connecting to Claude via Vertex AI...", end=' ')
        claude_ok = False
        try:
            claude_client = AnthropicVertex(project_id=GCP_PROJECT, region=CLAUDE_REGION)
            test_resp = claude_client.messages.create(
                model=CLAUDE_MODEL, max_tokens=64,
                messages=[{"role": "user", "content": "Return only: {\"test\": true}"}],
            )
            print(f"OK ({CLAUDE_MODEL})")
            claude_ok = True
        except Exception as e:
            print(f"\n  Claude not available: {e}")
            claude_client = None

        market_context_str = ""
        if market_ctx:
            market_context_str = (
                f"\nMARKET CONTEXT:\n"
                f"- Futures: {market_ctx.get('futures', 'unknown')}\n"
                f"- Theme: {market_ctx.get('overnight_theme', 'unknown')}\n"
                f"- Risk: {market_ctx.get('risk_sentiment', 'unknown')}\n"
                f"- VIX: {market_ctx.get('vix_level', 'unknown')}"
            )

        # Build prompts for each candidate
        stage2_prompts = {}
        cand_map = {c['ticker']: c for c in deep_candidates}
        for cand in deep_candidates:
            ticker = cand.get('ticker', '?')
            pd_info = price_data.get(ticker)
            price_str = ""
            if pd_info and pd_info.get('change_pct') is not None:
                price_str = (
                    f"\nACTUAL PRICE DATA (Polygon):\n"
                    f"- Current: ${pd_info.get('current_price', 'N/A')}\n"
                    f"- Previous close: ${pd_info.get('prev_close', 'N/A')}\n"
                    f"- Change: {float(pd_info['change_pct']):+.1f}%\n"
                )
            fd = fund_data.get(ticker, {})
            if fd:
                if 'float_shares' in fd:
                    price_str += f"\n- Float: {fd['float_shares']/1e6:.1f}M shares"
                if 'short_pct_float' in fd:
                    price_str += f"\n- Short interest: {fd['short_pct_float']:.1f}%"
                if 'market_cap' in fd:
                    price_str += f"\n- Market cap: ${fd['market_cap']/1e9:.1f}B"

            pattern_str = fetch_similar_patterns(str(cand.get('catalyst_type', '')).lower(), ticker)
            sources_str = ""
            sources = cand.get('_sources', [])
            if sources:
                sources_str = "\n\nSOURCE ARTICLES:\n"
                for j, src in enumerate(sources[:5]):
                    sources_str += f"  {j+1}. {src.get('title', '')} — {src.get('url', '')}\n"

            bounce_str = ""
            if ticker in gap_context:
                _gc = gap_context[ticker]
                bounce_str = (
                    f"\n\n20-DAY CONTEXT:\n"
                    f"- 20d high: ${_gc['high_20d']}, drawdown: {_gc['drawdown_pct']:.0f}%\n"
                    f"- Zone: {_gc['zone']}\n"
                )

            prompt = (
                f"Analyze {ticker} ({cand.get('company', 'unknown')}) for a potential "
                f"trade today ({today_str}).\n\n"
                f"DISCOVERY:\n"
                f"- Catalyst: {cand.get('catalyst', 'N/A')}\n"
                f"- Type: {cand.get('catalyst_type', 'unknown')}\n"
                f"- Direction: {cand.get('direction', '?')}\n"
                f"- Expected move: {cand['expected_move_pct']:+.1f}%\n"
                f"- Strength: {cand.get('catalyst_strength', 0)}/10\n"
                f"- News time: {cand.get('news_time', 'unknown')}\n"
                f"- Headline: {cand.get('source_headline', 'N/A')}\n"
                f"{price_str}{bounce_str}{sources_str}{market_context_str}{pattern_str}\n\n"
                f"Be specific about remaining move after entry."
            )
            stage2_prompts[ticker] = prompt

        # Run deep analysis in parallel
        MAX_WORKERS_DEEP = 3
        print(f"\n  Submitting {len(stage2_prompts)} analyses (max {MAX_WORKERS_DEEP} workers)...")

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
            _result = gemini_search(_prompt, DEEP_SYSTEM)
            _meta['model_used'] = GEMINI_MODEL if 'error' not in _result else ''
            _result['_s2_meta'] = _meta
            return _ticker, _result

        deep_results = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS_DEEP) as executor:
            deep_futures = {executor.submit(_deep_analyze_one, (t, p)): t
                            for t, p in stage2_prompts.items()}
            for fut in as_completed(deep_futures):
                _tk = deep_futures[fut]
                try:
                    _, _res = fut.result()
                    deep_results[_tk] = _res
                    if 'error' not in _res:
                        print(f"    + {_tk} done")
                    else:
                        print(f"    x {_tk} error: {_res['error'][:60]}")
                except Exception as e:
                    deep_results[_tk] = {'error': str(e)[:200]}

        # Post-process results
        for cand in deep_candidates:
            ticker = cand.get('ticker', '?')
            result = deep_results.get(ticker)
            if not result or 'error' in result:
                continue
            pd_info = price_data.get(ticker)
            claude_total = float(result.get('total_catalyst_impact_pct', 0))
            polygon_absorbed = float(pd_info.get('change_pct', 0)) if pd_info and 'change_pct' in pd_info else None
            if polygon_absorbed is not None and abs(claude_total) > 0.1:
                same_direction = (claude_total > 0 and polygon_absorbed > 0) or (claude_total < 0 and polygon_absorbed < 0)
                computed_remaining = claude_total - polygon_absorbed if same_direction else claude_total
                computed_remaining = max(min(computed_remaining, abs(claude_total)), -abs(claude_total))
                result['move_already_absorbed_pct'] = polygon_absorbed
                result['remaining_move_pct'] = round(computed_remaining, 2)
                result['expected_move_pct'] = round(computed_remaining, 2)
                if same_direction and abs(computed_remaining) < 3.0 and abs(polygon_absorbed / claude_total) > 0.80:
                    result['move_is_exhausted'] = True
                elif not same_direction:
                    result['move_is_exhausted'] = True

            remaining = float(result.get('remaining_move_pct', 0))
            strength = result.get('catalyst_strength', result.get('confidence', 5))
            freshness = result.get('freshness', 5)
            tradability = abs(remaining) * int(strength) * int(freshness) / 100.0
            result['tradability_score'] = round(tradability, 2)

        elapsed_s2 = time.time() - t1_start - elapsed_s1
        _s2_ok = sum(1 for r in deep_results.values() if 'error' not in r)
        print(f"\n  Stage 2 complete: {elapsed_s2:.0f}s ({_s2_ok}/{len(deep_candidates)} succeeded)")

    else:
        deep_results = {}
        cand_map = {}

    # Build final entries list
    all_entries = []
    for ticker, r in deep_results.items():
        if 'error' in r:
            continue
        disc = cand_map.get(ticker, {})
        _remaining = float(r.get('remaining_move_pct', 0))
        _priced = float(r.get('move_already_absorbed_pct', r.get('move_already_priced_pct', 0)))
        entry = {
            'ticker': ticker,
            'company': disc.get('company', ''),
            'direction': r.get('direction', 'NEUTRAL'),
            'confidence': _safe_int(r.get('confidence', 0), 0),
            'expected_move_pct': _safe_float(r.get('expected_move_pct', 0)),
            'move_already_priced_pct': _priced,
            'remaining_move_pct': _remaining,
            'move_is_exhausted': bool(r.get('move_is_exhausted', False)),
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
            'news_time': disc.get('news_time', ''),
            'theme': disc.get('_theme', ''),
            'headline': disc.get('source_headline', ''),
            'in_deploy': False,
            'change_pct': float(price_data.get(ticker, {}).get('change_pct', 0)) if price_data.get(ticker) else 0,
            '_price_data': price_data.get(ticker),
        }
        all_entries.append(entry)

    # Verify event timing
    if _POLYGON_OK:
        print(f"\n  Verifying event timing via Polygon...")
        for e in all_entries:
            verify_and_correct_event_timing(e, today_str, now_et, yesterday_str)

    # Filter stale news
    def _is_fresh(entry):
        news_dt = _parse_news_time(str(entry.get('news_time', '')))
        if not news_dt:
            return True
        age_hours = (now_pt - news_dt.astimezone(PST)).total_seconds() / 3600
        return age_hours <= MAX_NEWS_AGE_HOURS

    all_entries = [e for e in all_entries if _is_fresh(e)]

    # Apply score adjustments
    all_entries = apply_score_adjustments(all_entries, now_pt, market_ctx)

    # Quality gates
    all_entries, n_gated = apply_quality_gates(all_entries, market_ctx, gap_context, now_et, now_pt)
    if n_gated:
        print(f"\n  Quality gates: {n_gated} candidates gated")

    # ── STAGE 3: T+2 filters + selection ──
    print(f"\n\n{'='*W}")
    print(f"  Stage 3: T+2 gap-fade filters")
    print(f"{'='*W}")

    # Fetch PM bars for SHORT candidates
    short_entries = [e for e in all_entries if e.get('direction') == 'SHORT']
    short_tickers = [e['ticker'] for e in short_entries]
    print(f"\n  Fetching PM bars for {len(short_tickers)} SHORT candidates...")
    pm_data = fetch_pm_bars_for_candidates(short_tickers, today_str) if short_tickers else {}

    # Apply T+2 filters
    all_entries = apply_t2_filters(all_entries, pm_data)

    n_qualified = sum(1 for e in all_entries if e.get('passed_all'))
    print(f"  Qualified after T+2 filters: {n_qualified}")

    # Select top 2
    selected = select_top2(all_entries)
    print(f"  Selected: {len(selected)}")

    # Add T+2 dates
    t1_date, t2_date = get_t1_t2_dates(today_str)
    for c in selected:
        c['t1_date'] = t1_date
        c['t2_date'] = t2_date

    # Write to DB
    runtime = time.time() - t_start
    run_meta = {
        'runtime': runtime,
        'n_themes': len(THEMATIC_QUERIES),
        'n_discovered': len(all_discovered),
        'n_deep': len(deep_candidates) if deep_candidates else 0,
        'n_qualified': n_qualified,
        'futures': str(market_ctx.get('futures', ''))[:500],
        'risk': str(market_ctx.get('risk_sentiment', ''))[:200],
        'vix': str(market_ctx.get('vix_level', ''))[:50],
    }

    if not dry_run:
        write_to_db(today_str, all_entries, selected, run_meta)
        log_to_bq(all_entries, selected, market_ctx, today_str, time_str)

    # Save JSON output
    local_path = f"/tmp/agent2_scan_{today_str}.json"
    output_data = {
        'scan_date': today_str,
        'n_discovered': len(all_discovered),
        'n_deep_analyzed': len(deep_results),
        'n_qualified': n_qualified,
        'n_selected': len(selected),
        'selected': [{k: v for k, v in s.items() if k != '_price_data'} for s in selected],
    }
    with open(local_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    try:
        gcs_path = f"{OUTPUT_DIR}/agent2_{today_str}.json"
        bucket.blob(gcs_path).upload_from_filename(local_path)
        print(f"\n  Saved: gs://{BUCKET_NAME}/{gcs_path}")
    except Exception:
        pass

    # Print watchlist
    print_watchlist(selected, all_entries, today_str)

    print(f"\n{'='*W}")
    print(f"  AGENT 2 COMPLETE")
    print(f"  Total time: {runtime:.0f}s ({runtime/60:.1f} min)")
    print(f"  Discovered: {len(all_discovered)} | Deep: {len(deep_results)} | "
          f"Qualified: {n_qualified} | Selected: {len(selected)}")
    print(f"{'='*W}")

    return selected


# ==============================================================================
# CLI DISPATCH
# ==============================================================================
def parse_cli_args():
    try:
        get_ipython()
        return None
    except NameError:
        pass

    parser = argparse.ArgumentParser(description='Agent 2 — Discovery + T+2 Watchlist')
    sub = parser.add_subparsers(dest='command')

    scan_p = sub.add_parser('scan', help='Run discovery scan')
    scan_p.add_argument('--date', type=str, default=None)
    scan_p.add_argument('--dry-run', action='store_true')

    fill_p = sub.add_parser('fill', help='Fill T+2 outcomes')
    fill_p.add_argument('--date', type=str, default=None)

    report_p = sub.add_parser('report', help='Performance report')
    report_p.add_argument('--weeks', type=int, default=REPORT_WEEKS)

    log_p = sub.add_parser('log_entry', help='Log a trade entry')
    log_p.add_argument('--ticker', required=True)
    log_p.add_argument('--entry_price', type=float, required=True)
    log_p.add_argument('--entry_time', type=str, default=None)
    log_p.add_argument('--entry_type', type=str, default=None)
    log_p.add_argument('--size_usd', type=float, default=None)
    log_p.add_argument('--notes', type=str, default=None)

    args = parser.parse_args()
    if args.command is None:
        args.command = 'scan'
        args.date = None
        args.dry_run = False
    return args


# ==============================================================================
# MAIN
# ==============================================================================
args = parse_cli_args()

if args is not None:
    init_db()
    if args.command == 'scan':
        run_scan(scan_date_str=args.date, dry_run=args.dry_run)
    elif args.command == 'fill':
        fill_outcomes(fill_date_str=args.date)
    elif args.command == 'report':
        run_report(weeks=args.weeks)
    elif args.command == 'log_entry':
        log_entry(args.ticker, args.entry_price, args.entry_time,
                  args.entry_type, args.size_usd, args.notes)
else:
    init_db()
    if MODE == 'scan':
        run_scan(scan_date_str=SCAN_DATE, dry_run=DRY_RUN)
    elif MODE == 'fill':
        fill_outcomes(fill_date_str=SCAN_DATE)
    elif MODE == 'report':
        run_report(weeks=REPORT_WEEKS)
    else:
        print(f"Unknown MODE: {MODE}")
