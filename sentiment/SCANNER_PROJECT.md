# Pre-Market Sentiment Scanner — Project Documentation

## Overview

A dual-scanner system that identifies intraday trading opportunities by combining LLM-powered news analysis with verified exchange data. Designed for a trader who enters positions at **6:25 AM PT** (market open 6:30 AM PT) and holds for **30-60 minutes**, targeting **2-4% intraday spikes**.

Two scanners run daily before market open:

| Scanner | File | Purpose | Input | Tickers |
|---------|------|---------|-------|---------|
| **Premarket** | `premarket_scanner.py` (1603 lines) | Find catalysts for tracked stocks | Fixed 68-ticker deploy list | Known universe |
| **Discovery** | `discovery_scanner.py` (1589 lines) | Discover movers outside deploy universe | 14 thematic search queries | Any US equity |

**Version:** V2 (deployed 2026-09-07). V1 had critical false-positive issues from Gemini price hallucinations and stale timestamp misattribution.

---

## Architecture

```
                    STAGE 0                STAGE 1                 STAGE 1.5              STAGE 2
                ┌──────────┐         ┌──────────────┐         ┌──────────────┐      ┌──────────────┐
                │  Market   │         │  Broad Scan   │         │    Price     │      │    Deep      │
                │  Context  │         │  Gemini Flash │         │  Verification│      │   Analysis   │
                │  (Gemini) │         │  + Google     │         │  (Polygon)   │      │   (Claude)   │
                │           │         │  Search       │         │              │      │              │
                │ Futures   │         │ 68 tickers    │         │ 1-min bars   │      │ Remaining    │
                │ VIX       │    ──►  │ parallel (5x) │    ──►  │ prev close   │ ──►  │ move calc    │
                │ Macro     │         │ catalyst 0-10 │         │ change %     │      │ direction    │
                │ Sentiment │         │ news_time     │         │              │      │ confidence   │
                └──────────┘         └──────────────┘         └──────────────┘      └──────────────┘
                                           │                                              │
                                      Top 15 (score>=3)                              Shortlist
                                                                                     5 LONG + 5 SHORT
                                                                                          │
                                                                                          ▼
                                                                                 ┌──────────────┐
                                                                                 │  Post-Process │
                                                                                 │  Polygon      │
                                                                                 │  Verification │
                                                                                 │               │
                                                                                 │ Event timing  │
                                                                                 │ Absorption    │
                                                                                 │ Freshness     │
                                                                                 │ Stale filter  │
                                                                                 └──────────────┘
                                                                                          │
                                                                                          ▼
                                                                                 ┌──────────────┐
                                                                                 │   Output      │
                                                                                 │               │
                                                                                 │ GCS JSON      │
                                                                                 │ BigQuery log  │
                                                                                 │ Console cards │
                                                                                 └──────────────┘
```

---

## Pipeline Stages (Premarket Scanner)

### Stage 0: Market Context
Single Gemini + Google Search call. Returns futures direction, overnight theme, sector moves, macro events (jobs report, Fed), VIX level, and top premarket movers. Fed into Stage 2 prompts as context.

### Stage 1: Broad Scan
Scans all 68 deploy tickers in parallel (5 workers) using Gemini 2.5 Flash with Google Search grounding. Each ticker gets a structured query asking for news from the last 4-16 hours. Returns:
- `catalyst_score` (0-10) calibrated against a 5-tier weight scale
- `direction` (LONG/SHORT/NEUTRAL)
- `headline`, `catalyst_type`, `news_time`
- Up to 3 `key_articles` with source and publish time

Results ranked by catalyst_score. Top 15 with score >= 3 advance to Stage 2.

### Stage 1.5: Price Verification (V2)
Parallel Polygon API calls for all deep candidates. **Primary source** for price data (replaces Gemini-based price lookup from V1). Returns verified `prev_close`, `current_price`, `change_pct` from actual exchange data. Falls back to Gemini search only if Polygon fails (result tagged `(unverified)`).

**Why this exists:** V1 used Gemini for price data, which hallucinated prices in 3 of 5 analyzed false positives:
- LUNR: Gemini claimed -9.1%, actual was +0.34%
- BLZE: Gemini claimed +68%, actual was -0.94%
- FLNC: Gemini claimed -3.2%, actual was +2.64%

### Stage 2: Deep Analysis
Sequential Claude Sonnet 4.6 calls (via Vertex AI, Gemini fallback). Each candidate receives:
- Scan results (headline, catalyst, direction)
- Verified price data from Polygon
- Earnings verification (if catalyst is earnings-type)
- Market context from Stage 0

The deep analysis prompt enforces the **remaining move** concept: separating total catalyst impact from what's already absorbed pre-market. Returns confidence (1-10), remaining_move_pct, move_is_exhausted flag, reasoning, bull/bear cases, M&A role (acquirer vs target), and news timeline.

### Post-Processing: Polygon Verification (V2)
Runs `verify_and_correct_event_timing()` on all shortlisted candidates:

1. **Event timing**: Scans Polygon 5-min bars to find when price first moved >3% from prior close. If this is >1 hour older than the article timestamp, overrides `news_time` with the actual event time.
2. **Absorption**: Calculates `absorbed_frac = 1 - (remaining / total_move)`. Marks `move_is_exhausted` if >80% absorbed or price reversed past origin.
3. **Stale filter**: Hard cutoff at `MAX_NEWS_AGE_HOURS = 16`. Any candidate whose corrected news_time is older gets dropped.
4. **Freshness multiplier**: Time-decay applied to composite score:

| News Age | Multiplier |
|----------|-----------|
| < 2h | 1.5x |
| 2-4h | 1.2x |
| 4-6h | 1.0x |
| 6-8h | 0.5x |
| 8-12h | 0.2x |
| 12-16h | 0.08x |
| > 16h | 0.02x (effectively filtered) |

**Why this exists:** V1 used article publish timestamps as event time. Articles about stale events (e.g., earnings from 28h ago) appeared fresh because the article was new. This caused false positives on PL, HUT, AVAV, ASTS, and MARA.

---

## Discovery Scanner Differences

| Aspect | Premarket Scanner | Discovery Scanner |
|--------|------------------|-------------------|
| Stage 1 input | Fixed 68-ticker list | 14 thematic search queries |
| Goal | Catalysts for tracked stocks | Stocks with >5% moves outside deploy universe |
| Stage 1 output | `catalyst_score` per ticker | `discovered[]` array of tickers from search |
| Ranking metric | `composite = confidence * abs(remaining_move)` | `tradability_score = remaining * strength * freshness / 100` |
| Deep pool | Top 15 (score >= 3) | Top 20 (prioritizing non-deploy tickers) |
| Extra fields | Earnings verification | `liquidity_risk`, `market_cap_tier`, `theme`, `in_deploy` flag |
| BQ table | `premarket_scanner` | `discovery_scanner` |
| Workers / delay | 5 / 0.3s | 4 / 0.5s |

### Discovery Thematic Queries (14)
`stocks_moving_now`, `afterhours_movers`, `earnings_today`, `biggest_gainers`, `biggest_losers`, `fda_regulatory`, `mergers_acquisitions`, `analyst_actions`, `halt_resume`, `short_squeeze`, `offerings_dilution`, `contracts_deals`, `bankruptcy_fraud`, `insider_activity`

---

## Scoring Formula

**Composite score** = `confidence * abs(remaining_move_pct) * freshness_multiplier`

Where:
- `confidence` (1-10): Claude's assessment of trade conviction
- `remaining_move_pct`: Expected % move after 6:25 AM PT entry
- `freshness_multiplier`: Time-decay based on corrected news age

Final output: top 5 LONG + top 5 SHORT by composite score.

---

## Configuration

```python
GCP_PROJECT      = "yanina-everything"
GCP_LOCATION     = "us-central1"
GEMINI_MODEL     = "gemini-2.5-flash"
CLAUDE_MODEL     = "claude-sonnet-4-6"
CLAUDE_REGION    = "us-east5"
BUCKET_NAME      = "time-series-bucket-ys"
OUTPUT_DIR       = "Data/Scanner"
BQ_SCANNER_TABLE = "ys-time-series.scoring_logs.premarket_scanner"
POLYGON_API_KEY  = "2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs"
SCANNER_VERSION  = "V2"

MAX_WORKERS_SCAN   = 5
TOP_N_DEEP         = 15
MAX_NEWS_AGE_HOURS = 16
FINAL_LONG         = 5
FINAL_SHORT        = 5
GEMINI_DELAY_S     = 0.3
MAX_RETRIES        = 3
```

---

## LLM Prompts

### SCAN_SYSTEM (Stage 1 — Gemini)
Stock news analyst. Find most recent news for a ticker. 5-tier catalyst weight scale:
- **Tier 1** (8-10): Earnings surprise, FDA, massive contract, M&A target
- **Tier 2** (6-8): Analyst upgrade/downgrade with new PT, offering, regulatory
- **Tier 3** (4-6): Pre-market move >3%, partnership, CEO change
- **Tier 4** (2-4): Routine reiteration, insider Form 4, conference, routine filings
- **Tier 5** (0-2): No news, aggregator noise, law firm investigations

M&A rules: Acquirer stocks don't spike. Vote/regulatory clearance of existing deals is procedural (score <= 3).

### DEEP_SYSTEM (Stage 2 — Claude)
Senior pre-market trading analyst. Core concept: **remaining move vs total impact**. Separates total catalyst magnitude from pre-market absorption. Includes:
- Noise filters (law firm investigations, aggregator articles, price-reporting articles)
- M&A analysis (acquirer vs target, deal materiality)
- Earnings direction rules (miss = SHORT unless >15% drop + minor miss)
- 5-tier confidence scale matching catalyst tiers

### MARKET_SYSTEM (Stage 0 — Gemini)
Market overview: futures, overnight theme, sectors, macro events, VIX, premarket movers.

### PRICE_SYSTEM / EARNINGS_VERIFY_SYSTEM (Helpers — Gemini)
Fallback price lookup (only used if Polygon fails) and earnings date verification for earnings-type catalysts.

---

## Polygon Tools (V2)

| Function | Purpose | Input | Output |
|----------|---------|-------|--------|
| `fetch_price_data(ticker)` | Current price + prev close | Ticker | `{current_price, prev_close, change_pct, source}` |
| `get_prev_session_close(ticker)` | T-1 closing price | Ticker, ref_date | `{close, date}` |
| `get_actual_event_time(ticker)` | When price first moved >3% | Ticker, ref_date, threshold | `{event_time_dt, prev_close, peak_price, current_price}` |
| `calculate_absorption(prev, peak, current)` | How much of move is left | 3 prices | `{total_move_pct, remaining_pct, absorbed_pct, is_exhausted}` |
| `verify_and_correct_event_timing(entry)` | Orchestrator: correct timestamps + absorption | Shortlist entry dict | Mutates entry in-place with `polygon_*` fields |

---

## Output

### Console
Summary tables (one-line per candidate) + detailed cards with catalyst, reasoning, timeline, bull/bear cases, risk factors, and source articles.

### GCS
`gs://time-series-bucket-ys/Data/Scanner/scan_YYYY-MM-DD.json`

Contains: `scan_date`, `scan_time_pt`, `market_context`, `scan_results` (all 68 tickers), `deep_results` (top 15), `shortlist_long` (top 5), `shortlist_short` (top 5).

### BigQuery
`ys-time-series.scoring_logs.premarket_scanner` — one row per scored candidate.

Key columns: `scan_timestamp`, `ticker`, `direction`, `rank_in_direction`, `selected`, `confidence`, `expected_move_pct`, `remaining_move_pct`, `move_already_absorbed_pct`, `total_catalyst_impact_pct`, `move_is_exhausted`, `composite_score`, `catalyst_type`, `reasoning`, `freshness_multiplier`, `scanner_version`, `acquirer_or_target`, `deal_materiality`, `catalyst_is_new`, `news_timeline_json`.

All deep-analyzed AND scan-only tickers are logged (scan-only rows have confidence=0) for coverage tracking and backtesting.

---

## V1 to V2 Changelog

### Problems identified in V1 (Sep 4, 2026 analysis)

7 false positives analyzed, all eliminated by V2:

| Ticker | Root Cause | Gemini Claimed | Polygon Actual | V2 Fix |
|--------|-----------|---------------|----------------|--------|
| PL | Stale earnings (Sep 3 AH) | news fresh | 28h old event | Event timing correction |
| HUT | Stale deal + wrong prev close | close $80.68 | close $88.09 | Event timing + price fix |
| AVAV | Stale contract + absorption underreport | 0.2% absorbed | 91% absorbed | Event timing + absorption |
| LUNR | Hallucinated -9.1% drop | -9.1% | +0.34% | Polygon price verification |
| ASTS | Stale earnings (Aug 9) + fabricated impact | -8.0% impact | +2.21% | Price fix + no spike detected |
| BLZE | Fully fabricated +68% catalyst | +68% | -0.94% | Polygon price verification |
| FLNC | Stale downgrade + wrong price direction | -3.2% | +2.64% | Price fix + event timing |

### V2 fixes

1. **Polygon price verification** (`fetch_price_data`): Primary source for prev_close, current_price, change_pct. Eliminates Gemini price hallucinations entirely.

2. **Event timing correction** (`verify_and_correct_event_timing`): Polygon 5-min bars detect when price first moved >3%. If >1h older than article timestamp, overrides `news_time`. Catches stale catalysts reported as fresh.

3. **Absorption verification** (`calculate_absorption`): Runs on all candidates (V1 had a 5% threshold gate that skipped low-absorption candidates). Catches exhausted moves the LLM underreports.

4. **Auto-install**: `polygon-api-client` package installs automatically in Colab.

5. **Version tagging**: `SCANNER_VERSION = 'V2'` logged to BQ and GCS for A/B comparison.

---

## Runtime

- **Premarket scanner**: ~12-13 min for 68 tickers (Stage 1: ~6 min, Stage 2: ~6 min)
- **Discovery scanner**: ~15-18 min (14 queries + up to 20 deep analyses)
- **Polygon calls**: ~2s for price verification (parallel), ~15s for event timing verification (sequential)

---

## Dependencies

```
google-genai          # Gemini 2.5 Flash via Vertex AI
anthropic[vertex]     # Claude Sonnet via Vertex AI
google-cloud-bigquery # BQ logging
google-cloud-storage  # GCS output
polygon-api-client    # Polygon.io market data (V2)
yfinance              # Yahoo Finance fallback for prices + float/short interest data
pandas, numpy, pytz   # Data handling
```

All dependencies auto-install via try/except + pip for Colab compatibility.

---

## V2.1 Changes (2026-09-07)

### 1. No LLM fallback for prices
Removed Gemini as fallback price source in `fetch_price_data`. Chain is now: **Polygon -> yfinance -> None**. No LLM ever touches price data.

### 2. Claude Structured Outputs (tool-use)
Stage 2 `claude_analyze` now uses forced tool-use (`tool_choice: {"type": "tool", "name": "record_analysis"}`) with a full JSON schema (`DEEP_ANALYSIS_TOOL`). This guarantees valid JSON with correct keys/types, eliminating regex parse failures. Note: Gemini's `response_schema` is NOT compatible with Google Search grounding, so Stage 1 still uses regex parsing.

### 3. Python math for remaining move
After Claude returns `total_catalyst_impact_pct`, the remaining move is computed in Python: `remaining = claude_total_impact - polygon_absorbed`. This prevents LLM arithmetic errors. Exhaustion is also recomputed: if >80% of the total impact is absorbed, the move is marked exhausted.

### 4. Float and short interest injection
Stage 1.5 now fetches `floatShares`, `shortPercentOfFloat`, and `marketCap` from yfinance for each deep candidate. These are injected into Claude's prompt as "FUNDAMENTAL DATA" so the model can properly assess short squeeze potential and position sizing context.

### 5. Discovery scanner deduplication
When a ticker is found by multiple thematic queries, all contexts are now concatenated (`_all_themes`, `_all_catalysts`) and injected into Claude's prompt as "MULTIPLE DISCOVERY THEMES" instead of silently overwriting with the higher-move entry.
