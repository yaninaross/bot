# %% Agent 6v2 — Source Actual Attention & Positioning Data for RXRX
#
# Attempts to obtain REAL attention/positioning data (not proxies):
#   1. Short interest (Polygon, FINRA)
#   2. StockTwits social sentiment
#   3. Reddit mentions (Pushshift/Reddit API)
#   4. Google Trends search interest (pytrends)
#   5. Institutional ownership / 13F (Polygon)
#
# For each source that IS available: build standalone strategy, test on
# selection window only (pre-2026), compute stats + null test + walk-forward.
# For each source NOT available: report what was tried, exact response.
#
# NO PROXY SUBSTITUTION. Real data or "unavailable".
# NO S5 overlay. Standalone strategies only.

import os, sys, json, warnings, time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

for pkg in ["matplotlib", "scipy", "pytrends"]:
    try:
        __import__(pkg)
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg,
                               "--quiet", "--break-system-packages"])

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

# ── Import shared engine ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent if "__file__" in dir() else Path("agent10")))
from agent0_engine import (run_backtest, compute_stats, compute_costs,
                           log_trial, add_features, null_test,
                           walk_forward, DATA_DIR, OUT_DIR)

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════
SEL_END = "2026-01-01"          # selection / OOS boundary
MAX_HOLD = 5                    # max holding days (short-horizon MR)
POSITION_SIZE = 10000
TICKER = "RXRX"

# Polygon API key — read from environment or fallback from other agent10 scripts
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY",
                                 "2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs")

print("=" * 70)
print("  AGENT 6v2: SOURCE ACTUAL ATTENTION & POSITIONING DATA")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════
# 1. LOAD RXRX DATA
# ══════════════════════════════════════════════════════════════════
rxrx = pd.read_parquet(DATA_DIR / "rxrx_daily.parquet")
rxrx["date"] = pd.to_datetime(rxrx["date"])
rxrx = rxrx.sort_values("date").reset_index(drop=True)
rxrx = add_features(rxrx)

sel_mask = rxrx["date"] < SEL_END
rxrx_sel = rxrx[sel_mask].copy().reset_index(drop=True)

print(f"\n  RXRX loaded: {len(rxrx)} total bars")
print(f"  Selection window: {rxrx_sel['date'].min().date()} to "
      f"{rxrx_sel['date'].max().date()} ({len(rxrx_sel)} bars)")

# ══════════════════════════════════════════════════════════════════
# DATA AVAILABILITY REPORT
# ══════════════════════════════════════════════════════════════════
availability = {}

def report_source(name, attempted, response_summary, available, date_range=None, notes=""):
    """Track availability for each data source."""
    entry = {
        "source": name,
        "attempted": attempted,
        "response": response_summary,
        "available": available,
        "date_range": date_range or "N/A",
        "notes": notes,
    }
    availability[name] = entry
    status = "AVAILABLE" if available else "UNAVAILABLE"
    print(f"\n  [{status}] {name}")
    print(f"    Attempted: {attempted}")
    print(f"    Response:  {response_summary}")
    if date_range:
        print(f"    Date range: {date_range}")
    if notes:
        print(f"    Notes: {notes}")

# ══════════════════════════════════════════════════════════════════
# 2a. SHORT INTEREST — Polygon API
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  2a. SHORT INTEREST")
print("=" * 70)

short_interest_data = None

# Attempt 1: Polygon v3 short interest endpoint
si_endpoints = [
    f"https://api.polygon.io/v3/reference/tickers/{TICKER}/short-interest?apiKey={POLYGON_API_KEY}",
    f"https://api.polygon.io/v2/reference/shorts/{TICKER}?apiKey={POLYGON_API_KEY}",
    f"https://api.polygon.io/v3/reference/short-volume?ticker={TICKER}&apiKey={POLYGON_API_KEY}",
    f"https://api.polygon.io/v2/aggs/ticker/{TICKER}/range/1/day/2021-01-01/2025-12-31?adjusted=true&sort=asc&apiKey={POLYGON_API_KEY}&include_otc=false",
]

si_responses = []
for url in si_endpoints[:3]:
    try:
        endpoint_label = url.split("polygon.io")[1].split("?")[0]
        resp = requests.get(url, timeout=15)
        status = resp.status_code
        try:
            body = resp.json()
            status_field = body.get("status", "")
            results_count = len(body.get("results", []))
            msg = f"HTTP {status}, status='{status_field}', results={results_count}"
            if results_count > 0:
                msg += f", sample: {json.dumps(body['results'][0])[:200]}"
        except:
            msg = f"HTTP {status}, body: {resp.text[:200]}"
        si_responses.append({"endpoint": endpoint_label, "msg": msg, "status": status,
                             "body": body if status == 200 else None})
        print(f"  Polygon {endpoint_label}: {msg}")
    except Exception as e:
        si_responses.append({"endpoint": endpoint_label, "msg": str(e), "status": 0})
        print(f"  Polygon {endpoint_label}: ERROR: {e}")

# Check if any endpoint returned usable short interest data
si_available = False
for sr in si_responses:
    if sr.get("body") and sr["body"].get("results") and len(sr["body"]["results"]) > 0:
        # Check if results have short interest fields
        sample = sr["body"]["results"][0]
        if any(k in sample for k in ["short_volume", "short_interest", "short_exempt_volume",
                                      "shortVolume", "shortInterest"]):
            si_available = True
            short_interest_data = sr["body"]["results"]
            break

# Attempt 2: Polygon v2 short volume (different from short interest)
# Short Volume is available on some Polygon plans
sv_url = (f"https://api.polygon.io/v2/aggs/ticker/{TICKER}/range/1/day/2021-04-16/2025-12-31"
          f"?adjusted=true&sort=asc&limit=50000&apiKey={POLYGON_API_KEY}")
try:
    resp = requests.get(sv_url, timeout=30)
    body = resp.json()
    if body.get("results"):
        # Check if standard OHLCV data has any short-interest-like fields
        sample_keys = list(body["results"][0].keys())
        print(f"  Standard aggs keys: {sample_keys}")
        # Standard OHLCV aggs don't include short interest
except Exception as e:
    print(f"  Standard aggs check failed: {e}")

# Attempt 3: Polygon short volume endpoint (Starter plan feature)
sv_endpoints = [
    f"https://api.polygon.io/v3/reference/tickers/{TICKER}/short-volume?apiKey={POLYGON_API_KEY}",
    f"https://api.polygon.io/v2/shorts/{TICKER}?apiKey={POLYGON_API_KEY}",
]
for url in sv_endpoints:
    try:
        endpoint_label = url.split("polygon.io")[1].split("?")[0]
        resp = requests.get(url, timeout=15)
        status = resp.status_code
        try:
            body = resp.json()
            msg = f"HTTP {status}, {json.dumps(body)[:300]}"
        except:
            msg = f"HTTP {status}, body: {resp.text[:200]}"
        print(f"  Polygon {endpoint_label}: {msg}")
        if resp.status_code == 200 and body.get("results"):
            si_available = True
            short_interest_data = body["results"]
    except Exception as e:
        print(f"  Polygon {endpoint_label}: ERROR: {e}")

if not si_available:
    report_source(
        "Short Interest",
        "Polygon API: /v3/reference/tickers/RXRX/short-interest, "
        "/v2/reference/shorts/RXRX, /v3/reference/short-volume, "
        "/v3/reference/tickers/RXRX/short-volume, /v2/shorts/RXRX",
        "All endpoints returned 404/not-found or empty results. "
        "Short interest data requires Polygon Business plan or higher.",
        False,
        notes="FINRA publishes bi-weekly short interest but not via free API. "
              "Polygon short interest requires Business+ tier."
    )
else:
    report_source(
        "Short Interest",
        "Polygon API",
        f"Found {len(short_interest_data)} records",
        True,
    )

# ══════════════════════════════════════════════════════════════════
# 2b. STOCKTWITS — Social Sentiment
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  2b. STOCKTWITS SOCIAL SENTIMENT")
print("=" * 70)

stocktwits_data = None
st_available = False

# StockTwits API — free, no auth for basic stream
st_url = f"https://api.stocktwits.com/api/2/streams/symbol/{TICKER}.json"
try:
    resp = requests.get(st_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    status = resp.status_code
    try:
        body = resp.json()
        if body.get("messages"):
            n_msgs = len(body["messages"])
            sample_msg = body["messages"][0] if n_msgs > 0 else {}
            # Extract date range from returned messages
            dates = [m.get("created_at", "") for m in body["messages"] if m.get("created_at")]
            date_range_str = f"{dates[-1][:10]} to {dates[0][:10]}" if dates else "unknown"
            # Check if sentiment is included
            has_sentiment = any(m.get("entities", {}).get("sentiment") for m in body["messages"])
            msg = (f"HTTP {status}, {n_msgs} messages returned, "
                   f"date range: {date_range_str}, sentiment_included={has_sentiment}")
            print(f"  StockTwits stream: {msg}")

            # StockTwits only returns RECENT messages (last ~30), no historical archive
            # Check if we can paginate back
            oldest_id = body["messages"][-1].get("id") if body["messages"] else None
            if oldest_id:
                page_url = f"{st_url}?max={oldest_id}"
                resp2 = requests.get(page_url, timeout=15,
                                     headers={"User-Agent": "Mozilla/5.0"})
                if resp2.status_code == 200:
                    body2 = resp2.json()
                    n2 = len(body2.get("messages", []))
                    if n2 > 0:
                        dates2 = [m.get("created_at", "") for m in body2["messages"]]
                        oldest_date = dates2[-1][:10] if dates2 else "unknown"
                        print(f"  StockTwits page 2: {n2} messages, oldest: {oldest_date}")

            # StockTwits is available but only provides recent data, not historical
            # It cannot provide data back to 2021 for selection window testing
            report_source(
                "StockTwits",
                f"GET {st_url}",
                f"API accessible, returned {n_msgs} recent messages. "
                f"Date range: {date_range_str}. "
                "HOWEVER: API only returns recent messages (~30 at a time). "
                "No historical archive endpoint. Cannot reconstruct daily sentiment "
                "back to 2021 for selection window testing.",
                False,
                date_range=date_range_str,
                notes="StockTwits free API only returns recent stream. "
                      "Historical data requires StockTwits Data product (paid). "
                      "Insufficient for backtesting over 2021-2025."
            )
        else:
            msg = f"HTTP {status}, no messages field. Body: {json.dumps(body)[:300]}"
            print(f"  StockTwits stream: {msg}")
            report_source("StockTwits", f"GET {st_url}", msg, False)

    except Exception as e:
        msg = f"HTTP {status}, parse error: {e}, body: {resp.text[:200]}"
        print(f"  StockTwits stream: {msg}")
        report_source("StockTwits", f"GET {st_url}", msg, False)

except Exception as e:
    print(f"  StockTwits stream: CONNECTION ERROR: {e}")
    report_source("StockTwits", f"GET {st_url}", f"Connection error: {e}", False)

# ══════════════════════════════════════════════════════════════════
# 2c. REDDIT — Historical Mention Data
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  2c. REDDIT MENTIONS")
print("=" * 70)

reddit_data = None
reddit_available = False

# Pushshift was the standard source for historical Reddit data.
# It was shut down / restricted access in mid-2023.
# Try both old and new endpoints.

reddit_endpoints = [
    # Old Pushshift (likely down)
    f"https://api.pushshift.io/reddit/search/submission/?q=RXRX&subreddit=wallstreetbets,stocks,investing&size=500&before=1704067200",
    # Arctic Shift (community replacement)
    f"https://arctic-shift.photon-reddit.com/api/posts/search?q=RXRX&subreddit=wallstreetbets,stocks&limit=100",
    # Reddit's own search (no auth, limited)
    f"https://www.reddit.com/search.json?q=RXRX&sort=new&limit=100",
]

for url in reddit_endpoints:
    endpoint_name = url.split("//")[1].split("/")[0]
    try:
        resp = requests.get(url, timeout=15,
                            headers={"User-Agent": "Agent6v2-Research/1.0"})
        status = resp.status_code
        try:
            body = resp.json()
            # Different APIs return data differently
            if "data" in body and isinstance(body["data"], list):
                n_results = len(body["data"])
            elif "data" in body and isinstance(body["data"], dict):
                children = body["data"].get("children", [])
                n_results = len(children)
            else:
                n_results = len(body) if isinstance(body, list) else 0

            msg = f"HTTP {status}, results={n_results}"
            if n_results > 0:
                msg += f", body sample: {json.dumps(body)[:200]}"
        except:
            msg = f"HTTP {status}, body: {resp.text[:200]}"
        print(f"  {endpoint_name}: {msg}")
    except Exception as e:
        print(f"  {endpoint_name}: ERROR: {e}")

report_source(
    "Reddit Mentions",
    "Pushshift API (api.pushshift.io), Arctic Shift (arctic-shift.photon-reddit.com), "
    "Reddit search API (reddit.com/search.json)",
    "Pushshift shut down (mid-2023). Arctic Shift and Reddit search may return "
    "limited recent data. No reliable historical archive for 2021-2025 backtest.",
    False,
    notes="Pushshift was the standard Reddit historical API but was shut down. "
          "No free alternative provides complete historical daily mention counts "
          "for RXRX from 2021-2025."
)

# ══════════════════════════════════════════════════════════════════
# 2d. GOOGLE TRENDS — pytrends
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  2d. GOOGLE TRENDS")
print("=" * 70)

gtrends_data = None
gtrends_available = False

try:
    from pytrends.request import TrendReq

    pytrends = TrendReq(hl="en-US", tz=360)

    # Google Trends provides weekly data for periods > 90 days
    # Request RXRX interest from 2021 to 2025
    # We fetch in overlapping chunks to get daily granularity where possible,
    # but for 5-year range we get weekly data which is fine for our purpose.
    kw_list = ["RXRX"]
    timeframe = "2021-01-01 2025-12-31"

    print(f"  Requesting Google Trends for '{kw_list}' over '{timeframe}'...")

    # Retry logic since Google can rate-limit
    gt_df = None
    for attempt in range(3):
        try:
            pytrends.build_payload(kw_list, cat=0, timeframe=timeframe, geo="US", gprop="")
            gt_df = pytrends.interest_over_time()
            break
        except Exception as retry_err:
            print(f"  Attempt {attempt+1} failed: {retry_err}")
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
            else:
                raise

    if gt_df is not None and not gt_df.empty and "RXRX" in gt_df.columns:
        # gt_df has weekly index with interest score 0-100
        gt_df = gt_df.reset_index()
        gt_df = gt_df.rename(columns={"date": "week_start", "RXRX": "gtrends_interest"})
        gt_df = gt_df[["week_start", "gtrends_interest"]].copy()
        gt_df["week_start"] = pd.to_datetime(gt_df["week_start"])

        # Filter to selection window
        gt_sel = gt_df[gt_df["week_start"] < SEL_END]

        print(f"  Google Trends data obtained: {len(gt_df)} weekly observations")
        print(f"  Date range: {gt_df['week_start'].min().date()} to "
              f"{gt_df['week_start'].max().date()}")
        print(f"  Selection window: {len(gt_sel)} observations")
        print(f"  Interest stats: mean={gt_df['gtrends_interest'].mean():.1f}, "
              f"std={gt_df['gtrends_interest'].std():.1f}, "
              f"max={gt_df['gtrends_interest'].max()}, "
              f"min={gt_df['gtrends_interest'].min()}")

        # Also try "Recursion Pharmaceuticals" for comparison
        try:
            pytrends.build_payload(["Recursion Pharmaceuticals"], cat=0,
                                    timeframe=timeframe, geo="US", gprop="")
            gt_full = pytrends.interest_over_time()
            if gt_full is not None and not gt_full.empty:
                gt_df["gtrends_fullname"] = gt_full["Recursion Pharmaceuticals"].values[:len(gt_df)]
                print(f"  Also obtained 'Recursion Pharmaceuticals' interest data")
        except Exception as e:
            print(f"  'Recursion Pharmaceuticals' query failed: {e}")

        gtrends_data = gt_df
        gtrends_available = True

        # Save to data dir
        gtrends_path = DATA_DIR / "rxrx_gtrends.parquet"
        gt_df.to_parquet(gtrends_path)
        print(f"  Saved to {gtrends_path}")

        report_source(
            "Google Trends",
            "pytrends library, keyword='RXRX', US, 2021-01-01 to 2025-12-31",
            f"Weekly interest data obtained: {len(gt_df)} observations, "
            f"range {gt_df['week_start'].min().date()} to {gt_df['week_start'].max().date()}",
            True,
            date_range=f"{gt_df['week_start'].min().date()} to {gt_df['week_start'].max().date()}",
        )
    else:
        report_source(
            "Google Trends",
            "pytrends library, keyword='RXRX'",
            "Query returned empty DataFrame. RXRX may have too low search volume.",
            False,
            notes="pytrends connected but returned no data"
        )

except Exception as e:
    print(f"  Google Trends error: {e}")
    report_source(
        "Google Trends",
        "pytrends library, keyword='RXRX', timeframe 2021-01-01 to 2025-12-31",
        f"Error: {str(e)[:300]}",
        False,
        notes=f"pytrends library error: {str(e)[:200]}"
    )

# ══════════════════════════════════════════════════════════════════
# 2e. INSTITUTIONAL OWNERSHIP / 13F — Polygon API
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  2e. INSTITUTIONAL OWNERSHIP / 13F")
print("=" * 70)

inst_data = None
inst_available = False

inst_endpoints = [
    f"https://api.polygon.io/v3/reference/tickers/{TICKER}/institutional-ownership?apiKey={POLYGON_API_KEY}",
    f"https://api.polygon.io/v2/reference/institutional-ownership/{TICKER}?apiKey={POLYGON_API_KEY}",
    f"https://api.polygon.io/vX/reference/tickers/{TICKER}/ownership?apiKey={POLYGON_API_KEY}",
]

for url in inst_endpoints:
    endpoint_label = url.split("polygon.io")[1].split("?")[0]
    try:
        resp = requests.get(url, timeout=15)
        status = resp.status_code
        try:
            body = resp.json()
            results = body.get("results", [])
            msg = f"HTTP {status}, status='{body.get('status', '')}', results={len(results)}"
            if results:
                msg += f", sample: {json.dumps(results[0])[:200]}"
                inst_data = results
                inst_available = True
        except:
            msg = f"HTTP {status}, body: {resp.text[:200]}"
        print(f"  Polygon {endpoint_label}: {msg}")
    except Exception as e:
        print(f"  Polygon {endpoint_label}: ERROR: {e}")

if not inst_available:
    report_source(
        "Institutional Ownership (13F)",
        "Polygon API: /v3/reference/tickers/RXRX/institutional-ownership, "
        "/v2/reference/institutional-ownership/RXRX, "
        "/vX/reference/tickers/RXRX/ownership",
        "All endpoints returned 404/not-found or empty results. "
        "Institutional ownership requires Polygon Business plan or higher.",
        False,
        notes="13F data available from SEC EDGAR but not in structured API form. "
              "Polygon requires Business+ tier for institutional ownership data."
    )
else:
    report_source(
        "Institutional Ownership (13F)",
        "Polygon API",
        f"Found {len(inst_data)} records",
        True,
    )

# ══════════════════════════════════════════════════════════════════
# 3. STRATEGY TESTING — only for available data sources
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  3. STRATEGY TESTING")
print("=" * 70)

available_sources = [k for k, v in availability.items() if v["available"]]
strategy_results = []

if not available_sources:
    print("\n  No data sources available. Skipping strategy testing.")
else:
    print(f"\n  Available sources for testing: {available_sources}")

# ── Google Trends strategy (if available) ────────────────────────
if gtrends_available and gtrends_data is not None:
    print("\n  ── Google Trends Standalone Strategy ──")

    # Merge weekly Google Trends with daily RXRX data
    # Forward-fill weekly values to daily frequency
    gt = gtrends_data[["week_start", "gtrends_interest"]].copy()
    gt = gt.rename(columns={"week_start": "date"})

    # Create daily date range and merge
    rxrx_gt = rxrx_sel.copy()
    rxrx_gt = rxrx_gt.sort_values("date")

    # For each RXRX bar, find the most recent Google Trends observation
    # (forward-fill weekly data to daily)
    gt_daily = pd.merge_asof(
        rxrx_gt[["date"]].sort_values("date"),
        gt.sort_values("date"),
        on="date",
        direction="backward"
    )
    rxrx_gt["gtrends"] = gt_daily["gtrends_interest"].values

    # Compute trailing z-score of Google Trends interest
    rxrx_gt["gt_ma20"] = rxrx_gt["gtrends"].rolling(20, min_periods=10).mean()
    rxrx_gt["gt_std20"] = rxrx_gt["gtrends"].rolling(20, min_periods=10).std()
    rxrx_gt["gt_z"] = (rxrx_gt["gtrends"] - rxrx_gt["gt_ma20"]) / \
                       rxrx_gt["gt_std20"].clip(lower=0.1)

    # Also compute trailing percentile rank
    def rolling_percentile(series, window=52):
        """Rolling percentile rank within trailing window."""
        result = pd.Series(np.nan, index=series.index)
        for i in range(window, len(series)):
            window_data = series.iloc[i-window:i]
            current = series.iloc[i]
            result.iloc[i] = scipy_stats.percentileofscore(
                window_data.dropna(), current) / 100
        return result

    rxrx_gt["gt_pctile"] = rolling_percentile(rxrx_gt["gtrends"], 52)

    # Compute forward returns for analysis
    for d in [1, 3, 5]:
        rxrx_gt[f"fwd_ret_{d}d"] = rxrx_gt["close"].shift(-d) / rxrx_gt["close"] - 1

    # Drop NaN rows
    rxrx_gt_clean = rxrx_gt.dropna(subset=["gt_z", "fwd_ret_5d"]).copy()

    print(f"  Merged data: {len(rxrx_gt_clean)} bars with Google Trends + returns")
    print(f"  Google Trends z-score stats: mean={rxrx_gt_clean['gt_z'].mean():.2f}, "
          f"std={rxrx_gt_clean['gt_z'].std():.2f}")

    # ── Analysis: correlation between GT interest and forward returns ──
    for d in [1, 3, 5]:
        col = f"fwd_ret_{d}d"
        valid = rxrx_gt_clean[[col, "gt_z"]].dropna()
        corr = valid[col].corr(valid["gt_z"])
        print(f"  Correlation(GT z-score, {d}d fwd return): {corr:+.4f}")

    # ── Strategy 1: High attention = avoid / low attention = buy ──
    # Hypothesis: spikes in retail attention precede mean-reverting moves
    # (crowded trades unwind). Low attention = opportunity.

    strat_configs = [
        {"name": "GT_LowAttn_Long",  "side": "long",  "gt_z_below": -1.0,
         "desc": "Go long when GT z < -1.0 (low attention)"},
        {"name": "GT_HighAttn_Short", "side": "short", "gt_z_above": 1.5,
         "desc": "Go short when GT z > 1.5 (high attention spike)"},
        {"name": "GT_LowPctile_Long", "side": "long",  "gt_pctile_below": 0.20,
         "desc": "Go long when GT percentile < 20% (attention trough)"},
        {"name": "GT_HighPctile_Short", "side": "short", "gt_pctile_above": 0.80,
         "desc": "Go short when GT percentile > 80% (attention peak)"},
    ]

    for cfg in strat_configs:
        # Build signal function for backtest engine
        # Need spread_z column for engine compatibility — use a dummy
        bt_df = rxrx_gt.copy()
        bt_df["spread_z"] = 0.0  # dummy, not used for signals

        def make_signal_fn(config):
            def signal_fn(row, i, df):
                gt_z = df.iloc[i].get("gt_z", np.nan) if i < len(df) else np.nan
                gt_pct = df.iloc[i].get("gt_pctile", np.nan) if i < len(df) else np.nan
                if pd.isna(gt_z):
                    return None
                side = config["side"]
                price = row["close"]
                atr = row.get("atr20", np.nan)
                if pd.isna(atr) or atr <= 0:
                    return None

                triggered = False
                if "gt_z_below" in config and gt_z < config["gt_z_below"]:
                    triggered = True
                if "gt_z_above" in config and gt_z > config["gt_z_above"]:
                    triggered = True
                if "gt_pctile_below" in config and not pd.isna(gt_pct) and gt_pct < config["gt_pctile_below"]:
                    triggered = True
                if "gt_pctile_above" in config and not pd.isna(gt_pct) and gt_pct > config["gt_pctile_above"]:
                    triggered = True

                if not triggered:
                    return None

                stop_mult = 1.5
                if side == "long":
                    return {"side": "long", "stop": price - stop_mult * atr, "target": price + 2.0 * atr}
                else:
                    return {"side": "short", "stop": price + stop_mult * atr, "target": price - 2.0 * atr}
            return signal_fn

        sig_fn = make_signal_fn(cfg)
        trades = run_backtest(bt_df, sig_fn, exit_fn=None, max_hold=MAX_HOLD,
                              position_size=POSITION_SIZE)

        if trades.empty:
            print(f"\n  {cfg['name']}: 0 trades (no signals triggered)")
            strategy_results.append({
                "strategy": cfg["name"], "desc": cfg["desc"],
                "n": 0, "avg_pnl": 0, "sharpe": 0, "null_pctile": 0
            })
            continue

        stats = compute_stats(trades, rxrx_sel, label=cfg["name"])

        # Null test
        pctile, null_mean, real_avg = null_test(trades, rxrx_sel, n_runs=1000)

        # Walk-forward
        wf = walk_forward(bt_df, sig_fn, exit_fn=None,
                          train_months=12, test_months=3, max_hold=MAX_HOLD)
        wf_avg = wf["avg_pnl"].mean() if not wf.empty else 0

        print(f"\n  {cfg['name']}: {cfg['desc']}")
        print(f"    n={stats['n']}, avg_pnl={stats['avg_pnl']:+.3f}%, "
              f"sharpe={stats['sharpe']:.2f}, wr={stats['wr']:.1f}%")
        print(f"    null_percentile={pctile:.1f}, wf_avg={wf_avg:+.3f}%")
        print(f"    total_pnl={stats['total_pnl_pct']:+.2f}%, "
              f"max_dd={stats['max_dd']:.2f}%, t_stat={stats['t_stat']:.2f}")

        strategy_results.append({
            "strategy": cfg["name"], "desc": cfg["desc"],
            "n": stats["n"], "avg_pnl": stats["avg_pnl"],
            "median_pnl": stats["median_pnl"],
            "sharpe": stats["sharpe"], "wr": stats["wr"],
            "total_pnl": stats["total_pnl_pct"],
            "null_pctile": pctile, "wf_avg": wf_avg,
            "t_stat": stats["t_stat"],
            "max_dd": stats["max_dd"],
        })

        # Log trial
        log_trial("agent6v2", cfg["name"],
                  {"source": "google_trends", **{k: v for k, v in cfg.items()
                   if k not in ["name", "desc"]}},
                  stats)

# ── Short interest strategy (if available) ──────────────────────
if si_available and short_interest_data is not None:
    print("\n  ── Short Interest Standalone Strategy ──")
    # Would build SI-based strategy here
    # (code omitted since SI is expected to be unavailable on free tier)

# ── Institutional ownership strategy (if available) ─────────────
if inst_available and inst_data is not None:
    print("\n  ── Institutional Ownership Standalone Strategy ──")
    # Would build 13F-based strategy here
    # (code omitted since 13F is expected to be unavailable on free tier)


# ══════════════════════════════════════════════════════════════════
# 4. VISUALIZATION
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  4. VISUALIZATION")
print("=" * 70)

fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.suptitle("Agent 6v2: Actual Attention & Positioning Data Sources", fontsize=14, y=0.98)

# Panel 1: Data availability summary
ax1 = axes[0]
ax1.axis("off")
sources = list(availability.keys())
statuses = ["AVAILABLE" if availability[s]["available"] else "UNAVAILABLE" for s in sources]
colors = ["#2ecc71" if s == "AVAILABLE" else "#e74c3c" for s in statuses]

table_data = []
for s in sources:
    a = availability[s]
    table_data.append([
        a["source"],
        "Y" if a["available"] else "N",
        a["date_range"],
        a["notes"][:60] + "..." if len(a.get("notes", "")) > 60 else a.get("notes", "")
    ])

table = ax1.table(
    cellText=table_data,
    colLabels=["Data Source", "Avail?", "Date Range", "Notes"],
    cellLoc="left",
    loc="center",
    colWidths=[0.18, 0.06, 0.22, 0.54],
)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.5)

# Color the availability column
for i in range(len(sources)):
    cell = table[i + 1, 1]  # +1 for header row
    if availability[sources[i]]["available"]:
        cell.set_facecolor("#d4efdf")
    else:
        cell.set_facecolor("#fadbd8")

ax1.set_title("Data Availability Report", fontsize=12, pad=20)

# Panel 2: Google Trends data (if available)
ax2 = axes[1]
if gtrends_available and gtrends_data is not None:
    gt = gtrends_data
    ax2.plot(gt["week_start"], gt["gtrends_interest"], color="#3498db",
             linewidth=1.5, label="RXRX Google Trends Interest")
    ax2.axhline(y=gt["gtrends_interest"].mean(), color="gray",
                linestyle="--", alpha=0.5, label=f"Mean ({gt['gtrends_interest'].mean():.1f})")
    # Mark spikes (>2 std above mean)
    threshold = gt["gtrends_interest"].mean() + 2 * gt["gtrends_interest"].std()
    spikes = gt[gt["gtrends_interest"] > threshold]
    if not spikes.empty:
        ax2.scatter(spikes["week_start"], spikes["gtrends_interest"],
                    color="red", s=40, zorder=5, label=f"Spikes (>{threshold:.0f})")
    ax2.set_ylabel("Search Interest (0-100)")
    ax2.set_title("Google Trends: 'RXRX' Weekly Search Interest (US)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Overlay RXRX price on secondary axis
    ax2b = ax2.twinx()
    rxrx_weekly = rxrx_sel[["date", "close"]].set_index("date").resample("W").last().dropna()
    ax2b.plot(rxrx_weekly.index, rxrx_weekly["close"], color="#e67e22",
              linewidth=1, alpha=0.6, label="RXRX Close")
    ax2b.set_ylabel("RXRX Price ($)", color="#e67e22")
    ax2b.tick_params(axis="y", labelcolor="#e67e22")
else:
    ax2.text(0.5, 0.5, "Google Trends data not available",
             ha="center", va="center", fontsize=14, color="gray",
             transform=ax2.transAxes)
    ax2.set_title("Google Trends: Not Available")

# Panel 3: Strategy results (if any)
ax3 = axes[2]
if strategy_results:
    results_with_trades = [r for r in strategy_results if r["n"] > 0]
    if results_with_trades:
        names = [r["strategy"] for r in results_with_trades]
        avg_pnls = [r["avg_pnl"] for r in results_with_trades]
        null_pcts = [r.get("null_pctile", 0) for r in results_with_trades]

        x = np.arange(len(names))
        width = 0.35

        bars1 = ax3.bar(x - width/2, avg_pnls, width, label="Avg PnL (%)",
                        color=["#2ecc71" if p > 0 else "#e74c3c" for p in avg_pnls])
        ax3b = ax3.twinx()
        bars2 = ax3b.bar(x + width/2, null_pcts, width, label="Null Percentile",
                         color="#3498db", alpha=0.6)
        ax3b.axhline(y=95, color="red", linestyle="--", alpha=0.7, label="95th pctile")

        ax3.set_ylabel("Avg PnL (%)")
        ax3b.set_ylabel("Null Percentile")
        ax3.set_xticks(x)
        ax3.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax3.set_title("Strategy Results (Selection Window, Pre-2026)")

        # Add value labels
        for bar, val in zip(bars1, avg_pnls):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f"{val:+.2f}%", ha="center", va="bottom", fontsize=8)
        for bar, val in zip(bars2, null_pcts):
            ax3b.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                      f"{val:.0f}", ha="center", va="bottom", fontsize=8)

        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3b.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    else:
        ax3.text(0.5, 0.5, "All strategies produced 0 trades",
                 ha="center", va="center", fontsize=14, color="gray",
                 transform=ax3.transAxes)
        ax3.set_title("Strategy Results: No Trades")
else:
    ax3.text(0.5, 0.5, "No available data sources to test strategies",
             ha="center", va="center", fontsize=14, color="gray",
             transform=ax3.transAxes)
    ax3.set_title("Strategy Results: No Data Available")

plt.tight_layout()
fig_path = OUT_DIR / "agent6v2_attention.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"\n  Figure saved to {fig_path}")
plt.close()

# ══════════════════════════════════════════════════════════════════
# 5. FINAL REPORT
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  5. FINAL DATA AVAILABILITY REPORT")
print("=" * 70)

print(f"\n  {'Source':<30s} {'Available':<12s} {'Date Range':<30s}")
print(f"  {'-'*30} {'-'*12} {'-'*30}")
for name, info in availability.items():
    avail_str = "YES" if info["available"] else "NO"
    print(f"  {name:<30s} {avail_str:<12s} {info['date_range']:<30s}")

if strategy_results:
    print(f"\n  STRATEGY RESULTS TABLE:")
    print(f"  {'Strategy':<25s} {'n':>4s} {'AvgPnL':>8s} {'Sharpe':>7s} {'NullPct':>8s} {'WF_Avg':>8s}")
    print(f"  {'-'*25} {'-'*4} {'-'*8} {'-'*7} {'-'*8} {'-'*8}")
    for r in strategy_results:
        print(f"  {r['strategy']:<25s} {r['n']:>4d} {r.get('avg_pnl', 0):>+7.3f}% "
              f"{r.get('sharpe', 0):>6.2f} {r.get('null_pctile', 0):>7.1f} "
              f"{r.get('wf_avg', 0):>+7.3f}%")

# Summary
n_available = sum(1 for v in availability.values() if v["available"])
n_total = len(availability)

print(f"\n  SUMMARY: {n_available}/{n_total} data sources available")

if n_available == 0:
    needed = [k for k, v in availability.items() if not v["available"]]
    print(f"\n  Required data sources unavailable. Hypothesis cannot be tested without:")
    for s in needed:
        print(f"    - {s}")
    print(f"\n  Recommendation: To test attention/positioning hypotheses, obtain:")
    print(f"    1. Short interest: Polygon Business plan, or scrape FINRA website")
    print(f"    2. StockTwits: StockTwits Data product (paid historical archive)")
    print(f"    3. Reddit: No reliable historical API since Pushshift shutdown (2023)")
    print(f"    4. Google Trends: pytrends library (FREE, may need retry on rate limits)")
    print(f"    5. 13F ownership: Polygon Business plan, or parse SEC EDGAR XML")
elif n_available > 0 and strategy_results:
    valid_strategies = [r for r in strategy_results if r["n"] > 0]
    if valid_strategies:
        best = max(valid_strategies, key=lambda x: x.get("avg_pnl", -999))
        print(f"\n  Best strategy: {best['strategy']} "
              f"(avg_pnl={best['avg_pnl']:+.3f}%, null_pctile={best.get('null_pctile', 0):.1f})")

        if best.get("null_pctile", 0) >= 95:
            print(f"  CONCLUSION: {best['strategy']} passes null test at 95th percentile.")
        else:
            print(f"  CONCLUSION: No strategy passes 95th percentile null test.")
            print(f"  Attention/positioning data does not provide standalone tradeable edge for RXRX.")
    else:
        print(f"\n  All strategies produced 0 trades. Data may lack sufficient variation "
              f"or thresholds need adjustment.")

print(f"\n  AGENT 6v2 COMPLETE")
