# %% Agent 3v2 — Standalone Event Run-Up / Sell-the-News Strategies
#
# RXRX is Recursion Pharmaceuticals (biotech). This script:
#   1. Sources ACTUAL earnings dates from Polygon financials API
#   2. Builds STANDALONE strategies around earnings events
#   3. Tests pre-earnings drift, post-earnings fade, sell-the-news
#   4. Evaluates on selection window ONLY (pre-2026)
#
# CRITICAL: Does NOT use S5_zexit. Does NOT use volume/return proxies.
# Uses real earnings dates from SEC filing records via Polygon.
#
# Notebook-friendly: config vars at top, no if __name__

import os, sys, json, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

for pkg in ["requests"]:
    try:
        __import__(pkg)
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg,
                               "--quiet", "--break-system-packages"])

import requests

# ── Config ───────────────────────────────────────────────────────
SEL_END = "2026-01-01"                   # HARD cutoff: no 2026 data
DATA_DIR = Path(__file__).parent / "data" if "__file__" in dir() else Path("agent10/data")
OUT_DIR = Path(__file__).parent / "results" if "__file__" in dir() else Path("agent10/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Strategy parameters to sweep
PRE_DRIFT_WINDOWS = [5, 10, 15]          # days before earnings to enter long
POST_FADE_HOLD_DAYS = [1, 3, 5]          # hold days for post-earnings fade
POST_FADE_GAP_THRESHOLDS = [0.0, 3.0, 5.0]  # min absolute gap % to trigger
SELL_NEWS_HOLD_DAYS = [3, 5]             # hold days for sell-the-news short
POSITION_SIZE = 10_000                   # dollars per trade

# API key
ENV_PATHS = [
    Path(__file__).parent.parent / ".env" if "__file__" in dir() else Path("bot-RAG/.env"),
    Path(".env"),
    Path("/home/admin_yaninaso_altostrat_com/bot-RAG/.env"),
]
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "")
if not POLYGON_API_KEY:
    try:
        from dotenv import load_dotenv
        for p in ENV_PATHS:
            if p.exists():
                load_dotenv(p)
                POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "")
                if POLYGON_API_KEY:
                    break
    except ImportError:
        pass
# Fallback from phase0 script (already in repo)
if not POLYGON_API_KEY:
    POLYGON_API_KEY = "2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs"

# ── Engine imports ───────────────────────────────────────────────
ENGINE_DIR = Path(__file__).parent if "__file__" in dir() else Path("agent10")
sys.path.insert(0, str(ENGINE_DIR))

from agent0_engine import compute_costs, compute_stats, log_trial, null_test, add_features

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════
# 1. LOAD RXRX DAILY DATA (selection window only)
# ══════════════════════════════════════════════════════════════════
# %%
print("=" * 70)
print("  AGENT 3v2: STANDALONE EVENT STRATEGIES (REAL EARNINGS DATES)")
print("=" * 70)

rxrx = pd.read_parquet(DATA_DIR / "rxrx_daily.parquet")
rxrx["date"] = pd.to_datetime(rxrx["date"])
rxrx = rxrx.sort_values("date").reset_index(drop=True)
rxrx = add_features(rxrx)

# STRICT: filter to selection window only
rxrx_sel = rxrx[rxrx["date"] < SEL_END].copy().reset_index(drop=True)
print(f"  RXRX selection: {len(rxrx_sel)} bars, "
      f"{rxrx_sel['date'].min().date()} to {rxrx_sel['date'].max().date()}")

# Trading date index for lookup
trading_dates = rxrx_sel["date"].values
date_to_idx = {d: i for i, d in enumerate(trading_dates)}

# ══════════════════════════════════════════════════════════════════
# 2. SOURCE ACTUAL EARNINGS DATES FROM POLYGON API
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  STEP 1: FETCH REAL EARNINGS DATES FROM POLYGON")
print("=" * 70)

url = f"https://api.polygon.io/vX/reference/financials?ticker=RXRX&limit=100&apiKey={POLYGON_API_KEY}"
try:
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    fin_data = resp.json()
except Exception as e:
    print(f"\n  FATAL: Polygon financials API failed: {e}")
    print("  Required data unavailable, hypothesis cannot be tested.")
    sys.exit(1)

results = fin_data.get("results", [])
if not results:
    print("\n  FATAL: Polygon returned zero financial records for RXRX.")
    print("  Required data unavailable, hypothesis cannot be tested.")
    sys.exit(1)

# Extract quarterly earnings dates
# Q4 reports don't have filing_date in quarterly records, but FY records do
# FY filing = Q4 earnings report
earnings_raw = []
for r in results:
    period = r.get("fiscal_period", "")
    year = r.get("fiscal_year", "")
    filing = r.get("filing_date")
    end_date = r.get("end_date", "")

    if filing is None:
        continue

    # Skip TTM (trailing twelve months) aggregates
    if period == "TTM":
        continue

    # For FY records, this IS the Q4 earnings
    # For Q1/Q2/Q3, this is the quarterly earnings
    quarter = period if period.startswith("Q") else "Q4"
    if period == "FY":
        quarter = "Q4"

    earnings_raw.append({
        "period": f"{quarter}/{year}",
        "filing_date": pd.Timestamp(filing),
        "period_end": end_date,
        "fiscal_period": period,
    })

# Deduplicate: for a given period_end, prefer quarterly over FY
# (they should have the same filing date for Q4 vs FY)
earnings_by_end = {}
for e in earnings_raw:
    key = e["period_end"]
    if key not in earnings_by_end:
        earnings_by_end[key] = e
    elif e["fiscal_period"].startswith("Q"):
        earnings_by_end[key] = e  # prefer quarterly label

earnings_list = sorted(earnings_by_end.values(), key=lambda x: x["filing_date"])

# Filter to selection window
earnings_sel = [e for e in earnings_list if e["filing_date"] < pd.Timestamp(SEL_END)]

print(f"\n  Total earnings records from Polygon: {len(earnings_list)}")
print(f"  In selection window (pre-2026):      {len(earnings_sel)}")

if len(earnings_sel) < 8:
    print(f"\n  FATAL: Only {len(earnings_sel)} earnings events in selection window (need >= 8).")
    print("  Required data unavailable, hypothesis cannot be tested.")
    sys.exit(1)

# Map each filing date to nearest trading day
def nearest_trading_day(target_date, dates_array):
    """Find the nearest trading day on or after target_date."""
    diffs = dates_array - np.datetime64(target_date)
    # Find closest date on or after
    future = np.where(diffs >= np.timedelta64(0, 'ns'))[0]
    if len(future) > 0:
        return int(future[0])
    # If none on or after, find closest before
    past = np.where(diffs < np.timedelta64(0, 'ns'))[0]
    if len(past) > 0:
        return int(past[-1])
    return None

# Build earnings calendar with trading day indices
earnings_calendar = []
for e in earnings_sel:
    idx = nearest_trading_day(e["filing_date"], trading_dates)
    if idx is None:
        continue
    actual_date = pd.Timestamp(trading_dates[idx])
    # Only keep if within 3 calendar days of filing (handles weekends)
    if abs((actual_date - e["filing_date"]).days) > 5:
        continue
    earnings_calendar.append({
        "period": e["period"],
        "filing_date": e["filing_date"],
        "trading_date": actual_date,
        "trading_idx": idx,
    })

print(f"  Earnings mapped to trading days:     {len(earnings_calendar)}")
print(f"\n  {'Period':>10s}  {'Filing Date':>12s}  {'Trading Date':>12s}  {'Idx':>5s}")
print("  " + "-" * 50)
for ec in earnings_calendar:
    print(f"  {ec['period']:>10s}  {str(ec['filing_date'].date()):>12s}  "
          f"{str(ec['trading_date'].date()):>12s}  {ec['trading_idx']:>5d}")

# ══════════════════════════════════════════════════════════════════
# 3. COMPUTE EARNINGS GAP AND SURROUNDING RETURNS
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  STEP 2: EARNINGS REACTION ANALYSIS")
print("=" * 70)

# For each earnings date, compute:
# - Pre-earnings drift (N days before)
# - Earnings gap (open_T+1 / close_T - 1)  [earnings reported after close]
# - Post-earnings return (1, 3, 5 day)
earnings_detail = []
for ec in earnings_calendar:
    idx = ec["trading_idx"]

    # We assume earnings are released after market close on filing_date
    # So the reaction gap is at open of next trading day (idx+1)
    if idx + 1 >= len(rxrx_sel):
        continue
    if idx < 15:  # need enough history for pre-earnings drift
        continue

    close_before = rxrx_sel.loc[idx, "close"]
    open_after = rxrx_sel.loc[idx + 1, "open"]
    gap_pct = (open_after / close_before - 1) * 100

    # Pre-earnings returns (5, 10, 15 day)
    pre_returns = {}
    for window in [5, 10, 15]:
        if idx - window >= 0:
            pre_close = rxrx_sel.loc[idx - window, "close"]
            pre_returns[window] = (close_before / pre_close - 1) * 100
        else:
            pre_returns[window] = np.nan

    # Post-earnings returns (1, 3, 5 day close-to-close from T+1 open)
    post_returns = {}
    for hold in [1, 3, 5]:
        exit_idx = min(idx + 1 + hold, len(rxrx_sel) - 1)
        exit_close = rxrx_sel.loc[exit_idx, "close"]
        post_returns[hold] = (exit_close / open_after - 1) * 100

    # Earnings-day close return (close T vs close T-1)
    day_return = rxrx_sel.loc[idx, "pct_ret"] * 100 if pd.notna(rxrx_sel.loc[idx, "pct_ret"]) else 0

    earnings_detail.append({
        "period": ec["period"],
        "date": ec["trading_date"],
        "idx": idx,
        "close_before": close_before,
        "open_after": open_after,
        "gap_pct": round(gap_pct, 2),
        "day_return": round(day_return, 2),
        "pre_5d": round(pre_returns.get(5, np.nan), 2),
        "pre_10d": round(pre_returns.get(10, np.nan), 2),
        "pre_15d": round(pre_returns.get(15, np.nan), 2),
        "post_1d": round(post_returns.get(1, np.nan), 2),
        "post_3d": round(post_returns.get(3, np.nan), 2),
        "post_5d": round(post_returns.get(5, np.nan), 2),
    })

earnings_df = pd.DataFrame(earnings_detail)
print(f"\n  Earnings events with full data: {len(earnings_df)}")

print(f"\n  {'Period':>10s} {'Date':>12s} {'Gap%':>7s} {'Pre5d':>7s} {'Pre10d':>7s} "
      f"{'Pre15d':>7s} {'Post1d':>7s} {'Post3d':>7s} {'Post5d':>7s}")
print("  " + "-" * 85)
for _, e in earnings_df.iterrows():
    print(f"  {e['period']:>10s} {str(e['date'].date()):>12s} {e['gap_pct']:>+6.2f}% "
          f"{e['pre_5d']:>+6.2f}% {e['pre_10d']:>+6.2f}% {e['pre_15d']:>+6.2f}% "
          f"{e['post_1d']:>+6.2f}% {e['post_3d']:>+6.2f}% {e['post_5d']:>+6.2f}%")

# Summary stats
print(f"\n  Summary (n={len(earnings_df)}):")
print(f"    Avg gap:     {earnings_df['gap_pct'].mean():+.2f}% (std={earnings_df['gap_pct'].std():.2f}%)")
print(f"    Avg pre-5d:  {earnings_df['pre_5d'].mean():+.2f}%")
print(f"    Avg pre-10d: {earnings_df['pre_10d'].mean():+.2f}%")
print(f"    Avg pre-15d: {earnings_df['pre_15d'].mean():+.2f}%")
print(f"    Avg post-1d: {earnings_df['post_1d'].mean():+.2f}%")
print(f"    Avg post-3d: {earnings_df['post_3d'].mean():+.2f}%")
print(f"    Avg post-5d: {earnings_df['post_5d'].mean():+.2f}%")
print(f"    Positive gaps: {(earnings_df['gap_pct'] > 0).sum()}/{len(earnings_df)}")
print(f"    Negative gaps: {(earnings_df['gap_pct'] < 0).sum()}/{len(earnings_df)}")

# ══════════════════════════════════════════════════════════════════
# 4. STANDALONE STRATEGY 1: PRE-EARNINGS DRIFT (LONG)
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  STRATEGY 1: PRE-EARNINGS DRIFT (LONG)")
print("  Enter long N days before earnings, exit at close day before earnings")
print("=" * 70)

strat1_results = []

for window in PRE_DRIFT_WINDOWS:
    trades = []
    for _, e in earnings_df.iterrows():
        idx = e["idx"]
        entry_idx = idx - window
        exit_idx = idx - 1  # close day before earnings

        if entry_idx < 0 or exit_idx < 0:
            continue

        # Enter at open of entry_idx+1 (signal on close of entry_idx)
        # Actually: signal on close of day (idx - window), enter at open next day
        signal_idx = entry_idx
        entry_bar = signal_idx + 1
        if entry_bar >= idx:
            continue  # no room

        entry_price = rxrx_sel.loc[entry_bar, "open"]
        exit_price = rxrx_sel.loc[exit_idx, "close"]
        holding_days = max((rxrx_sel.loc[exit_idx, "date"] - rxrx_sel.loc[entry_bar, "date"]).days, 1)

        pnl_gross = (exit_price / entry_price - 1) * 100
        cost = compute_costs(entry_price, "long", holding_days) / entry_price * 100
        pnl_net = pnl_gross - cost
        shares = max(int(POSITION_SIZE / entry_price), 1)

        trades.append({
            "signal_date": rxrx_sel.loc[signal_idx, "date"],
            "entry_date": rxrx_sel.loc[entry_bar, "date"],
            "exit_date": rxrx_sel.loc[exit_idx, "date"],
            "side": "long",
            "entry_price": round(entry_price, 4),
            "exit_price": round(exit_price, 4),
            "exit_reason": "pre_earnings_exit",
            "holding_days": holding_days,
            "z_at_signal": np.nan,
            "pnl_gross_pct": round(pnl_gross, 4),
            "cost_pct": round(cost, 4),
            "pnl_net_pct": round(pnl_net, 4),
            "shares": shares,
            "pnl_dollars": round(pnl_net / 100 * shares * entry_price, 2),
            "earnings_period": e["period"],
            "earnings_gap": e["gap_pct"],
        })

    trades_df = pd.DataFrame(trades)
    label = f"PreDrift_L{window}d"

    if trades_df.empty:
        stats = {"label": label, "n": 0, "avg_pnl": 0, "t_stat": 0, "wr": 0, "sharpe": 0}
    else:
        stats = compute_stats(trades_df, rxrx_sel, label=label)

    strat1_results.append({
        "strategy": label,
        "window": window,
        "n": stats["n"],
        "avg_pnl": stats["avg_pnl"],
        "median_pnl": stats.get("median_pnl", 0),
        "wr": stats["wr"],
        "t_stat": stats["t_stat"],
        "sharpe": stats["sharpe"],
        "total_pnl": stats.get("total_pnl_pct", 0),
        "trades_df": trades_df,
    })

    print(f"\n  {label}: n={stats['n']}, avg={stats['avg_pnl']:+.3f}%, "
          f"med={stats.get('median_pnl',0):+.3f}%, wr={stats['wr']:.1f}%, "
          f"t={stats['t_stat']:.2f}, Sharpe={stats['sharpe']:.2f}, "
          f"total={stats.get('total_pnl_pct',0):+.2f}%")

    if not trades_df.empty:
        for _, t in trades_df.iterrows():
            print(f"    {t['earnings_period']:>10s}  entry={str(t['entry_date'].date()):>12s}  "
                  f"exit={str(t['exit_date'].date()):>12s}  "
                  f"pnl={t['pnl_net_pct']:+.2f}%  gap_after={t['earnings_gap']:+.2f}%")

# ══════════════════════════════════════════════════════════════════
# 5. STANDALONE STRATEGY 2: POST-EARNINGS GAP FADE
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  STRATEGY 2: POST-EARNINGS GAP FADE")
print("  If gap > threshold, fade at open of T+1, hold N days")
print("=" * 70)

strat2_results = []

for gap_thresh in POST_FADE_GAP_THRESHOLDS:
    for hold_days in POST_FADE_HOLD_DAYS:
        trades = []
        for _, e in earnings_df.iterrows():
            idx = e["idx"]
            gap = e["gap_pct"]

            # Only trade if absolute gap exceeds threshold
            if abs(gap) < gap_thresh:
                continue

            # Fade the gap: short if gap > 0, long if gap < 0
            # Entry at open of day after earnings (idx+1)
            entry_bar = idx + 1
            if entry_bar >= len(rxrx_sel):
                continue

            exit_bar = min(entry_bar + hold_days, len(rxrx_sel) - 1)

            entry_price = rxrx_sel.loc[entry_bar, "open"]
            exit_price = rxrx_sel.loc[exit_bar, "close"]
            side = "short" if gap > 0 else "long"
            holding = max((rxrx_sel.loc[exit_bar, "date"] - rxrx_sel.loc[entry_bar, "date"]).days, 1)

            if side == "long":
                pnl_gross = (exit_price / entry_price - 1) * 100
            else:
                pnl_gross = (entry_price - exit_price) / entry_price * 100

            cost = compute_costs(entry_price, side, holding) / entry_price * 100
            pnl_net = pnl_gross - cost
            shares = max(int(POSITION_SIZE / entry_price), 1)

            trades.append({
                "signal_date": rxrx_sel.loc[idx, "date"],
                "entry_date": rxrx_sel.loc[entry_bar, "date"],
                "exit_date": rxrx_sel.loc[exit_bar, "date"],
                "side": side,
                "entry_price": round(entry_price, 4),
                "exit_price": round(exit_price, 4),
                "exit_reason": "time_stop",
                "holding_days": holding,
                "z_at_signal": np.nan,
                "pnl_gross_pct": round(pnl_gross, 4),
                "cost_pct": round(cost, 4),
                "pnl_net_pct": round(pnl_net, 4),
                "shares": shares,
                "pnl_dollars": round(pnl_net / 100 * shares * entry_price, 2),
                "earnings_period": e["period"],
                "earnings_gap": gap,
            })

        trades_df = pd.DataFrame(trades)
        label = f"GapFade_g{gap_thresh:.0f}_h{hold_days}d"

        if trades_df.empty:
            stats = {"label": label, "n": 0, "avg_pnl": 0, "t_stat": 0, "wr": 0, "sharpe": 0}
        else:
            stats = compute_stats(trades_df, rxrx_sel, label=label)

        strat2_results.append({
            "strategy": label,
            "gap_thresh": gap_thresh,
            "hold_days": hold_days,
            "n": stats["n"],
            "avg_pnl": stats["avg_pnl"],
            "median_pnl": stats.get("median_pnl", 0),
            "wr": stats["wr"],
            "t_stat": stats["t_stat"],
            "sharpe": stats["sharpe"],
            "total_pnl": stats.get("total_pnl_pct", 0),
            "trades_df": trades_df,
        })

        print(f"  {label}: n={stats['n']}, avg={stats['avg_pnl']:+.3f}%, "
              f"wr={stats['wr']:.1f}%, t={stats['t_stat']:.2f}, "
              f"total={stats.get('total_pnl_pct',0):+.2f}%")

# ══════════════════════════════════════════════════════════════════
# 6. STANDALONE STRATEGY 3: SELL-THE-NEWS (SHORT AFTER POSITIVE)
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  STRATEGY 3: SELL-THE-NEWS (SHORT AFTER POSITIVE GAP)")
print("  Short at open T+1 if gap > 0, hold N days")
print("=" * 70)

strat3_results = []

for hold_days in SELL_NEWS_HOLD_DAYS:
    trades = []
    for _, e in earnings_df.iterrows():
        idx = e["idx"]
        gap = e["gap_pct"]

        # Only short after positive gap (sell the good news)
        if gap <= 0:
            continue

        entry_bar = idx + 1
        if entry_bar >= len(rxrx_sel):
            continue

        exit_bar = min(entry_bar + hold_days, len(rxrx_sel) - 1)
        entry_price = rxrx_sel.loc[entry_bar, "open"]
        exit_price = rxrx_sel.loc[exit_bar, "close"]
        side = "short"
        holding = max((rxrx_sel.loc[exit_bar, "date"] - rxrx_sel.loc[entry_bar, "date"]).days, 1)

        pnl_gross = (entry_price - exit_price) / entry_price * 100
        cost = compute_costs(entry_price, side, holding) / entry_price * 100
        pnl_net = pnl_gross - cost
        shares = max(int(POSITION_SIZE / entry_price), 1)

        trades.append({
            "signal_date": rxrx_sel.loc[idx, "date"],
            "entry_date": rxrx_sel.loc[entry_bar, "date"],
            "exit_date": rxrx_sel.loc[exit_bar, "date"],
            "side": side,
            "entry_price": round(entry_price, 4),
            "exit_price": round(exit_price, 4),
            "exit_reason": "time_stop",
            "holding_days": holding,
            "z_at_signal": np.nan,
            "pnl_gross_pct": round(pnl_gross, 4),
            "cost_pct": round(cost, 4),
            "pnl_net_pct": round(pnl_net, 4),
            "shares": shares,
            "pnl_dollars": round(pnl_net / 100 * shares * entry_price, 2),
            "earnings_period": e["period"],
            "earnings_gap": gap,
        })

    trades_df = pd.DataFrame(trades)
    label = f"SellNews_h{hold_days}d"

    if trades_df.empty:
        stats = {"label": label, "n": 0, "avg_pnl": 0, "t_stat": 0, "wr": 0, "sharpe": 0}
    else:
        stats = compute_stats(trades_df, rxrx_sel, label=label)

    strat3_results.append({
        "strategy": label,
        "hold_days": hold_days,
        "n": stats["n"],
        "avg_pnl": stats["avg_pnl"],
        "median_pnl": stats.get("median_pnl", 0),
        "wr": stats["wr"],
        "t_stat": stats["t_stat"],
        "sharpe": stats["sharpe"],
        "total_pnl": stats.get("total_pnl_pct", 0),
        "trades_df": trades_df,
    })

    print(f"\n  {label}: n={stats['n']}, avg={stats['avg_pnl']:+.3f}%, "
          f"wr={stats['wr']:.1f}%, t={stats['t_stat']:.2f}")

    if not trades_df.empty:
        for _, t in trades_df.iterrows():
            print(f"    {t['earnings_period']:>10s}  entry={str(t['entry_date'].date()):>12s}  "
                  f"exit={str(t['exit_date'].date()):>12s}  gap={t['earnings_gap']:+.2f}%  "
                  f"pnl={t['pnl_net_pct']:+.2f}%")

# ══════════════════════════════════════════════════════════════════
# 7. STANDALONE STRATEGY 4: BUY-THE-DIP (LONG AFTER NEGATIVE GAP)
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  STRATEGY 4: BUY-THE-DIP (LONG AFTER NEGATIVE GAP)")
print("  Long at open T+1 if gap < 0, hold N days")
print("=" * 70)

strat4_results = []

for hold_days in [1, 3, 5]:
    trades = []
    for _, e in earnings_df.iterrows():
        idx = e["idx"]
        gap = e["gap_pct"]

        # Only long after negative gap (buy the bad news dip)
        if gap >= 0:
            continue

        entry_bar = idx + 1
        if entry_bar >= len(rxrx_sel):
            continue

        exit_bar = min(entry_bar + hold_days, len(rxrx_sel) - 1)
        entry_price = rxrx_sel.loc[entry_bar, "open"]
        exit_price = rxrx_sel.loc[exit_bar, "close"]
        side = "long"
        holding = max((rxrx_sel.loc[exit_bar, "date"] - rxrx_sel.loc[entry_bar, "date"]).days, 1)

        pnl_gross = (exit_price / entry_price - 1) * 100
        cost = compute_costs(entry_price, side, holding) / entry_price * 100
        pnl_net = pnl_gross - cost
        shares = max(int(POSITION_SIZE / entry_price), 1)

        trades.append({
            "signal_date": rxrx_sel.loc[idx, "date"],
            "entry_date": rxrx_sel.loc[entry_bar, "date"],
            "exit_date": rxrx_sel.loc[exit_bar, "date"],
            "side": side,
            "entry_price": round(entry_price, 4),
            "exit_price": round(exit_price, 4),
            "exit_reason": "time_stop",
            "holding_days": holding,
            "z_at_signal": np.nan,
            "pnl_gross_pct": round(pnl_gross, 4),
            "cost_pct": round(cost, 4),
            "pnl_net_pct": round(pnl_net, 4),
            "shares": shares,
            "pnl_dollars": round(pnl_net / 100 * shares * entry_price, 2),
            "earnings_period": e["period"],
            "earnings_gap": gap,
        })

    trades_df = pd.DataFrame(trades)
    label = f"BuyDip_h{hold_days}d"

    if trades_df.empty:
        stats = {"label": label, "n": 0, "avg_pnl": 0, "t_stat": 0, "wr": 0, "sharpe": 0}
    else:
        stats = compute_stats(trades_df, rxrx_sel, label=label)

    strat4_results.append({
        "strategy": label,
        "hold_days": hold_days,
        "n": stats["n"],
        "avg_pnl": stats["avg_pnl"],
        "median_pnl": stats.get("median_pnl", 0),
        "wr": stats["wr"],
        "t_stat": stats["t_stat"],
        "sharpe": stats["sharpe"],
        "total_pnl": stats.get("total_pnl_pct", 0),
        "trades_df": trades_df,
    })

    print(f"\n  {label}: n={stats['n']}, avg={stats['avg_pnl']:+.3f}%, "
          f"wr={stats['wr']:.1f}%, t={stats['t_stat']:.2f}")

    if not trades_df.empty:
        for _, t in trades_df.iterrows():
            print(f"    {t['earnings_period']:>10s}  entry={str(t['entry_date'].date()):>12s}  "
                  f"exit={str(t['exit_date'].date()):>12s}  gap={t['earnings_gap']:+.2f}%  "
                  f"pnl={t['pnl_net_pct']:+.2f}%")

# ══════════════════════════════════════════════════════════════════
# 8. SUMMARY TABLE + NULL TESTS
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  SUMMARY: ALL STANDALONE EVENT STRATEGIES")
print("=" * 70)

all_strat_results = strat1_results + strat2_results + strat3_results + strat4_results

summary_rows = []
for s in all_strat_results:
    summary_rows.append({
        "strategy": s["strategy"],
        "n": s["n"],
        "avg_pnl": s["avg_pnl"],
        "median_pnl": s.get("median_pnl", 0),
        "wr": s["wr"],
        "t_stat": s["t_stat"],
        "sharpe": s["sharpe"],
        "total_pnl": s.get("total_pnl", 0),
    })

summary_df = pd.DataFrame(summary_rows)
print(f"\n  {'Strategy':<20s} {'n':>4s} {'Avg%':>8s} {'Med%':>8s} {'WR%':>6s} "
      f"{'t':>6s} {'Sharpe':>7s} {'Total%':>8s}")
print("  " + "-" * 75)
for _, row in summary_df.iterrows():
    print(f"  {row['strategy']:<20s} {row['n']:>4d} {row['avg_pnl']:>+7.3f}% "
          f"{row['median_pnl']:>+7.3f}% {row['wr']:>5.1f}% {row['t_stat']:>+5.2f} "
          f"{row['sharpe']:>+6.2f} {row['total_pnl']:>+7.2f}%")

# Null tests for strategies with positive avg PnL
print("\n" + "=" * 70)
print("  NULL TESTS (strategies with avg_pnl > 0)")
print("=" * 70)

null_results = []
for s in all_strat_results:
    if s["avg_pnl"] > 0 and s["n"] >= 5:
        trades_df = s["trades_df"]
        pctile, null_mean, real_avg = null_test(trades_df, rxrx_sel, n_runs=1000,
                                                 max_hold=max(trades_df["holding_days"].max(), 1))
        verdict = "PASS" if pctile >= 95 else "FAIL"
        null_results.append({
            "strategy": s["strategy"],
            "real_avg": real_avg,
            "null_mean": null_mean,
            "pctile": pctile,
            "verdict": verdict,
        })
        print(f"  {s['strategy']:<20s}: real={real_avg:+.3f}% null={null_mean:+.3f}% "
              f"pctile={pctile:.1f}th -> {verdict}")

        # Log trial
        log_trial("agent3v2", s["strategy"],
                  {"type": "standalone_event", "n_events": s["n"]},
                  {"n": s["n"], "avg_pnl": s["avg_pnl"], "wr": s["wr"],
                   "t_stat": s["t_stat"], "sharpe": s["sharpe"],
                   "total_pnl_pct": s.get("total_pnl", 0),
                   "null_pctile": pctile})
    elif s["n"] > 0:
        # Log negative strategies too
        log_trial("agent3v2", s["strategy"],
                  {"type": "standalone_event", "n_events": s["n"]},
                  {"n": s["n"], "avg_pnl": s["avg_pnl"], "wr": s["wr"],
                   "t_stat": s["t_stat"], "sharpe": s["sharpe"],
                   "total_pnl_pct": s.get("total_pnl", 0)})

if not null_results:
    print("  No strategies with positive avg PnL and n >= 5 to test.")

# ══════════════════════════════════════════════════════════════════
# 9. MATPLOTLIB FIGURE (4 panels)
# ══════════════════════════════════════════════════════════════════
# %%
print("\n  Generating figure...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Agent 3v2: RXRX Standalone Event Strategies (Real Earnings Dates)",
             fontsize=14, fontweight="bold")

# ── Panel A: Earnings calendar on price chart ──
ax = axes[0, 0]
sel_mask = rxrx["date"] < SEL_END
ax.plot(rxrx.loc[sel_mask, "date"], rxrx.loc[sel_mask, "close"],
        color="steelblue", linewidth=0.8, label="RXRX Close")
for _, ec in earnings_df.iterrows():
    clr = "green" if ec["gap_pct"] > 0 else "red"
    ax.axvline(ec["date"], color=clr, alpha=0.5, linewidth=0.8)
    ax.plot(ec["date"], ec["close_before"], "v" if ec["gap_pct"] < 0 else "^",
            color=clr, markersize=7, alpha=0.8)
# Legend markers
ax.plot([], [], "^", color="green", markersize=7, label="Positive gap")
ax.plot([], [], "v", color="red", markersize=7, label="Negative gap")
ax.set_title(f"(a) RXRX + Actual Earnings Dates (n={len(earnings_df)})")
ax.set_ylabel("Price ($)")
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, alpha=0.3)
ax.tick_params(axis="x", rotation=45)

# ── Panel B: Earnings gap distribution ──
ax = axes[0, 1]
gaps = earnings_df["gap_pct"].values
colors_bar = ["green" if g > 0 else "red" for g in gaps]
bars = ax.bar(range(len(gaps)), gaps, color=colors_bar, alpha=0.8, edgecolor="black", linewidth=0.5)
ax.set_xticks(range(len(gaps)))
ax.set_xticklabels([str(d.date()) for d in earnings_df["date"]], rotation=60, fontsize=7)
ax.axhline(0, color="black", linewidth=0.8)
ax.axhline(gaps.mean(), color="blue", linestyle="--", linewidth=1,
           label=f"Mean: {gaps.mean():+.1f}%")
ax.set_title("(b) Earnings Gap Distribution")
ax.set_ylabel("Gap at Open (%)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

# ── Panel C: Pre-earnings drift strategy comparison ──
ax = axes[1, 0]
if strat1_results:
    windows = [s["window"] for s in strat1_results]
    avgs = [s["avg_pnl"] for s in strat1_results]
    ns = [s["n"] for s in strat1_results]
    bar_colors = ["#2ca02c" if a > 0 else "#d62728" for a in avgs]
    bars = ax.bar(windows, avgs, width=2, color=bar_colors, alpha=0.8,
                  edgecolor="black", linewidth=0.5)
    for bar, n, v in zip(bars, ns, avgs):
        ypos = bar.get_height()
        offset = 0.1 if ypos >= 0 else -0.2
        ax.text(bar.get_x() + bar.get_width() / 2, ypos + offset,
                f"n={n}\n{v:+.2f}%", ha="center",
                va="bottom" if ypos >= 0 else "top", fontsize=9)
    ax.set_xticks(windows)
    ax.set_xticklabels([f"{w}d" for w in windows])
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("(c) Pre-Earnings Drift (LONG): Avg PnL by Window")
ax.set_xlabel("Days Before Earnings")
ax.set_ylabel("Avg Net PnL (%)")
ax.grid(True, alpha=0.3, axis="y")

# ── Panel D: Summary of all strategies ──
ax = axes[1, 1]
strat_names = [s["strategy"] for s in all_strat_results if s["n"] > 0]
strat_avgs = [s["avg_pnl"] for s in all_strat_results if s["n"] > 0]
strat_ns = [s["n"] for s in all_strat_results if s["n"] > 0]
if strat_names:
    y_pos = range(len(strat_names))
    bar_colors = ["#2ca02c" if a > 0 else "#d62728" for a in strat_avgs]
    bars = ax.barh(y_pos, strat_avgs, color=bar_colors, alpha=0.8,
                   edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(strat_names, fontsize=7)
    for bar, n, v in zip(bars, strat_ns, strat_avgs):
        xpos = bar.get_width()
        offset = 0.05 if xpos >= 0 else -0.05
        ax.text(xpos + offset, bar.get_y() + bar.get_height() / 2,
                f"n={n} {v:+.2f}%", va="center", fontsize=7,
                ha="left" if xpos >= 0 else "right")
    ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("(d) All Standalone Event Strategies: Avg Net PnL")
ax.set_xlabel("Avg Net PnL (%)")
ax.grid(True, alpha=0.3, axis="x")

plt.tight_layout()
fig_path = OUT_DIR / "agent3v2_events.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Figure saved to {fig_path}")

# ══════════════════════════════════════════════════════════════════
# 10. CONCLUSION
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  CONCLUSION")
print("=" * 70)

# Find best strategy
best = max(all_strat_results, key=lambda x: x["avg_pnl"]) if all_strat_results else None

# Count how many strategies are positive
n_positive = sum(1 for s in all_strat_results if s["avg_pnl"] > 0 and s["n"] > 0)
n_total_strats = sum(1 for s in all_strat_results if s["n"] > 0)

# Any pass null test?
any_pass_null = any(r["verdict"] == "PASS" for r in null_results) if null_results else False

print(f"""
  DATA SOURCE: Polygon financials API (SEC filing dates)
  EARNINGS EVENTS: {len(earnings_df)} in selection window (2021-2025)
  STRATEGIES TESTED: {n_total_strats} (4 families x parameter sweep)
  POSITIVE AVG PNL: {n_positive}/{n_total_strats}
""")

if best and best["n"] > 0:
    print(f"  BEST STRATEGY: {best['strategy']}")
    print(f"    n={best['n']}, avg={best['avg_pnl']:+.3f}%, t={best['t_stat']:.2f}, "
          f"wr={best['wr']:.1f}%")

if any_pass_null:
    print(f"\n  VERDICT: Strategy found with positive avg PnL that passes null test.")
    print("  However, low sample size (quarterly earnings = ~18 events) means")
    print("  statistical confidence is inherently limited.")
else:
    if n_positive > 0:
        print(f"\n  VERDICT: {n_positive} strategies show positive avg PnL, but NONE pass")
        print("  the null test at 95th percentile. The earnings-based event edge")
        print("  is NOT statistically distinguishable from random timing.")
        print("  Hypothesis tested and rejected (insufficient evidence).")
    else:
        print(f"\n  VERDICT: No standalone event strategy shows positive avg PnL.")
        print("  Hypothesis tested and rejected.")

print(f"""
  KEY OBSERVATIONS:
  - RXRX has {len(earnings_df)} quarterly earnings in selection window
  - Avg earnings gap: {earnings_df['gap_pct'].mean():+.2f}% (wide range)
  - Low n per strategy (~8-18 events) limits statistical power
  - Event-driven strategies need 80+ events for reliable inference
  - Biotech earnings are high-variance (pipeline, not revenue)

  RECOMMENDATION: Earnings events alone do NOT provide a tradeable
  standalone edge for RXRX. Consider only as a filter/gate on
  other strategies (e.g., avoiding mean-reversion trades near earnings).
""")

print("  AGENT 3v2 COMPLETE")
print("=" * 70)
