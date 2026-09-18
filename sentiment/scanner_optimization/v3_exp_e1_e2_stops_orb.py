"""
V3 Scanner Optimization: E1 (ATR Trailing Stop Design) + E2 (ORB Filter Refinement)
====================================================================================

E1: Simulates trailing stops from high-water-mark at k * ATR across multiple
    time horizons. Compares to fixed-stop results and no-SL baselines.
    E1b: Hybrid trail-30min-then-hold-to-T+2 for SHORT trades.

E2: Tests different Opening Range Breakout timeframes (2/5/10/15 min) and
    breakout strength thresholds with optional volume filter. Measures
    agreement with LLM direction and PnL impact.

Data: 316 trades from v2_features_full.csv + v3_minute_bars.json
"""

import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path("os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../') + '/sentiment/scanner_optimization")
RESULTS = BASE / "results"
CSV_PATH = RESULTS / "v2_features_full.csv"
BARS_PATH = RESULTS / "v3_minute_bars.json"
DAILY_PATH = RESULTS / "v2_daily_data.json"

OUT_E1_PNG = BASE / "v3_e1_trailing_stops.png"
OUT_E2_PNG = BASE / "v3_e2_orb_refinement.png"
OUT_JSON = RESULTS / "v3_e1_e2_results.json"

# ---------------------------------------------------------------------------
# Chart palette (design-system neutral, validated)
# ---------------------------------------------------------------------------
PAL_BLUE = "#3566A0"
PAL_RED = "#C0392B"
PAL_GREEN = "#27864E"
PAL_AMBER = "#D4850A"
PAL_PURPLE = "#7B5EA7"
PAL_TEAL = "#1D8A8A"
PAL_GREY = "#6B7280"
PAL_LIGHT_BLUE = "#A3C4E9"
PAL_LIGHT_RED = "#E6A8A1"
PAL_SURFACE = "#FFFFFF"
PAL_GRID = "#E5E7EB"
PAL_TEXT = "#1F2937"
PAL_MUTED = "#9CA3AF"

LONG_COLOR = PAL_BLUE
SHORT_COLOR = PAL_RED
NOSL_COLOR = PAL_GREY
TRAIL_COLOR = PAL_GREEN
FIXED_COLOR = PAL_AMBER
AGREE_COLOR = PAL_GREEN
DISAGREE_COLOR = PAL_RED

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def trimmed_mean(arr, lo=5, hi=95):
    """Trimmed mean clipping at lo/hi percentiles."""
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 3:
        return np.nan
    low, high = np.percentile(a, [lo, hi])
    trimmed = a[(a >= low) & (a <= high)]
    return float(np.mean(trimmed)) if len(trimmed) > 0 else np.nan


def fmt(v, dec=2):
    """Format a number for display."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "  n/a"
    return f"{v:+.{dec}f}" if v != 0 else f" 0.{'0' * dec}"


def low_n_flag(n, threshold=20):
    return " *LOW-N*" if n < threshold else ""


def time_to_minutes(t_str):
    """Convert HH:MM to minutes since midnight."""
    h, m = map(int, t_str.split(":"))
    return h * 60 + m


def minutes_to_time(mins):
    """Convert minutes since midnight to HH:MM."""
    return f"{mins // 60:02d}:{mins % 60:02d}"


def get_market_bars(bars_dict, start_time="09:30", end_time="16:00"):
    """Return sorted list of (time_str, bar_dict) within market hours."""
    start_m = time_to_minutes(start_time)
    end_m = time_to_minutes(end_time)
    result = []
    for t_str, bar in sorted(bars_dict.items()):
        t_m = time_to_minutes(t_str)
        if start_m <= t_m <= end_m:
            result.append((t_str, bar))
    return result


def horizon_end_time(entry_time, horizon_minutes):
    """Compute the end time string for a given horizon."""
    entry_m = time_to_minutes(entry_time)
    end_m = entry_m + horizon_minutes
    if end_m > time_to_minutes("16:00"):
        return "16:00"
    return minutes_to_time(end_m)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("=" * 80)
print("V3 Scanner Optimization: E1 (ATR Trailing Stops) + E2 (ORB Refinement)")
print("=" * 80)
print()

df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} trades from CSV")

with open(BARS_PATH) as f:
    bars_data = json.load(f)
ticker_bars = bars_data["ticker_bars"]
spy_bars = bars_data.get("spy_bars", {})
print(f"Loaded minute bars for {len(ticker_bars)} ticker-dates")

with open(DAILY_PATH) as f:
    daily_data = json.load(f)
print(f"Loaded daily data for {len(daily_data)} tickers")
print()

# Precompute gap_pct quartiles
df["gap_q"] = pd.qcut(df["gap_pct"], 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])

# =========================================================================
# E1: ATR Trailing Stop Simulation
# =========================================================================
print("=" * 80)
print("E1: ATR TRAILING STOP DESIGN")
print("=" * 80)
print()

K_VALUES = [0.5, 0.75, 1.0, 1.5]
HORIZONS = {"30min": 30, "60min": 60, "120min": 120, "EOD": 390}  # 390 = full day

e1_results = {}  # (k, horizon) -> list of per-trade results


def simulate_trailing_stop(row, bars, k, horizon_min):
    """
    Simulate a trailing stop for one trade.

    Returns dict with:
        pnl: realized PnL %
        stopped: bool
        exit_time: str
        hwm_pct: high water mark as % from entry (for LONG)
    """
    entry_time = row["entry_time_et"]
    entry_price = row["entry_price"]
    sign = row["sign"]  # +1 LONG, -1 SHORT
    atr_pct = row["atr_pct"]
    atr_dollars = entry_price * atr_pct / 100.0

    entry_m = time_to_minutes(entry_time)
    end_m = min(entry_m + horizon_min, time_to_minutes("16:00"))

    # Get bars in the trading window
    market = get_market_bars(bars, entry_time, minutes_to_time(end_m))
    if not market:
        return None

    is_long = sign > 0
    hwm = entry_price  # high water mark (LONG) or low water mark (SHORT)
    trail_stop = None
    stopped = False
    exit_price = None
    exit_time = None

    for t_str, bar in market:
        c = bar["c"]
        h = bar["h"]
        lo = bar["l"]

        if is_long:
            # Update HWM
            if c > hwm:
                hwm = c
            if h > hwm:
                hwm = h
            # Trailing stop below HWM
            trail_stop = hwm - k * atr_dollars
            # Check if low hits stop
            if lo <= trail_stop:
                exit_price = trail_stop
                exit_time = t_str
                stopped = True
                break
        else:
            # SHORT: track low water mark
            if c < hwm:
                hwm = c
            if lo < hwm:
                hwm = lo
            # Trailing stop above LWM
            trail_stop = hwm + k * atr_dollars
            # Check if high hits stop
            if h >= trail_stop:
                exit_price = trail_stop
                exit_time = t_str
                stopped = True
                break

    if not stopped:
        # Exit at last bar's close
        exit_price = market[-1][1]["c"]
        exit_time = market[-1][0]

    # Compute PnL (sign-adjusted)
    pnl = (exit_price - entry_price) / entry_price * 100.0 * sign

    hwm_pct = abs(hwm - entry_price) / entry_price * 100.0

    return {
        "pnl": pnl,
        "stopped": stopped,
        "exit_time": exit_time,
        "hwm_pct": hwm_pct,
    }


print("Simulating trailing stops...")
for k in K_VALUES:
    for h_name, h_min in HORIZONS.items():
        key = (k, h_name)
        results = []
        for idx, row in df.iterrows():
            tk = f"{row['ticker']}__{row['eval_date']}"
            bars = ticker_bars.get(tk)
            if bars is None:
                continue
            res = simulate_trailing_stop(row, bars, k, h_min)
            if res is not None:
                res["direction"] = row["direction"]
                res["sign"] = row["sign"]
                res["gap_q"] = row["gap_q"]
                res["ticker"] = row["ticker"]
                res["idx"] = idx
                results.append(res)
        e1_results[key] = results
        n = len(results)
        n_stopped = sum(1 for r in results if r["stopped"])
        avg_pnl = np.mean([r["pnl"] for r in results]) if results else np.nan
        print(f"  k={k:.2f} {h_name:>5s}: n={n:3d}, stopped={n_stopped:3d} ({n_stopped/n*100:.0f}%), avg PnL={avg_pnl:+.3f}%")

print()

# ---------------------------------------------------------------------------
# E1: Compute metrics tables
# ---------------------------------------------------------------------------

def compute_metrics(pnl_list):
    """Return dict of metrics for a list of PnL values."""
    a = np.array(pnl_list, dtype=float)
    a = a[np.isfinite(a)]
    n = len(a)
    if n == 0:
        return {"n": 0, "avg": np.nan, "median": np.nan, "tmean": np.nan,
                "wr": np.nan, "max_dd": np.nan}
    return {
        "n": n,
        "avg": float(np.mean(a)),
        "median": float(np.median(a)),
        "tmean": trimmed_mean(a),
        "wr": float(np.mean(a > 0) * 100),
        "max_dd": float(np.min(a)),
    }


# Build metrics for each (k, horizon) x (ALL, LONG, SHORT) x gap_q
print("-" * 80)
print("E1 RESULTS: Trailing Stop Performance")
print("-" * 80)
print()

# Reference no-SL PnLs from CSV
nosl_map = {"30min": "pnl_30m_nosl", "60min": "pnl_60m_nosl",
            "120min": "pnl_120m_nosl", "EOD": "pnl_eod_nosl"}

# Reference fixed-SL PnLs (best from V2 was 5% SL at 60min)
fixed_sl_cols = {
    "60min": {"1.5%": "pnl_60m_sl1.5", "2.0%": "pnl_60m_sl2.0",
              "2.5%": "pnl_60m_sl2.5", "3.0%": "pnl_60m_sl3.0",
              "4.0%": "pnl_60m_sl4.0", "5.0%": "pnl_60m_sl5.0"},
    "30min": {"1.5%": "pnl_30m_sl1.5", "2.0%": "pnl_30m_sl2.0",
              "2.5%": "pnl_30m_sl2.5", "3.0%": "pnl_30m_sl3.0",
              "4.0%": "pnl_30m_sl4.0", "5.0%": "pnl_30m_sl5.0"},
}

e1_summary = {}

for h_name in HORIZONS:
    print(f"\n{'='*60}")
    print(f"  Horizon: {h_name}")
    print(f"{'='*60}")

    # No-SL reference
    nosl_col = nosl_map.get(h_name)
    if nosl_col and nosl_col in df.columns:
        nosl_all = compute_metrics(df[nosl_col].dropna())
        nosl_long = compute_metrics(df.loc[df["sign"] == 1, nosl_col].dropna())
        nosl_short = compute_metrics(df.loc[df["sign"] == -1, nosl_col].dropna())
    else:
        nosl_all = nosl_long = nosl_short = {"n": 0, "avg": np.nan, "median": np.nan, "tmean": np.nan, "wr": np.nan, "max_dd": np.nan}

    print(f"\n  No-SL reference:")
    print(f"    ALL:   avg={fmt(nosl_all['avg'])}% tmean={fmt(nosl_all['tmean'])}% wr={fmt(nosl_all['wr'],1)}% n={nosl_all['n']}")
    print(f"    LONG:  avg={fmt(nosl_long['avg'])}% tmean={fmt(nosl_long['tmean'])}% wr={fmt(nosl_long['wr'],1)}% n={nosl_long['n']}")
    print(f"    SHORT: avg={fmt(nosl_short['avg'])}% tmean={fmt(nosl_short['tmean'])}% wr={fmt(nosl_short['wr'],1)}% n={nosl_short['n']}")

    # Fixed SL reference (for 30m/60m only)
    if h_name in fixed_sl_cols:
        print(f"\n  Fixed SL reference ({h_name}):")
        for sl_name, sl_col in fixed_sl_cols[h_name].items():
            if sl_col in df.columns:
                m = compute_metrics(df[sl_col].dropna())
                print(f"    SL {sl_name}: avg={fmt(m['avg'])}% tmean={fmt(m['tmean'])}% wr={fmt(m['wr'],1)}%")

    print(f"\n  Trailing Stop results:")
    header = f"  {'k':>5s} | {'Slice':>6s} | {'n':>4s} | {'Avg PnL':>8s} | {'TMean':>8s} | {'Median':>8s} | {'WR':>6s} | {'MaxDD':>8s} | {'vs NoSL':>8s}"
    print(header)
    print("  " + "-" * len(header.strip()))

    for k in K_VALUES:
        key = (k, h_name)
        results = e1_results[key]
        if not results:
            continue

        for slice_name, filt_fn in [("ALL", lambda r: True),
                                     ("LONG", lambda r: r["direction"] == "LONG"),
                                     ("SHORT", lambda r: r["direction"] == "SHORT")]:
            subset = [r for r in results if filt_fn(r)]
            pnls = [r["pnl"] for r in subset]
            m = compute_metrics(pnls)

            ref = {"ALL": nosl_all, "LONG": nosl_long, "SHORT": nosl_short}[slice_name]
            delta = m["avg"] - ref["avg"] if not np.isnan(m["avg"]) and not np.isnan(ref["avg"]) else np.nan

            flag = low_n_flag(m["n"])
            print(f"  {k:5.2f} | {slice_name:>6s} | {m['n']:4d} | {fmt(m['avg'])}% | {fmt(m['tmean'])}% | {fmt(m['median'])}% | {fmt(m['wr'],1)}% | {fmt(m['max_dd'])}% | {fmt(delta)}%{flag}")

            e1_summary[f"trail_k{k}_{h_name}_{slice_name}"] = {
                "k": k, "horizon": h_name, "slice": slice_name,
                **m, "delta_vs_nosl": delta if not np.isnan(delta) else None,
            }

        print()

# ---------------------------------------------------------------------------
# E1: Gap quartile breakdown for best config
# ---------------------------------------------------------------------------
print("\n" + "-" * 80)
print("E1: Gap Quartile Breakdown")
print("-" * 80)

# Find best trailing config by avg PnL (ALL)
best_key = None
best_avg = -999
for k in K_VALUES:
    for h_name in HORIZONS:
        key = (k, h_name)
        results = e1_results[key]
        pnls = [r["pnl"] for r in results]
        avg = np.mean(pnls) if pnls else -999
        if avg > best_avg:
            best_avg = avg
            best_key = key

print(f"\nBest trailing config (ALL): k={best_key[0]}, horizon={best_key[1]}, avg PnL={best_avg:+.3f}%")
print()

best_results = e1_results[best_key]
for gq in ["Q1_low", "Q2", "Q3", "Q4_high"]:
    subset = [r for r in best_results if r["gap_q"] == gq]
    pnls = [r["pnl"] for r in subset]
    m = compute_metrics(pnls)
    flag = low_n_flag(m["n"])
    print(f"  {gq:>8s}: n={m['n']:3d}, avg={fmt(m['avg'])}%, tmean={fmt(m['tmean'])}%, wr={fmt(m['wr'],1)}%{flag}")

# Also by direction within gap quartile
for direction in ["LONG", "SHORT"]:
    print(f"\n  {direction}:")
    for gq in ["Q1_low", "Q2", "Q3", "Q4_high"]:
        subset = [r for r in best_results if r["gap_q"] == gq and r["direction"] == direction]
        pnls = [r["pnl"] for r in subset]
        m = compute_metrics(pnls)
        flag = low_n_flag(m["n"])
        print(f"    {gq:>8s}: n={m['n']:3d}, avg={fmt(m['avg'])}%, tmean={fmt(m['tmean'])}%, wr={fmt(m['wr'],1)}%{flag}")

# ---------------------------------------------------------------------------
# E1b: Time-stop hybrid for SHORT trades
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("E1b: TIME-STOP HYBRID (Trail 30min then hold to T+2 close) -- SHORT only")
print("=" * 80)
print()

short_df = df[df["sign"] == -1].copy()
e1b_results = {}

for k in K_VALUES:
    hybrid_pnls = []
    for idx, row in short_df.iterrows():
        tk = f"{row['ticker']}__{row['eval_date']}"
        bars = ticker_bars.get(tk)
        if bars is None:
            continue

        # Simulate trailing stop for first 30 minutes
        res_30 = simulate_trailing_stop(row, bars, k, 30)
        if res_30 is None:
            continue

        if res_30["stopped"]:
            # Stopped within 30 min -- take that PnL
            hybrid_pnls.append({"pnl": res_30["pnl"], "exit": "trail_stop", "idx": idx})
        else:
            # Not stopped in 30 min -- hold to T+2 close
            t2_pnl = row.get("pnl_t2_close")
            if pd.notna(t2_pnl):
                hybrid_pnls.append({"pnl": t2_pnl, "exit": "t2_close", "idx": idx})
            else:
                # Fallback: use EOD no-SL if T+2 not available
                eod_pnl = row.get("pnl_eod_nosl")
                if pd.notna(eod_pnl):
                    hybrid_pnls.append({"pnl": eod_pnl, "exit": "eod_fallback", "idx": idx})

    e1b_results[k] = hybrid_pnls
    pnls_arr = [r["pnl"] for r in hybrid_pnls]
    m = compute_metrics(pnls_arr)
    n_stopped = sum(1 for r in hybrid_pnls if r["exit"] == "trail_stop")
    n_t2 = sum(1 for r in hybrid_pnls if r["exit"] == "t2_close")
    n_fb = sum(1 for r in hybrid_pnls if r["exit"] == "eod_fallback")
    print(f"  k={k:.2f}: n={m['n']}, avg={fmt(m['avg'])}%, tmean={fmt(m['tmean'])}%, wr={fmt(m['wr'],1)}%")
    print(f"         stopped_30m={n_stopped}, held_t2={n_t2}, eod_fallback={n_fb}")

    e1_summary[f"hybrid_k{k}_SHORT"] = {
        "k": k, "horizon": "30m_trail+T2", "slice": "SHORT",
        **m, "n_stopped_30m": n_stopped, "n_held_t2": n_t2,
    }

# SHORT no-SL references
print(f"\n  SHORT references:")
for ref_name, ref_col in [("30m no-SL", "pnl_30m_nosl"), ("60m no-SL", "pnl_60m_nosl"),
                            ("EOD no-SL", "pnl_eod_nosl"), ("T+2 close", "pnl_t2_close")]:
    vals = short_df[ref_col].dropna()
    m = compute_metrics(vals)
    print(f"    {ref_name:>12s}: n={m['n']:3d}, avg={fmt(m['avg'])}%, tmean={fmt(m['tmean'])}%, wr={fmt(m['wr'],1)}%")


# =========================================================================
# E2: ORB Filter Refinement
# =========================================================================
print("\n" + "=" * 80)
print("E2: ORB FILTER REFINEMENT")
print("=" * 80)
print()

ORB_TIMEFRAMES = {
    "2min": (time_to_minutes("09:30"), time_to_minutes("09:31")),
    "5min": (time_to_minutes("09:30"), time_to_minutes("09:34")),
    "10min": (time_to_minutes("09:30"), time_to_minutes("09:39")),
    "15min": (time_to_minutes("09:30"), time_to_minutes("09:44")),
}

BREAKOUT_THRESHOLDS = [0.0, 0.1, 0.25, 0.5]
VOL_FILTER_OPTIONS = [False, True]


def compute_orb(bars, tf_start_m, tf_end_m, entry_price):
    """
    Compute ORB high/low for given timeframe, breakout direction,
    breakout strength, and volume stats.

    Returns dict or None if insufficient bars.
    """
    orb_highs = []
    orb_lows = []
    orb_volumes = []

    for t_str, bar in sorted(bars.items()):
        t_m = time_to_minutes(t_str)
        if tf_start_m <= t_m <= tf_end_m:
            orb_highs.append(bar["h"])
            orb_lows.append(bar["l"])
            orb_volumes.append(bar["v"])

    if not orb_highs:
        return None

    or_high = max(orb_highs)
    or_low = min(orb_lows)
    avg_orb_vol = np.mean(orb_volumes)

    # First bar AFTER the ORB window
    first_after = None
    first_after_vol = None
    for t_str, bar in sorted(bars.items()):
        t_m = time_to_minutes(t_str)
        if t_m > tf_end_m and t_m <= time_to_minutes("16:00"):
            first_after = bar
            first_after_vol = bar["v"]
            break

    if first_after is None:
        return None

    close_after = first_after["c"]

    # Breakout direction
    if close_after > or_high:
        bo_dir = "LONG"
        bo_strength = (close_after - or_high) / entry_price * 100
    elif close_after < or_low:
        bo_dir = "SHORT"
        bo_strength = (or_low - close_after) / entry_price * 100
    else:
        bo_dir = "INSIDE"
        bo_strength = 0.0

    vol_ratio = first_after_vol / avg_orb_vol if avg_orb_vol > 0 else 0.0

    return {
        "or_high": or_high,
        "or_low": or_low,
        "bo_dir": bo_dir,
        "bo_strength": bo_strength,
        "vol_ratio": vol_ratio,
        "avg_orb_vol": avg_orb_vol,
        "first_after_vol": first_after_vol,
    }


print("Computing ORB across timeframes...")
e2_orb_data = {}  # (tf_name) -> list of per-trade ORB results

for tf_name, (tf_start, tf_end) in ORB_TIMEFRAMES.items():
    orb_results = []
    for idx, row in df.iterrows():
        tk = f"{row['ticker']}__{row['eval_date']}"
        bars = ticker_bars.get(tk)
        if bars is None:
            continue

        orb = compute_orb(bars, tf_start, tf_end, row["entry_price"])
        if orb is None:
            continue

        orb["idx"] = idx
        orb["ticker"] = row["ticker"]
        orb["eval_date"] = row["eval_date"]
        orb["llm_dir"] = row["direction"]
        orb["sign"] = row["sign"]
        orb["pnl_60m"] = row["pnl_60m_nosl"]
        orb["pnl_30m"] = row["pnl_30m_nosl"]
        orb["pnl_eod"] = row["pnl_eod_nosl"]
        orb["entry_time"] = row["entry_time_et"]
        orb_results.append(orb)

    e2_orb_data[tf_name] = orb_results
    print(f"  {tf_name}: {len(orb_results)} trades computed")

print()

# ---------------------------------------------------------------------------
# E2: Analysis
# ---------------------------------------------------------------------------
print("-" * 80)
print("E2 RESULTS: ORB Filter Agreement and PnL Impact")
print("-" * 80)
print()

e2_summary = {}

header = f"  {'TF':>5s} | {'Thresh':>6s} | {'VolF':>4s} | {'n':>4s} | {'Agree':>5s} | {'Dis':>4s} | {'Inside':>6s} | {'AgreeR':>6s} | {'Agr PnL':>8s} | {'Dis PnL':>8s} | {'Spread':>8s} | {'Agr WR':>6s} | {'Dis WR':>6s}"
print(header)
print("  " + "-" * len(header.strip()))

for tf_name in ORB_TIMEFRAMES:
    for thresh in BREAKOUT_THRESHOLDS:
        for vol_filt in VOL_FILTER_OPTIONS:
            orb_results = e2_orb_data[tf_name]

            agree_pnls = []
            disagree_pnls = []
            inside_pnls = []
            n_total = 0

            for orb in orb_results:
                # Apply breakout strength threshold
                if orb["bo_dir"] == "INSIDE":
                    inside_pnls.append(orb["pnl_60m"])
                    n_total += 1
                    continue

                if orb["bo_strength"] < thresh:
                    # Below threshold = treat as inside
                    inside_pnls.append(orb["pnl_60m"])
                    n_total += 1
                    continue

                # Apply volume filter
                if vol_filt and orb["vol_ratio"] < 1.5:
                    inside_pnls.append(orb["pnl_60m"])
                    n_total += 1
                    continue

                n_total += 1

                # Check agreement (only count non-NaN PnL)
                pnl_val = orb["pnl_60m"]
                if pd.notna(pnl_val):
                    if orb["bo_dir"] == orb["llm_dir"]:
                        agree_pnls.append(pnl_val)
                    else:
                        disagree_pnls.append(pnl_val)
                else:
                    # NaN PnL -- still count for total/agreement rate
                    if orb["bo_dir"] == orb["llm_dir"]:
                        agree_pnls.append(pnl_val)
                    else:
                        disagree_pnls.append(pnl_val)

            # Filter NaN for metrics
            agree_clean = [v for v in agree_pnls if pd.notna(v)]
            disagree_clean = [v for v in disagree_pnls if pd.notna(v)]
            inside_clean = [v for v in inside_pnls if pd.notna(v)]
            n_agree = len(agree_clean)
            n_disagree = len(disagree_clean)
            n_inside = len(inside_clean)
            agree_rate = n_agree / (n_agree + n_disagree) * 100 if (n_agree + n_disagree) > 0 else 0

            avg_agree = np.mean(agree_clean) if agree_clean else np.nan
            avg_disagree = np.mean(disagree_clean) if disagree_clean else np.nan
            spread = avg_agree - avg_disagree if not np.isnan(avg_agree) and not np.isnan(avg_disagree) else np.nan

            wr_agree = np.mean(np.array(agree_clean) > 0) * 100 if agree_clean else np.nan
            wr_disagree = np.mean(np.array(disagree_clean) > 0) * 100 if disagree_clean else np.nan

            vf_label = "Y" if vol_filt else "N"
            flag_a = low_n_flag(n_agree)
            flag_d = low_n_flag(n_disagree)

            print(f"  {tf_name:>5s} | {thresh:6.2f} | {vf_label:>4s} | {n_total:4d} | {n_agree:5d} | {n_disagree:4d} | {n_inside:6d} | {agree_rate:5.1f}% | {fmt(avg_agree)}% | {fmt(avg_disagree)}% | {fmt(spread)}% | {fmt(wr_agree,1)}% | {fmt(wr_disagree,1)}%{flag_a}{flag_d}")

            key = f"{tf_name}_t{thresh}_vf{int(vol_filt)}"
            e2_summary[key] = {
                "timeframe": tf_name, "threshold": thresh, "vol_filter": vol_filt,
                "n_total": n_total, "n_agree": n_agree, "n_disagree": n_disagree,
                "n_inside": n_inside, "agree_rate": agree_rate,
                "avg_agree_pnl": float(avg_agree) if not np.isnan(avg_agree) else None,
                "avg_disagree_pnl": float(avg_disagree) if not np.isnan(avg_disagree) else None,
                "spread": float(spread) if not np.isnan(spread) else None,
                "wr_agree": float(wr_agree) if not np.isnan(wr_agree) else None,
                "wr_disagree": float(wr_disagree) if not np.isnan(wr_disagree) else None,
            }

    print()

# ---------------------------------------------------------------------------
# E2: Trimmed mean analysis for top configs
# ---------------------------------------------------------------------------
print("\nE2: Trimmed Mean (5-95%) for key configs:")
print(f"  {'TF':>5s} | {'Thresh':>6s} | {'VolF':>4s} | {'Agr TMean':>9s} | {'Dis TMean':>9s} | {'Spr TMean':>9s}")
print("  " + "-" * 65)

for tf_name in ORB_TIMEFRAMES:
    for thresh in BREAKOUT_THRESHOLDS:
        for vol_filt in [False]:  # Just no-vol-filter for trimmed mean table
            orb_results = e2_orb_data[tf_name]
            agree_pnls = []
            disagree_pnls = []
            for orb in orb_results:
                if orb["bo_dir"] == "INSIDE" or orb["bo_strength"] < thresh:
                    continue
                if not pd.notna(orb["pnl_60m"]):
                    continue
                if orb["bo_dir"] == orb["llm_dir"]:
                    agree_pnls.append(orb["pnl_60m"])
                else:
                    disagree_pnls.append(orb["pnl_60m"])

            tm_a = trimmed_mean(agree_pnls)
            tm_d = trimmed_mean(disagree_pnls)
            tm_s = tm_a - tm_d if not np.isnan(tm_a) and not np.isnan(tm_d) else np.nan
            print(f"  {tf_name:>5s} | {thresh:6.2f} |    N | {fmt(tm_a)}%   | {fmt(tm_d)}%   | {fmt(tm_s)}%")

# ---------------------------------------------------------------------------
# E2: Direction-specific ORB analysis
# ---------------------------------------------------------------------------
print("\n" + "-" * 80)
print("E2: Direction-Specific ORB Analysis (best timeframe = 5min, thresh=0.0)")
print("-" * 80)

for tf_name in ["5min"]:
    orb_results = e2_orb_data[tf_name]
    for direction in ["LONG", "SHORT"]:
        dir_results = [o for o in orb_results if o["llm_dir"] == direction]
        agree = [o["pnl_60m"] for o in dir_results if o["bo_dir"] == direction and pd.notna(o["pnl_60m"])]
        disagree = [o["pnl_60m"] for o in dir_results if o["bo_dir"] != "INSIDE" and o["bo_dir"] != direction and pd.notna(o["pnl_60m"])]
        inside = [o["pnl_60m"] for o in dir_results if o["bo_dir"] == "INSIDE" and pd.notna(o["pnl_60m"])]
        print(f"\n  {direction} trades ({len(dir_results)} total):")
        print(f"    ORB agrees:    n={len(agree):3d}, avg={fmt(np.mean(agree) if agree else np.nan)}%, tmean={fmt(trimmed_mean(agree))}%, wr={fmt(np.mean(np.array(agree)>0)*100 if agree else np.nan,1)}%{low_n_flag(len(agree))}")
        print(f"    ORB disagrees: n={len(disagree):3d}, avg={fmt(np.mean(disagree) if disagree else np.nan)}%, tmean={fmt(trimmed_mean(disagree))}%, wr={fmt(np.mean(np.array(disagree)>0)*100 if disagree else np.nan,1)}%{low_n_flag(len(disagree))}")
        print(f"    ORB inside:    n={len(inside):3d}, avg={fmt(np.mean(inside) if inside else np.nan)}%, tmean={fmt(trimmed_mean(inside))}%, wr={fmt(np.mean(np.array(inside)>0)*100 if inside else np.nan,1)}%{low_n_flag(len(inside))}")


# =========================================================================
# CHARTS
# =========================================================================

# ---------------------------------------------------------------------------
# E1 Charts
# ---------------------------------------------------------------------------
print("\n\nGenerating E1 charts...")

fig = plt.figure(figsize=(18, 16))
fig.patch.set_facecolor(PAL_SURFACE)
fig.suptitle("E1: ATR Trailing Stop Analysis", fontsize=16, fontweight="bold",
             color=PAL_TEXT, y=0.98)
gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3, top=0.93, bottom=0.06,
                       left=0.07, right=0.95)

# --- Panel 1: Heatmap k x horizon -> avg PnL (LONG) ---
ax1 = fig.add_subplot(gs[0, 0])
hm_data_long = np.zeros((len(K_VALUES), len(HORIZONS)))
h_names = list(HORIZONS.keys())
for i, k in enumerate(K_VALUES):
    for j, h_name in enumerate(h_names):
        results = e1_results[(k, h_name)]
        pnls = [r["pnl"] for r in results if r["direction"] == "LONG"]
        hm_data_long[i, j] = np.mean(pnls) if pnls else np.nan

vmax = max(abs(np.nanmin(hm_data_long)), abs(np.nanmax(hm_data_long)))
vmax = max(vmax, 0.1)
im1 = ax1.imshow(hm_data_long, cmap="RdYlGn", aspect="auto",
                 vmin=-vmax, vmax=vmax)
ax1.set_xticks(range(len(h_names)))
ax1.set_xticklabels(h_names, fontsize=9, color=PAL_TEXT)
ax1.set_yticks(range(len(K_VALUES)))
ax1.set_yticklabels([f"k={k}" for k in K_VALUES], fontsize=9, color=PAL_TEXT)
ax1.set_title("LONG: Avg PnL by k x Horizon", fontsize=11, fontweight="bold",
              color=PAL_TEXT, pad=8)
for i in range(len(K_VALUES)):
    for j in range(len(h_names)):
        val = hm_data_long[i, j]
        if not np.isnan(val):
            ax1.text(j, i, f"{val:+.2f}", ha="center", va="center",
                     fontsize=9, color=PAL_TEXT, fontweight="bold")
cb1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
cb1.set_label("Avg PnL %", fontsize=9, color=PAL_TEXT)

# --- Panel 2: Heatmap k x horizon -> avg PnL (SHORT) ---
ax2 = fig.add_subplot(gs[0, 1])
hm_data_short = np.zeros((len(K_VALUES), len(HORIZONS)))
for i, k in enumerate(K_VALUES):
    for j, h_name in enumerate(h_names):
        results = e1_results[(k, h_name)]
        pnls = [r["pnl"] for r in results if r["direction"] == "SHORT"]
        hm_data_short[i, j] = np.mean(pnls) if pnls else np.nan

vmax2 = max(abs(np.nanmin(hm_data_short)), abs(np.nanmax(hm_data_short)))
vmax2 = max(vmax2, 0.1)
im2 = ax2.imshow(hm_data_short, cmap="RdYlGn", aspect="auto",
                 vmin=-vmax2, vmax=vmax2)
ax2.set_xticks(range(len(h_names)))
ax2.set_xticklabels(h_names, fontsize=9, color=PAL_TEXT)
ax2.set_yticks(range(len(K_VALUES)))
ax2.set_yticklabels([f"k={k}" for k in K_VALUES], fontsize=9, color=PAL_TEXT)
ax2.set_title("SHORT: Avg PnL by k x Horizon", fontsize=11, fontweight="bold",
              color=PAL_TEXT, pad=8)
for i in range(len(K_VALUES)):
    for j in range(len(h_names)):
        val = hm_data_short[i, j]
        if not np.isnan(val):
            ax2.text(j, i, f"{val:+.2f}", ha="center", va="center",
                     fontsize=9, color=PAL_TEXT, fontweight="bold")
cb2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
cb2.set_label("Avg PnL %", fontsize=9, color=PAL_TEXT)

# --- Panel 3: Trail PnL vs Fixed-SL PnL scatter for best k ---
ax3 = fig.add_subplot(gs[1, 0])

# Best k at 60min (most comparable to fixed SL data)
best_k_60 = None
best_avg_60 = -999
for k in K_VALUES:
    results = e1_results[(k, "60min")]
    pnls = [r["pnl"] for r in results]
    avg = np.mean(pnls) if pnls else -999
    if avg > best_avg_60:
        best_avg_60 = avg
        best_k_60 = k

trail_60_results = e1_results[(best_k_60, "60min")]
trail_pnls = {r["idx"]: r["pnl"] for r in trail_60_results}

# Compare to best fixed SL (5.0% at 60min)
fixed_col = "pnl_60m_sl5.0"
scatter_trail = []
scatter_fixed = []
scatter_colors = []

for idx, row in df.iterrows():
    if idx in trail_pnls and pd.notna(row[fixed_col]):
        scatter_trail.append(trail_pnls[idx])
        scatter_fixed.append(row[fixed_col])
        scatter_colors.append(LONG_COLOR if row["sign"] > 0 else SHORT_COLOR)

ax3.scatter(scatter_fixed, scatter_trail, c=scatter_colors, alpha=0.45, s=18,
            edgecolors="none", rasterized=True)
lim = max(abs(min(scatter_trail + scatter_fixed)), abs(max(scatter_trail + scatter_fixed))) * 1.1
lim = min(lim, 30)
ax3.plot([-lim, lim], [-lim, lim], color=PAL_MUTED, linewidth=1, linestyle="--",
         alpha=0.6, zorder=0)
ax3.set_xlim(-lim, lim)
ax3.set_ylim(-lim, lim)
ax3.set_xlabel(f"Fixed SL 5.0% PnL (60min) %", fontsize=9, color=PAL_TEXT)
ax3.set_ylabel(f"Trailing k={best_k_60} PnL (60min) %", fontsize=9, color=PAL_TEXT)
ax3.set_title(f"Trail k={best_k_60} vs Fixed SL 5% (60min)", fontsize=11,
              fontweight="bold", color=PAL_TEXT, pad=8)
ax3.axhline(0, color=PAL_GRID, linewidth=0.5, zorder=0)
ax3.axvline(0, color=PAL_GRID, linewidth=0.5, zorder=0)
ax3.set_facecolor(PAL_SURFACE)
ax3.grid(True, color=PAL_GRID, linewidth=0.5, alpha=0.5)
# Legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=LONG_COLOR,
                          markersize=7, label='LONG'),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor=SHORT_COLOR,
                          markersize=7, label='SHORT')]
ax3.legend(handles=legend_elements, loc="upper left", fontsize=8, framealpha=0.8)

# --- Panel 4: E1b Hybrid results bar chart (SHORT only) ---
ax4 = fig.add_subplot(gs[1, 1])
hybrid_avgs = []
hybrid_labels = []
hybrid_tmeans = []
for k in K_VALUES:
    pnls = [r["pnl"] for r in e1b_results[k]]
    hybrid_avgs.append(np.mean(pnls) if pnls else 0)
    hybrid_tmeans.append(trimmed_mean(pnls))
    hybrid_labels.append(f"k={k}")

# Add reference bars
ref_cols_short = [("30m noSL", "pnl_30m_nosl"), ("60m noSL", "pnl_60m_nosl"),
                  ("T+2 close", "pnl_t2_close")]
for ref_name, ref_col in ref_cols_short:
    vals = short_df[ref_col].dropna()
    hybrid_avgs.append(float(np.mean(vals)) if len(vals) > 0 else 0)
    hybrid_tmeans.append(trimmed_mean(vals))
    hybrid_labels.append(ref_name)

x = np.arange(len(hybrid_labels))
width = 0.35
# Replace NaN with 0 for plotting
hybrid_avgs_plot = [v if not np.isnan(v) else 0 for v in hybrid_avgs]
hybrid_tmeans_plot = [v if not np.isnan(v) else 0 for v in hybrid_tmeans]
bars1 = ax4.bar(x - width/2, hybrid_avgs_plot, width, color=TRAIL_COLOR, alpha=0.8,
                edgecolor="none", label="Mean")
bars2 = ax4.bar(x + width/2, hybrid_tmeans_plot, width, color=PAL_TEAL, alpha=0.8,
                edgecolor="none", label="Trimmed Mean")
ax4.set_xticks(x)
ax4.set_xticklabels(hybrid_labels, fontsize=8, rotation=30, ha="right", color=PAL_TEXT)
ax4.set_ylabel("PnL %", fontsize=9, color=PAL_TEXT)
ax4.set_title("E1b: SHORT Hybrid Trail+T2 vs References", fontsize=11,
              fontweight="bold", color=PAL_TEXT, pad=8)
ax4.axhline(0, color=PAL_TEXT, linewidth=0.5)
ax4.grid(True, axis="y", color=PAL_GRID, linewidth=0.5, alpha=0.5)
ax4.set_facecolor(PAL_SURFACE)
ax4.legend(fontsize=8, framealpha=0.8)

# Value labels on bars
for bar in bars1:
    h = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2, h + (0.05 if h >= 0 else -0.15),
             f"{h:+.2f}", ha="center", va="bottom" if h >= 0 else "top",
             fontsize=7, color=PAL_TEXT)

# --- Panel 5: Cumulative PnL curve for best trailing vs no-SL ---
ax5 = fig.add_subplot(gs[2, 0])

# Best overall trailing config
best_trail = e1_results[best_key]
# Sort by trade index for temporal ordering
best_trail_sorted = sorted(best_trail, key=lambda r: r["idx"])
trail_cum = np.cumsum([r["pnl"] for r in best_trail_sorted])

# No-SL reference for same horizon
nosl_col = nosl_map[best_key[1]]
nosl_vals = []
for r in best_trail_sorted:
    idx = r["idx"]
    val = df.loc[idx, nosl_col]
    nosl_vals.append(val if pd.notna(val) else 0)
nosl_cum = np.cumsum(nosl_vals)

x_trades = np.arange(1, len(trail_cum) + 1)
ax5.plot(x_trades, trail_cum, color=TRAIL_COLOR, linewidth=2,
         label=f"Trail k={best_key[0]} ({best_key[1]})")
ax5.plot(x_trades, nosl_cum, color=NOSL_COLOR, linewidth=2, linestyle="--",
         label=f"No-SL ({best_key[1]})")
ax5.fill_between(x_trades, trail_cum, nosl_cum,
                 where=trail_cum >= nosl_cum, color=TRAIL_COLOR, alpha=0.1)
ax5.fill_between(x_trades, trail_cum, nosl_cum,
                 where=trail_cum < nosl_cum, color=SHORT_COLOR, alpha=0.1)
ax5.set_xlabel("Trade #", fontsize=9, color=PAL_TEXT)
ax5.set_ylabel("Cumulative PnL %", fontsize=9, color=PAL_TEXT)
ax5.set_title(f"Cumulative PnL: Trail k={best_key[0]} vs No-SL ({best_key[1]})",
              fontsize=11, fontweight="bold", color=PAL_TEXT, pad=8)
ax5.legend(fontsize=9, framealpha=0.8)
ax5.grid(True, color=PAL_GRID, linewidth=0.5, alpha=0.5)
ax5.set_facecolor(PAL_SURFACE)
ax5.axhline(0, color=PAL_MUTED, linewidth=0.5)

# --- Panel 6: Trailing stop rate by k x horizon ---
ax6 = fig.add_subplot(gs[2, 1])
stop_rates = np.zeros((len(K_VALUES), len(HORIZONS)))
for i, k in enumerate(K_VALUES):
    for j, h_name in enumerate(h_names):
        results = e1_results[(k, h_name)]
        if results:
            stop_rates[i, j] = sum(1 for r in results if r["stopped"]) / len(results) * 100

im6 = ax6.imshow(stop_rates, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
ax6.set_xticks(range(len(h_names)))
ax6.set_xticklabels(h_names, fontsize=9, color=PAL_TEXT)
ax6.set_yticks(range(len(K_VALUES)))
ax6.set_yticklabels([f"k={k}" for k in K_VALUES], fontsize=9, color=PAL_TEXT)
ax6.set_title("Stop Trigger Rate (%)", fontsize=11, fontweight="bold",
              color=PAL_TEXT, pad=8)
for i in range(len(K_VALUES)):
    for j in range(len(h_names)):
        ax6.text(j, i, f"{stop_rates[i,j]:.0f}%", ha="center", va="center",
                 fontsize=10, color=PAL_TEXT, fontweight="bold")
cb6 = plt.colorbar(im6, ax=ax6, shrink=0.8)
cb6.set_label("Stop Rate %", fontsize=9, color=PAL_TEXT)

plt.savefig(str(OUT_E1_PNG), dpi=150, facecolor=PAL_SURFACE)
plt.close()
print(f"  Saved: {OUT_E1_PNG}")


# ---------------------------------------------------------------------------
# E2 Charts
# ---------------------------------------------------------------------------
print("Generating E2 charts...")

fig2 = plt.figure(figsize=(18, 14))
fig2.patch.set_facecolor(PAL_SURFACE)
fig2.suptitle("E2: ORB Filter Refinement", fontsize=16, fontweight="bold",
              color=PAL_TEXT, y=0.98)
gs2 = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3, top=0.92, bottom=0.08,
                        left=0.07, right=0.95)

# --- Panel 1: Grouped bar chart: TF x threshold -> agree PnL, disagree PnL ---
ax_e1 = fig2.add_subplot(gs2[0, 0])
tf_names = list(ORB_TIMEFRAMES.keys())
n_tf = len(tf_names)
n_thresh = len(BREAKOUT_THRESHOLDS)
group_width = 0.8
bar_width = group_width / (n_thresh * 2)
x_base = np.arange(n_tf)

for t_idx, thresh in enumerate(BREAKOUT_THRESHOLDS):
    agree_vals = []
    disagree_vals = []
    for tf_name in tf_names:
        key = f"{tf_name}_t{thresh}_vf0"
        s = e2_summary.get(key, {})
        agree_vals.append(s.get("avg_agree_pnl") or 0)
        disagree_vals.append(s.get("avg_disagree_pnl") or 0)

    offset_a = -group_width/2 + t_idx * 2 * bar_width + bar_width * 0.5
    offset_d = offset_a + bar_width

    ax_e1.bar(x_base + offset_a, agree_vals, bar_width, color=AGREE_COLOR,
              alpha=0.5 + t_idx * 0.15, edgecolor="none",
              label=f"Agree t={thresh}" if t_idx == 0 else "")
    ax_e1.bar(x_base + offset_d, disagree_vals, bar_width, color=DISAGREE_COLOR,
              alpha=0.5 + t_idx * 0.15, edgecolor="none",
              label=f"Disagree t={thresh}" if t_idx == 0 else "")

ax_e1.set_xticks(x_base)
ax_e1.set_xticklabels(tf_names, fontsize=9, color=PAL_TEXT)
ax_e1.set_ylabel("Avg PnL % (60min)", fontsize=9, color=PAL_TEXT)
ax_e1.set_title("Agree vs Disagree PnL by ORB Timeframe", fontsize=11,
                fontweight="bold", color=PAL_TEXT, pad=8)
ax_e1.axhline(0, color=PAL_TEXT, linewidth=0.5)
ax_e1.grid(True, axis="y", color=PAL_GRID, linewidth=0.5, alpha=0.5)
ax_e1.set_facecolor(PAL_SURFACE)

# Custom legend
from matplotlib.patches import Patch
legend_items = [Patch(facecolor=AGREE_COLOR, alpha=0.8, label="Agree"),
                Patch(facecolor=DISAGREE_COLOR, alpha=0.8, label="Disagree")]
ax_e1.legend(handles=legend_items, fontsize=8, framealpha=0.8)

# --- Panel 2: Best ORB config vs current 5min ORB comparison ---
ax_e2 = fig2.add_subplot(gs2[0, 1])

# Find best ORB config by spread
best_orb_key = None
best_spread = -999
for key, s in e2_summary.items():
    sp = s.get("spread")
    n_a = s.get("n_agree", 0)
    n_d = s.get("n_disagree", 0)
    if sp is not None and n_a >= 10 and n_d >= 10 and sp > best_spread:
        best_spread = sp
        best_orb_key = key

# Current 5min ORB (thresh=0)
current_key = "5min_t0.0_vf0"
current = e2_summary.get(current_key, {})
best_orb = e2_summary.get(best_orb_key, {})

comparison_labels = [
    f"Current\n5min t=0.0",
    f"Best\n{best_orb.get('timeframe','?')} t={best_orb.get('threshold','?')}",
]
comp_agree = [current.get("avg_agree_pnl") or 0, best_orb.get("avg_agree_pnl") or 0]
comp_disagree = [current.get("avg_disagree_pnl") or 0, best_orb.get("avg_disagree_pnl") or 0]
comp_spread = [current.get("spread") or 0, best_orb.get("spread") or 0]

x_comp = np.arange(len(comparison_labels))
w = 0.25
b1 = ax_e2.bar(x_comp - w, comp_agree, w, color=AGREE_COLOR, alpha=0.8,
               edgecolor="none", label="Agree PnL")
b2 = ax_e2.bar(x_comp, comp_disagree, w, color=DISAGREE_COLOR, alpha=0.8,
               edgecolor="none", label="Disagree PnL")
b3 = ax_e2.bar(x_comp + w, comp_spread, w, color=PAL_PURPLE, alpha=0.8,
               edgecolor="none", label="Spread")

ax_e2.set_xticks(x_comp)
ax_e2.set_xticklabels(comparison_labels, fontsize=9, color=PAL_TEXT)
ax_e2.set_ylabel("PnL %", fontsize=9, color=PAL_TEXT)
ax_e2.set_title("Current 5min ORB vs Best Config", fontsize=11,
                fontweight="bold", color=PAL_TEXT, pad=8)
ax_e2.axhline(0, color=PAL_TEXT, linewidth=0.5)
ax_e2.legend(fontsize=8, framealpha=0.8)
ax_e2.grid(True, axis="y", color=PAL_GRID, linewidth=0.5, alpha=0.5)
ax_e2.set_facecolor(PAL_SURFACE)

# Value labels
for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        if h != 0:
            ax_e2.text(bar.get_x() + bar.get_width()/2,
                       h + (0.03 if h >= 0 else -0.08),
                       f"{h:+.2f}", ha="center",
                       va="bottom" if h >= 0 else "top",
                       fontsize=7, color=PAL_TEXT)

# --- Panel 3: Heatmap timeframe x threshold -> PnL spread ---
ax_e3 = fig2.add_subplot(gs2[1, 0])
spread_data = np.zeros((len(tf_names), len(BREAKOUT_THRESHOLDS)))
for i, tf in enumerate(tf_names):
    for j, thresh in enumerate(BREAKOUT_THRESHOLDS):
        key = f"{tf}_t{thresh}_vf0"
        s = e2_summary.get(key, {})
        sp = s.get("spread")
        spread_data[i, j] = sp if sp is not None else np.nan

if np.all(np.isnan(spread_data)):
    vmax3 = 1.0
else:
    vmax3 = max(abs(np.nanmin(spread_data)), abs(np.nanmax(spread_data)))
    vmax3 = max(vmax3, 0.1)
im3 = ax_e3.imshow(spread_data, cmap="RdYlGn", aspect="auto",
                   vmin=-vmax3, vmax=vmax3)
ax_e3.set_xticks(range(len(BREAKOUT_THRESHOLDS)))
ax_e3.set_xticklabels([f"{t:.2f}%" for t in BREAKOUT_THRESHOLDS], fontsize=9,
                      color=PAL_TEXT)
ax_e3.set_yticks(range(len(tf_names)))
ax_e3.set_yticklabels(tf_names, fontsize=9, color=PAL_TEXT)
ax_e3.set_xlabel("Breakout Threshold", fontsize=9, color=PAL_TEXT)
ax_e3.set_title("PnL Spread (Agree - Disagree) by TF x Threshold",
                fontsize=11, fontweight="bold", color=PAL_TEXT, pad=8)
for i in range(len(tf_names)):
    for j in range(len(BREAKOUT_THRESHOLDS)):
        val = spread_data[i, j]
        if not np.isnan(val):
            # Check n for low-n flag
            key = f"{tf_names[i]}_t{BREAKOUT_THRESHOLDS[j]}_vf0"
            s = e2_summary.get(key, {})
            n_a = s.get("n_agree", 0)
            n_d = s.get("n_disagree", 0)
            flag = "*" if n_a < 20 or n_d < 20 else ""
            ax_e3.text(j, i, f"{val:+.2f}{flag}", ha="center", va="center",
                       fontsize=9, color=PAL_TEXT, fontweight="bold")
cb3 = plt.colorbar(im3, ax=ax_e3, shrink=0.8)
cb3.set_label("PnL Spread %", fontsize=9, color=PAL_TEXT)

# --- Panel 4: Agreement rate and n counts ---
ax_e4 = fig2.add_subplot(gs2[1, 1])
agree_rates = []
n_breakouts = []
labels_e4 = []
for tf in tf_names:
    for thresh in BREAKOUT_THRESHOLDS:
        key = f"{tf}_t{thresh}_vf0"
        s = e2_summary.get(key, {})
        agree_rates.append(s.get("agree_rate", 0))
        n_breakouts.append(s.get("n_agree", 0) + s.get("n_disagree", 0))
        labels_e4.append(f"{tf}\nt={thresh}")

x_e4 = np.arange(len(labels_e4))
color_map = [PAL_BLUE, PAL_RED, PAL_GREEN, PAL_AMBER] * len(tf_names)

ax_e4_twin = ax_e4.twinx()

ax_e4.bar(x_e4, n_breakouts, 0.6, color=PAL_LIGHT_BLUE, alpha=0.6,
          edgecolor="none", label="n breakouts")
ax_e4_twin.plot(x_e4, agree_rates, color=PAL_PURPLE, linewidth=2,
                marker="o", markersize=5, label="Agree Rate")

ax_e4.set_xticks(x_e4)
ax_e4.set_xticklabels(labels_e4, fontsize=6, rotation=45, ha="right",
                      color=PAL_TEXT)
ax_e4.set_ylabel("n Breakouts", fontsize=9, color=PAL_TEXT)
ax_e4_twin.set_ylabel("Agree Rate %", fontsize=9, color=PAL_PURPLE)
ax_e4.set_title("Breakout Count and Agreement Rate", fontsize=11,
                fontweight="bold", color=PAL_TEXT, pad=8)
ax_e4.set_facecolor(PAL_SURFACE)
ax_e4.grid(True, axis="y", color=PAL_GRID, linewidth=0.5, alpha=0.5)
ax_e4_twin.set_ylim(30, 70)
ax_e4_twin.axhline(50, color=PAL_MUTED, linewidth=0.5, linestyle="--")

# Combined legend
lines1, labels1 = ax_e4.get_legend_handles_labels()
lines2, labels2 = ax_e4_twin.get_legend_handles_labels()
ax_e4.legend(lines1 + lines2, labels1 + labels2, fontsize=8, framealpha=0.8,
             loc="upper right")

plt.savefig(str(OUT_E2_PNG), dpi=150, facecolor=PAL_SURFACE)
plt.close()
print(f"  Saved: {OUT_E2_PNG}")


# =========================================================================
# Save JSON results
# =========================================================================
output = {
    "experiment": "V3 E1 (ATR Trailing Stops) + E2 (ORB Refinement)",
    "n_trades": len(df),
    "e1_summary": e1_summary,
    "e1_best_config": {
        "k": best_key[0], "horizon": best_key[1],
        "avg_pnl": best_avg,
    },
    "e1b_hybrid_short": {
        k: {
            "k": k,
            "avg_pnl": float(np.mean([r["pnl"] for r in e1b_results[k]])) if e1b_results[k] else None,
            "tmean_pnl": trimmed_mean([r["pnl"] for r in e1b_results[k]]),
            "n": len(e1b_results[k]),
            "n_stopped": sum(1 for r in e1b_results[k] if r["exit"] == "trail_stop"),
        }
        for k in K_VALUES
    },
    "e2_summary": e2_summary,
    "e2_best_config": {
        "key": best_orb_key,
        **(best_orb if best_orb else {}),
    },
}

with open(str(OUT_JSON), "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nSaved results JSON: {OUT_JSON}")


# =========================================================================
# Final summary
# =========================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\nE1 Best Trailing Stop: k={best_key[0]}, {best_key[1]} -> avg PnL {best_avg:+.3f}%")

# Best by direction
for direction in ["LONG", "SHORT"]:
    best_d_key = None
    best_d_avg = -999
    for k in K_VALUES:
        for h_name in HORIZONS:
            results = e1_results[(k, h_name)]
            pnls = [r["pnl"] for r in results if r["direction"] == direction]
            avg = np.mean(pnls) if pnls else -999
            if avg > best_d_avg:
                best_d_avg = avg
                best_d_key = (k, h_name)
    print(f"  Best {direction}: k={best_d_key[0]}, {best_d_key[1]} -> avg PnL {best_d_avg:+.3f}%")

# E1b best
best_hybrid_k = max(K_VALUES, key=lambda k: np.mean([r["pnl"] for r in e1b_results[k]]) if e1b_results[k] else -999)
best_hybrid_avg = np.mean([r["pnl"] for r in e1b_results[best_hybrid_k]])
print(f"\nE1b Best Hybrid (SHORT): k={best_hybrid_k} -> avg PnL {best_hybrid_avg:+.3f}%")

# E2 best
if best_orb_key:
    print(f"\nE2 Best ORB Config: {best_orb_key}")
    print(f"  Agree PnL: {fmt(best_orb.get('avg_agree_pnl'))}%")
    print(f"  Disagree PnL: {fmt(best_orb.get('avg_disagree_pnl'))}%")
    print(f"  Spread: {fmt(best_orb.get('spread'))}%")
    print(f"  Agree Rate: {best_orb.get('agree_rate', 0):.1f}%")

# Compare trail vs no-SL conclusion
print(f"\nE1 Conclusion:")
nosl_60_avg = np.mean(df["pnl_60m_nosl"].dropna())
trail_60_avg = np.mean([r["pnl"] for r in e1_results[(best_k_60, "60min")]])
delta = trail_60_avg - nosl_60_avg
if delta > 0:
    print(f"  Trailing stop k={best_k_60} at 60min ADDS {delta:+.3f}% vs no-SL ({trail_60_avg:+.3f}% vs {nosl_60_avg:+.3f}%)")
else:
    print(f"  Trailing stop k={best_k_60} at 60min DESTROYS {delta:+.3f}% vs no-SL ({trail_60_avg:+.3f}% vs {nosl_60_avg:+.3f}%)")
    print(f"  -> Trailing stops, like fixed stops, destroy value in this dataset.")

print(f"\nE2 Conclusion:")
if best_orb and best_orb.get("spread") and best_orb["spread"] > 0:
    print(f"  ORB agreement adds {best_orb['spread']:+.3f}% spread.")
    if best_orb.get("n_agree", 0) < 20 or best_orb.get("n_disagree", 0) < 20:
        print(f"  WARNING: Low-n in some cells. Not robust.")
else:
    print(f"  ORB filter shows no reliable spread.")

print("\nDone.")
