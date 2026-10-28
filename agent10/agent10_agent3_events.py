# %% Agent 3 — Event Run-Up / Sell-the-News Analysis for RXRX
#
# Investigates whether biotech event patterns (earnings, pipeline,
# conferences) explain z-score dislocations that S5_zexit trades.
#
# 1. Builds event calendar via heuristic detection (volume + return spikes)
# 2. Measures run-up / sell-the-news patterns
# 3. Splits S5_zexit trades into near-event vs far-from-event buckets
# 4. Tests event-avoidance gate on S5_zexit
#
# Notebook-friendly: config vars at top, no if __name__

import os, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────────
SEL_END = "2026-01-01"                   # selection / OOS cutoff
EVENT_VOL_MULT = 2.0                     # volume > 2x 20d avg
EVENT_RET_MULT = 2.0                     # |return| > 2x 20d avg |return|
EVENT_LOOKBACK = 20                      # trailing window for thresholds
EVENT_PROXIMITY_DAYS = 5                 # "near event" trade window
EVENT_GATE_DAYS = 3                      # skip signals within +/- N days
PRE_EVENT_WINDOW = 10                    # days before event to measure drift
POST_EVENT_WINDOW = 5                    # days after event to measure drift
EVENT_STUDY_WINDOW = 10                  # +/- days for event study plot
S5_K_ENTRY = 1.5                         # z-score entry threshold
S5_Z_EXIT = 0.5                          # z-score exit threshold
S5_MAX_HOLD = 10                         # max holding period
S5_LOOKBACK = 60                         # spread lookback
MIN_EVENT_GAP = 3                        # min days between events (dedup)

DATA_DIR = Path(__file__).parent / "data" if "__file__" in dir() else Path("agent10/data")
OUT_DIR = Path(__file__).parent / "results" if "__file__" in dir() else Path("agent10/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Engine imports ───────────────────────────────────────────────
ENGINE_DIR = Path(__file__).parent if "__file__" in dir() else Path("agent10")
sys.path.insert(0, str(ENGINE_DIR))

from agent0_engine import (
    build_spread, run_backtest, compute_stats, compute_costs,
    log_trial, add_features, s5_zexit_signal, s5_zexit_exit,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ══════════════════════════════════════════════════════════════════
# 1. LOAD DATA + BUILD SPREAD
# ══════════════════════════════════════════════════════════════════
# %%
print("=" * 70)
print("  AGENT 3: EVENT RUN-UP / SELL-THE-NEWS ANALYSIS")
print("=" * 70)

rxrx = pd.read_parquet(DATA_DIR / "rxrx_daily.parquet")
rxrx["date"] = pd.to_datetime(rxrx["date"])
rxrx = rxrx.sort_values("date").reset_index(drop=True)
rxrx = add_features(rxrx)

ctx = pd.read_parquet(DATA_DIR / "context_daily.parquet")
ctx["date"] = pd.to_datetime(ctx["date"])
arkg = ctx[ctx["ticker"] == "ARKG"].sort_values("date").reset_index(drop=True)

print(f"  RXRX: {len(rxrx)} bars, {rxrx['date'].min().date()} to {rxrx['date'].max().date()}")
print(f"  ARKG: {len(arkg)} bars")

# Build canonical spread
print("  Building OLS spread (lookback=60)...")
spread = build_spread(rxrx, arkg, "ARKG", S5_LOOKBACK)
print(f"  Spread: {len(spread)} bars with z-scores")

# ══════════════════════════════════════════════════════════════════
# 2. BUILD EVENT CALENDAR (heuristic + known events)
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  STEP 1: EVENT DETECTION")
print("=" * 70)

# --- 2a. Heuristic detection: volume + return anomalies (trailing only) ---
rxrx_ev = rxrx.copy()
rxrx_ev["abs_ret"] = rxrx_ev["pct_ret"].abs()
rxrx_ev["vol_ma20"] = rxrx_ev["volume"].rolling(EVENT_LOOKBACK, min_periods=EVENT_LOOKBACK).mean()
rxrx_ev["ret_ma20"] = rxrx_ev["abs_ret"].rolling(EVENT_LOOKBACK, min_periods=EVENT_LOOKBACK).mean()

# Shift by 1 to use strictly trailing data (avoid lookahead)
rxrx_ev["vol_ratio"] = rxrx_ev["volume"] / rxrx_ev["vol_ma20"].shift(1)
rxrx_ev["ret_ratio"] = rxrx_ev["abs_ret"] / rxrx_ev["ret_ma20"].shift(1)

# Event day = both volume and return are abnormal
rxrx_ev["is_event_heuristic"] = (
    (rxrx_ev["vol_ratio"] >= EVENT_VOL_MULT) &
    (rxrx_ev["ret_ratio"] >= EVENT_RET_MULT) &
    rxrx_ev["vol_ratio"].notna()
)

# Deduplicate: if multiple events within MIN_EVENT_GAP days, keep the strongest
heuristic_events = rxrx_ev[rxrx_ev["is_event_heuristic"]].copy()
heuristic_events = heuristic_events.sort_values("date").reset_index(drop=True)

deduped_dates = []
last_event_date = pd.Timestamp("1900-01-01")
for _, row in heuristic_events.iterrows():
    if (row["date"] - last_event_date).days >= MIN_EVENT_GAP:
        deduped_dates.append(row["date"])
        last_event_date = row["date"]

heuristic_event_dates = set(deduped_dates)

# --- 2b. Load known events from parquet ---
known_events = pd.DataFrame()
known_event_path = DATA_DIR / "rxrx_events.parquet"
if known_event_path.exists():
    known_events = pd.read_parquet(known_event_path)
    known_events["date"] = pd.to_datetime(known_events["date"])
    print(f"  Known events loaded: {len(known_events)}")
    for _, row in known_events.iterrows():
        print(f"    {row['date'].date()} | {row['type']} | {row['detail']}")
    known_event_dates = set(known_events["date"].tolist())
else:
    known_event_dates = set()
    print("  No known events file found.")

# --- 2c. Try Polygon API for dividends/splits ---
polygon_events = []
try:
    from dotenv import load_dotenv
    for env_path in [Path(".env"), Path("bot-RAG/.env"), Path(__file__).parent.parent / ".env"]:
        if env_path.exists():
            load_dotenv(env_path)
            break
except:
    pass

POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "")
if POLYGON_API_KEY:
    import urllib.request, json
    for event_type in ["dividends", "splits"]:
        try:
            url = f"https://api.polygon.io/v3/reference/{event_type}?ticker=RXRX&apiKey={POLYGON_API_KEY}"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                for r in data.get("results", []):
                    d = pd.Timestamp(r.get("ex_dividend_date") or r.get("execution_date", ""))
                    if pd.notna(d):
                        polygon_events.append({"date": d, "type": event_type,
                                                "detail": str(r)})
            print(f"  Polygon {event_type}: {len([e for e in polygon_events if e['type'] == event_type])} found")
        except Exception as e:
            print(f"  Polygon {event_type} API: {e}")
else:
    print("  No POLYGON_API_KEY found, skipping API calls.")

polygon_event_dates = set(pd.Timestamp(e["date"]) for e in polygon_events)

# --- 2d. Combine all event dates ---
all_event_dates = sorted(heuristic_event_dates | known_event_dates | polygon_event_dates)
print(f"\n  Total events detected: {len(all_event_dates)}")
print(f"    Heuristic (vol+ret spike): {len(heuristic_event_dates)}")
print(f"    Known (parquet):           {len(known_event_dates)}")
print(f"    Polygon API:               {len(polygon_event_dates)}")

# ══════════════════════════════════════════════════════════════════
# 3. EVENT CALENDAR TABLE
# ══════════════════════════════════════════════════════════════════
# %%
print("\n  ── Event Calendar ──")
event_records = []
for ed in all_event_dates:
    idx_arr = rxrx.index[rxrx["date"] == ed]
    if len(idx_arr) == 0:
        # Try nearest trading day
        nearest_idx = (rxrx["date"] - ed).abs().idxmin()
        if abs((rxrx.loc[nearest_idx, "date"] - ed).days) > 3:
            continue
        idx = nearest_idx
    else:
        idx = idx_arr[0]

    row = rxrx.loc[idx]
    vol_r = row["volume"] / rxrx.loc[max(0, idx - EVENT_LOOKBACK):idx - 1]["volume"].mean() \
        if idx > EVENT_LOOKBACK else np.nan
    day_ret = row["pct_ret"] * 100 if pd.notna(row["pct_ret"]) else 0

    # Source classification
    sources = []
    if ed in heuristic_event_dates:
        sources.append("heuristic")
    if ed in known_event_dates:
        sources.append("known")
    if ed in polygon_event_dates:
        sources.append("polygon")

    # Known event detail
    detail = ""
    if ed in known_event_dates:
        match = known_events[known_events["date"] == ed]
        if not match.empty:
            detail = match.iloc[0]["detail"]

    event_records.append({
        "date": ed,
        "close": round(row["close"], 2),
        "return_pct": round(day_ret, 2),
        "vol_ratio": round(vol_r, 2) if pd.notna(vol_r) else np.nan,
        "source": "+".join(sources),
        "detail": detail,
    })

events_df = pd.DataFrame(event_records).sort_values("date").reset_index(drop=True)
print(f"\n  {'Date':>12s} {'Close':>7s} {'Ret%':>7s} {'VolR':>6s} {'Source':>16s} {'Detail'}")
print("  " + "-" * 80)
for _, ev in events_df.iterrows():
    print(f"  {str(ev['date'].date()):>12s} {ev['close']:>7.2f} {ev['return_pct']:>+6.2f}% "
          f"{ev['vol_ratio']:>6.1f}x {ev['source']:>16s} {ev['detail'][:40]}")

# ══════════════════════════════════════════════════════════════════
# 4. RUN-UP / SELL-THE-NEWS CLASSIFICATION
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  STEP 2: RUN-UP / SELL-THE-NEWS PATTERNS")
print("=" * 70)

pattern_records = []
for ed in all_event_dates:
    idx_arr = rxrx.index[rxrx["date"] == ed]
    if len(idx_arr) == 0:
        continue
    idx = idx_arr[0]

    # Pre-event drift: cumulative return t-PRE_EVENT_WINDOW to t-1
    pre_start = max(0, idx - PRE_EVENT_WINDOW)
    pre_end = max(0, idx - 1)
    if pre_end <= pre_start or pre_start >= len(rxrx):
        continue

    pre_close_start = rxrx.loc[pre_start, "close"]
    pre_close_end = rxrx.loc[pre_end, "close"]
    pre_drift = (pre_close_end - pre_close_start) / pre_close_start * 100

    # Event-day return
    event_ret = rxrx.loc[idx, "pct_ret"] * 100 if pd.notna(rxrx.loc[idx, "pct_ret"]) else 0

    # Post-event drift: cumulative return t+1 to t+POST_EVENT_WINDOW
    post_start = min(idx + 1, len(rxrx) - 1)
    post_end = min(idx + POST_EVENT_WINDOW, len(rxrx) - 1)
    if post_start >= len(rxrx) or post_end <= post_start:
        continue

    post_close_start = rxrx.loc[post_start, "close"]
    post_close_end = rxrx.loc[post_end, "close"]
    post_drift = (post_close_end - post_close_start) / post_close_start * 100

    # Classification
    is_runup_stn = (pre_drift > 0) and (post_drift < 0)
    is_selloff_btn = (pre_drift < 0) and (post_drift > 0)
    if is_runup_stn:
        pattern = "RUN-UP/STN"
    elif is_selloff_btn:
        pattern = "SELLOFF/BTN"
    elif pre_drift > 0 and post_drift > 0:
        pattern = "SUSTAINED_UP"
    elif pre_drift < 0 and post_drift < 0:
        pattern = "SUSTAINED_DN"
    else:
        pattern = "MIXED"

    pattern_records.append({
        "date": ed,
        "pre_drift_pct": round(pre_drift, 2),
        "event_ret_pct": round(event_ret, 2),
        "post_drift_pct": round(post_drift, 2),
        "pattern": pattern,
    })

patterns_df = pd.DataFrame(pattern_records)
if not patterns_df.empty:
    # Split by selection/OOS
    sel_patterns = patterns_df[patterns_df["date"] < SEL_END]
    oos_patterns = patterns_df[patterns_df["date"] >= SEL_END]

    print(f"\n  {'Date':>12s} {'Pre-10d':>8s} {'Event':>8s} {'Post-5d':>8s} {'Pattern':>14s}")
    print("  " + "-" * 60)
    for _, p in patterns_df.iterrows():
        marker = " OOS" if p["date"] >= pd.Timestamp(SEL_END) else ""
        print(f"  {str(p['date'].date()):>12s} {p['pre_drift_pct']:>+7.2f}% {p['event_ret_pct']:>+7.2f}% "
              f"{p['post_drift_pct']:>+7.2f}% {p['pattern']:>14s}{marker}")

    print(f"\n  Pattern counts (selection window):")
    if not sel_patterns.empty:
        for pat, cnt in sel_patterns["pattern"].value_counts().items():
            print(f"    {pat}: {cnt}")
    print(f"\n  Pattern counts (OOS 2026+):")
    if not oos_patterns.empty:
        for pat, cnt in oos_patterns["pattern"].value_counts().items():
            print(f"    {pat}: {cnt}")

    n_runup_stn = (patterns_df["pattern"] == "RUN-UP/STN").sum()
    n_total = len(patterns_df)
    print(f"\n  Run-up/sell-the-news: {n_runup_stn}/{n_total} events ({n_runup_stn/n_total*100:.0f}%)")
else:
    print("  No patterns computed (insufficient event data).")

# ══════════════════════════════════════════════════════════════════
# 5. RUN S5_ZEXIT BASELINE
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  STEP 3: S5_ZEXIT BASELINE TRADES")
print("=" * 70)

def sig_fn(row, i, df):
    return s5_zexit_signal(row, i, df, k_entry=S5_K_ENTRY)

def exit_fn(sig, row, j, df):
    return s5_zexit_exit(sig, row, j, df, z_exit_thresh=S5_Z_EXIT)

all_trades = run_backtest(spread, sig_fn, exit_fn, max_hold=S5_MAX_HOLD)
sel_trades = all_trades[all_trades["entry_date"] < SEL_END].copy()
oos_trades = all_trades[all_trades["entry_date"] >= SEL_END].copy()

sel_stats_base = compute_stats(sel_trades, label="S5_zexit Selection")
oos_stats_base = compute_stats(oos_trades, label="S5_zexit OOS")

print(f"  Baseline S5_zexit:")
print(f"    Selection: n={sel_stats_base['n']}, avg={sel_stats_base['avg_pnl']:+.3f}%, "
      f"Sharpe={sel_stats_base['sharpe']}, t={sel_stats_base['t_stat']:.2f}")
print(f"    OOS 2026:  n={oos_stats_base['n']}, avg={oos_stats_base['avg_pnl']:+.3f}%, "
      f"Sharpe={oos_stats_base['sharpe']}, t={oos_stats_base['t_stat']:.2f}")

# ══════════════════════════════════════════════════════════════════
# 6. Z-SCORE INTERACTION: NEAR-EVENT vs FAR-FROM-EVENT
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  STEP 4: NEAR-EVENT vs FAR-FROM-EVENT TRADE PERFORMANCE")
print("=" * 70)

# Build event proximity column for all trades
event_dates_ts = pd.DatetimeIndex([pd.Timestamp(d) for d in all_event_dates])

def min_event_distance(trade_date):
    """Minimum calendar day distance from trade entry to any event."""
    if len(event_dates_ts) == 0:
        return 999
    dists = np.abs((event_dates_ts - trade_date).days)
    return int(dists.min())

all_trades["event_dist"] = all_trades["entry_date"].apply(min_event_distance)
all_trades["near_event"] = all_trades["event_dist"] <= EVENT_PROXIMITY_DAYS

# Split: selection window
sel_near = all_trades[(all_trades["entry_date"] < SEL_END) & all_trades["near_event"]]
sel_far = all_trades[(all_trades["entry_date"] < SEL_END) & ~all_trades["near_event"]]

sel_near_stats = compute_stats(sel_near, label="Selection Near-Event")
sel_far_stats = compute_stats(sel_far, label="Selection Far-Event")

print(f"\n  SELECTION WINDOW:")
print(f"    Near-event  (within {EVENT_PROXIMITY_DAYS}d): n={sel_near_stats['n']}, "
      f"avg={sel_near_stats['avg_pnl']:+.3f}%, wr={sel_near_stats['wr']:.1f}%")
print(f"    Far-from-event:               n={sel_far_stats['n']}, "
      f"avg={sel_far_stats['avg_pnl']:+.3f}%, wr={sel_far_stats['wr']:.1f}%")

# Split: OOS
oos_near = all_trades[(all_trades["entry_date"] >= SEL_END) & all_trades["near_event"]]
oos_far = all_trades[(all_trades["entry_date"] >= SEL_END) & ~all_trades["near_event"]]

oos_near_stats = compute_stats(oos_near, label="OOS Near-Event")
oos_far_stats = compute_stats(oos_far, label="OOS Far-Event")

print(f"\n  OOS (2026+):")
print(f"    Near-event  (within {EVENT_PROXIMITY_DAYS}d): n={oos_near_stats['n']}, "
      f"avg={oos_near_stats['avg_pnl']:+.3f}%, wr={oos_near_stats['wr']:.1f}%")
print(f"    Far-from-event:               n={oos_far_stats['n']}, "
      f"avg={oos_far_stats['avg_pnl']:+.3f}%, wr={oos_far_stats['wr']:.1f}%")

# Statistical test: near vs far (selection window)
if sel_near_stats["n"] >= 3 and sel_far_stats["n"] >= 3:
    from scipy.stats import mannwhitneyu
    u_stat, u_pval = mannwhitneyu(
        sel_near["pnl_net_pct"].values,
        sel_far["pnl_net_pct"].values,
        alternative="two-sided"
    )
    print(f"\n  Mann-Whitney U (near vs far, selection): U={u_stat:.1f}, p={u_pval:.4f} "
          f"({'SIGNIFICANT' if u_pval < 0.05 else 'NOT significant'})")
else:
    u_pval = 1.0
    print(f"\n  Too few trades for statistical test (near={sel_near_stats['n']}, far={sel_far_stats['n']})")

# ══════════════════════════════════════════════════════════════════
# 7. EVENT-GATED VARIANT: SKIP SIGNALS NEAR EVENTS
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  STEP 5: EVENT-AVOIDANCE GATE (skip within +/-{} days)".format(EVENT_GATE_DAYS))
print("=" * 70)

# Build a boolean mask on the spread dataframe for event proximity
spread_dates = spread["date"].values

def is_near_event(d, gate_days=EVENT_GATE_DAYS):
    """Check if a date is within gate_days of any event."""
    if len(event_dates_ts) == 0:
        return False
    dists = np.abs((event_dates_ts - d).days)
    return int(dists.min()) <= gate_days

spread["near_event"] = spread["date"].apply(is_near_event)
n_gated_bars = spread["near_event"].sum()
print(f"  Bars near events (gated out): {n_gated_bars}/{len(spread)} "
      f"({n_gated_bars/len(spread)*100:.1f}%)")

# Event-gated signal function
def sig_fn_gated(row, i, df):
    """S5_zexit signal that skips bars near events."""
    if row.get("near_event", False):
        return None
    return s5_zexit_signal(row, i, df, k_entry=S5_K_ENTRY)

gated_trades = run_backtest(spread, sig_fn_gated, exit_fn, max_hold=S5_MAX_HOLD)
sel_gated = gated_trades[gated_trades["entry_date"] < SEL_END]
oos_gated = gated_trades[gated_trades["entry_date"] >= SEL_END]

sel_gated_stats = compute_stats(sel_gated, label="S5_zexit+EventGate Selection")
oos_gated_stats = compute_stats(oos_gated, label="S5_zexit+EventGate OOS")

print(f"\n  Comparison (Selection):")
print(f"    Baseline:    n={sel_stats_base['n']}, avg={sel_stats_base['avg_pnl']:+.3f}%, "
      f"Sharpe={sel_stats_base['sharpe']}, t={sel_stats_base['t_stat']:.2f}")
print(f"    Event-gated: n={sel_gated_stats['n']}, avg={sel_gated_stats['avg_pnl']:+.3f}%, "
      f"Sharpe={sel_gated_stats['sharpe']}, t={sel_gated_stats['t_stat']:.2f}")
sel_delta_avg = sel_gated_stats["avg_pnl"] - sel_stats_base["avg_pnl"]
print(f"    Delta avg PnL: {sel_delta_avg:+.3f}%")

print(f"\n  Comparison (OOS 2026+):")
print(f"    Baseline:    n={oos_stats_base['n']}, avg={oos_stats_base['avg_pnl']:+.3f}%, "
      f"Sharpe={oos_stats_base['sharpe']}, t={oos_stats_base['t_stat']:.2f}")
print(f"    Event-gated: n={oos_gated_stats['n']}, avg={oos_gated_stats['avg_pnl']:+.3f}%, "
      f"Sharpe={oos_gated_stats['sharpe']}, t={oos_gated_stats['t_stat']:.2f}")
oos_delta_avg = oos_gated_stats["avg_pnl"] - oos_stats_base["avg_pnl"]
print(f"    Delta avg PnL: {oos_delta_avg:+.3f}%")

# ══════════════════════════════════════════════════════════════════
# 8. EVENT STUDY: AVERAGE CUMULATIVE RETURN AROUND EVENTS
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  STEP 6: EVENT STUDY (average cumulative return +/-{} days)".format(EVENT_STUDY_WINDOW))
print("=" * 70)

cum_returns_matrix = []
for ed in all_event_dates:
    idx_arr = rxrx.index[rxrx["date"] == ed]
    if len(idx_arr) == 0:
        continue
    idx = idx_arr[0]

    start_idx = idx - EVENT_STUDY_WINDOW
    end_idx = idx + EVENT_STUDY_WINDOW
    if start_idx < 0 or end_idx >= len(rxrx):
        continue

    # Cumulative return relative to t-1 close (event study convention)
    base_price = rxrx.loc[idx - 1, "close"]
    returns = []
    for offset in range(-EVENT_STUDY_WINDOW, EVENT_STUDY_WINDOW + 1):
        ret = (rxrx.loc[idx + offset, "close"] - base_price) / base_price * 100
        returns.append(ret)
    cum_returns_matrix.append(returns)

offsets = list(range(-EVENT_STUDY_WINDOW, EVENT_STUDY_WINDOW + 1))

if cum_returns_matrix:
    avg_cum_ret = np.mean(cum_returns_matrix, axis=0)
    std_cum_ret = np.std(cum_returns_matrix, axis=0)
    n_events_study = len(cum_returns_matrix)

    print(f"  Events in study: {n_events_study}")
    print(f"  Average cumulative return profile:")
    for i, (off, ret) in enumerate(zip(offsets, avg_cum_ret)):
        marker = " <-- EVENT" if off == 0 else ""
        print(f"    t{off:+d}: {ret:+.2f}% (+/- {std_cum_ret[i]:.2f}%){marker}")
else:
    avg_cum_ret = np.zeros(len(offsets))
    std_cum_ret = np.zeros(len(offsets))
    n_events_study = 0
    print("  No events with sufficient data for event study.")

# ══════════════════════════════════════════════════════════════════
# 9. LOG TRIALS
# ══════════════════════════════════════════════════════════════════
# %%
log_trial("agent3", "S5_zexit_baseline",
          {"k": S5_K_ENTRY, "ze": S5_Z_EXIT, "max_hold": S5_MAX_HOLD},
          sel_stats_base)

log_trial("agent3", "S5_zexit_event_gate",
          {"k": S5_K_ENTRY, "ze": S5_Z_EXIT, "max_hold": S5_MAX_HOLD,
           "event_gate_days": EVENT_GATE_DAYS},
          sel_gated_stats)

log_trial("agent3", "S5_zexit_near_event",
          {"k": S5_K_ENTRY, "ze": S5_Z_EXIT, "proximity_days": EVENT_PROXIMITY_DAYS},
          sel_near_stats)

log_trial("agent3", "S5_zexit_far_event",
          {"k": S5_K_ENTRY, "ze": S5_Z_EXIT, "proximity_days": EVENT_PROXIMITY_DAYS},
          sel_far_stats)

print(f"\n  Logged 4 trials to {OUT_DIR / 'trials_log.csv'}")

# ══════════════════════════════════════════════════════════════════
# 10. MATPLOTLIB FIGURE (4 panels)
# ══════════════════════════════════════════════════════════════════
# %%
print("\n  Generating figure...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Agent 3: RXRX Event Run-Up / Sell-the-News Analysis", fontsize=14, fontweight="bold")

# ── Panel A: RXRX price with event days marked ──
ax = axes[0, 0]
ax.plot(rxrx["date"], rxrx["close"], color="steelblue", linewidth=0.8, label="RXRX Close")
# Mark events
for ed in all_event_dates:
    idx_arr = rxrx.index[rxrx["date"] == ed]
    if len(idx_arr) > 0:
        idx = idx_arr[0]
        ax.axvline(rxrx.loc[idx, "date"], color="red", alpha=0.4, linewidth=0.7)
        ax.plot(rxrx.loc[idx, "date"], rxrx.loc[idx, "close"], "rv", markersize=5, alpha=0.7)
# Mark selection/OOS cutoff
ax.axvline(pd.Timestamp(SEL_END), color="black", linestyle="--", linewidth=1.2, label="OOS cutoff")
ax.set_title("(a) RXRX Price + Detected Event Days (red)")
ax.set_ylabel("Price ($)")
ax.legend(loc="upper left", fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.tick_params(axis="x", rotation=45)
ax.grid(True, alpha=0.3)

# ── Panel B: Average cumulative return around events ──
ax = axes[0, 1]
if n_events_study > 0:
    ax.fill_between(offsets,
                    avg_cum_ret - std_cum_ret,
                    avg_cum_ret + std_cum_ret,
                    alpha=0.2, color="steelblue")
    ax.plot(offsets, avg_cum_ret, "o-", color="steelblue", linewidth=2, markersize=4,
            label=f"Mean (n={n_events_study})")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Event Day")
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Days Relative to Event")
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend(fontsize=8)
else:
    ax.text(0.5, 0.5, "No events with sufficient data", ha="center", va="center",
            transform=ax.transAxes)
ax.set_title("(b) Event Study: Avg Cumulative Return +/-10d")
ax.grid(True, alpha=0.3)

# ── Panel C: Near-event vs far-from-event trade PnL ──
ax = axes[1, 0]
categories = ["Near Event\n(Selection)", "Far Event\n(Selection)",
              "Near Event\n(OOS)", "Far Event\n(OOS)"]
avg_pnls = [sel_near_stats["avg_pnl"], sel_far_stats["avg_pnl"],
            oos_near_stats["avg_pnl"], oos_far_stats["avg_pnl"]]
counts = [sel_near_stats["n"], sel_far_stats["n"],
          oos_near_stats["n"], oos_far_stats["n"]]
colors_bar = ["#d62728" if v < 0 else "#2ca02c" for v in avg_pnls]

bars = ax.bar(categories, avg_pnls, color=colors_bar, alpha=0.8, edgecolor="black", linewidth=0.5)
for bar, n, v in zip(bars, counts, avg_pnls):
    ypos = bar.get_height()
    offset = 0.05 if ypos >= 0 else -0.15
    ax.text(bar.get_x() + bar.get_width() / 2, ypos + offset,
            f"n={n}\n{v:+.2f}%", ha="center", va="bottom" if ypos >= 0 else "top", fontsize=8)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title(f"(c) Z-Score Signal PnL: Near ({EVENT_PROXIMITY_DAYS}d) vs Far from Events")
ax.set_ylabel("Avg Net PnL (%)")
ax.grid(True, alpha=0.3, axis="y")

# ── Panel D: S5_zexit with vs without event gate ──
ax = axes[1, 1]
# Cumulative PnL comparison
def cum_pnl_series(trades_df, label_prefix):
    """Build cumulative PnL series from trades."""
    if trades_df.empty:
        return pd.Series(dtype=float)
    sorted_t = trades_df.sort_values("entry_date")
    cum = sorted_t["pnl_net_pct"].cumsum()
    cum.index = sorted_t["entry_date"]
    return cum

baseline_cum = cum_pnl_series(all_trades, "Baseline")
gated_cum = cum_pnl_series(gated_trades, "Gated")

if not baseline_cum.empty:
    ax.plot(baseline_cum.index, baseline_cum.values, "-", color="steelblue", linewidth=1.5,
            label=f"Baseline (n={len(all_trades)}, avg={compute_stats(all_trades)['avg_pnl']:+.2f}%)")
if not gated_cum.empty:
    ax.plot(gated_cum.index, gated_cum.values, "-", color="darkorange", linewidth=1.5,
            label=f"Event Gate +/-{EVENT_GATE_DAYS}d (n={len(gated_trades)}, "
                  f"avg={compute_stats(gated_trades)['avg_pnl']:+.2f}%)")
ax.axvline(pd.Timestamp(SEL_END), color="black", linestyle="--", linewidth=1.2, label="OOS cutoff")
ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)
ax.set_title("(d) S5_zexit Cumulative PnL: Baseline vs Event-Gated")
ax.set_ylabel("Cumulative Net PnL (%)")
ax.legend(loc="upper left", fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.tick_params(axis="x", rotation=45)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = OUT_DIR / "agent3_events.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Figure saved to {fig_path}")

# ══════════════════════════════════════════════════════════════════
# 11. CONCLUSION
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  CONCLUSION")
print("=" * 70)

# Determine if events drive the edge
near_far_sel_diff = sel_near_stats["avg_pnl"] - sel_far_stats["avg_pnl"]
gate_sel_diff = sel_gated_stats["avg_pnl"] - sel_stats_base["avg_pnl"]
gate_oos_diff = oos_gated_stats["avg_pnl"] - oos_stats_base["avg_pnl"]

print(f"""
  EVENT DETECTION SUMMARY:
    Total events detected: {len(all_event_dates)}
    Run-up/sell-the-news pattern: {n_runup_stn}/{n_total} ({n_runup_stn/max(n_total,1)*100:.0f}%)

  DO EVENTS DRIVE THE Z-SCORE EDGE?
    Near-event avg PnL (selection):  {sel_near_stats['avg_pnl']:+.3f}% (n={sel_near_stats['n']})
    Far-from-event avg PnL (sel):    {sel_far_stats['avg_pnl']:+.3f}% (n={sel_far_stats['n']})
    Difference:                      {near_far_sel_diff:+.3f}%
    Mann-Whitney p-value:            {u_pval:.4f}
""")

if abs(near_far_sel_diff) < 0.3 or u_pval > 0.10:
    event_conclusion = "NO - events do NOT significantly drive the z-score edge"
    event_detail = ("Near-event and far-from-event trades show similar performance. "
                    "The z-score mean-reversion signal works regardless of event proximity.")
elif near_far_sel_diff > 0.3 and u_pval <= 0.10:
    event_conclusion = "PARTIALLY - near-event trades are STRONGER"
    event_detail = ("Events create larger z-score dislocations that revert more profitably. "
                    "However, the far-from-event edge still exists, so events amplify but don't create the signal.")
elif near_far_sel_diff < -0.3 and u_pval <= 0.10:
    event_conclusion = "NEGATIVE - near-event trades HURT performance"
    event_detail = ("Events create z-score signals that fail to revert normally. "
                    "Event avoidance may improve the strategy.")
else:
    event_conclusion = "INCONCLUSIVE"
    event_detail = "Insufficient evidence to determine event impact."

print(f"  VERDICT: {event_conclusion}")
print(f"  DETAIL: {event_detail}")

print(f"""
  DOES EVENT AVOIDANCE HELP?
    Selection: baseline avg={sel_stats_base['avg_pnl']:+.3f}% -> gated avg={sel_gated_stats['avg_pnl']:+.3f}% (delta={gate_sel_diff:+.3f}%)
    OOS 2026:  baseline avg={oos_stats_base['avg_pnl']:+.3f}% -> gated avg={oos_gated_stats['avg_pnl']:+.3f}% (delta={gate_oos_diff:+.3f}%)
""")

if gate_sel_diff > 0.1 and gate_oos_diff > 0:
    gate_conclusion = "YES - event avoidance improves PnL in both selection and OOS"
elif gate_sel_diff > 0.1 and gate_oos_diff <= 0:
    gate_conclusion = "OVERFIT - event avoidance helps selection but NOT OOS"
elif gate_sel_diff <= 0.1 and gate_oos_diff <= 0:
    gate_conclusion = "NO - event avoidance does not help (or hurts)"
else:
    gate_conclusion = "MARGINAL - small improvement, not robust"

print(f"  EVENT GATE VERDICT: {gate_conclusion}")
print(f"\n  AGENT 3 COMPLETE")
print("=" * 70)
