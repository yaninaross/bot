# %% Agent 10 — Daily Signal Monitor for S5_zexit on RXRX
#
# Purpose: Compute current z-score and print today's signal status.
# This is a MONITORING script, not an auto-trading bot.
# Run daily after market close to check for entry/exit signals.
#
# Canonical config (frozen):
#   Hedge: ARKG, OLS 60-day lookback
#   Entry: z > 1.5 -> SHORT, z < -1.5 -> LONG (MOO next day)
#   Exit: z crosses +/-0.5, OR 1.5 ATR stop, OR max_hold days
#
# Usage: Run as a notebook cell or `python agent10_daily_signal.py`

import sys, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data" if "__file__" in dir() else Path("agent10/data")
LOOKBACK = 60
K_ENTRY = 1.5
Z_EXIT = 0.5
MAX_HOLD = 10  # Change to 20 if deploying with extended hold
ATR_STOP_MULT = 1.5
KILL_SWITCH_WINDOW = 5  # Check last N trades
KILL_SWITCH_THRESHOLD = 0.0  # Avg PnL below this triggers warning
CORR_KILL_THRESHOLD = 0.50  # RXRX-ARKG correlation below this triggers warning

# ── Import engine ────────────────────────────────────────────────
try:
    from agent0_engine import build_spread, add_features, s5_zexit_signal
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent if "__file__" in dir() else Path("agent10")))
    from agent0_engine import build_spread, add_features, s5_zexit_signal

# ── Load data ────────────────────────────────────────────────────
print("=" * 60)
print("  RXRX S5_zexit Daily Signal Monitor")
print(f"  Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

rxrx = pd.read_parquet(DATA_DIR / "rxrx_daily.parquet")
rxrx["date"] = pd.to_datetime(rxrx["date"])
rxrx = rxrx.sort_values("date").reset_index(drop=True)

ctx = pd.read_parquet(DATA_DIR / "context_daily.parquet")
ctx["date"] = pd.to_datetime(ctx["date"])
arkg = ctx[ctx["ticker"] == "ARKG"].sort_values("date").reset_index(drop=True)

rxrx = add_features(rxrx)

print(f"\n  RXRX data: {rxrx['date'].min().date()} to {rxrx['date'].max().date()} ({len(rxrx)} bars)")
print(f"  ARKG data: {arkg['date'].min().date()} to {arkg['date'].max().date()} ({len(arkg)} bars)")
print(f"  Latest RXRX close: ${rxrx['close'].iloc[-1]:.2f}")
print(f"  Latest ARKG close: ${arkg['close'].iloc[-1]:.2f}")

# ── Build spread ─────────────────────────────────────────────────
spread = build_spread(rxrx, arkg, "ARKG", LOOKBACK)

latest = spread.iloc[-1]
prev = spread.iloc[-2] if len(spread) > 1 else latest

z_now = latest["spread_z"]
z_prev = prev["spread_z"]
spread_now = latest["spread"]
beta = latest["hedge_beta"]
r2 = latest["r_squared"]
atr = latest["atr20"]
price = latest["close"]

print(f"\n  ── Current Spread Status ──")
print(f"  Date:         {latest['date'].date()}")
print(f"  Z-score:      {z_now:+.4f}  (prev: {z_prev:+.4f}, delta: {z_now - z_prev:+.4f})")
print(f"  Spread:       {spread_now:+.6f}")
print(f"  Hedge beta:   {beta:.4f}")
print(f"  R-squared:    {r2:.4f}")
print(f"  ATR(20):      ${atr:.4f}")
print(f"  Price:        ${price:.2f}")

# ── Signal evaluation ────────────────────────────────────────────
print(f"\n  ── Signal Status ──")

if pd.isna(z_now):
    signal = "FLAT"
    action = "Z-score is NaN -- insufficient data"
elif z_now > K_ENTRY:
    signal = "SHORT"
    stop = price + ATR_STOP_MULT * atr
    action = f"ENTER SHORT at tomorrow's open. Stop: ${stop:.2f} (+{ATR_STOP_MULT * atr:.2f}). Exit when z < +{Z_EXIT}"
elif z_now < -K_ENTRY:
    signal = "LONG"
    stop = price - ATR_STOP_MULT * atr
    action = f"ENTER LONG at tomorrow's open. Stop: ${stop:.2f} (-{ATR_STOP_MULT * atr:.2f}). Exit when z > -{Z_EXIT}"
elif abs(z_now) < Z_EXIT:
    signal = "FLAT (in exit zone)"
    action = "Z-score inside exit band -- if in a trade, EXIT at tomorrow's open"
else:
    signal = "FLAT (between zones)"
    action = f"No signal. Z is between entry ({K_ENTRY}) and exit ({Z_EXIT}) thresholds"

signal_indicator = {"SHORT": ">>>  SHORT  <<<", "LONG": ">>>  LONG  <<<"}.get(signal, f"---  {signal}  ---")

print(f"  Signal:  {signal_indicator}")
print(f"  Action:  {action}")

# ── Z-score history (last 10 days) ───────────────────────────────
print(f"\n  ── Z-Score History (last 10 bars) ──")
print(f"  {'Date':>12s}  {'Z':>8s}  {'Spread':>10s}  {'Close':>8s}  {'Signal':>8s}")
for i in range(max(len(spread) - 10, 0), len(spread)):
    row = spread.iloc[i]
    z = row["spread_z"]
    sig = "SHORT" if z > K_ENTRY else ("LONG" if z < -K_ENTRY else "")
    print(f"  {str(row['date'].date()):>12s}  {z:>+8.4f}  {row['spread']:>+10.6f}  ${row['close']:>7.2f}  {sig:>8s}")

# ── Correlation health check ─────────────────────────────────────
print(f"\n  ── Health Checks ──")

# 60-day rolling correlation
recent_rxrx = rxrx.set_index("date")["close"].iloc[-LOOKBACK:]
recent_arkg = arkg.set_index("date")["close"].reindex(recent_rxrx.index, method="ffill")
if len(recent_rxrx) > 10 and len(recent_arkg.dropna()) > 10:
    corr_60d = recent_rxrx.corr(recent_arkg)
    corr_status = "OK" if corr_60d >= CORR_KILL_THRESHOLD else "WARNING -- BELOW THRESHOLD"
    print(f"  RXRX-ARKG 60d correlation: {corr_60d:.3f}  [{corr_status}]")
    if corr_60d < CORR_KILL_THRESHOLD:
        print(f"  *** KILL-SWITCH WARNING: Correlation {corr_60d:.3f} < {CORR_KILL_THRESHOLD} ***")
        print(f"  *** The hedge relationship may have broken down. Review before trading. ***")

# ── Kill-switch: check recent trade performance ──────────────────
# This reads from a trade log if one exists; otherwise skips
TRADE_LOG_PATH = Path(__file__).parent / "results" / "live_trades.csv" if "__file__" in dir() \
    else Path("agent10/results/live_trades.csv")

if TRADE_LOG_PATH.exists():
    try:
        live_trades = pd.read_csv(TRADE_LOG_PATH)
        if len(live_trades) >= KILL_SWITCH_WINDOW:
            recent_avg = live_trades["pnl_net_pct"].tail(KILL_SWITCH_WINDOW).mean()
            if recent_avg < KILL_SWITCH_THRESHOLD:
                print(f"\n  *** KILL-SWITCH WARNING ***")
                print(f"  *** Last {KILL_SWITCH_WINDOW} trades avg PnL: {recent_avg:+.2f}% (threshold: {KILL_SWITCH_THRESHOLD}%) ***")
                print(f"  *** PAUSE TRADING and review before next entry ***")
            else:
                print(f"  Last {KILL_SWITCH_WINDOW} trades avg PnL: {recent_avg:+.2f}%  [OK]")
        else:
            print(f"  Live trade log has {len(live_trades)} trades (need {KILL_SWITCH_WINDOW} for kill-switch check)")
    except Exception as e:
        print(f"  Could not read trade log: {e}")
else:
    print(f"  No live trade log found at {TRADE_LOG_PATH}")
    print(f"  (Create this file to enable kill-switch monitoring)")

# ── Consecutive stop check ───────────────────────────────────────
# Check backtest trades for recent stop pattern
from agent0_engine import run_backtest, s5_zexit_exit

def _sig_fn(row, i, df):
    return s5_zexit_signal(row, i, df, k_entry=K_ENTRY)

def _exit_fn(sig, row, j, df):
    return s5_zexit_exit(sig, row, j, df, z_exit_thresh=Z_EXIT)

bt_trades = run_backtest(spread, _sig_fn, _exit_fn, max_hold=MAX_HOLD)
if not bt_trades.empty:
    recent_bt = bt_trades.tail(5)
    consec_stops = 0
    for _, t in recent_bt.iterrows():
        if t["exit_reason"] == "stop":
            consec_stops += 1
        else:
            consec_stops = 0
    if consec_stops >= 3:
        print(f"\n  *** WARNING: {consec_stops} consecutive stopped-out trades ***")
        print(f"  *** Consider pausing until signal quality improves ***")
    else:
        last_5_avg = recent_bt["pnl_net_pct"].mean()
        print(f"  Last 5 backtest trades avg PnL: {last_5_avg:+.2f}%  "
              f"[{'OK' if last_5_avg > 0 else 'CAUTION'}]")

# ── Summary line ─────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"  SUMMARY: z={z_now:+.3f}  signal={signal}  price=${price:.2f}")
print(f"  Config: ARKG hedge, k={K_ENTRY}, z_exit={Z_EXIT}, max_hold={MAX_HOLD}d")
print(f"{'=' * 60}")
