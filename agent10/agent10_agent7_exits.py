# %% Agent 7 — Exit Engineering for S5_zexit
#
# Tests whether the EXIT mechanism of S5_zexit can be improved.
# Covers: MFE/MAE analysis, trailing stops, partial exits,
#         z-exit threshold sweep, cooldown period, max hold sweep.
#
# Imports canonical engine from agent0_engine.py.
# Notebook-friendly: no if __name__, config vars at top.

import sys, os
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────
AGENT10_DIR = Path("/home/admin_yaninaso_altostrat_com/bot-RAG/agent10")
DATA_DIR = AGENT10_DIR / "data"
OUT_DIR = AGENT10_DIR / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEL_END = "2026-01-01"        # OOS cutoff
K_ENTRY = 1.5                 # z-score entry threshold
Z_EXIT  = 0.5                 # z-score exit threshold (baseline)
MAX_HOLD = 10                 # max holding days (baseline)
LOOKBACK = 60                 # OLS + rolling z lookback
POSITION_SIZE = 10000         # $ per trade

# ensure engine is importable
if str(AGENT10_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT10_DIR))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agent0_engine import (
    build_spread, run_backtest, compute_stats, compute_costs,
    log_trial, add_features, s5_zexit_signal, s5_zexit_exit,
    daily_mtm_equity, compute_sharpe, walk_forward,
)

# ══════════════════════════════════════════════════════════════════
# 1. LOAD DATA + BUILD SPREAD
# ══════════════════════════════════════════════════════════════════

print("=" * 72)
print("  AGENT 7: EXIT ENGINEERING")
print("=" * 72)

rxrx = pd.read_parquet(DATA_DIR / "rxrx_daily.parquet")
rxrx["date"] = pd.to_datetime(rxrx["date"])
rxrx = rxrx.sort_values("date").reset_index(drop=True)

ctx = pd.read_parquet(DATA_DIR / "context_daily.parquet")
ctx["date"] = pd.to_datetime(ctx["date"])
arkg = ctx[ctx["ticker"] == "ARKG"].sort_values("date").reset_index(drop=True)

rxrx = add_features(rxrx)
spread = build_spread(rxrx, arkg, "ARKG", LOOKBACK)

print(f"  RXRX: {len(rxrx)} bars, {rxrx['date'].min().date()} to {rxrx['date'].max().date()}")
print(f"  Spread: {len(spread)} bars")


# ══════════════════════════════════════════════════════════════════
# 2. BASELINE BACKTEST
# ══════════════════════════════════════════════════════════════════

def baseline_signal(row, i, df):
    return s5_zexit_signal(row, i, df, k_entry=K_ENTRY)

def baseline_exit(sig, row, j, df):
    return s5_zexit_exit(sig, row, j, df, z_exit_thresh=Z_EXIT)

baseline_trades = run_backtest(spread, baseline_signal, baseline_exit,
                               max_hold=MAX_HOLD, position_size=POSITION_SIZE)

sel_baseline = baseline_trades[baseline_trades["entry_date"] < SEL_END].copy()
oos_baseline = baseline_trades[baseline_trades["entry_date"] >= SEL_END].copy()

sel_stats = compute_stats(sel_baseline, label="Baseline SEL")
oos_stats = compute_stats(oos_baseline, label="Baseline OOS")

print(f"\n  BASELINE S5_zexit:")
print(f"    Selection: n={sel_stats['n']}, avg={sel_stats['avg_pnl']:+.3f}%, "
      f"total={sel_stats['total_pnl_pct']:+.1f}%")
print(f"    2026 OOS:  n={oos_stats['n']}, avg={oos_stats['avg_pnl']:+.3f}%, "
      f"total={oos_stats['total_pnl_pct']:+.1f}%")


# ══════════════════════════════════════════════════════════════════
# 3. MFE / MAE ANALYSIS
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  MFE / MAE ANALYSIS")
print("=" * 72)


def compute_mfe_mae(trades_df, spread_df):
    """Compute MFE, MAE, and capture ratio for each trade."""
    results = []
    for _, t in trades_df.iterrows():
        entry_date = t["entry_date"]
        exit_date = t["exit_date"]
        entry_price = t["entry_price"]
        side = t["side"]

        # Get bars during the trade (from entry bar to exit bar inclusive)
        mask = (spread_df["date"] >= entry_date) & (spread_df["date"] <= exit_date)
        trade_bars = spread_df[mask]

        if trade_bars.empty:
            results.append({"mfe_pct": 0, "mae_pct": 0, "capture_ratio": 0})
            continue

        if side == "long":
            # MFE: max high vs entry
            max_high = trade_bars["high"].max()
            mfe_pct = (max_high - entry_price) / entry_price * 100
            # MAE: entry vs min low
            min_low = trade_bars["low"].min()
            mae_pct = (entry_price - min_low) / entry_price * 100
        else:  # short
            # MFE: entry vs min low
            min_low = trade_bars["low"].min()
            mfe_pct = (entry_price - min_low) / entry_price * 100
            # MAE: max high vs entry
            max_high = trade_bars["high"].max()
            mae_pct = (max_high - entry_price) / entry_price * 100

        actual_pnl = t["pnl_gross_pct"]
        capture_ratio = actual_pnl / mfe_pct if mfe_pct > 0 else 0

        results.append({
            "mfe_pct": round(mfe_pct, 3),
            "mae_pct": round(mae_pct, 3),
            "capture_ratio": round(capture_ratio, 3),
            "pnl_gross_pct": round(actual_pnl, 3),
            "pnl_net_pct": round(t["pnl_net_pct"], 3),
            "side": side,
            "exit_reason": t["exit_reason"],
            "entry_date": entry_date,
            "holding_days": t["holding_days"],
        })
    return pd.DataFrame(results)


mfe_mae_all = compute_mfe_mae(baseline_trades, spread)
mfe_mae_sel = mfe_mae_all[mfe_mae_all["entry_date"] < SEL_END]
mfe_mae_oos = mfe_mae_all[mfe_mae_all["entry_date"] >= SEL_END]

print("\n  Selection Window MFE/MAE Summary:")
print(f"    Avg MFE: {mfe_mae_sel['mfe_pct'].mean():.2f}%")
print(f"    Avg MAE: {mfe_mae_sel['mae_pct'].mean():.2f}%")
print(f"    Avg Capture Ratio: {mfe_mae_sel['capture_ratio'].mean():.3f}")
print(f"    Median Capture Ratio: {mfe_mae_sel['capture_ratio'].median():.3f}")

print("\n  2026 OOS MFE/MAE Summary:")
print(f"    Avg MFE: {mfe_mae_oos['mfe_pct'].mean():.2f}%")
print(f"    Avg MAE: {mfe_mae_oos['mae_pct'].mean():.2f}%")
print(f"    Avg Capture Ratio: {mfe_mae_oos['capture_ratio'].mean():.3f}")
print(f"    Median Capture Ratio: {mfe_mae_oos['capture_ratio'].median():.3f}")

# MAE vs outcome correlation
from scipy import stats as scipy_stats
corr_mae_pnl, p_mae = scipy_stats.pearsonr(mfe_mae_all["mae_pct"], mfe_mae_all["pnl_net_pct"])
corr_mfe_pnl, p_mfe = scipy_stats.pearsonr(mfe_mae_all["mfe_pct"], mfe_mae_all["pnl_net_pct"])
print(f"\n  Correlations (all trades):")
print(f"    MAE vs Net PnL: r={corr_mae_pnl:.3f} (p={p_mae:.4f})")
print(f"    MFE vs Net PnL: r={corr_mfe_pnl:.3f} (p={p_mfe:.4f})")

# By exit reason
print("\n  Capture Ratio by Exit Reason:")
for reason in mfe_mae_all["exit_reason"].unique():
    sub = mfe_mae_all[mfe_mae_all["exit_reason"] == reason]
    print(f"    {reason:>10s}: n={len(sub):>3d}, avg_capture={sub['capture_ratio'].mean():.3f}, "
          f"avg_MFE={sub['mfe_pct'].mean():.2f}%, avg_MAE={sub['mae_pct'].mean():.2f}%")

# Per-trade detail
print("\n  Per-Trade MFE/MAE (all trades):")
print(f"  {'Entry':>12s} {'Side':>6s} {'MFE%':>7s} {'MAE%':>7s} {'Capt':>6s} "
      f"{'Net%':>8s} {'Exit':>10s}")
for _, r in mfe_mae_all.iterrows():
    print(f"  {str(r['entry_date'].date()):>12s} {r['side']:>6s} "
          f"{r['mfe_pct']:>+7.2f} {r['mae_pct']:>7.2f} {r['capture_ratio']:>6.3f} "
          f"{r['pnl_net_pct']:>+7.2f}% {r['exit_reason']:>10s}")


# ══════════════════════════════════════════════════════════════════
# 4. CUSTOM BACKTEST ENGINE FOR EXIT VARIANTS
# ══════════════════════════════════════════════════════════════════
# The standard run_backtest has a fixed stop-loss check. For trailing
# stops and other variants, we need a custom loop.

def run_exit_variant(spread_df, signal_fn, exit_variant="baseline",
                     max_hold=10, position_size=10000,
                     z_exit_thresh=0.5, trail_atr=None, ratchet=False,
                     cooldown_days=0):
    """Flexible backtest engine supporting multiple exit variants.

    exit_variant options:
      "baseline"  - fixed 1.5 ATR stop + z-exit
      "trail_T1"  - trailing 1.0 ATR from best price
      "trail_T2"  - trailing 1.5 ATR from best price
      "trail_T3"  - trailing 2.0 ATR from best price
      "trail_T4"  - ratcheting: start 1.5 ATR, tighten to 1.0 after MFE > 1 ATR
    """
    trades = []
    in_trade_until = pd.Timestamp("1900-01-01")
    last_exit_reason = None
    cooldown_until = pd.Timestamp("1900-01-01")

    for i in range(120, len(spread_df)):
        sig = signal_fn(spread_df.iloc[i], i, spread_df)
        if sig is None:
            continue

        entry_idx = i + 1
        if entry_idx >= len(spread_df):
            continue

        entry_date = spread_df["date"].iloc[entry_idx]
        if entry_date <= in_trade_until:
            continue

        # Cooldown: skip if within cooldown window after a stop
        if cooldown_days > 0 and entry_date <= cooldown_until:
            continue

        entry_price = spread_df["open"].iloc[entry_idx]
        side = sig["side"]
        atr = spread_df.iloc[i].get("atr20", np.nan)
        if pd.isna(atr) or atr <= 0:
            continue

        # Initial stop: anchored to signal bar's close (matches s5_zexit_signal)
        signal_close = spread_df.iloc[i]["close"]
        if side == "short":
            initial_stop = signal_close + 1.5 * atr
        else:
            initial_stop = signal_close - 1.5 * atr

        exit_price = exit_date = exit_reason = None

        # Tracking variables for trailing stop
        best_price = entry_price  # best favorable price seen
        current_stop = initial_stop
        mfe_so_far = 0  # in price units

        for j in range(entry_idx + 1, min(entry_idx + max_hold + 1, len(spread_df))):
            row = spread_df.iloc[j]
            z = row.get("spread_z", np.nan)

            # Update best price and trailing stop BEFORE checking stop hit
            if exit_variant.startswith("trail") or exit_variant == "ratchet":
                if side == "long":
                    if row["high"] > best_price:
                        best_price = row["high"]
                    mfe_so_far = best_price - entry_price
                else:
                    if row["low"] < best_price:
                        best_price = row["low"]
                    mfe_so_far = entry_price - best_price

                # Compute trail width
                if exit_variant == "trail_T1":
                    trail_width = 1.0 * atr
                elif exit_variant == "trail_T2":
                    trail_width = 1.5 * atr
                elif exit_variant == "trail_T3":
                    trail_width = 2.0 * atr
                elif exit_variant == "trail_T4" or exit_variant == "ratchet":
                    # Ratchet: start at 1.5 ATR, tighten to 1.0 after MFE > 1 ATR
                    if mfe_so_far > 1.0 * atr:
                        trail_width = 1.0 * atr
                    else:
                        trail_width = 1.5 * atr
                else:
                    trail_width = 1.5 * atr

                # Update trailing stop
                if side == "long":
                    new_stop = best_price - trail_width
                    current_stop = max(current_stop, new_stop)
                else:
                    new_stop = best_price + trail_width
                    current_stop = min(current_stop, new_stop)

            # Check z-exit
            if not pd.isna(z):
                if side == "long" and z >= -z_exit_thresh:
                    exit_price = row["close"]
                    exit_reason = "z_cross"
                    exit_date = row["date"]
                    break
                if side == "short" and z <= z_exit_thresh:
                    exit_price = row["close"]
                    exit_reason = "z_cross"
                    exit_date = row["date"]
                    break

            # Check stop loss (use current_stop which may be trailing)
            if side == "short" and row["high"] >= current_stop:
                exit_price = current_stop
                exit_reason = "stop"
                exit_date = row["date"]
                break
            if side == "long" and row["low"] <= current_stop:
                exit_price = current_stop
                exit_reason = "stop"
                exit_date = row["date"]
                break

        # Time stop if no exit triggered
        if exit_price is None:
            exit_idx = min(entry_idx + max_hold, len(spread_df) - 1)
            exit_price = spread_df["close"].iloc[exit_idx]
            exit_date = spread_df["date"].iloc[exit_idx]
            exit_reason = "time_stop"

        in_trade_until = exit_date
        last_exit_reason = exit_reason

        # Set cooldown after stops
        if cooldown_days > 0 and exit_reason == "stop":
            cooldown_until = exit_date + pd.Timedelta(days=cooldown_days)

        holding_days = max((exit_date - entry_date).days, 1)
        pnl_pct = ((exit_price - entry_price) / entry_price * 100) if side == "long" \
                  else ((entry_price - exit_price) / entry_price * 100)
        cost_pct = compute_costs(entry_price, side, holding_days) / entry_price * 100
        pnl_net = pnl_pct - cost_pct
        shares = max(int(position_size / entry_price), 1)

        trades.append({
            "signal_date": spread_df["date"].iloc[i],
            "entry_date": entry_date, "exit_date": exit_date,
            "side": side, "entry_price": round(entry_price, 4),
            "exit_price": round(exit_price, 4), "exit_reason": exit_reason,
            "holding_days": holding_days,
            "z_at_signal": round(spread_df["spread_z"].iloc[i], 4),
            "pnl_gross_pct": round(pnl_pct, 4), "cost_pct": round(cost_pct, 4),
            "pnl_net_pct": round(pnl_net, 4),
            "shares": shares, "pnl_dollars": round(pnl_net / 100 * shares * entry_price, 2),
        })

    return pd.DataFrame(trades)


def run_partial_exit(spread_df, signal_fn, partial_variant="P1",
                     max_hold=10, position_size=10000):
    """Partial exit variants. Computes blended PnL.

    P1: 50% at z-cross (0.5), 50% with trailing stop (1.5 ATR) until time/full stop
    P2: 50% at z=0, 50% at z-cross (0.5)
    """
    trades = []
    in_trade_until = pd.Timestamp("1900-01-01")

    for i in range(120, len(spread_df)):
        sig = signal_fn(spread_df.iloc[i], i, spread_df)
        if sig is None:
            continue

        entry_idx = i + 1
        if entry_idx >= len(spread_df):
            continue

        entry_date = spread_df["date"].iloc[entry_idx]
        if entry_date <= in_trade_until:
            continue

        entry_price = spread_df["open"].iloc[entry_idx]
        side = sig["side"]
        atr = spread_df.iloc[i].get("atr20", np.nan)
        if pd.isna(atr) or atr <= 0:
            continue

        # Initial stop: anchored to signal bar's close (matches s5_zexit_signal)
        signal_close = spread_df.iloc[i]["close"]
        if side == "short":
            initial_stop = signal_close + 1.5 * atr
        else:
            initial_stop = signal_close - 1.5 * atr

        # Set z thresholds for partial exits
        if partial_variant == "P1":
            first_z_thresh = 0.5    # exit first 50% at z-cross
            second_z_thresh = None  # hold with trailing stop
        elif partial_variant == "P2":
            first_z_thresh = 0.0    # exit first 50% at z=0
            second_z_thresh = 0.5   # exit remainder at z-cross 0.5

        first_exit_price = None
        first_exit_date = None
        first_exit_reason = None
        second_exit_price = None
        second_exit_date = None
        second_exit_reason = None

        best_price = entry_price
        current_stop = initial_stop
        first_exited = False

        for j in range(entry_idx + 1, min(entry_idx + max_hold + 1, len(spread_df))):
            row = spread_df.iloc[j]
            z = row.get("spread_z", np.nan)

            # Update trailing stop for second leg (P1)
            if first_exited and partial_variant == "P1":
                if side == "long":
                    if row["high"] > best_price:
                        best_price = row["high"]
                    new_stop = best_price - 1.5 * atr
                    current_stop = max(current_stop, new_stop)
                else:
                    if row["low"] < best_price:
                        best_price = row["low"]
                    new_stop = best_price + 1.5 * atr
                    current_stop = min(current_stop, new_stop)

            # Check stop loss (affects both legs if first not yet exited,
            # or second leg if first already exited)
            stop_hit = False
            if side == "short" and row["high"] >= current_stop:
                stop_hit = True
            if side == "long" and row["low"] <= current_stop:
                stop_hit = True

            if stop_hit:
                if not first_exited:
                    first_exit_price = current_stop
                    first_exit_date = row["date"]
                    first_exit_reason = "stop"
                second_exit_price = current_stop
                second_exit_date = row["date"]
                second_exit_reason = "stop"
                break

            if pd.isna(z):
                continue

            # First exit check
            if not first_exited:
                trigger_first = False
                if side == "long" and z >= -first_z_thresh:
                    trigger_first = True
                if side == "short" and z <= first_z_thresh:
                    trigger_first = True

                if trigger_first:
                    first_exit_price = row["close"]
                    first_exit_date = row["date"]
                    first_exit_reason = "z_cross_partial"
                    first_exited = True
                    # For P2, continue looking for second z-cross
                    # For P1, continue with trailing stop for second leg
                    if partial_variant == "P1":
                        best_price = row["close"] if side == "long" else row["close"]
                    continue

            # Second exit check (only after first exited)
            if first_exited:
                if partial_variant == "P2" and second_z_thresh is not None:
                    trigger_second = False
                    if side == "long" and z >= -second_z_thresh:
                        trigger_second = True
                    if side == "short" and z <= second_z_thresh:
                        trigger_second = True
                    if trigger_second:
                        second_exit_price = row["close"]
                        second_exit_date = row["date"]
                        second_exit_reason = "z_cross"
                        break

        # Handle time stop for any unfilled exits
        exit_idx = min(entry_idx + max_hold, len(spread_df) - 1)
        time_stop_price = spread_df["close"].iloc[exit_idx]
        time_stop_date = spread_df["date"].iloc[exit_idx]

        if first_exit_price is None:
            first_exit_price = time_stop_price
            first_exit_date = time_stop_date
            first_exit_reason = "time_stop"
        if second_exit_price is None:
            second_exit_price = time_stop_price
            second_exit_date = time_stop_date
            second_exit_reason = "time_stop"

        in_trade_until = max(first_exit_date, second_exit_date)

        # Compute blended PnL
        if side == "long":
            pnl1_pct = (first_exit_price - entry_price) / entry_price * 100
            pnl2_pct = (second_exit_price - entry_price) / entry_price * 100
        else:
            pnl1_pct = (entry_price - first_exit_price) / entry_price * 100
            pnl2_pct = (entry_price - second_exit_price) / entry_price * 100

        blended_pnl = 0.5 * pnl1_pct + 0.5 * pnl2_pct
        overall_exit_date = max(first_exit_date, second_exit_date)
        holding_days = max((overall_exit_date - entry_date).days, 1)
        cost_pct = compute_costs(entry_price, side, holding_days) / entry_price * 100
        pnl_net = blended_pnl - cost_pct
        shares = max(int(position_size / entry_price), 1)

        trades.append({
            "signal_date": spread_df["date"].iloc[i],
            "entry_date": entry_date,
            "exit_date": overall_exit_date,
            "side": side, "entry_price": round(entry_price, 4),
            "exit_price": round((first_exit_price + second_exit_price) / 2, 4),
            "exit_reason": f"{first_exit_reason}+{second_exit_reason}",
            "holding_days": holding_days,
            "z_at_signal": round(spread_df["spread_z"].iloc[i], 4),
            "pnl_gross_pct": round(blended_pnl, 4),
            "cost_pct": round(cost_pct, 4),
            "pnl_net_pct": round(pnl_net, 4),
            "shares": shares,
            "pnl_dollars": round(pnl_net / 100 * shares * entry_price, 2),
        })

    return pd.DataFrame(trades)


# Helper to split and report stats
def split_stats(trades_df, label):
    """Split trades into selection/OOS and compute stats."""
    if trades_df.empty:
        return (compute_stats(trades_df, label=f"{label} SEL"),
                compute_stats(trades_df, label=f"{label} OOS"))
    sel = trades_df[trades_df["entry_date"] < SEL_END]
    oos = trades_df[trades_df["entry_date"] >= SEL_END]
    return (compute_stats(sel, label=f"{label} SEL"),
            compute_stats(oos, label=f"{label} OOS"))


# ══════════════════════════════════════════════════════════════════
# 5. TRAILING STOP VARIANTS
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  TRAILING STOP VARIANTS")
print("=" * 72)

trail_variants = {
    "Baseline": {"exit_variant": "baseline"},
    "T1 Trail 1.0 ATR": {"exit_variant": "trail_T1"},
    "T2 Trail 1.5 ATR": {"exit_variant": "trail_T2"},
    "T3 Trail 2.0 ATR": {"exit_variant": "trail_T3"},
    "T4 Ratchet": {"exit_variant": "trail_T4"},
}

trail_results = {}
for name, params in trail_variants.items():
    trades = run_exit_variant(spread, baseline_signal,
                              exit_variant=params["exit_variant"],
                              max_hold=MAX_HOLD, z_exit_thresh=Z_EXIT)
    sel_s, oos_s = split_stats(trades, name)
    trail_results[name] = {"sel": sel_s, "oos": oos_s, "trades": trades}

print(f"\n  {'Variant':<20s} {'SEL n':>6s} {'SEL avg%':>9s} {'SEL tot%':>9s} "
      f"{'OOS n':>6s} {'OOS avg%':>9s} {'OOS tot%':>9s} {'OOS WR':>7s}")
print("  " + "-" * 80)
for name, r in trail_results.items():
    s, o = r["sel"], r["oos"]
    print(f"  {name:<20s} {s['n']:>6d} {s['avg_pnl']:>+9.3f} {s['total_pnl_pct']:>+9.1f} "
          f"{o['n']:>6d} {o['avg_pnl']:>+9.3f} {o['total_pnl_pct']:>+9.1f} {o['wr']:>6.1f}%")

# Log trailing stop trials
for name, r in trail_results.items():
    log_trial("agent7", f"trail_{name}", {"variant": name}, r["oos"])


# ══════════════════════════════════════════════════════════════════
# 6. PARTIAL EXIT VARIANTS
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  PARTIAL EXIT VARIANTS")
print("=" * 72)

partial_results = {}
for pvar in ["P1", "P2"]:
    trades = run_partial_exit(spread, baseline_signal, partial_variant=pvar,
                              max_hold=MAX_HOLD)
    sel_s, oos_s = split_stats(trades, pvar)
    partial_results[pvar] = {"sel": sel_s, "oos": oos_s, "trades": trades}

print(f"\n  {'Variant':<25s} {'SEL n':>6s} {'SEL avg%':>9s} {'SEL tot%':>9s} "
      f"{'OOS n':>6s} {'OOS avg%':>9s} {'OOS tot%':>9s}")
print("  " + "-" * 80)

# Print baseline for reference
print(f"  {'Baseline (z=0.5)':<25s} {sel_stats['n']:>6d} {sel_stats['avg_pnl']:>+9.3f} "
      f"{sel_stats['total_pnl_pct']:>+9.1f} {oos_stats['n']:>6d} "
      f"{oos_stats['avg_pnl']:>+9.3f} {oos_stats['total_pnl_pct']:>+9.1f}")
for pvar, r in partial_results.items():
    s, o = r["sel"], r["oos"]
    desc = "P1: 50% z-cross + trail" if pvar == "P1" else "P2: 50% z=0 + z=0.5"
    print(f"  {desc:<25s} {s['n']:>6d} {s['avg_pnl']:>+9.3f} {s['total_pnl_pct']:>+9.1f} "
          f"{o['n']:>6d} {o['avg_pnl']:>+9.3f} {o['total_pnl_pct']:>+9.1f}")

for pvar, r in partial_results.items():
    log_trial("agent7", f"partial_{pvar}", {"variant": pvar}, r["oos"])


# ══════════════════════════════════════════════════════════════════
# 7. Z-EXIT THRESHOLD SWEEP
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  Z-EXIT THRESHOLD SWEEP")
print("=" * 72)

z_thresholds = [0.0, 0.25, 0.5, 0.75, 1.0]
z_results = {}

for zt in z_thresholds:
    trades = run_exit_variant(spread, baseline_signal,
                              exit_variant="baseline",
                              max_hold=MAX_HOLD, z_exit_thresh=zt)
    sel_s, oos_s = split_stats(trades, f"z_exit={zt}")
    z_results[zt] = {"sel": sel_s, "oos": oos_s, "trades": trades}

# Asymmetric: overshoot exit — exit when z reverses past -0.5 for shorts
# (i.e., z must go from >1.5 to <-0.5 to exit)
def overshoot_exit_fn(sig, row, j, df):
    """Overshoot: short exits only when z goes below -0.5 (past zero)."""
    z = row.get("spread_z", np.nan)
    if pd.isna(z):
        return None
    if sig["side"] == "long" and z >= 0.5:  # long exits when z goes above +0.5
        return ("z_overshoot", row["close"])
    if sig["side"] == "short" and z <= -0.5:  # short exits when z goes below -0.5
        return ("z_overshoot", row["close"])
    return None

overshoot_trades = run_backtest(spread, baseline_signal, overshoot_exit_fn,
                                max_hold=MAX_HOLD, position_size=POSITION_SIZE)
overshoot_sel_s, overshoot_oos_s = split_stats(overshoot_trades, "Overshoot")
z_results["overshoot"] = {"sel": overshoot_sel_s, "oos": overshoot_oos_s,
                           "trades": overshoot_trades}

print(f"\n  {'Z-exit thresh':<16s} {'SEL n':>6s} {'SEL avg%':>9s} {'SEL tot%':>9s} "
      f"{'SEL hold':>8s} {'OOS n':>6s} {'OOS avg%':>9s} {'OOS tot%':>9s} {'OOS hold':>8s}")
print("  " + "-" * 90)
for zt in z_thresholds:
    r = z_results[zt]
    s, o = r["sel"], r["oos"]
    marker = " <-- baseline" if zt == 0.5 else ""
    print(f"  {zt:<16.2f} {s['n']:>6d} {s['avg_pnl']:>+9.3f} {s['total_pnl_pct']:>+9.1f} "
          f"{s['avg_hold']:>8.1f} {o['n']:>6d} {o['avg_pnl']:>+9.3f} {o['total_pnl_pct']:>+9.1f} "
          f"{o['avg_hold']:>8.1f}{marker}")

r = z_results["overshoot"]
s, o = r["sel"], r["oos"]
print(f"  {'overshoot':<16s} {s['n']:>6d} {s['avg_pnl']:>+9.3f} {s['total_pnl_pct']:>+9.1f} "
      f"{s['avg_hold']:>8.1f} {o['n']:>6d} {o['avg_pnl']:>+9.3f} {o['total_pnl_pct']:>+9.1f} "
      f"{o['avg_hold']:>8.1f}")

for zt in z_thresholds:
    log_trial("agent7", f"z_exit_{zt}", {"z_exit_thresh": zt}, z_results[zt]["oos"])
log_trial("agent7", "z_exit_overshoot", {"z_exit_thresh": "overshoot"}, z_results["overshoot"]["oos"])


# ══════════════════════════════════════════════════════════════════
# 8. COOLDOWN PERIOD
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  COOLDOWN PERIOD AFTER STOPS")
print("=" * 72)

cooldown_values = [0, 3, 5, 10]
cooldown_results = {}

for cd in cooldown_values:
    trades = run_exit_variant(spread, baseline_signal,
                              exit_variant="baseline",
                              max_hold=MAX_HOLD, z_exit_thresh=Z_EXIT,
                              cooldown_days=cd)
    sel_s, oos_s = split_stats(trades, f"cooldown={cd}d")
    cooldown_results[cd] = {"sel": sel_s, "oos": oos_s, "trades": trades}

print(f"\n  {'Cooldown':>10s} {'SEL n':>6s} {'SEL avg%':>9s} {'SEL tot%':>9s} "
      f"{'OOS n':>6s} {'OOS avg%':>9s} {'OOS tot%':>9s} {'OOS WR':>7s}")
print("  " + "-" * 70)
for cd in cooldown_values:
    r = cooldown_results[cd]
    s, o = r["sel"], r["oos"]
    marker = " <-- baseline" if cd == 0 else ""
    print(f"  {cd:>10d}d {s['n']:>6d} {s['avg_pnl']:>+9.3f} {s['total_pnl_pct']:>+9.1f} "
          f"{o['n']:>6d} {o['avg_pnl']:>+9.3f} {o['total_pnl_pct']:>+9.1f} {o['wr']:>6.1f}%{marker}")

for cd in cooldown_values:
    log_trial("agent7", f"cooldown_{cd}d", {"cooldown_days": cd}, cooldown_results[cd]["oos"])


# ══════════════════════════════════════════════════════════════════
# 9. MAX HOLD SWEEP
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  MAX HOLD SWEEP")
print("=" * 72)

max_hold_values = [5, 7, 10, 15, 20]
maxhold_results = {}

for mh in max_hold_values:
    trades = run_exit_variant(spread, baseline_signal,
                              exit_variant="baseline",
                              max_hold=mh, z_exit_thresh=Z_EXIT)
    sel_s, oos_s = split_stats(trades, f"max_hold={mh}")
    maxhold_results[mh] = {"sel": sel_s, "oos": oos_s, "trades": trades}

print(f"\n  {'Max Hold':>10s} {'SEL n':>6s} {'SEL avg%':>9s} {'SEL tot%':>9s} "
      f"{'SEL hold':>8s} {'OOS n':>6s} {'OOS avg%':>9s} {'OOS tot%':>9s} {'OOS hold':>8s}")
print("  " + "-" * 80)
for mh in max_hold_values:
    r = maxhold_results[mh]
    s, o = r["sel"], r["oos"]
    marker = " <-- baseline" if mh == 10 else ""
    print(f"  {mh:>10d}d {s['n']:>6d} {s['avg_pnl']:>+9.3f} {s['total_pnl_pct']:>+9.1f} "
          f"{s['avg_hold']:>8.1f} {o['n']:>6d} {o['avg_pnl']:>+9.3f} {o['total_pnl_pct']:>+9.1f} "
          f"{o['avg_hold']:>8.1f}{marker}")

for mh in max_hold_values:
    log_trial("agent7", f"max_hold_{mh}d", {"max_hold": mh}, maxhold_results[mh]["oos"])


# ══════════════════════════════════════════════════════════════════
# 10. IDENTIFY BEST VARIANT + WALK-FORWARD
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  BEST VARIANT IDENTIFICATION + WALK-FORWARD")
print("=" * 72)

# Collect all variants with their selection-window stats
all_variants = []

# Baseline
all_variants.append(("Baseline", sel_stats, oos_stats, baseline_trades))

# Trailing stops
for name, r in trail_results.items():
    if name != "Baseline":
        all_variants.append((name, r["sel"], r["oos"], r["trades"]))

# Partial exits
for pvar, r in partial_results.items():
    desc = f"P1 partial" if pvar == "P1" else f"P2 partial"
    all_variants.append((desc, r["sel"], r["oos"], r["trades"]))

# Z-exit sweep (excluding baseline which is z=0.5)
for zt in z_thresholds:
    if zt != 0.5:
        r = z_results[zt]
        all_variants.append((f"z_exit={zt}", r["sel"], r["oos"], r["trades"]))
all_variants.append(("z_overshoot", z_results["overshoot"]["sel"],
                     z_results["overshoot"]["oos"], z_results["overshoot"]["trades"]))

# Cooldown (excluding baseline which is cd=0)
for cd in cooldown_values:
    if cd != 0:
        r = cooldown_results[cd]
        all_variants.append((f"cooldown={cd}d", r["sel"], r["oos"], r["trades"]))

# Max hold (excluding baseline which is mh=10)
for mh in max_hold_values:
    if mh != 10:
        r = maxhold_results[mh]
        all_variants.append((f"max_hold={mh}d", r["sel"], r["oos"], r["trades"]))

# Sort by selection-window avg PnL (conservative: pick based on in-sample)
all_variants.sort(key=lambda x: x[1]["avg_pnl"], reverse=True)

print("\n  ALL VARIANTS RANKED BY SELECTION-WINDOW AVG PnL:")
print(f"  {'Rank':>4s}  {'Variant':<25s} {'SEL avg%':>9s} {'SEL n':>6s} "
      f"{'OOS avg%':>9s} {'OOS n':>6s} {'OOS WR':>7s}")
print("  " + "-" * 75)
for rank, (name, sel_s, oos_s, _) in enumerate(all_variants, 1):
    marker = " ***" if name == "Baseline" else ""
    print(f"  {rank:>4d}  {name:<25s} {sel_s['avg_pnl']:>+9.3f} {sel_s['n']:>6d} "
          f"{oos_s['avg_pnl']:>+9.3f} {oos_s['n']:>6d} {oos_s['wr']:>6.1f}%{marker}")

best_name, best_sel, best_oos, best_trades = all_variants[0]
print(f"\n  BEST VARIANT (by selection avg): {best_name}")
print(f"    Selection: n={best_sel['n']}, avg={best_sel['avg_pnl']:+.3f}%")
print(f"    2026 OOS:  n={best_oos['n']}, avg={best_oos['avg_pnl']:+.3f}%")

# Walk-forward test on best variant
print(f"\n  Walk-Forward Test for '{best_name}':")

# Build the appropriate signal/exit functions for the best variant
# (We use the selection-window spread for walk-forward)
if best_name == "Baseline":
    wf_sig = baseline_signal
    wf_exit = baseline_exit
elif best_name.startswith("z_exit="):
    zt_best = float(best_name.split("=")[1])
    def wf_exit(sig, row, j, df, _zt=zt_best):
        return s5_zexit_exit(sig, row, j, df, z_exit_thresh=_zt)
    wf_sig = baseline_signal
elif best_name == "z_overshoot":
    wf_sig = baseline_signal
    wf_exit = overshoot_exit_fn
else:
    # For trailing/cooldown/partial/maxhold, use baseline for WF
    # (walk_forward uses run_backtest which doesn't support these variants)
    wf_sig = baseline_signal
    wf_exit = baseline_exit
    print(f"    NOTE: Walk-forward uses standard run_backtest; "
          f"trailing/cooldown/partial variants not directly testable via walk_forward().")

# Determine max_hold for best variant
wf_max_hold = MAX_HOLD
if best_name.startswith("max_hold="):
    wf_max_hold = int(best_name.split("=")[1].replace("d", ""))

wf_results = walk_forward(spread, wf_sig, wf_exit,
                          train_months=12, test_months=3, max_hold=wf_max_hold)
if not wf_results.empty:
    print(f"    {'Test Start':>12s} {'Test End':>12s} {'n':>4s} {'avg%':>8s} {'total%':>8s}")
    for _, w in wf_results.iterrows():
        print(f"    {str(w['test_start'].date()):>12s} {str(w['test_end'].date()):>12s} "
              f"{w['n']:>4.0f} {w['avg_pnl']:>+7.2f}% {w['total_pnl']:>+7.2f}%")
    wf_avg = wf_results["avg_pnl"].mean()
    wf_positive = (wf_results["avg_pnl"] > 0).sum()
    wf_total = len(wf_results)
    print(f"    WF OOS average: {wf_avg:+.2f}%  "
          f"({wf_positive}/{wf_total} positive folds)  "
          f"{'PASS' if wf_avg > 0 and wf_positive >= wf_total * 0.5 else 'FAIL'}")
else:
    print("    No walk-forward folds generated.")


# Also run walk-forward on baseline for comparison
print(f"\n  Walk-Forward Test for Baseline (comparison):")
wf_baseline = walk_forward(spread, baseline_signal, baseline_exit,
                           train_months=12, test_months=3, max_hold=MAX_HOLD)
if not wf_baseline.empty:
    for _, w in wf_baseline.iterrows():
        print(f"    {str(w['test_start'].date()):>12s} {str(w['test_end'].date()):>12s} "
              f"{w['n']:>4.0f} {w['avg_pnl']:>+7.2f}% {w['total_pnl']:>+7.2f}%")
    wf_bl_avg = wf_baseline["avg_pnl"].mean()
    print(f"    WF OOS average: {wf_bl_avg:+.2f}%")


# ══════════════════════════════════════════════════════════════════
# 11. MATPLOTLIB FIGURE
# ══════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Agent 7 — Exit Engineering Analysis", fontsize=14, fontweight="bold")

# Panel A: MFE vs MAE scatter (color by win/loss)
ax = axes[0, 0]
wins_mask = mfe_mae_all["pnl_net_pct"] > 0
ax.scatter(mfe_mae_all.loc[wins_mask, "mae_pct"],
           mfe_mae_all.loc[wins_mask, "mfe_pct"],
           c="green", alpha=0.7, s=60, label="Win", edgecolors="black", linewidths=0.5)
ax.scatter(mfe_mae_all.loc[~wins_mask, "mae_pct"],
           mfe_mae_all.loc[~wins_mask, "mfe_pct"],
           c="red", alpha=0.7, s=60, label="Loss", edgecolors="black", linewidths=0.5)
ax.plot([0, mfe_mae_all["mae_pct"].max()],
        [0, mfe_mae_all["mae_pct"].max()],
        "k--", alpha=0.3, label="MFE=MAE")
ax.set_xlabel("MAE (max adverse excursion %)")
ax.set_ylabel("MFE (max favorable excursion %)")
ax.set_title("A) MFE vs MAE (color by win/loss)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel B: Capture ratio distribution
ax = axes[0, 1]
ax.hist(mfe_mae_sel["capture_ratio"].clip(-1, 2), bins=25, alpha=0.6,
        color="steelblue", label=f"Selection (n={len(mfe_mae_sel)})", edgecolor="black")
if len(mfe_mae_oos) > 0:
    ax.hist(mfe_mae_oos["capture_ratio"].clip(-1, 2), bins=15, alpha=0.6,
            color="orange", label=f"OOS 2026 (n={len(mfe_mae_oos)})", edgecolor="black")
ax.axvline(mfe_mae_all["capture_ratio"].mean(), color="red", linestyle="--",
           label=f"Mean={mfe_mae_all['capture_ratio'].mean():.2f}")
ax.set_xlabel("Capture Ratio (actual PnL / MFE)")
ax.set_ylabel("Count")
ax.set_title("B) Capture Ratio Distribution")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel C: Z-exit threshold sweep: avg PnL by threshold
ax = axes[1, 0]
z_vals = z_thresholds
sel_avgs = [z_results[zt]["sel"]["avg_pnl"] for zt in z_vals]
oos_avgs = [z_results[zt]["oos"]["avg_pnl"] for zt in z_vals]

x = np.arange(len(z_vals))
width = 0.35
ax.bar(x - width/2, sel_avgs, width, label="Selection", color="steelblue", alpha=0.8)
ax.bar(x + width/2, oos_avgs, width, label="OOS 2026", color="orange", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([str(v) for v in z_vals])
ax.set_xlabel("Z-exit Threshold")
ax.set_ylabel("Avg PnL (%)")
ax.set_title("C) Z-Exit Threshold Sweep")
ax.legend(fontsize=9)
ax.axhline(0, color="black", linewidth=0.5)
ax.grid(True, alpha=0.3, axis="y")

# Panel D: Best exit variant equity curve vs baseline
ax = axes[1, 1]

# Compute cumulative PnL for baseline and best variant
baseline_all_sorted = baseline_trades.sort_values("exit_date")
baseline_cum = baseline_all_sorted["pnl_net_pct"].cumsum()

best_all_sorted = best_trades.sort_values("exit_date")
best_cum = best_all_sorted["pnl_net_pct"].cumsum()

ax.plot(baseline_all_sorted["exit_date"].values, baseline_cum.values,
        "b-", linewidth=1.5, label=f"Baseline (z_exit=0.5)")
if best_name != "Baseline":
    ax.plot(best_all_sorted["exit_date"].values, best_cum.values,
            "r-", linewidth=1.5, label=f"Best: {best_name}")
ax.axvline(pd.Timestamp(SEL_END), color="gray", linestyle="--", alpha=0.6,
           label="OOS cutoff")
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative Net PnL (%)")
ax.set_title(f"D) Equity Curve: Baseline vs Best ({best_name})")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(OUT_DIR / "agent7_exits.png", dpi=150, bbox_inches="tight")
print(f"\n  Figure saved to {OUT_DIR / 'agent7_exits.png'}")
plt.close()


# ══════════════════════════════════════════════════════════════════
# 12. CONCLUSION
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  CONCLUSION")
print("=" * 72)

# Check if any variant meaningfully beats baseline on BOTH sel and OOS
baseline_sel_avg = sel_stats["avg_pnl"]
baseline_oos_avg = oos_stats["avg_pnl"]
meaningful_threshold = 0.5  # must beat by at least 0.5% avg to be meaningful

better_variants = []
for name, sel_s, oos_s, _ in all_variants:
    if name == "Baseline":
        continue
    sel_delta = sel_s["avg_pnl"] - baseline_sel_avg
    oos_delta = oos_s["avg_pnl"] - baseline_oos_avg
    if sel_delta > meaningful_threshold and oos_delta > 0:
        better_variants.append((name, sel_delta, oos_delta, sel_s, oos_s))

if better_variants:
    print(f"\n  Variants that meaningfully beat baseline (>{meaningful_threshold}% SEL + positive OOS):")
    for name, sd, od, sel_s, oos_s in better_variants:
        print(f"    {name}: SEL +{sd:.2f}%, OOS +{od:.2f}%")
        print(f"      SEL: n={sel_s['n']}, avg={sel_s['avg_pnl']:+.3f}%, WR={sel_s['wr']:.1f}%")
        print(f"      OOS: n={oos_s['n']}, avg={oos_s['avg_pnl']:+.3f}%, WR={oos_s['wr']:.1f}%")
else:
    print("\n  NO variant meaningfully beats baseline on both selection AND OOS windows.")

# Also flag if any variant is WORSE
print(f"\n  Baseline reference:")
print(f"    Selection: n={sel_stats['n']}, avg={sel_stats['avg_pnl']:+.3f}%, "
      f"total={sel_stats['total_pnl_pct']:+.1f}%, WR={sel_stats['wr']:.1f}%")
print(f"    2026 OOS:  n={oos_stats['n']}, avg={oos_stats['avg_pnl']:+.3f}%, "
      f"total={oos_stats['total_pnl_pct']:+.1f}%, WR={oos_stats['wr']:.1f}%")

# MFE/MAE summary
print(f"\n  MFE/MAE KEY FINDINGS:")
print(f"    Median capture ratio: {mfe_mae_all['capture_ratio'].median():.3f} "
      f"(we capture {mfe_mae_all['capture_ratio'].median()*100:.1f}% of the available move)")
print(f"    Mean capture ratio is {mfe_mae_all['capture_ratio'].mean():.3f} "
      f"(skewed by stopped trades with large negative values)")
print(f"    MAE vs PnL correlation: r={corr_mae_pnl:.3f} (p={p_mae:.4f})")
print(f"    MFE vs PnL correlation: r={corr_mfe_pnl:.3f} (p={p_mfe:.4f})")
mae_stop = mfe_mae_all[mfe_mae_all["exit_reason"]=="stop"]["mae_pct"]
mae_zcross = mfe_mae_all[mfe_mae_all["exit_reason"]=="z_cross"]["mae_pct"]
print(f"    Stopped trades avg MAE: {mae_stop.mean():.2f}%  vs  z-cross trades avg MAE: {mae_zcross.mean():.2f}%")

# Final verdict
print(f"\n  FINAL VERDICT:")
if not better_variants:
    print("    The baseline z-exit mechanism (z-cross at +/-0.5, 1.5 ATR fixed stop,")
    print("    10-day max hold) is OPTIMAL or near-optimal. No exit variant tested")
    print("    produces a meaningful, consistent improvement across both selection")
    print("    and OOS windows. The z-score carries the edge; exit engineering")
    print("    cannot meaningfully improve upon the simple z-cross exit.")
else:
    best_v = better_variants[0]
    print(f"    BEST EXIT VARIANT: {best_v[0]}")
    print(f"    Improvement: SEL +{best_v[1]:.2f}%, OOS +{best_v[2]:.2f}%")
    print(f"    However, verify walk-forward stability before deploying.")

print("\n  AGENT 7 COMPLETE")
