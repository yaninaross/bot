# %% Agent 1 — Intraday Execution Variants for S5_zexit
#
# Compares MOO vs MOC vs Limit fill variants, plus 30-min intraday
# timing analysis for the S5_zexit RXRX/ARKG spread strategy.
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
Z_EXIT  = 0.5                 # z-score exit threshold
MAX_HOLD = 10                 # max holding days
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
    daily_mtm_equity, compute_sharpe,
)

# ══════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════

print("=" * 72)
print("  AGENT 1: INTRADAY EXECUTION VARIANTS")
print("=" * 72)

rxrx = pd.read_parquet(DATA_DIR / "rxrx_daily.parquet")
rxrx["date"] = pd.to_datetime(rxrx["date"])
rxrx = rxrx.sort_values("date").reset_index(drop=True)
rxrx = add_features(rxrx)

ctx = pd.read_parquet(DATA_DIR / "context_daily.parquet")
ctx["date"] = pd.to_datetime(ctx["date"])
arkg = ctx[ctx["ticker"] == "ARKG"].sort_values("date").reset_index(drop=True)

rxrx_30m = pd.read_parquet(DATA_DIR / "rxrx_30min.parquet")
rxrx_30m["date"] = pd.to_datetime(rxrx_30m["date"])
# Convert to ET for time-of-day analysis
rxrx_30m["date_et"] = rxrx_30m["date"].dt.tz_convert("US/Eastern")
rxrx_30m["time_et"] = rxrx_30m["date_et"].dt.time
rxrx_30m["cal_date"] = rxrx_30m["date_et"].dt.normalize().dt.tz_localize(None)

print(f"  RXRX daily: {len(rxrx)} bars, {rxrx['date'].min().date()} to {rxrx['date'].max().date()}")
print(f"  ARKG daily: {len(arkg)} bars")
print(f"  RXRX 30min: {len(rxrx_30m)} bars, {rxrx_30m['date'].min()} to {rxrx_30m['date'].max()}")

# ── Build canonical spread ──────────────────────────────────────
print("\n  Building canonical spread...")
spread_df = build_spread(rxrx, arkg, "ARKG", LOOKBACK)
print(f"  Spread: {len(spread_df)} rows, z-score range [{spread_df['spread_z'].min():.2f}, {spread_df['spread_z'].max():.2f}]")

# Add rolling std of spread for limit-order threshold calculation
spread_col = spread_df["spread"]
spread_df["spread_rolling_std"] = spread_col.rolling(LOOKBACK).std()
spread_df["spread_rolling_mean"] = spread_col.rolling(LOOKBACK).mean()

# ── Signal / exit functions ─────────────────────────────────────
def sig_fn(row, i, df):
    return s5_zexit_signal(row, i, df, k_entry=K_ENTRY)

def exit_fn(sig, row, j, df):
    return s5_zexit_exit(sig, row, j, df, z_exit_thresh=Z_EXIT)


# ══════════════════════════════════════════════════════════════════
# 2. BASELINE — MOO
# ══════════════════════════════════════════════════════════════════

print("\n" + "-" * 72)
print("  2. BASELINE: MARKET-ON-OPEN (MOO)")
print("-" * 72)

trades_moo = run_backtest(spread_df, sig_fn, exit_fn, max_hold=MAX_HOLD,
                          position_size=POSITION_SIZE, fill_type="market_on_open")

sel_moo = trades_moo[trades_moo["entry_date"] < SEL_END].copy()
oos_moo = trades_moo[trades_moo["entry_date"] >= SEL_END].copy()

stats_sel_moo = compute_stats(sel_moo, spread_df, label="MOO Selection")
stats_oos_moo = compute_stats(oos_moo, spread_df, label="MOO 2026")

print(f"  Selection: n={stats_sel_moo['n']}, avg={stats_sel_moo['avg_pnl']:+.2f}%, "
      f"total={stats_sel_moo['total_pnl_pct']:+.1f}%, Sharpe={stats_sel_moo['sharpe']}")
print(f"  2026 OOS:  n={stats_oos_moo['n']}, avg={stats_oos_moo['avg_pnl']:+.2f}%, "
      f"total={stats_oos_moo['total_pnl_pct']:+.1f}%, Sharpe={stats_oos_moo['sharpe']}")


# ══════════════════════════════════════════════════════════════════
# 3. MARKET-ON-CLOSE (MOC) VARIANT
# ══════════════════════════════════════════════════════════════════

print("\n" + "-" * 72)
print("  3. MARKET-ON-CLOSE (MOC) — enter at close of signal day")
print("-" * 72)

trades_moc = run_backtest(spread_df, sig_fn, exit_fn, max_hold=MAX_HOLD,
                          position_size=POSITION_SIZE, fill_type="market_on_close")

sel_moc = trades_moc[trades_moc["entry_date"] < SEL_END].copy()
oos_moc = trades_moc[trades_moc["entry_date"] >= SEL_END].copy()

stats_sel_moc = compute_stats(sel_moc, spread_df, label="MOC Selection")
stats_oos_moc = compute_stats(oos_moc, spread_df, label="MOC 2026")

print(f"  Selection: n={stats_sel_moc['n']}, avg={stats_sel_moc['avg_pnl']:+.2f}%, "
      f"total={stats_sel_moc['total_pnl_pct']:+.1f}%, Sharpe={stats_sel_moc['sharpe']}")
print(f"  2026 OOS:  n={stats_oos_moc['n']}, avg={stats_oos_moc['avg_pnl']:+.2f}%, "
      f"total={stats_oos_moc['total_pnl_pct']:+.1f}%, Sharpe={stats_oos_moc['sharpe']}")


# ══════════════════════════════════════════════════════════════════
# 4. LIMIT ORDER VARIANT
# ══════════════════════════════════════════════════════════════════

print("\n" + "-" * 72)
print("  4. LIMIT ORDER — fill at z-threshold price, MOO fallback")
print("-" * 72)

# For each signal, compute the RXRX price where z would equal exactly k_entry.
#
# Math:
#   spread_z = (spread - mean) / std
#   spread = log(RXRX) - OLS_predict(log(ARKG))
#   threshold_spread = mean + k_threshold * std     (for shorts: +k, longs: -k)
#   delta = threshold_spread - spread_actual = (k_threshold - z_actual) * std
#   threshold_price = close * exp(delta)
#
# For shorts (z > +k): delta < 0, so threshold_price < close (lower)
# For longs  (z < -k): delta > 0, so threshold_price > close (higher)

def run_limit_backtest(spread_df, k_entry=K_ENTRY, z_exit=Z_EXIT,
                       max_hold=MAX_HOLD, position_size=POSITION_SIZE):
    """Custom backtest with limit-order entry logic."""
    trades = []
    in_trade_until = pd.Timestamp("1900-01-01")

    for i in range(120, len(spread_df)):
        sig = sig_fn(spread_df.iloc[i], i, spread_df)
        if sig is None:
            continue

        entry_idx = i + 1   # next day (same as MOO)
        if entry_idx >= len(spread_df):
            continue

        signal_date = spread_df["date"].iloc[i]
        entry_date = spread_df["date"].iloc[entry_idx]
        if entry_date <= in_trade_until:
            continue

        side = sig["side"]
        z_signal = spread_df["spread_z"].iloc[i]
        std_val = spread_df["spread_rolling_std"].iloc[i]
        close_signal = spread_df["close"].iloc[i]

        if pd.isna(std_val) or std_val <= 0:
            # Fallback to MOO
            entry_price = spread_df["open"].iloc[entry_idx]
            fill_type_used = "moo_fallback"
        else:
            # Compute threshold price
            k_thresh = k_entry if side == "short" else -k_entry
            delta = (k_thresh - z_signal) * std_val
            threshold_price = close_signal * np.exp(delta)

            next_low = spread_df["low"].iloc[entry_idx]
            next_high = spread_df["high"].iloc[entry_idx]
            next_open = spread_df["open"].iloc[entry_idx]

            if side == "short":
                # Limit sell: want to sell at threshold_price or higher
                # If next day's HIGH >= threshold_price, we get filled at threshold_price
                if next_high >= threshold_price:
                    entry_price = threshold_price
                    fill_type_used = "limit_fill"
                else:
                    entry_price = next_open
                    fill_type_used = "moo_fallback"
            else:  # long
                # Limit buy: want to buy at threshold_price or lower
                # If next day's LOW <= threshold_price, we get filled at threshold_price
                if next_low <= threshold_price:
                    entry_price = threshold_price
                    fill_type_used = "limit_fill"
                else:
                    entry_price = next_open
                    fill_type_used = "moo_fallback"

        stop = sig.get("stop")
        target = sig.get("target")
        exit_price = exit_date = exit_reason = None

        for j in range(entry_idx + 1, min(entry_idx + max_hold + 1, len(spread_df))):
            row = spread_df.iloc[j]

            result = exit_fn(sig, row, j, spread_df)
            if result:
                exit_reason, exit_price = result
                exit_date = row["date"]
                break

            if stop is not None:
                if side == "short" and row["high"] >= stop:
                    exit_price, exit_reason, exit_date = stop, "stop", row["date"]
                    break
                if side == "long" and row["low"] <= stop:
                    exit_price, exit_reason, exit_date = stop, "stop", row["date"]
                    break

            if target is not None:
                if side == "long" and row["high"] >= target:
                    exit_price, exit_reason, exit_date = target, "target", row["date"]
                    break
                if side == "short" and row["low"] <= target:
                    exit_price, exit_reason, exit_date = target, "target", row["date"]
                    break

        if exit_price is None:
            exit_idx = min(entry_idx + max_hold, len(spread_df) - 1)
            exit_price = spread_df["close"].iloc[exit_idx]
            exit_date = spread_df["date"].iloc[exit_idx]
            exit_reason = "time_stop"

        in_trade_until = exit_date
        holding_days = max((exit_date - entry_date).days, 1)

        pnl_pct = ((exit_price - entry_price) / entry_price * 100) if side == "long" \
                  else ((entry_price - exit_price) / entry_price * 100)
        cost_pct = compute_costs(entry_price, side, holding_days) / entry_price * 100
        pnl_net = pnl_pct - cost_pct
        shares = max(int(position_size / entry_price), 1)

        trades.append({
            "signal_date": signal_date,
            "entry_date": entry_date, "exit_date": exit_date,
            "side": side, "entry_price": round(entry_price, 4),
            "exit_price": round(exit_price, 4), "exit_reason": exit_reason,
            "holding_days": holding_days,
            "z_at_signal": round(z_signal, 4),
            "pnl_gross_pct": round(pnl_pct, 4), "cost_pct": round(cost_pct, 4),
            "pnl_net_pct": round(pnl_net, 4),
            "shares": shares, "pnl_dollars": round(pnl_net / 100 * shares * entry_price, 2),
            "fill_type": fill_type_used,
            "threshold_price": round(threshold_price, 4) if 'threshold_price' in dir() else np.nan,
        })

    return pd.DataFrame(trades)

trades_limit = run_limit_backtest(spread_df)

sel_limit = trades_limit[trades_limit["entry_date"] < SEL_END].copy()
oos_limit = trades_limit[trades_limit["entry_date"] >= SEL_END].copy()

stats_sel_limit = compute_stats(sel_limit, spread_df, label="Limit Selection")
stats_oos_limit = compute_stats(oos_limit, spread_df, label="Limit 2026")

print(f"  Selection: n={stats_sel_limit['n']}, avg={stats_sel_limit['avg_pnl']:+.2f}%, "
      f"total={stats_sel_limit['total_pnl_pct']:+.1f}%, Sharpe={stats_sel_limit['sharpe']}")
print(f"  2026 OOS:  n={stats_oos_limit['n']}, avg={stats_oos_limit['avg_pnl']:+.2f}%, "
      f"total={stats_oos_limit['total_pnl_pct']:+.1f}%, Sharpe={stats_oos_limit['sharpe']}")

# Limit fill rate
if not trades_limit.empty:
    limit_fills = (trades_limit["fill_type"] == "limit_fill").sum()
    moo_fallbacks = (trades_limit["fill_type"] == "moo_fallback").sum()
    total = len(trades_limit)
    print(f"  Fill breakdown: {limit_fills}/{total} limit fills ({limit_fills/total*100:.0f}%), "
          f"{moo_fallbacks}/{total} MOO fallbacks ({moo_fallbacks/total*100:.0f}%)")


# ══════════════════════════════════════════════════════════════════
# 5. COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  COMPARISON TABLE: MOO vs MOC vs LIMIT")
print("=" * 72)

def print_row(label, stats):
    print(f"  {label:<22s}  n={stats['n']:>3d}  avg={stats['avg_pnl']:>+6.2f}%  "
          f"total={stats['total_pnl_pct']:>+7.1f}%  WR={stats['wr']:>5.1f}%  "
          f"Sharpe={stats['sharpe']:>5.2f}  PF={stats['pf']:>5.2f}  "
          f"DD={stats['max_dd']:>+6.1f}%  t={stats['t_stat']:>5.2f}")

print("\n  ── Selection Window (before 2026-01-01) ──")
print_row("MOO", stats_sel_moo)
print_row("MOC", stats_sel_moc)
print_row("Limit", stats_sel_limit)

print("\n  ── 2026 OOS ──")
print_row("MOO", stats_oos_moo)
print_row("MOC", stats_oos_moc)
print_row("Limit", stats_oos_limit)


# ══════════════════════════════════════════════════════════════════
# 6. ENTRY TIMING WITH 30-MIN BARS
# ══════════════════════════════════════════════════════════════════

print("\n" + "-" * 72)
print("  6. INTRADAY TIMING ANALYSIS (30-min bars)")
print("-" * 72)

# Regular trading hours: 9:30 to 16:00 ET (inclusive of bar starts)
from datetime import time as dtime
RTH_SLOTS = [dtime(h, m) for h in range(9, 16) for m in [0, 30]
             if (h, m) >= (9, 30)] + [dtime(16, 0)]
# Filter: 09:30, 10:00, 10:30, 11:00, 11:30, 12:00, 12:30, 13:00,
#          13:30, 14:00, 14:30, 15:00, 15:30, 16:00

print(f"  RTH time slots: {[str(t)[:5] for t in RTH_SLOTS]}")

# For each trade signal, find the intraday bars on the entry day
# and compare each bar's OPEN (fill at start of that period) to MOO.
# MOO = open of the 9:30 bar.

intraday_results = []

for _, trade in trades_moo.iterrows():
    entry_d = trade["entry_date"]
    side = trade["side"]
    moo_price = trade["entry_price"]
    z_signal = trade["z_at_signal"]

    # Get 30-min bars for the entry day
    day_bars = rxrx_30m[rxrx_30m["cal_date"] == pd.Timestamp(entry_d.date())].copy()
    if day_bars.empty:
        continue

    for slot in RTH_SLOTS:
        slot_bar = day_bars[day_bars["time_et"] == slot]
        if slot_bar.empty:
            continue

        # Use OPEN of this bar as the theoretical fill price at this time
        fill_price = slot_bar.iloc[0]["open"]

        # PnL improvement: positive = better fill than MOO
        if side == "short":
            # For shorts, higher fill = better (selling at higher price)
            improvement_pct = (fill_price - moo_price) / moo_price * 100
        else:
            # For longs, lower fill = better (buying at lower price)
            improvement_pct = (moo_price - fill_price) / moo_price * 100

        intraday_results.append({
            "entry_date": entry_d,
            "side": side,
            "z_at_signal": z_signal,
            "time_slot": slot,
            "moo_price": moo_price,
            "slot_price": fill_price,
            "improvement_pct": improvement_pct,
            "period": "OOS" if entry_d >= pd.Timestamp(SEL_END) else "Selection",
        })

intra_df = pd.DataFrame(intraday_results)

if not intra_df.empty:
    # Average improvement by time slot
    print("\n  Average fill improvement vs MOO by time slot (all trades):")
    print(f"  {'Time':>8s}  {'Avg Impr%':>10s}  {'Median':>8s}  {'Std':>8s}  {'N':>5s}")
    time_summary = []
    for slot in RTH_SLOTS:
        subset = intra_df[intra_df["time_slot"] == slot]
        if subset.empty:
            continue
        avg_imp = subset["improvement_pct"].mean()
        med_imp = subset["improvement_pct"].median()
        std_imp = subset["improvement_pct"].std()
        n = len(subset)
        print(f"  {str(slot)[:5]:>8s}  {avg_imp:>+10.3f}  {med_imp:>+8.3f}  {std_imp:>8.3f}  {n:>5d}")
        time_summary.append({
            "time_slot": slot, "avg_improvement": avg_imp,
            "median_improvement": med_imp, "std": std_imp, "n": n,
        })

    # Best and worst intraday fill per signal
    print("\n  Best and worst 30-min fill per trade:")
    print(f"  {'Entry Date':>12s} {'Side':>6s}  {'MOO':>8s}  {'Best Fill':>10s} {'At':>8s}  "
          f"{'Worst Fill':>10s} {'At':>8s}  {'Range %':>8s}")

    for entry_d in sorted(intra_df["entry_date"].unique()):
        day_data = intra_df[intra_df["entry_date"] == entry_d]
        best = day_data.loc[day_data["improvement_pct"].idxmax()]
        worst = day_data.loc[day_data["improvement_pct"].idxmin()]
        range_pct = best["improvement_pct"] - worst["improvement_pct"]
        print(f"  {str(entry_d.date()):>12s} {best['side']:>6s}  "
              f"${best['moo_price']:>7.2f}  "
              f"${best['slot_price']:>8.2f} {str(best['time_slot'])[:5]:>6s}  "
              f"${worst['slot_price']:>8.2f} {str(worst['time_slot'])[:5]:>6s}  "
              f"{range_pct:>+7.2f}%")

    # OOS only
    oos_intra = intra_df[intra_df["period"] == "OOS"]
    if not oos_intra.empty:
        print("\n  Average fill improvement vs MOO by time slot (2026 OOS only):")
        print(f"  {'Time':>8s}  {'Avg Impr%':>10s}  {'N':>5s}")
        for slot in RTH_SLOTS:
            subset = oos_intra[oos_intra["time_slot"] == slot]
            if subset.empty:
                continue
            print(f"  {str(slot)[:5]:>8s}  {subset['improvement_pct'].mean():>+10.3f}  {len(subset):>5d}")

    time_summary_df = pd.DataFrame(time_summary)
else:
    print("  No intraday data available for signal dates.")
    time_summary_df = pd.DataFrame()


# ══════════════════════════════════════════════════════════════════
# 7. MOC vs MOO FILL COMPARISON (trade-by-trade)
# ══════════════════════════════════════════════════════════════════

print("\n" + "-" * 72)
print("  7. MOC vs MOO — TRADE-BY-TRADE FILL COMPARISON")
print("-" * 72)

# Match MOC and MOO trades by signal date
if not trades_moc.empty and not trades_moo.empty:
    merged_trades = trades_moo[["signal_date", "side", "entry_price", "pnl_net_pct", "z_at_signal"]].rename(
        columns={"entry_price": "moo_entry", "pnl_net_pct": "moo_pnl"}
    ).merge(
        trades_moc[["signal_date", "entry_price", "pnl_net_pct"]].rename(
            columns={"entry_price": "moc_entry", "pnl_net_pct": "moc_pnl"}
        ),
        on="signal_date", how="inner"
    )

    merged_trades["fill_diff_pct"] = (merged_trades["moc_entry"] - merged_trades["moo_entry"]) / merged_trades["moo_entry"] * 100
    merged_trades["pnl_diff"] = merged_trades["moc_pnl"] - merged_trades["moo_pnl"]

    # For scatter plot: z-score at signal vs fill improvement
    # "improvement" = better fill for the given side
    merged_trades["fill_improvement"] = np.where(
        merged_trades["side"] == "short",
        merged_trades["fill_diff_pct"],      # shorts: higher price = better
        -merged_trades["fill_diff_pct"],      # longs: lower price = better
    )

    print(f"\n  Matched trades: {len(merged_trades)}")
    print(f"  Avg MOC fill improvement: {merged_trades['fill_improvement'].mean():+.3f}%")
    print(f"  Avg PnL diff (MOC - MOO): {merged_trades['pnl_diff'].mean():+.3f}%")

    # Split by side
    for s in ["short", "long"]:
        sub = merged_trades[merged_trades["side"] == s]
        if sub.empty:
            continue
        print(f"    {s.upper():>6s}: n={len(sub)}, avg fill improvement={sub['fill_improvement'].mean():+.3f}%, "
              f"avg PnL diff={sub['pnl_diff'].mean():+.3f}%")


# ══════════════════════════════════════════════════════════════════
# 8. LOG TRIALS
# ══════════════════════════════════════════════════════════════════

print("\n  Logging trials...")

for label, stats in [
    ("MOO_sel", stats_sel_moo), ("MOO_oos", stats_oos_moo),
    ("MOC_sel", stats_sel_moc), ("MOC_oos", stats_oos_moc),
    ("Limit_sel", stats_sel_limit), ("Limit_oos", stats_oos_limit),
]:
    log_trial("agent1", f"S5_zexit_{label}",
              {"fill_type": label.split("_")[0], "k_entry": K_ENTRY,
               "z_exit": Z_EXIT, "max_hold": MAX_HOLD},
              stats)


# ══════════════════════════════════════════════════════════════════
# 9. MATPLOTLIB FIGURE
# ══════════════════════════════════════════════════════════════════

print("\n  Generating plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Agent 1: S5_zexit Execution Variants", fontsize=14, fontweight="bold")

# ── Panel A: Selection Window Equity Curves ─────────────────────
ax = axes[0, 0]
ax.set_title("(a) Selection Window Equity Curves")

for label, trades, color, ls in [
    ("MOO", sel_moo, "#1f77b4", "-"),
    ("MOC", sel_moc, "#ff7f0e", "--"),
    ("Limit", sel_limit, "#2ca02c", "-."),
]:
    if not trades.empty:
        eq = np.cumsum(trades["pnl_net_pct"].values)
        ax.plot(range(len(eq)), eq, label=label, color=color, linestyle=ls, linewidth=1.5)
        # Add dots at each trade
        ax.scatter(range(len(eq)), eq, s=15, color=color, zorder=5)

ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
ax.set_xlabel("Trade Number")
ax.set_ylabel("Cumulative Net PnL (%)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── Panel B: 2026 OOS Equity Curves ────────────────────────────
ax = axes[0, 1]
ax.set_title("(b) 2026 OOS Equity Curves")

for label, trades, color, ls in [
    ("MOO", oos_moo, "#1f77b4", "-"),
    ("MOC", oos_moc, "#ff7f0e", "--"),
    ("Limit", oos_limit, "#2ca02c", "-."),
]:
    if not trades.empty:
        eq = np.cumsum(trades["pnl_net_pct"].values)
        dates = trades["entry_date"].values
        ax.plot(dates, eq, label=f"{label} (avg {trades['pnl_net_pct'].mean():+.2f}%)",
                color=color, linestyle=ls, linewidth=1.5)
        ax.scatter(dates, eq, s=15, color=color, zorder=5)

ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative Net PnL (%)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
# Rotate x labels
for lbl in ax.get_xticklabels():
    lbl.set_rotation(30)
    lbl.set_ha("right")

# ── Panel C: Intraday Fill Improvement by Time Slot ─────────────
ax = axes[1, 0]
ax.set_title("(c) Intraday Fill Improvement vs MOO by Time")

if not time_summary_df.empty:
    time_labels = [str(t)[:5] for t in time_summary_df["time_slot"]]
    avg_vals = time_summary_df["avg_improvement"].values
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in avg_vals]
    bars = ax.bar(range(len(time_labels)), avg_vals, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(time_labels)))
    ax.set_xticklabels(time_labels, rotation=45, ha="right", fontsize=8)

    # Add error bars
    ax.errorbar(range(len(time_labels)), avg_vals,
                yerr=time_summary_df["std"].values / np.sqrt(time_summary_df["n"].values),
                fmt="none", ecolor="black", capsize=3, linewidth=1)

    ax.axhline(0, color="gray", linewidth=1, linestyle="-")
    ax.set_xlabel("Time of Day (ET)")
    ax.set_ylabel("Avg Fill Improvement vs MOO (%)")
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate best slot
    best_idx = np.argmax(avg_vals)
    ax.annotate(f"Best: {time_labels[best_idx]}\n{avg_vals[best_idx]:+.3f}%",
                xy=(best_idx, avg_vals[best_idx]),
                xytext=(best_idx + 1.5, avg_vals[best_idx] + 0.1),
                fontsize=8, arrowprops=dict(arrowstyle="->", color="black"),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
else:
    ax.text(0.5, 0.5, "No 30-min data for signal dates", transform=ax.transAxes,
            ha="center", va="center", fontsize=12, color="gray")

# ── Panel D: Z-score at Signal vs Fill Improvement (MOC vs MOO) ──
ax = axes[1, 1]
ax.set_title("(d) Z-Score at Signal vs MOC Fill Improvement")

if 'merged_trades' in dir() and not merged_trades.empty:
    shorts = merged_trades[merged_trades["side"] == "short"]
    longs = merged_trades[merged_trades["side"] == "long"]

    if not shorts.empty:
        ax.scatter(shorts["z_at_signal"], shorts["fill_improvement"],
                   c="#d62728", marker="v", s=50, label=f"Short (n={len(shorts)})",
                   edgecolors="black", linewidth=0.5, alpha=0.7)
    if not longs.empty:
        ax.scatter(longs["z_at_signal"], longs["fill_improvement"],
                   c="#1f77b4", marker="^", s=50, label=f"Long (n={len(longs)})",
                   edgecolors="black", linewidth=0.5, alpha=0.7)

    ax.axhline(0, color="gray", linewidth=1, linestyle=":")
    ax.axvline(K_ENTRY, color="red", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axvline(-K_ENTRY, color="blue", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel("Z-Score at Signal")
    ax.set_ylabel("MOC Fill Improvement vs MOO (%)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add correlation
    if len(merged_trades) > 5:
        from scipy.stats import pearsonr
        r, p = pearsonr(merged_trades["z_at_signal"].abs(), merged_trades["fill_improvement"])
        ax.text(0.05, 0.95, f"r(|z|, imp) = {r:.3f}\np = {p:.3f}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
else:
    ax.text(0.5, 0.5, "No matched MOC/MOO trades", transform=ax.transAxes,
            ha="center", va="center", fontsize=12, color="gray")

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig_path = OUT_DIR / "agent1_execution.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Figure saved: {fig_path}")


# ══════════════════════════════════════════════════════════════════
# 10. CONCLUSION
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  CONCLUSION")
print("=" * 72)

# Determine best method
methods = {
    "MOO": (stats_oos_moo, stats_sel_moo),
    "MOC": (stats_oos_moc, stats_sel_moc),
    "Limit": (stats_oos_limit, stats_sel_limit),
}

best_oos = max(methods.items(), key=lambda x: x[1][0]["avg_pnl"])
best_sel = max(methods.items(), key=lambda x: x[1][1]["avg_pnl"])

print(f"""
  EXECUTION VARIANT COMPARISON SUMMARY
  =====================================

  Selection Window:
    MOO:   n={stats_sel_moo['n']:>3d}  avg={stats_sel_moo['avg_pnl']:>+6.2f}%  Sharpe={stats_sel_moo['sharpe']:>5.2f}
    MOC:   n={stats_sel_moc['n']:>3d}  avg={stats_sel_moc['avg_pnl']:>+6.2f}%  Sharpe={stats_sel_moc['sharpe']:>5.2f}
    Limit: n={stats_sel_limit['n']:>3d}  avg={stats_sel_limit['avg_pnl']:>+6.2f}%  Sharpe={stats_sel_limit['sharpe']:>5.2f}
    Best in selection: {best_sel[0]}

  2026 OOS:
    MOO:   n={stats_oos_moo['n']:>3d}  avg={stats_oos_moo['avg_pnl']:>+6.2f}%  Sharpe={stats_oos_moo['sharpe']:>5.2f}
    MOC:   n={stats_oos_moc['n']:>3d}  avg={stats_oos_moc['avg_pnl']:>+6.2f}%  Sharpe={stats_oos_moc['sharpe']:>5.2f}
    Limit: n={stats_oos_limit['n']:>3d}  avg={stats_oos_limit['avg_pnl']:>+6.2f}%  Sharpe={stats_oos_limit['sharpe']:>5.2f}
    Best in 2026 OOS: {best_oos[0]}

  MOO vs best ({best_oos[0]}) improvement in 2026:
    avg PnL: {stats_oos_moo['avg_pnl']:+.2f}% -> {best_oos[1][0]['avg_pnl']:+.2f}% ({best_oos[1][0]['avg_pnl'] - stats_oos_moo['avg_pnl']:+.2f}% delta)
    Sharpe:  {stats_oos_moo['sharpe']:.2f} -> {best_oos[1][0]['sharpe']:.2f}""")

if not time_summary_df.empty:
    best_time = time_summary_df.loc[time_summary_df["avg_improvement"].idxmax()]
    worst_time = time_summary_df.loc[time_summary_df["avg_improvement"].idxmin()]
    print(f"""
  INTRADAY TIMING:
    Best time slot:  {str(best_time['time_slot'])[:5]} ET  ({best_time['avg_improvement']:+.3f}% avg improvement)
    Worst time slot: {str(worst_time['time_slot'])[:5]} ET  ({worst_time['avg_improvement']:+.3f}% avg improvement)
    Intraday range: {best_time['avg_improvement'] - worst_time['avg_improvement']:.3f}% avg swing""")

if not trades_limit.empty:
    fill_rate = (trades_limit["fill_type"] == "limit_fill").mean() * 100
    print(f"""
  LIMIT ORDER:
    Fill rate: {fill_rate:.0f}% ({(trades_limit['fill_type'] == 'limit_fill').sum()}/{len(trades_limit)} trades)
    Limit fills capture the z-threshold crossing price.""")

print(f"""
  RECOMMENDATION: {best_oos[0]} is the best execution variant for 2026 OOS.
""")

print("  AGENT 1 COMPLETE")
