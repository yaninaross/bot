# %% Agent 2 — Overnight vs Intraday Decomposition
#
# Decomposes RXRX daily returns into overnight (prev close -> open)
# and intraday (open -> close) components. Tests whether the z-score
# signal predicts one component better than the other. Compares
# MOC vs MOO entry. Analyzes AM vs PM session predictive power.
#
# Imports from agent0_engine.py (canonical engine).

import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

for pkg in ["matplotlib", "statsmodels", "scipy"]:
    try:
        __import__(pkg)
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg,
                               "--quiet", "--break-system-packages"])

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Engine imports ───────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent if "__file__" in dir() else Path("agent10")))
from agent0_engine import (
    build_spread, run_backtest, compute_stats, compute_costs,
    log_trial, s5_zexit_signal, s5_zexit_exit, add_features,
    DATA_DIR, OUT_DIR,
)

# ── Config ───────────────────────────────────────────────────────
SEL_END = "2026-01-01"           # OOS starts here
K_ENTRY = 1.5                    # z-score threshold
Z_EXIT  = 0.5                    # z-exit threshold
MAX_HOLD = 10                    # max holding days
LOOKBACK = 60                    # OLS lookback
POSITION_SIZE = 10000
FIG_PATH = OUT_DIR / "agent2_overnight_decomp.png"

# ══════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════

print("=" * 70)
print("  AGENT 2: OVERNIGHT vs INTRADAY DECOMPOSITION")
print("=" * 70)

rxrx = pd.read_parquet(DATA_DIR / "rxrx_daily.parquet")
rxrx["date"] = pd.to_datetime(rxrx["date"])
rxrx = rxrx.sort_values("date").reset_index(drop=True)
rxrx = add_features(rxrx)

ctx = pd.read_parquet(DATA_DIR / "context_daily.parquet")
ctx["date"] = pd.to_datetime(ctx["date"])
arkg = ctx[ctx["ticker"] == "ARKG"].sort_values("date").reset_index(drop=True)

print(f"  RXRX: {len(rxrx)} bars, {rxrx['date'].min().date()} to {rxrx['date'].max().date()}")

# Build canonical spread
spread = build_spread(rxrx, arkg, "ARKG", LOOKBACK)
print(f"  Spread built: {len(spread)} bars")

# ══════════════════════════════════════════════════════════════════
# 2. DECOMPOSE RETURNS INTO OVERNIGHT + INTRADAY
# ══════════════════════════════════════════════════════════════════

print("\n" + "-" * 70)
print("  PART 1: RETURN DECOMPOSITION")
print("-" * 70)

spread["prev_close"] = spread["close"].shift(1)
spread["overnight_ret"] = (spread["open"] - spread["prev_close"]) / spread["prev_close"]
spread["intraday_ret"]  = (spread["close"] - spread["open"]) / spread["open"]
spread["total_ret"]     = (spread["close"] - spread["prev_close"]) / spread["prev_close"]

# Rolling stats (60-day window)
ROLL_W = 60
for comp in ["overnight_ret", "intraday_ret"]:
    spread[f"{comp}_roll_mean"] = spread[comp].rolling(ROLL_W).mean()
    spread[f"{comp}_roll_vol"]  = spread[comp].rolling(ROLL_W).std()
    spread[f"{comp}_roll_sr"]   = (
        spread[f"{comp}_roll_mean"] / spread[f"{comp}_roll_vol"].replace(0, np.nan)
    ) * np.sqrt(252)

# Summary stats
valid = spread.dropna(subset=["overnight_ret", "intraday_ret"])
sel_mask = valid["date"] < SEL_END
oos_mask = valid["date"] >= SEL_END

print(f"\n  {'Period':<15s} {'Component':<12s} {'Mean%':>8s} {'Vol%':>8s} {'SR':>8s} {'Skew':>8s}")
print(f"  {'-'*55}")
for period_name, mask in [("Selection", sel_mask), ("OOS 2026", oos_mask), ("All", valid.index == valid.index)]:
    subset = valid[mask] if isinstance(mask, pd.Series) else valid
    for comp_name, col in [("Overnight", "overnight_ret"), ("Intraday", "intraday_ret")]:
        vals = subset[col].values * 100
        mn = np.mean(vals)
        vol = np.std(vals)
        sr = mn / vol * np.sqrt(252) if vol > 0 else 0
        sk = float(pd.Series(vals).skew())
        print(f"  {period_name:<15s} {comp_name:<12s} {mn:>+7.3f}% {vol:>7.3f}% {sr:>+7.2f} {sk:>+7.2f}")

# ══════════════════════════════════════════════════════════════════
# 3. BASELINE S5_ZEXIT TRADES — DECOMPOSE PnL
# ══════════════════════════════════════════════════════════════════

print("\n" + "-" * 70)
print("  PART 2: PnL DECOMPOSITION BY TRADE (MOO baseline)")
print("-" * 70)

def sig_fn(row, i, df):
    return s5_zexit_signal(row, i, df, k_entry=K_ENTRY)

def exit_fn(sig, row, j, df):
    return s5_zexit_exit(sig, row, j, df, z_exit_thresh=Z_EXIT)

trades_moo = run_backtest(spread, sig_fn, exit_fn, max_hold=MAX_HOLD,
                          position_size=POSITION_SIZE, fill_type="market_on_open")

# For each trade, decompose PnL into overnight + intraday components
def decompose_trade_pnl(trade_row, spread_df):
    """Decompose a trade's gross PnL into overnight and intraday parts."""
    entry_date = trade_row["entry_date"]
    exit_date  = trade_row["exit_date"]
    side       = trade_row["side"]
    entry_px   = trade_row["entry_price"]

    mask = (spread_df["date"] >= entry_date) & (spread_df["date"] <= exit_date)
    bars = spread_df[mask].copy()
    if len(bars) < 1:
        return 0.0, 0.0

    overnight_sum = 0.0
    intraday_sum  = 0.0

    for idx in range(len(bars)):
        row = bars.iloc[idx]
        if idx == 0:
            # First day: entry is at open, so intraday = open -> close
            intraday_sum += (row["close"] - row["open"]) / entry_px * 100
        else:
            # Overnight: prev_close -> open
            prev_close = bars.iloc[idx - 1]["close"]
            overnight_sum += (row["open"] - prev_close) / entry_px * 100
            if row["date"] == exit_date:
                # Last day: partial intraday = open -> exit_price
                intraday_sum += (trade_row["exit_price"] - row["open"]) / entry_px * 100
            else:
                # Full day: open -> close
                intraday_sum += (row["close"] - row["open"]) / entry_px * 100

    if side == "short":
        overnight_sum = -overnight_sum
        intraday_sum  = -intraday_sum

    return overnight_sum, intraday_sum


decomp_records = []
for _, t in trades_moo.iterrows():
    on_pnl, id_pnl = decompose_trade_pnl(t, spread)
    decomp_records.append({
        "signal_date": t["signal_date"],
        "entry_date": t["entry_date"],
        "exit_date": t["exit_date"],
        "side": t["side"],
        "z_at_signal": t["z_at_signal"],
        "pnl_gross_pct": t["pnl_gross_pct"],
        "overnight_pnl_pct": round(on_pnl, 4),
        "intraday_pnl_pct": round(id_pnl, 4),
        "pnl_net_pct": t["pnl_net_pct"],
        "cost_pct": t["cost_pct"],
        "holding_days": t["holding_days"],
    })

decomp_df = pd.DataFrame(decomp_records)

# Split into selection / OOS
decomp_sel = decomp_df[decomp_df["entry_date"] < SEL_END]
decomp_oos = decomp_df[decomp_df["entry_date"] >= SEL_END]

print(f"\n  Total trades: {len(decomp_df)} (Selection: {len(decomp_sel)}, OOS: {len(decomp_oos)})")

# Print stats by side and period
print(f"\n  {'Period':<12s} {'Side':<6s} {'N':>4s} {'ON avg%':>9s} {'ID avg%':>9s} "
      f"{'ON tot%':>9s} {'ID tot%':>9s} {'ON share':>9s}")
print(f"  {'-'*70}")

for period_name, df_sub in [("Selection", decomp_sel), ("OOS 2026", decomp_oos)]:
    for side in ["long", "short"]:
        s = df_sub[df_sub["side"] == side]
        if len(s) == 0:
            continue
        on_avg = s["overnight_pnl_pct"].mean()
        id_avg = s["intraday_pnl_pct"].mean()
        on_tot = s["overnight_pnl_pct"].sum()
        id_tot = s["intraday_pnl_pct"].sum()
        total  = on_tot + id_tot
        on_share = on_tot / total * 100 if total != 0 else 0
        print(f"  {period_name:<12s} {side:<6s} {len(s):>4d} {on_avg:>+8.3f}% {id_avg:>+8.3f}% "
              f"{on_tot:>+8.2f}% {id_tot:>+8.2f}% {on_share:>+8.1f}%")

# OOS trade-by-trade table
if len(decomp_oos) > 0:
    print(f"\n  OOS 2026 trade-by-trade decomposition:")
    print(f"  {'Entry':>12s} {'Side':<6s} {'z':>7s} {'Gross%':>8s} {'ON%':>8s} {'ID%':>8s} "
          f"{'Net%':>8s} {'Days':>4s}")
    for _, t in decomp_oos.iterrows():
        print(f"  {str(t['entry_date'].date()):>12s} {t['side']:<6s} {t['z_at_signal']:>+6.2f} "
              f"{t['pnl_gross_pct']:>+7.2f}% {t['overnight_pnl_pct']:>+7.2f}% "
              f"{t['intraday_pnl_pct']:>+7.2f}% {t['pnl_net_pct']:>+7.2f}% {t['holding_days']:>4d}")

# ══════════════════════════════════════════════════════════════════
# 4. Z-SCORE vs OVERNIGHT/INTRADAY RETURN ANALYSIS
# ══════════════════════════════════════════════════════════════════

print("\n" + "-" * 70)
print("  PART 3: Z-SCORE PREDICTIVE POWER — OVERNIGHT vs INTRADAY")
print("-" * 70)

# For each date where z-score is extreme, compute next-day overnight and intraday return
spread_valid = spread.dropna(subset=["spread_z", "overnight_ret", "intraday_ret"]).copy()

# Forward returns: next day's overnight and intraday
spread_valid["fwd_overnight"] = spread_valid["overnight_ret"].shift(-1)
spread_valid["fwd_intraday"]  = spread_valid["intraday_ret"].shift(-1)
spread_valid["fwd_total"]     = spread_valid["total_ret"].shift(-1)

# Extreme z-score observations
for period_name, pmask in [("Selection", spread_valid["date"] < SEL_END),
                            ("OOS 2026", spread_valid["date"] >= SEL_END)]:
    pdata = spread_valid[pmask].dropna(subset=["fwd_overnight", "fwd_intraday"])

    short_sig = pdata["spread_z"] > K_ENTRY
    long_sig  = pdata["spread_z"] < -K_ENTRY

    print(f"\n  {period_name}:")
    for label, mask, flip in [("SHORT (z>{:.1f})".format(K_ENTRY), short_sig, -1),
                               ("LONG  (z<{:.1f})".format(-K_ENTRY), long_sig, 1)]:
        s = pdata[mask]
        if len(s) == 0:
            print(f"    {label}: no signals")
            continue
        on_ret = s["fwd_overnight"].values * 100 * flip
        id_ret = s["fwd_intraday"].values * 100 * flip
        tot_ret = s["fwd_total"].values * 100 * flip
        print(f"    {label}  n={len(s)}")
        print(f"      Next-day overnight:  avg={np.mean(on_ret):+.3f}%  med={np.median(on_ret):+.3f}%  "
              f"pos={np.mean(on_ret > 0)*100:.0f}%")
        print(f"      Next-day intraday:   avg={np.mean(id_ret):+.3f}%  med={np.median(id_ret):+.3f}%  "
              f"pos={np.mean(id_ret > 0)*100:.0f}%")
        print(f"      Next-day total:      avg={np.mean(tot_ret):+.3f}%  med={np.median(tot_ret):+.3f}%  "
              f"pos={np.mean(tot_ret > 0)*100:.0f}%")


# ══════════════════════════════════════════════════════════════════
# 5. MOC VARIANT — ENTER AT CLOSE OF SIGNAL DAY
# ══════════════════════════════════════════════════════════════════

print("\n" + "-" * 70)
print("  PART 4: MOC vs MOO COMPARISON")
print("-" * 70)

trades_moc = run_backtest(spread, sig_fn, exit_fn, max_hold=MAX_HOLD,
                          position_size=POSITION_SIZE, fill_type="market_on_close")

# Split both by period
oos_cutoff = pd.Timestamp(SEL_END)
for period_name, cutoff_fn in [("Selection", lambda d: d < oos_cutoff),
                                ("OOS 2026", lambda d: d >= oos_cutoff)]:
    moo_sub = trades_moo[trades_moo["entry_date"].apply(cutoff_fn)]
    moc_sub = trades_moc[trades_moc["entry_date"].apply(cutoff_fn)]

    moo_stats = compute_stats(moo_sub, label=f"MOO {period_name}")
    moc_stats = compute_stats(moc_sub, label=f"MOC {period_name}")

    print(f"\n  {period_name}:")
    print(f"    {'Fill':<5s} {'N':>4s} {'Avg%':>8s} {'Med%':>8s} {'WR':>6s} {'PF':>6s} "
          f"{'Sharpe':>7s} {'Tot%':>8s} {'t':>6s}")
    for st in [moo_stats, moc_stats]:
        print(f"    {st['label'].split()[0]:<5s} {st['n']:>4d} {st['avg_pnl']:>+7.2f}% "
              f"{st['median_pnl']:>+7.2f}% {st['wr']:>5.1f}% {st['pf']:>5.2f} "
              f"{st['sharpe']:>+6.2f} {st['total_pnl_pct']:>+7.1f}% {st['t_stat']:>+5.2f}")


# ══════════════════════════════════════════════════════════════════
# 6. AM vs PM SESSION ANALYSIS (30-min bars)
# ══════════════════════════════════════════════════════════════════

print("\n" + "-" * 70)
print("  PART 5: AM vs PM SESSION ANALYSIS (30-min bars)")
print("-" * 70)

intra = pd.read_parquet(DATA_DIR / "rxrx_30min.parquet")
intra["date"] = pd.to_datetime(intra["date"])

# Convert to US/Eastern for session identification
intra["date_et"] = intra["date"].dt.tz_convert("US/Eastern")
intra["trade_date"] = intra["date_et"].dt.date
intra["hour_et"] = intra["date_et"].dt.hour
intra["minute_et"] = intra["date_et"].dt.minute
intra["time_et"] = intra["hour_et"] + intra["minute_et"] / 60.0

# Regular hours: 9:30 - 16:00 ET
rth = intra[(intra["time_et"] >= 9.5) & (intra["time_et"] < 16.0)].copy()

# AM session: first 3 bars (9:30, 10:00, 10:30) = 9:30-11:00
# PM session: last 3 bars (14:30, 15:00, 15:30) = 14:30-16:00
am_bars = rth[(rth["time_et"] >= 9.5) & (rth["time_et"] < 11.0)]
pm_bars = rth[(rth["time_et"] >= 14.5) & (rth["time_et"] < 16.0)]

# Compute session returns per day
def session_return(bars_df):
    """Compute session open-to-close return per trade_date."""
    records = []
    for td, grp in bars_df.groupby("trade_date"):
        grp = grp.sort_values("date_et")
        if len(grp) < 1:
            continue
        sess_open = grp.iloc[0]["open"]
        sess_close = grp.iloc[-1]["close"]
        ret = (sess_close - sess_open) / sess_open * 100
        records.append({"trade_date": td, "session_ret": ret})
    return pd.DataFrame(records)

am_rets = session_return(am_bars).rename(columns={"session_ret": "am_ret"})
pm_rets = session_return(pm_bars).rename(columns={"session_ret": "pm_ret"})

# Merge with spread z-scores
spread_for_merge = spread[["date", "spread_z"]].copy()
spread_for_merge["trade_date"] = spread_for_merge["date"].dt.date

session_df = am_rets.merge(pm_rets, on="trade_date", how="inner")
session_df = session_df.merge(spread_for_merge, on="trade_date", how="inner")
session_df["trade_date_dt"] = pd.to_datetime(session_df["trade_date"])

print(f"  Session data: {len(session_df)} days with z-scores and AM/PM returns")

# Analyze by signal type
for period_name, pmask in [("Selection", session_df["trade_date_dt"] < SEL_END),
                            ("OOS 2026", session_df["trade_date_dt"] >= SEL_END)]:
    pdata = session_df[pmask]

    # Signals: use next-day returns (no lookahead)
    pdata = pdata.copy()
    pdata["fwd_am_ret"] = pdata["am_ret"].shift(-1)
    pdata["fwd_pm_ret"] = pdata["pm_ret"].shift(-1)
    pdata = pdata.dropna(subset=["fwd_am_ret", "fwd_pm_ret", "spread_z"])

    short_sig = pdata["spread_z"] > K_ENTRY
    long_sig  = pdata["spread_z"] < -K_ENTRY

    print(f"\n  {period_name}:")
    print(f"    {'Signal':<20s} {'N':>4s} {'AM avg%':>9s} {'PM avg%':>9s} "
          f"{'AM pos%':>8s} {'PM pos%':>8s}")
    for label, mask, flip in [("SHORT (z>{:.1f})".format(K_ENTRY), short_sig, -1),
                               ("LONG  (z<{:.1f})".format(-K_ENTRY), long_sig, 1)]:
        s = pdata[mask]
        if len(s) == 0:
            print(f"    {label:<20s}   0     n/a       n/a       n/a       n/a")
            continue
        am = s["fwd_am_ret"].values * flip
        pm = s["fwd_pm_ret"].values * flip
        print(f"    {label:<20s} {len(s):>4d} {np.mean(am):>+8.3f}% {np.mean(pm):>+8.3f}% "
              f"{np.mean(am > 0)*100:>7.0f}% {np.mean(pm > 0)*100:>7.0f}%")


# ══════════════════════════════════════════════════════════════════
# 7. MATPLOTLIB FIGURE — 4 PANELS
# ══════════════════════════════════════════════════════════════════

print("\n" + "-" * 70)
print("  GENERATING FIGURE...")
print("-" * 70)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Agent 2: Overnight vs Intraday Decomposition — RXRX S5_zexit",
             fontsize=14, fontweight="bold")

# ── Panel A: Stacked bar — overnight vs intraday PnL by trade ───
ax = axes[0, 0]
# Show OOS trades (12 trades) + a representative sample from selection
if len(decomp_oos) > 0:
    plot_df = decomp_oos.copy()
    plot_df["trade_label"] = [f"{str(d.date())[5:]}\n{s[0].upper()}"
                               for d, s in zip(plot_df["entry_date"], plot_df["side"])]
    x = np.arange(len(plot_df))
    width = 0.6
    on_vals = plot_df["overnight_pnl_pct"].values
    id_vals = plot_df["intraday_pnl_pct"].values

    # Color: positive=green, negative=red within each component
    on_colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in on_vals]
    id_colors = ["#27ae60" if v >= 0 else "#c0392b" for v in id_vals]

    bars1 = ax.bar(x, on_vals, width, label="Overnight", color="#3498db", alpha=0.8, edgecolor="white")
    bars2 = ax.bar(x, id_vals, width, bottom=on_vals, label="Intraday", color="#e67e22", alpha=0.8, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["trade_label"].values, fontsize=7, rotation=45, ha="right")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("PnL (%)")
    ax.legend(fontsize=8)
    ax.set_title("A) OOS 2026 PnL Decomposition by Trade", fontsize=10)
else:
    ax.text(0.5, 0.5, "No OOS trades", transform=ax.transAxes, ha="center")
    ax.set_title("A) OOS PnL Decomposition", fontsize=10)

# ── Panel B: Scatter — z-score vs overnight/intraday return ─────
ax = axes[0, 1]
scatter_data = spread_valid.dropna(subset=["fwd_overnight", "fwd_intraday", "spread_z"])
# Use all data but color by period
sel_sc = scatter_data[scatter_data["date"] < SEL_END]
oos_sc = scatter_data[scatter_data["date"] >= SEL_END]

# Only plot extreme z-scores for clarity
for df_sc, alpha, marker, period_label in [
    (sel_sc, 0.3, "o", "Sel"),
    (oos_sc, 0.9, "D", "OOS"),
]:
    extreme = df_sc[(df_sc["spread_z"].abs() > 0.5)]
    ax.scatter(extreme["spread_z"], extreme["fwd_overnight"] * 100,
               alpha=alpha, s=15, marker=marker, c="#3498db",
               label=f"Overnight ({period_label})")
    ax.scatter(extreme["spread_z"], extreme["fwd_intraday"] * 100,
               alpha=alpha, s=15, marker=marker, c="#e67e22",
               label=f"Intraday ({period_label})")

ax.axhline(0, color="gray", linewidth=0.5)
ax.axvline(K_ENTRY, color="red", linewidth=0.5, linestyle="--", alpha=0.5)
ax.axvline(-K_ENTRY, color="green", linewidth=0.5, linestyle="--", alpha=0.5)
ax.set_xlabel("Z-score at signal")
ax.set_ylabel("Next-day return (%)")
ax.legend(fontsize=7, ncol=2, loc="upper left")
ax.set_title("B) Z-score vs Next-day Component Returns", fontsize=10)

# ── Panel C: MOC vs MOO equity curves ───────────────────────────
ax = axes[1, 0]

def equity_curve(trades_df):
    """Simple cumulative PnL curve."""
    if trades_df.empty:
        return pd.Series(dtype=float)
    sorted_t = trades_df.sort_values("entry_date")
    cum_pnl = sorted_t["pnl_net_pct"].cumsum()
    return pd.Series(cum_pnl.values, index=sorted_t["entry_date"].values)

moo_eq = equity_curve(trades_moo)
moc_eq = equity_curve(trades_moc)

if len(moo_eq) > 0:
    ax.plot(moo_eq.index, moo_eq.values, color="#2ecc71", linewidth=1.5,
            label=f"MOO (n={len(trades_moo)}, tot={trades_moo['pnl_net_pct'].sum():+.1f}%)")
if len(moc_eq) > 0:
    ax.plot(moc_eq.index, moc_eq.values, color="#e74c3c", linewidth=1.5,
            label=f"MOC (n={len(trades_moc)}, tot={trades_moc['pnl_net_pct'].sum():+.1f}%)")

ax.axhline(0, color="gray", linewidth=0.5)
# Mark OOS boundary
oos_line = pd.Timestamp(SEL_END)
ax.axvline(oos_line, color="black", linewidth=1, linestyle="--", alpha=0.5, label="OOS start")
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative PnL (%)")
ax.legend(fontsize=8, loc="upper left")
ax.set_title("C) MOC vs MOO Equity Curves", fontsize=10)
ax.tick_params(axis="x", rotation=30)

# ── Panel D: AM vs PM session return by signal type ─────────────
ax = axes[1, 1]

# Aggregate AM vs PM returns by signal type for bar chart
bar_data = []
for period_name, pmask in [("Sel", session_df["trade_date_dt"] < SEL_END),
                            ("OOS", session_df["trade_date_dt"] >= SEL_END)]:
    pdata = session_df[pmask].copy()
    pdata["fwd_am_ret"] = pdata["am_ret"].shift(-1)
    pdata["fwd_pm_ret"] = pdata["pm_ret"].shift(-1)
    pdata = pdata.dropna(subset=["fwd_am_ret", "fwd_pm_ret", "spread_z"])

    for label, sig_mask, flip in [("SHORT", pdata["spread_z"] > K_ENTRY, -1),
                                   ("LONG", pdata["spread_z"] < -K_ENTRY, 1)]:
        s = pdata[sig_mask]
        if len(s) > 0:
            bar_data.append({
                "group": f"{period_name}\n{label}",
                "AM_avg": np.mean(s["fwd_am_ret"].values * flip),
                "PM_avg": np.mean(s["fwd_pm_ret"].values * flip),
                "n": len(s),
            })

if bar_data:
    bar_df = pd.DataFrame(bar_data)
    x = np.arange(len(bar_df))
    width = 0.35
    ax.bar(x - width/2, bar_df["AM_avg"], width, label="AM (9:30-11:00)",
           color="#3498db", alpha=0.8, edgecolor="white")
    ax.bar(x + width/2, bar_df["PM_avg"], width, label="PM (14:30-16:00)",
           color="#e67e22", alpha=0.8, edgecolor="white")
    ax.set_xticks(x)
    labels_with_n = [f"{g}\n(n={n})" for g, n in zip(bar_df["group"], bar_df["n"])]
    ax.set_xticklabels(labels_with_n, fontsize=8)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_ylabel("Avg Return (%)")
    ax.legend(fontsize=8)
ax.set_title("D) AM vs PM Session Returns by Signal Type", fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
print(f"  Figure saved to: {FIG_PATH}")
plt.close()


# ══════════════════════════════════════════════════════════════════
# 8. LOG TRIALS
# ══════════════════════════════════════════════════════════════════

print("\n" + "-" * 70)
print("  LOGGING TRIALS")
print("-" * 70)

# MOO OOS stats
moo_oos = trades_moo[trades_moo["entry_date"] >= SEL_END]
moo_oos_stats = compute_stats(moo_oos, label="MOO OOS")
log_trial("agent2", "S5_zexit_MOO",
          {"k_entry": K_ENTRY, "z_exit": Z_EXIT, "fill": "MOO", "max_hold": MAX_HOLD},
          moo_oos_stats)

# MOC OOS stats
moc_oos = trades_moc[trades_moc["entry_date"] >= SEL_END]
moc_oos_stats = compute_stats(moc_oos, label="MOC OOS")
log_trial("agent2", "S5_zexit_MOC",
          {"k_entry": K_ENTRY, "z_exit": Z_EXIT, "fill": "MOC", "max_hold": MAX_HOLD},
          moc_oos_stats)

# Overnight decomposition summary
if len(decomp_oos) > 0:
    on_share = decomp_oos["overnight_pnl_pct"].sum()
    id_share = decomp_oos["intraday_pnl_pct"].sum()
    total_gross = on_share + id_share
    log_trial("agent2", "overnight_decomp_OOS",
              {"overnight_total_pct": round(on_share, 2),
               "intraday_total_pct": round(id_share, 2),
               "overnight_share_pct": round(on_share / total_gross * 100, 1) if total_gross != 0 else 0},
              moo_oos_stats)

print("  Trials logged.")


# ══════════════════════════════════════════════════════════════════
# 9. CONCLUSION
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  CONCLUSION")
print("=" * 70)

# Compute key numbers for conclusion
if len(decomp_oos) > 0:
    oos_on_total = decomp_oos["overnight_pnl_pct"].sum()
    oos_id_total = decomp_oos["intraday_pnl_pct"].sum()
    oos_gross = oos_on_total + oos_id_total
    oos_on_avg = decomp_oos["overnight_pnl_pct"].mean()
    oos_id_avg = decomp_oos["intraday_pnl_pct"].mean()
else:
    oos_on_total = oos_id_total = oos_gross = oos_on_avg = oos_id_avg = 0

# Same for selection
if len(decomp_sel) > 0:
    sel_on_total = decomp_sel["overnight_pnl_pct"].sum()
    sel_id_total = decomp_sel["intraday_pnl_pct"].sum()
    sel_on_avg = decomp_sel["overnight_pnl_pct"].mean()
    sel_id_avg = decomp_sel["intraday_pnl_pct"].mean()
else:
    sel_on_total = sel_id_total = sel_on_avg = sel_id_avg = 0

print(f"""
  PnL DECOMPOSITION:
    Selection window:
      Overnight avg: {sel_on_avg:+.3f}%   Intraday avg: {sel_id_avg:+.3f}%
      Overnight tot: {sel_on_total:+.2f}%   Intraday tot: {sel_id_total:+.2f}%

    OOS 2026:
      Overnight avg: {oos_on_avg:+.3f}%   Intraday avg: {oos_id_avg:+.3f}%
      Overnight tot: {oos_on_total:+.2f}%   Intraday tot: {oos_id_total:+.2f}%
""")

# Determine which drives the edge
if abs(oos_on_total) > abs(oos_id_total):
    driver = "OVERNIGHT"
    driver_pct = oos_on_total / oos_gross * 100 if oos_gross != 0 else 0
else:
    driver = "INTRADAY"
    driver_pct = oos_id_total / oos_gross * 100 if oos_gross != 0 else 0

print(f"  DOMINANT COMPONENT: {driver} ({driver_pct:+.0f}% of gross PnL in OOS)")

# MOC verdict
moc_oos_avg = moc_oos["pnl_net_pct"].mean() if len(moc_oos) > 0 else 0
moo_oos_avg = moo_oos["pnl_net_pct"].mean() if len(moo_oos) > 0 else 0
moc_better = moc_oos_avg > moo_oos_avg
print(f"""
  MOC vs MOO:
    MOO OOS avg: {moo_oos_avg:+.3f}% (n={len(moo_oos)})
    MOC OOS avg: {moc_oos_avg:+.3f}% (n={len(moc_oos)})
    Verdict: {'MOC is BETTER — worth further testing' if moc_better else 'MOO is BETTER — stick with default'}
""")

# AM vs PM verdict
print("  AM vs PM SESSION:")
if bar_data:
    for bd in bar_data:
        if "OOS" in bd["group"]:
            print(f"    {bd['group'].replace(chr(10), ' ')}: AM={bd['AM_avg']:+.3f}%  PM={bd['PM_avg']:+.3f}%  n={bd['n']}")
print()

print("  AGENT 2 COMPLETE")
print("=" * 70)
