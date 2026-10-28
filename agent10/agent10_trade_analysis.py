# %% Agent 10 — Trade-Level PnL Analysis
# Concrete trading statistics: what would you make/lose trading these?
#
# For each viable strategy, shows:
#   - Full trade list with entry/exit dates, side, PnL
#   - Total PnL, avg trade PnL, median, best/worst trade
#   - Win rate, profit factor, payoff ratio
#   - Max consecutive wins/losses
#   - Monthly and yearly PnL breakdown
#   - Equity curve
#   - Long vs Short breakdown
#
# Requires: Phase 0 data
# Run as Colab cell.

import os, sys, json, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

for pkg in ["statsmodels"]:
    try:
        __import__(pkg)
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg,
                               "--quiet", "--break-system-packages"])

import statsmodels.api as sm

# ── Config ─────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data" if "__file__" in dir() else Path("agent10/data")
OUT_DIR = Path(__file__).parent / "results" if "__file__" in dir() else Path("agent10/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HOLDOUT_START = "2026-04-01"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

HALF_SPREAD_BPS = 10
SLIPPAGE_BPS = 7.5
COMMISSION_PER_SHARE = 0.005
SHORT_BORROW_ANNUAL_PCT = 5.0

POSITION_SIZE_USD = 10000  # dollars per trade

# ── Load Data ──────────────────────────────────────────────────────
# %%
print("Loading data...")
rxrx = pd.read_parquet(DATA_DIR / "rxrx_daily.parquet")
rxrx["date"] = pd.to_datetime(rxrx["date"])
rxrx = rxrx.sort_values("date").reset_index(drop=True)

rxrx["log_price"] = np.log(rxrx["close"])
rxrx["log_ret"] = rxrx["log_price"].diff()
rxrx["pct_ret"] = rxrx["close"].pct_change()

def compute_atr(df, period=20):
    high = df["high"]
    low = df["low"]
    close = df["close"].shift(1)
    tr = pd.concat([high - low, (high - close).abs(), (low - close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def rsi(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

rxrx["atr20"] = compute_atr(rxrx, 20)
rxrx["rsi2"] = rsi(rxrx["close"], 2)
rxrx["ibs"] = (rxrx["close"] - rxrx["low"]) / (rxrx["high"] - rxrx["low"])

for L in [10, 20, 30]:
    rxrx[f"z_{L}"] = (
        (rxrx["log_price"] - rxrx["log_price"].rolling(L).mean()) /
        rxrx["log_price"].rolling(L).std()
    )

rxrx_dev = rxrx[rxrx["date"] < HOLDOUT_START].copy()
print(f"Development: {len(rxrx_dev)} bars, {rxrx_dev['date'].min().date()} to {rxrx_dev['date'].max().date()}")
print(f"Position size: ${POSITION_SIZE_USD:,.0f} per trade")


# ── Trade Execution Engine ─────────────────────────────────────────
# %%
def apply_costs(entry_price, side, holding_days):
    spread = entry_price * HALF_SPREAD_BPS / 10000 * 2
    slip = entry_price * SLIPPAGE_BPS / 10000 * 2
    comm = COMMISSION_PER_SHARE * 2
    borrow = 0
    if side == "short":
        borrow = entry_price * SHORT_BORROW_ANNUAL_PCT / 100 * holding_days / 252
    return spread + slip + comm + borrow


def run_strategy(df, signals):
    """Execute signals, return detailed trade list."""
    trades = []
    date_to_idx = {d: i for i, d in enumerate(df["date"])}

    for sig in signals:
        if sig["date"] not in date_to_idx:
            continue
        sig_idx = date_to_idx[sig["date"]]
        entry_idx = sig_idx + 1
        if entry_idx >= len(df):
            continue

        entry_price = df["open"].iloc[entry_idx]
        entry_date = df["date"].iloc[entry_idx]
        max_hold = sig.get("time_stop", 7)
        exit_price = None
        exit_reason = None

        for j in range(entry_idx + 1, min(entry_idx + max_hold + 1, len(df))):
            if sig["side"] == "long":
                if sig.get("stop") and df["low"].iloc[j] <= sig["stop"]:
                    exit_price = sig["stop"]
                    exit_reason = "stop"
                    break
                if sig.get("target") and df["high"].iloc[j] >= sig["target"]:
                    exit_price = sig["target"]
                    exit_reason = "target"
                    break
            else:
                if sig.get("stop") and df["high"].iloc[j] >= sig["stop"]:
                    exit_price = sig["stop"]
                    exit_reason = "stop"
                    break
                if sig.get("target") and df["low"].iloc[j] <= sig["target"]:
                    exit_price = sig["target"]
                    exit_reason = "target"
                    break

        if exit_price is None:
            exit_idx = min(entry_idx + max_hold, len(df) - 1)
            exit_price = df["close"].iloc[exit_idx]
            exit_date = df["date"].iloc[exit_idx]
            exit_reason = "time_stop"
        else:
            exit_date = df["date"].iloc[j]

        holding_days = (exit_date - entry_date).days
        if sig["side"] == "long":
            pnl_pct = (exit_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - exit_price) / entry_price * 100

        cost_per_share = apply_costs(entry_price, sig["side"], holding_days)
        cost_pct = cost_per_share / entry_price * 100
        pnl_net_pct = pnl_pct - cost_pct

        shares = int(POSITION_SIZE_USD / entry_price)
        pnl_dollars = pnl_net_pct / 100 * shares * entry_price

        trades.append({
            "entry_date": entry_date,
            "exit_date": exit_date,
            "side": sig["side"],
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "exit_reason": exit_reason,
            "holding_days": holding_days,
            "pnl_gross_pct": round(pnl_pct, 3),
            "cost_pct": round(cost_pct, 3),
            "pnl_net_pct": round(pnl_net_pct, 3),
            "shares": shares,
            "pnl_dollars": round(pnl_dollars, 2),
        })

    return pd.DataFrame(trades)


def print_stats(trades_df, label):
    """Print comprehensive trading statistics."""
    if trades_df.empty:
        print(f"\n{'='*70}")
        print(f"  {label}: NO TRADES")
        print(f"{'='*70}")
        return

    n = len(trades_df)
    pnls = trades_df["pnl_net_pct"].values
    dollars = trades_df["pnl_dollars"].values
    wins = trades_df[trades_df["pnl_net_pct"] > 0]
    losses = trades_df[trades_df["pnl_net_pct"] <= 0]

    wr = len(wins) / n * 100
    avg_pnl = np.mean(pnls)
    med_pnl = np.median(pnls)
    avg_win = wins["pnl_net_pct"].mean() if len(wins) > 0 else 0
    avg_loss = losses["pnl_net_pct"].mean() if len(losses) > 0 else 0
    best = np.max(pnls)
    worst = np.min(pnls)
    total_pnl_pct = np.sum(pnls)
    total_pnl_usd = np.sum(dollars)

    pf = abs(wins["pnl_net_pct"].sum() / losses["pnl_net_pct"].sum()) if len(losses) > 0 and losses["pnl_net_pct"].sum() != 0 else np.inf
    payoff = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
    expectancy = avg_pnl
    expectancy_r = (wr/100 * payoff) - (1 - wr/100) if payoff != np.inf else np.inf

    # Consecutive wins/losses
    max_consec_win = 0
    max_consec_loss = 0
    curr_win = 0
    curr_loss = 0
    for p in pnls:
        if p > 0:
            curr_win += 1
            curr_loss = 0
            max_consec_win = max(max_consec_win, curr_win)
        else:
            curr_loss += 1
            curr_win = 0
            max_consec_loss = max(max_consec_loss, curr_loss)

    # Equity curve stats
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    max_dd = dd.min()
    avg_hold = trades_df["holding_days"].mean()

    # Sharpe (per-trade, annualized assuming avg hold)
    if np.std(pnls) > 0 and avg_hold > 0:
        sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252 / avg_hold)
    else:
        sharpe = 0

    # Cost drag
    avg_cost = trades_df["cost_pct"].mean()
    total_cost_usd = (trades_df["cost_pct"] / 100 * trades_df["shares"] * trades_df["entry_price"]).sum()

    # Exit reason breakdown
    exit_counts = trades_df["exit_reason"].value_counts()

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  Total trades:        {n}")
    print(f"  Win rate:            {wr:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"  Avg PnL:             {avg_pnl:+.3f}%  (${avg_pnl/100*POSITION_SIZE_USD:+.2f})")
    print(f"  Median PnL:          {med_pnl:+.3f}%")
    print(f"  Best trade:          {best:+.2f}%")
    print(f"  Worst trade:         {worst:+.2f}%")
    print(f"  Avg win:             {avg_win:+.3f}%")
    print(f"  Avg loss:            {avg_loss:+.3f}%")
    print(f"  Payoff ratio:        {payoff:.2f}x")
    print(f"  Profit factor:       {pf:.2f}")
    print(f"  Expectancy (R):      {expectancy_r:+.3f}")
    print(f"  Sharpe (ann.):       {sharpe:.2f}")
    print(f"  ─────────────────────────────────────")
    print(f"  TOTAL PnL:           {total_pnl_pct:+.2f}%  (${total_pnl_usd:+,.2f})")
    print(f"  Max drawdown:        {max_dd:.2f}%")
    print(f"  Avg holding days:    {avg_hold:.1f}")
    print(f"  Max consec wins:     {max_consec_win}")
    print(f"  Max consec losses:   {max_consec_loss}")
    print(f"  ─────────────────────────────────────")
    print(f"  Avg cost/trade:      {avg_cost:.3f}%")
    print(f"  Total costs:         ${total_cost_usd:,.2f}")
    print(f"  Exit reasons:        {dict(exit_counts)}")


def print_yearly(trades_df, label):
    """Print year-by-year breakdown."""
    if trades_df.empty:
        return
    trades_df = trades_df.copy()
    trades_df["year"] = trades_df["entry_date"].dt.year

    print(f"\n  Year-by-year: {label}")
    print(f"  {'Year':>6s} {'N':>5s} {'WR':>6s} {'AvgPnL':>9s} {'TotalPnL':>10s} {'Total$':>12s} {'Best':>8s} {'Worst':>8s}")
    print(f"  {'-'*65}")

    for yr in sorted(trades_df["year"].unique()):
        yr_df = trades_df[trades_df["year"] == yr]
        n = len(yr_df)
        wr = sum(yr_df["pnl_net_pct"] > 0) / n * 100
        avg = yr_df["pnl_net_pct"].mean()
        total = yr_df["pnl_net_pct"].sum()
        total_usd = yr_df["pnl_dollars"].sum()
        best = yr_df["pnl_net_pct"].max()
        worst = yr_df["pnl_net_pct"].min()
        print(f"  {yr:>6d} {n:>5d} {wr:>5.0f}% {avg:>+8.2f}% {total:>+9.2f}% ${total_usd:>+10,.2f} {best:>+7.2f}% {worst:>+7.2f}%")

    print(f"  {'-'*65}")
    total_all = trades_df["pnl_net_pct"].sum()
    total_usd_all = trades_df["pnl_dollars"].sum()
    print(f"  {'TOTAL':>6s} {len(trades_df):>5d} {sum(trades_df['pnl_net_pct']>0)/len(trades_df)*100:>5.0f}% "
          f"{trades_df['pnl_net_pct'].mean():>+8.2f}% {total_all:>+9.2f}% ${total_usd_all:>+10,.2f}")


def print_monthly(trades_df, label):
    """Print monthly PnL summary."""
    if trades_df.empty:
        return
    trades_df = trades_df.copy()
    trades_df["month"] = trades_df["entry_date"].dt.to_period("M")

    print(f"\n  Monthly PnL: {label}")
    print(f"  {'Month':>10s} {'N':>4s} {'AvgPnL':>9s} {'TotalPnL':>10s} {'Total$':>12s}")
    print(f"  {'-'*50}")

    monthly = trades_df.groupby("month").agg(
        n=("pnl_net_pct", "count"),
        avg=("pnl_net_pct", "mean"),
        total=("pnl_net_pct", "sum"),
        total_usd=("pnl_dollars", "sum"),
    )

    for idx, row in monthly.iterrows():
        print(f"  {str(idx):>10s} {int(row['n']):>4d} {row['avg']:>+8.2f}% {row['total']:>+9.2f}% ${row['total_usd']:>+10,.2f}")

    # Months with trades vs total months
    first = trades_df["entry_date"].min()
    last = trades_df["entry_date"].max()
    total_months = (last.year - first.year) * 12 + (last.month - first.month) + 1
    active_months = len(monthly)
    print(f"\n  Active months: {active_months} / {total_months} ({active_months/total_months*100:.0f}%)")
    print(f"  Avg trades/month: {len(trades_df)/total_months:.1f}")


# ── Signal Generators ──────────────────────────────────────────────
# %%

def signals_s4c(df, n_consec=5, side="combined"):
    signals = []
    consec_down = np.zeros(len(df))
    consec_up = np.zeros(len(df))
    for i in range(1, len(df)):
        if df["close"].iloc[i] < df["close"].iloc[i-1]:
            consec_down[i] = consec_down[i-1] + 1
        if df["close"].iloc[i] > df["close"].iloc[i-1]:
            consec_up[i] = consec_up[i-1] + 1
    for i in range(20, len(df)):
        atr = df["atr20"].iloc[i]
        if pd.isna(atr):
            continue
        price = df["close"].iloc[i]
        if side in ("long", "combined") and consec_down[i] >= n_consec:
            signals.append({
                "date": df["date"].iloc[i], "side": "long",
                "stop": price - 1.5 * atr, "target": None, "time_stop": 5,
            })
        if side in ("short", "combined") and consec_up[i] >= n_consec:
            signals.append({
                "date": df["date"].iloc[i], "side": "short",
                "stop": price + 1.5 * atr, "target": None, "time_stop": 5,
            })
    return signals


def signals_s1(df, lookback=20, k_thresh=2.0, side="combined"):
    signals = []
    z_col = f"z_{lookback}"
    if z_col not in df.columns:
        df[z_col] = (
            (df["log_price"] - df["log_price"].rolling(lookback).mean()) /
            df["log_price"].rolling(lookback).std()
        )
    for i in range(lookback + 1, len(df)):
        z = df[z_col].iloc[i]
        if pd.isna(z):
            continue
        atr = df["atr20"].iloc[i]
        if pd.isna(atr):
            continue
        price = df["close"].iloc[i]
        if side in ("long", "combined") and z < -k_thresh:
            signals.append({
                "date": df["date"].iloc[i], "side": "long",
                "stop": price - 1.5 * atr,
                "target": price + abs(z) * atr * 0.5,
                "time_stop": 7,
            })
        if side in ("short", "combined") and z > k_thresh:
            signals.append({
                "date": df["date"].iloc[i], "side": "short",
                "stop": price + 1.5 * atr,
                "target": price - abs(z) * atr * 0.5,
                "time_stop": 7,
            })
    return signals


def signals_s4a(df, rsi_low=10, rsi_high=90, side="combined"):
    signals = []
    if "rsi2" not in df.columns:
        df["rsi2"] = rsi(df["close"], 2)
    for i in range(20, len(df)):
        r2 = df["rsi2"].iloc[i]
        if pd.isna(r2):
            continue
        atr = df["atr20"].iloc[i]
        if pd.isna(atr):
            continue
        price = df["close"].iloc[i]
        if side in ("long", "combined") and r2 < rsi_low:
            signals.append({
                "date": df["date"].iloc[i], "side": "long",
                "stop": price - 1.5 * atr, "target": None, "time_stop": 5,
            })
        if side in ("short", "combined") and r2 > rsi_high:
            signals.append({
                "date": df["date"].iloc[i], "side": "short",
                "stop": price + 1.5 * atr, "target": None, "time_stop": 5,
            })
    return signals


# ══════════════════════════════════════════════════════════════════
# RUN ALL STRATEGIES
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  TRADE-LEVEL PnL ANALYSIS — RXRX Development Period")
print(f"  Position size: ${POSITION_SIZE_USD:,} per trade")
print("=" * 70)

strategy_configs = [
    ("S4c combined (5 consec)", lambda df: signals_s4c(df, 5, "combined")),
    ("S4c SHORT only (5 consec)", lambda df: signals_s4c(df, 5, "short")),
    ("S4c LONG only (5 consec)", lambda df: signals_s4c(df, 5, "long")),
    ("S4c combined (4 consec)", lambda df: signals_s4c(df, 4, "combined")),
    ("S4c combined (3 consec)", lambda df: signals_s4c(df, 3, "combined")),
    ("S1 combined (L=20, k=2.0)", lambda df: signals_s1(df, 20, 2.0, "combined")),
    ("S1 combined (L=20, k=1.5)", lambda df: signals_s1(df, 20, 1.5, "combined")),
    ("S4a combined (RSI2 10/90)", lambda df: signals_s4a(df, 10, 90, "combined")),
    ("S4a LONG only (RSI2 < 10)", lambda df: signals_s4a(df, 10, 90, "long")),
]

all_trades = {}

for name, sig_func in strategy_configs:
    sigs = sig_func(rxrx_dev)
    trades_df = run_strategy(rxrx_dev, sigs)
    all_trades[name] = trades_df

    print_stats(trades_df, name)
    print_yearly(trades_df, name)

    # Long vs Short breakdown if combined
    if not trades_df.empty and trades_df["side"].nunique() > 1:
        longs = trades_df[trades_df["side"] == "long"]
        shorts = trades_df[trades_df["side"] == "short"]
        print(f"\n  Side breakdown: {name}")
        print(f"    LONG:  n={len(longs):3d}  avg={longs['pnl_net_pct'].mean():+.3f}%  "
              f"WR={sum(longs['pnl_net_pct']>0)/len(longs)*100:.0f}%  "
              f"total=${longs['pnl_dollars'].sum():+,.2f}" if len(longs) > 0 else "    LONG: 0 trades")
        print(f"    SHORT: n={len(shorts):3d}  avg={shorts['pnl_net_pct'].mean():+.3f}%  "
              f"WR={sum(shorts['pnl_net_pct']>0)/len(shorts)*100:.0f}%  "
              f"total=${shorts['pnl_dollars'].sum():+,.2f}" if len(shorts) > 0 else "    SHORT: 0 trades")


# ══════════════════════════════════════════════════════════════════
# EQUITY CURVES
# ══════════════════════════════════════════════════════════════════
# %%
print("\n\nPlotting equity curves...")

fig, axes = plt.subplots(3, 1, figsize=(16, 16))
fig.suptitle(f"RXRX Strategy Equity Curves — ${POSITION_SIZE_USD:,}/trade", fontsize=14, fontweight="bold")

# Top strategies equity curves (cumulative PnL %)
ax = axes[0]
for name in ["S4c combined (5 consec)", "S4c SHORT only (5 consec)", "S4c LONG only (5 consec)"]:
    df = all_trades[name]
    if df.empty:
        continue
    equity = df["pnl_net_pct"].cumsum()
    ax.plot(df["entry_date"], equity, label=name, lw=1.5)
ax.axhline(0, color="black", lw=0.5, ls="--")
ax.set_ylabel("Cumulative PnL (%)")
ax.set_title("S4c Variants — Cumulative PnL %")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Dollar equity curves
ax = axes[1]
for name in ["S4c combined (5 consec)", "S1 combined (L=20, k=2.0)", "S4a combined (RSI2 10/90)"]:
    df = all_trades[name]
    if df.empty:
        continue
    equity_usd = df["pnl_dollars"].cumsum()
    ax.plot(df["entry_date"], equity_usd, label=name, lw=1.5)
ax.axhline(0, color="black", lw=0.5, ls="--")
ax.set_ylabel("Cumulative PnL ($)")
ax.set_title(f"Best Strategies — Dollar PnL (${POSITION_SIZE_USD:,}/trade)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Drawdown
ax = axes[2]
for name in ["S4c combined (5 consec)", "S4c SHORT only (5 consec)"]:
    df = all_trades[name]
    if df.empty:
        continue
    equity = df["pnl_net_pct"].cumsum().values
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    ax.fill_between(df["entry_date"], dd, 0, alpha=0.3, label=name)
    ax.plot(df["entry_date"], dd, lw=0.8)
ax.set_ylabel("Drawdown (%)")
ax.set_title("Drawdown from Peak")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "trade_analysis_equity.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved trade_analysis_equity.png")


# ══════════════════════════════════════════════════════════════════
# TRADE DISTRIBUTION
# ══════════════════════════════════════════════════════════════════
# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Trade PnL Distributions", fontsize=14, fontweight="bold")

for idx, name in enumerate(["S4c combined (5 consec)", "S4c SHORT only (5 consec)",
                              "S1 combined (L=20, k=2.0)", "S4a combined (RSI2 10/90)"]):
    ax = axes.flat[idx]
    df = all_trades[name]
    if df.empty:
        ax.set_title(f"{name}: no trades")
        continue
    pnls = df["pnl_net_pct"].values
    colors = ["green" if p > 0 else "red" for p in pnls]
    ax.bar(range(len(pnls)), pnls, color=colors, alpha=0.7, width=1.0)
    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(np.mean(pnls), color="blue", ls="--", lw=1, label=f"Avg: {np.mean(pnls):+.2f}%")
    ax.set_title(f"{name} (n={len(pnls)})")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("PnL (%)")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR / "trade_analysis_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved trade_analysis_distribution.png")


# ══════════════════════════════════════════════════════════════════
# SUMMARY COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  STRATEGY COMPARISON SUMMARY")
print("=" * 70)
print(f"\n  {'Strategy':40s} {'N':>5s} {'WR':>6s} {'AvgPnL':>9s} {'TotPnL%':>9s} {'TotPnL$':>12s} {'Sharpe':>7s} {'MaxDD':>7s}")
print(f"  {'-'*95}")

for name, df in all_trades.items():
    if df.empty:
        print(f"  {name:40s}     0")
        continue
    n = len(df)
    wr = sum(df["pnl_net_pct"] > 0) / n * 100
    avg = df["pnl_net_pct"].mean()
    total = df["pnl_net_pct"].sum()
    total_usd = df["pnl_dollars"].sum()
    equity = df["pnl_net_pct"].cumsum().values
    peak = np.maximum.accumulate(equity)
    max_dd = (equity - peak).min()
    avg_hold = df["holding_days"].mean()
    sharpe = avg / (df["pnl_net_pct"].std() + 1e-10) * np.sqrt(252 / avg_hold) if avg_hold > 0 else 0

    print(f"  {name:40s} {n:>5d} {wr:>5.0f}% {avg:>+8.2f}% {total:>+8.1f}% ${total_usd:>+10,.0f} {sharpe:>7.2f} {max_dd:>+6.1f}%")

# Trade list for the best strategy
# %%
best_name = "S4c combined (5 consec)"
best_df = all_trades[best_name]

if not best_df.empty:
    print(f"\n\n{'='*70}")
    print(f"  FULL TRADE LIST: {best_name}")
    print(f"{'='*70}")
    print(f"  {'#':>3s} {'Entry':>12s} {'Exit':>12s} {'Side':>6s} {'EntryP':>8s} {'ExitP':>8s} "
          f"{'Reason':>10s} {'Days':>5s} {'PnL%':>8s} {'PnL$':>10s}")
    print(f"  {'-'*95}")

    for i, row in best_df.iterrows():
        print(f"  {i+1:>3d} {str(row['entry_date'].date()):>12s} {str(row['exit_date'].date()):>12s} "
              f"{row['side']:>6s} ${row['entry_price']:>6.2f} ${row['exit_price']:>6.2f} "
              f"{row['exit_reason']:>10s} {row['holding_days']:>5d} {row['pnl_net_pct']:>+7.2f}% "
              f"${row['pnl_dollars']:>+8.2f}")

    print(f"  {'-'*95}")
    print(f"  TOTAL: {len(best_df)} trades, "
          f"PnL: {best_df['pnl_net_pct'].sum():+.2f}% / "
          f"${best_df['pnl_dollars'].sum():+,.2f}")

print("\n\nTrade analysis complete.")
print("Review plots in agent10/results/")
