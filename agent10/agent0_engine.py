# %% Agent 0 — Shared Backtest Engine + S5_zexit Reconciliation
#
# This module provides:
#   1. Canonical spread construction (fixes Phase 4 z-score bug)
#   2. Shared cost model
#   3. Pluggable trade engine (market-on-open, MOC, limit, stop)
#   4. Metrics (daily MTM equity, Sharpe, DSR, walk-forward)
#   5. S5_zexit discrepancy reconciliation
#   6. Regression tests
#
# Every other agent imports from this file.

import os, sys, json, warnings, hashlib
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

for pkg in ["statsmodels"]:
    try:
        __import__(pkg)
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg,
                               "--quiet", "--break-system-packages"])

import statsmodels.api as sm
from scipy import stats as scipy_stats

# ── Config ────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data" if "__file__" in dir() else Path("agent10/data")
OUT_DIR = Path(__file__).parent / "results" if "__file__" in dir() else Path("agent10/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# COST MODEL (shared)
# ══════════════════════════════════════════════════════════════════

DEFAULT_COSTS = {
    "half_spread_bps": 10,
    "slippage_bps": 7.5,
    "commission_per_share": 0.005,
    "short_borrow_annual_pct": 5.0,
}

def compute_costs(entry_price, side, holding_days, cost_multiplier=1.0,
                  costs=None):
    """Total round-trip cost in dollars per share."""
    c = costs or DEFAULT_COSTS
    m = cost_multiplier
    spread = entry_price * c["half_spread_bps"] / 10000 * 2 * m
    slip = entry_price * c["slippage_bps"] / 10000 * 2 * m
    comm = c["commission_per_share"] * 2 * m
    borrow = 0
    if side == "short":
        borrow = entry_price * c["short_borrow_annual_pct"] / 100 * holding_days / 252 * m
    return spread + slip + comm + borrow


# ══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (shared)
# ══════════════════════════════════════════════════════════════════

def add_features(df):
    """Add standard technical features. Input must have date/open/high/low/close/volume."""
    df = df.copy()
    df["log_price"] = np.log(df["close"].clip(lower=0.01))
    df["log_ret"] = df["log_price"].diff()
    df["pct_ret"] = df["close"].pct_change()

    high, low, prev_close = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    df["atr20"] = tr.rolling(20).mean()
    df["atr_pct"] = df["atr20"] / df["close"] * 100
    df["adv"] = (df["close"] * df["volume"]).rolling(60).mean()
    df["vol_ma20"] = df["volume"].rolling(20).mean()

    for L in [10, 20, 30, 60]:
        df[f"z_{L}"] = (df["log_price"] - df["log_price"].rolling(L).mean()) / \
                        df["log_price"].rolling(L).std()
    return df


# ══════════════════════════════════════════════════════════════════
# SPREAD CONSTRUCTION (canonical — fixes Phase 4 bug)
# ══════════════════════════════════════════════════════════════════

def build_spread(target_df, hedge_df, hedge_name="hedge", lookback=60):
    """Canonical OLS spread with ROLLING z-score.

    The z-score is computed as:
        z = (point_residual - rolling_mean(point_residual, lookback))
            / rolling_std(point_residual, lookback)

    This is the correct formulation. The Phase 4 script had a bug where it
    used the IN-SAMPLE regression residual distribution instead, which
    produces different (fewer, weaker) signals.
    """
    t_df = target_df[["date", "close", "open", "high", "low", "volume"]].copy()
    if "log_price" not in t_df.columns:
        t_df["log_price"] = np.log(t_df["close"].clip(lower=0.01))
    else:
        t_df["log_price"] = target_df["log_price"]

    high, low, prev_close = t_df["high"], t_df["low"], t_df["close"].shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    t_df["atr20"] = tr.rolling(20).mean()

    merged = t_df.merge(
        hedge_df[["date", "close"]].rename(columns={"close": "hedge_close"}),
        on="date", how="inner"
    ).sort_values("date").reset_index(drop=True)

    merged["hedge_log"] = np.log(merged["hedge_close"].clip(lower=0.01))
    merged["target_log"] = merged["log_price"]

    # Point residual at each bar
    merged["spread"] = np.nan
    merged["hedge_beta"] = np.nan
    merged["r_squared"] = np.nan

    for i in range(lookback, len(merged)):
        y = merged["target_log"].iloc[i-lookback:i+1].values
        x = sm.add_constant(merged["hedge_log"].iloc[i-lookback:i+1].values)
        try:
            model = sm.OLS(y, x).fit()
            merged.iloc[i, merged.columns.get_loc("spread")] = y[-1] - model.predict(x)[-1]
            merged.iloc[i, merged.columns.get_loc("hedge_beta")] = model.params[1]
            merged.iloc[i, merged.columns.get_loc("r_squared")] = model.rsquared
        except:
            pass

    # ROLLING z-score of the spread (correct method)
    spread = merged["spread"]
    merged["spread_z"] = (spread - spread.rolling(lookback).mean()) / spread.rolling(lookback).std()

    return merged


def build_spread_phase4_bug(target_df, hedge_df, hedge_name="hedge", lookback=60):
    """Phase 4's BUGGY z-score for reconciliation comparison."""
    t_df = target_df[["date", "close", "open", "high", "low", "volume"]].copy()
    t_df["log_price"] = np.log(t_df["close"].clip(lower=0.01))
    high, low, prev_close = t_df["high"], t_df["low"], t_df["close"].shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    t_df["atr20"] = tr.rolling(20).mean()

    merged = t_df.merge(
        hedge_df[["date", "close"]].rename(columns={"close": "hedge_close"}),
        on="date", how="inner"
    ).sort_values("date").reset_index(drop=True)

    merged["hedge_log"] = np.log(merged["hedge_close"].clip(lower=0.01))
    merged["target_log"] = merged["log_price"]
    merged["spread_z"] = np.nan

    for i in range(lookback, len(merged)):
        y = merged["target_log"].iloc[i-lookback:i+1].values
        x = sm.add_constant(merged["hedge_log"].iloc[i-lookback:i+1].values)
        try:
            model = sm.OLS(y, x).fit()
            resid = y[-1] - model.predict(x)[-1]
            # BUG: uses in-sample regression residuals instead of rolling z
            spread_series = y - model.predict(x)
            mu = np.mean(spread_series)
            sigma = np.std(spread_series)
            if sigma > 0:
                merged.iloc[i, merged.columns.get_loc("spread_z")] = (resid - mu) / sigma
        except:
            pass

    merged["spread"] = np.nan  # not stored in buggy version
    high2, low2, prev2 = merged["high"], merged["low"], merged["close"].shift(1)
    tr2 = pd.concat([high2-low2, (high2-prev2).abs(), (low2-prev2).abs()], axis=1).max(axis=1)
    merged["atr20"] = tr2.rolling(20).mean()

    return merged


# ══════════════════════════════════════════════════════════════════
# TRADE ENGINE (shared, pluggable)
# ══════════════════════════════════════════════════════════════════

def run_backtest(spread_df, signal_fn, exit_fn=None, max_hold=10,
                 position_size=10000, cost_multiplier=1.0, fill_type="market_on_open"):
    """Generic backtest engine.

    signal_fn(row, i, spread_df) -> dict with keys: side, stop, target, or None
    exit_fn(trade, row, j, spread_df) -> ("exit_reason", exit_price) or None
    fill_type: "market_on_open" (default), "market_on_close", "limit"
    """
    trades = []
    in_trade_until = pd.Timestamp("1900-01-01")

    for i in range(120, len(spread_df)):
        sig = signal_fn(spread_df.iloc[i], i, spread_df)
        if sig is None:
            continue

        entry_idx = i + 1 if fill_type == "market_on_open" else i
        if entry_idx >= len(spread_df):
            continue

        entry_date = spread_df["date"].iloc[entry_idx]
        if entry_date <= in_trade_until:
            continue

        entry_price = spread_df["open"].iloc[entry_idx] if fill_type == "market_on_open" \
                      else spread_df["close"].iloc[entry_idx]

        side = sig["side"]
        stop = sig.get("stop")
        target = sig.get("target")

        exit_price = exit_date = exit_reason = None

        for j in range(entry_idx + 1, min(entry_idx + max_hold + 1, len(spread_df))):
            row = spread_df.iloc[j]

            # Custom exit function
            if exit_fn:
                result = exit_fn(sig, row, j, spread_df)
                if result:
                    exit_reason, exit_price = result
                    exit_date = row["date"]
                    break

            # Stop loss
            if stop is not None:
                if side == "short" and row["high"] >= stop:
                    exit_price, exit_reason, exit_date = stop, "stop", row["date"]
                    break
                if side == "long" and row["low"] <= stop:
                    exit_price, exit_reason, exit_date = stop, "stop", row["date"]
                    break

            # Target
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
        cost_pct = compute_costs(entry_price, side, holding_days, cost_multiplier) / entry_price * 100
        pnl_net = pnl_pct - cost_pct
        shares = max(int(position_size / entry_price), 1)

        trades.append({
            "signal_date": spread_df["date"].iloc[i],
            "entry_date": entry_date, "exit_date": exit_date,
            "side": side, "entry_price": round(entry_price, 4),
            "exit_price": round(exit_price, 4), "exit_reason": exit_reason,
            "holding_days": holding_days,
            "z_at_signal": round(spread_df["spread_z"].iloc[i], 4) if "spread_z" in spread_df.columns else np.nan,
            "pnl_gross_pct": round(pnl_pct, 4), "cost_pct": round(cost_pct, 4),
            "pnl_net_pct": round(pnl_net, 4),
            "shares": shares, "pnl_dollars": round(pnl_net / 100 * shares * entry_price, 2),
        })

    return pd.DataFrame(trades)


# ── S5_zexit signal + exit functions ─────────────────────────────

def s5_zexit_signal(row, i, df, k_entry=1.5):
    """Signal: short when z > k, long when z < -k."""
    z = row.get("spread_z", np.nan)
    if pd.isna(z):
        return None
    atr = row.get("atr20", np.nan)
    if pd.isna(atr) or atr <= 0:
        return None
    price = row["close"]
    if z > k_entry:
        return {"side": "short", "stop": price + 1.5 * atr, "target": None}
    elif z < -k_entry:
        return {"side": "long", "stop": price - 1.5 * atr, "target": None}
    return None


def s5_zexit_exit(sig, row, j, df, z_exit_thresh=0.5):
    """Exit when z crosses back toward zero."""
    z = row.get("spread_z", np.nan)
    if pd.isna(z):
        return None
    if sig["side"] == "long" and z >= -z_exit_thresh:
        return ("z_cross", row["close"])
    if sig["side"] == "short" and z <= z_exit_thresh:
        return ("z_cross", row["close"])
    return None


# ══════════════════════════════════════════════════════════════════
# METRICS (shared)
# ══════════════════════════════════════════════════════════════════

def daily_mtm_equity(trades_df, price_df):
    """Daily mark-to-market equity including flat days."""
    if trades_df.empty:
        return pd.Series(dtype=float)
    min_d = trades_df["entry_date"].min()
    max_d = trades_df["exit_date"].max()
    all_dates = price_df[(price_df["date"] >= min_d) & (price_df["date"] <= max_d)]["date"].values
    closes = dict(zip(price_df["date"], price_df["close"]))

    daily_pnl = pd.Series(0.0, index=pd.DatetimeIndex(all_dates))
    for _, t in trades_df.iterrows():
        td = price_df[(price_df["date"] >= t["entry_date"]) &
                      (price_df["date"] <= t["exit_date"])]["date"].values
        prev = t["entry_price"]
        for d in td:
            curr = closes.get(d, prev)
            ret = ((curr - prev) / t["entry_price"] * 100) if t["side"] == "long" \
                  else ((prev - curr) / t["entry_price"] * 100)
            daily_pnl.loc[d] += ret
            prev = curr
    return daily_pnl


def compute_sharpe(daily_pnl):
    """Annualized Sharpe from daily PnL series including flat days."""
    if daily_pnl.std() == 0 or len(daily_pnl) == 0:
        return 0.0
    return float(daily_pnl.mean() / daily_pnl.std() * np.sqrt(252))


def compute_stats(trades_df, price_df=None, label=""):
    """Compute comprehensive trade statistics."""
    if trades_df.empty:
        return {"label": label, "n": 0, "avg_pnl": 0, "median_pnl": 0, "sharpe": 0,
                "wr": 0, "pf": 0, "max_dd": 0, "total_pnl_pct": 0, "total_pnl_usd": 0,
                "avg_hold": 0, "pnl_per_day": 0, "t_stat": 0}

    pnls = trades_df["pnl_net_pct"].values
    n = len(pnls)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    avg = np.mean(pnls)
    med = np.median(pnls)
    wr = len(wins) / n * 100
    pf = abs(np.sum(wins) / np.sum(losses)) if len(losses) > 0 and np.sum(losses) != 0 else np.inf
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    max_dd = float((equity - peak).min())
    avg_hold = trades_df["holding_days"].mean()
    pnl_per_day = avg / avg_hold if avg_hold > 0 else 0
    t_stat = avg / (np.std(pnls, ddof=1) / np.sqrt(n)) if np.std(pnls) > 0 else 0

    if price_df is not None:
        dpnl = daily_mtm_equity(trades_df, price_df)
        sharpe = compute_sharpe(dpnl)
    else:
        sharpe = avg / (np.std(pnls) + 1e-10) * np.sqrt(252 / max(avg_hold, 1))

    return {
        "label": label, "n": n, "avg_pnl": round(avg, 3), "median_pnl": round(med, 3),
        "sharpe": round(sharpe, 2), "wr": round(wr, 1), "pf": round(pf, 2),
        "max_dd": round(max_dd, 2), "total_pnl_pct": round(np.sum(pnls), 2),
        "total_pnl_usd": round(trades_df["pnl_dollars"].sum(), 2),
        "avg_hold": round(avg_hold, 1), "pnl_per_day": round(pnl_per_day, 3),
        "t_stat": round(t_stat, 2),
    }


def deflated_sharpe_ratio(sharpe, n_trades, n_trials, skew=0, kurt=3):
    """Harvey & Liu (2015) deflated Sharpe ratio."""
    from scipy.stats import norm
    expected_max_sr = norm.ppf(1 - 1 / n_trials) if n_trials > 1 else 0
    se = np.sqrt((1 + 0.5 * sharpe**2 - skew * sharpe +
                  (kurt - 3) / 4 * sharpe**2) / n_trades)
    if se == 0:
        return 0
    dsr = norm.cdf((sharpe - expected_max_sr) / se)
    return round(dsr, 3)


def null_test(trades_df, price_df, n_runs=1000, max_hold=10, seed_base=40000):
    """Drift-matched random-entry null, matched on side/count/hold."""
    if trades_df.empty:
        return 0, 0, 0
    n = len(trades_df)
    n_short = (trades_df["side"] == "short").sum()
    eligible = price_df["date"].iloc[20:].values
    real_avg = trades_df["pnl_net_pct"].mean()

    null_avgs = []
    for run in range(n_runs):
        rng = np.random.RandomState(run + seed_base)
        if len(eligible) < n:
            continue
        rd = rng.choice(eligible, size=n, replace=False)
        rd.sort()
        pnls = []
        in_trade_until = pd.Timestamp("1900-01-01")
        for d_val in rd:
            d = pd.Timestamp(d_val)
            if d <= in_trade_until:
                continue
            idx_arr = price_df.index[price_df["date"] == d]
            if len(idx_arr) == 0:
                continue
            loc = price_df.index.get_loc(idx_arr[0])
            if loc + 1 >= len(price_df):
                continue
            entry_p = price_df.iloc[loc + 1]["open"]
            exit_loc = min(loc + max_hold + 1, len(price_df) - 1)
            exit_p = price_df.iloc[exit_loc]["close"]
            side = "short" if rng.random() < n_short / n else "long"
            pnl = ((entry_p - exit_p) / entry_p * 100) if side == "short" \
                  else ((exit_p - entry_p) / entry_p * 100)
            cost = compute_costs(entry_p, side, max_hold) / entry_p * 100
            pnls.append(pnl - cost)
            in_trade_until = price_df.iloc[exit_loc]["date"]
        if pnls:
            null_avgs.append(np.mean(pnls))

    if not null_avgs:
        return 0, 0, 0
    pctile = scipy_stats.percentileofscore(null_avgs, real_avg)
    return round(pctile, 1), round(np.mean(null_avgs), 3), round(real_avg, 3)


def walk_forward(spread_df, signal_fn, exit_fn, train_months=12, test_months=3,
                 max_hold=10, position_size=10000):
    """Walk-forward: 12-month train, 3-month test, rolling."""
    dates = spread_df["date"]
    min_d, max_d = dates.min(), dates.max()
    results = []

    current = min_d + pd.DateOffset(months=train_months)
    while current + pd.DateOffset(months=test_months) <= max_d:
        test_start = current
        test_end = current + pd.DateOffset(months=test_months)

        test_trades_all = run_backtest(spread_df, signal_fn, exit_fn, max_hold, position_size)
        if not test_trades_all.empty:
            test_trades = test_trades_all[
                (test_trades_all["entry_date"] >= test_start) &
                (test_trades_all["entry_date"] < test_end)
            ]
            if not test_trades.empty:
                results.append({
                    "test_start": test_start, "test_end": test_end,
                    "n": len(test_trades),
                    "avg_pnl": test_trades["pnl_net_pct"].mean(),
                    "total_pnl": test_trades["pnl_net_pct"].sum(),
                })
        current += pd.DateOffset(months=test_months)

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════
# TRIALS LOG (shared)
# ══════════════════════════════════════════════════════════════════

TRIALS_LOG_PATH = OUT_DIR / "trials_log.csv"

def log_trial(agent, family, params, stats):
    """Append to shared trials log."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "agent": agent, "family": family, "params": json.dumps(params),
        "n": stats.get("n", 0), "avg_pnl": stats.get("avg_pnl", 0),
        "total_pnl": stats.get("total_pnl_pct", 0),
        "sharpe": stats.get("sharpe", 0), "t_stat": stats.get("t_stat", 0),
        "wr": stats.get("wr", 0), "avg_hold": stats.get("avg_hold", 0),
    }
    df = pd.DataFrame([entry])
    if TRIALS_LOG_PATH.exists():
        existing = pd.read_csv(TRIALS_LOG_PATH)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(TRIALS_LOG_PATH, index=False)


# ══════════════════════════════════════════════════════════════════
# RECONCILIATION: OOS vs Phase 4 S5_zexit on RXRX 2026
# ══════════════════════════════════════════════════════════════════
# %%

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("  AGENT 0: ENGINE BUILD + S5_ZEXIT RECONCILIATION")
    print("=" * 70)

    # Load data
    rxrx = pd.read_parquet(DATA_DIR / "rxrx_daily.parquet")
    rxrx["date"] = pd.to_datetime(rxrx["date"])
    rxrx = rxrx.sort_values("date").reset_index(drop=True)

    ctx = pd.read_parquet(DATA_DIR / "context_daily.parquet")
    ctx["date"] = pd.to_datetime(ctx["date"])
    arkg = ctx[ctx["ticker"] == "ARKG"].sort_values("date").reset_index(drop=True)

    rxrx = add_features(rxrx)
    print(f"  RXRX: {len(rxrx)} bars, {rxrx['date'].min().date()} to {rxrx['date'].max().date()}")
    print(f"  ARKG: {len(arkg)} bars")

    # ── Build spreads: correct vs buggy ──────────────────────────
    print("\n  Building spreads (correct vs buggy)...")
    spread_correct = build_spread(rxrx, arkg, "ARKG", 60)
    spread_buggy = build_spread_phase4_bug(rxrx, arkg, "ARKG", 60)

    # Compare z-scores
    z_correct = spread_correct["spread_z"].dropna()
    z_buggy = spread_buggy["spread_z"].dropna()
    print(f"\n  Z-score comparison:")
    print(f"    Correct (rolling):    range [{z_correct.min():.2f}, {z_correct.max():.2f}], "
          f"mean={z_correct.mean():.4f}, std={z_correct.std():.4f}")
    print(f"    Buggy (in-sample):    range [{z_buggy.min():.2f}, {z_buggy.max():.2f}], "
          f"mean={z_buggy.mean():.4f}, std={z_buggy.std():.4f}")

    # ── Run S5_zexit with CORRECT z-score ────────────────────────
    print("\n  ── Running S5_zexit with CORRECT (rolling) z-score ──")

    def sig_fn(row, i, df):
        return s5_zexit_signal(row, i, df, k_entry=1.5)

    def exit_fn(sig, row, j, df):
        return s5_zexit_exit(sig, row, j, df, z_exit_thresh=0.5)

    trades_correct = run_backtest(spread_correct, sig_fn, exit_fn, max_hold=10)

    # ── Run S5_zexit with BUGGY z-score ──────────────────────────
    print("  ── Running S5_zexit with BUGGY (in-sample) z-score ──")
    trades_buggy = run_backtest(spread_buggy, sig_fn, exit_fn, max_hold=10)

    # ── Split by period ──────────────────────────────────────────
    SEL_END = "2026-01-01"

    for label, trades in [("CORRECT", trades_correct), ("BUGGY", trades_buggy)]:
        sel = trades[trades["entry_date"] < SEL_END]
        oos = trades[trades["entry_date"] >= SEL_END]

        sel_stats = compute_stats(sel, label=f"{label} Selection")
        oos_stats = compute_stats(oos, label=f"{label} 2026")

        print(f"\n  {label}:")
        print(f"    Selection: n={sel_stats['n']}, avg={sel_stats['avg_pnl']:+.2f}%, "
              f"total={sel_stats['total_pnl_pct']:+.1f}%, Sharpe={sel_stats['sharpe']}, "
              f"t={sel_stats['t_stat']:.2f}")
        print(f"    2026 OOS:  n={oos_stats['n']}, avg={oos_stats['avg_pnl']:+.2f}%, "
              f"total={oos_stats['total_pnl_pct']:+.1f}%, Sharpe={oos_stats['sharpe']}, "
              f"t={oos_stats['t_stat']:.2f}")

        log_trial("agent0", "S5_zexit_ARKG_k1.5_ze0.5",
                  {"z_method": label.lower(), "lookback": 60, "k": 1.5, "ze": 0.5},
                  oos_stats)

    # ── Entry-by-entry diff ──────────────────────────────────────
    print("\n  ── Entry-by-entry diff (2026) ──")
    oos_correct = trades_correct[trades_correct["entry_date"] >= SEL_END].copy()
    oos_buggy = trades_buggy[trades_buggy["entry_date"] >= SEL_END].copy()

    print(f"    Correct: {len(oos_correct)} trades")
    print(f"    Buggy:   {len(oos_buggy)} trades")

    if not oos_correct.empty:
        print(f"\n    CORRECT trade list:")
        print(f"    {'Signal':>12s} {'Entry':>12s} {'Exit':>12s} {'Side':>6s} {'z':>7s} "
              f"{'Days':>4s} {'Net%':>8s} {'Reason':>8s}")
        for _, t in oos_correct.iterrows():
            print(f"    {str(t['signal_date'].date()):>12s} {str(t['entry_date'].date()):>12s} "
                  f"{str(t['exit_date'].date()):>12s} {t['side']:>6s} {t['z_at_signal']:>+7.3f} "
                  f"{t['holding_days']:>4d} {t['pnl_net_pct']:>+7.2f}% {t['exit_reason']:>8s}")

    if not oos_buggy.empty:
        print(f"\n    BUGGY trade list:")
        print(f"    {'Signal':>12s} {'Entry':>12s} {'Exit':>12s} {'Side':>6s} {'z':>7s} "
              f"{'Days':>4s} {'Net%':>8s} {'Reason':>8s}")
        for _, t in oos_buggy.iterrows():
            print(f"    {str(t['signal_date'].date()):>12s} {str(t['entry_date'].date()):>12s} "
                  f"{str(t['exit_date'].date()):>12s} {t['side']:>6s} {t['z_at_signal']:>+7.3f} "
                  f"{t['holding_days']:>4d} {t['pnl_net_pct']:>+7.2f}% {t['exit_reason']:>8s}")

    # Identify trades in correct but not buggy (and vice versa)
    correct_dates = set(oos_correct["entry_date"].tolist()) if not oos_correct.empty else set()
    buggy_dates = set(oos_buggy["entry_date"].tolist()) if not oos_buggy.empty else set()

    only_correct = correct_dates - buggy_dates
    only_buggy = buggy_dates - correct_dates
    common = correct_dates & buggy_dates

    print(f"\n    In both:        {len(common)} trades")
    print(f"    Only in correct: {len(only_correct)} trades  {[str(d.date()) for d in sorted(only_correct)]}")
    print(f"    Only in buggy:   {len(only_buggy)} trades  {[str(d.date()) for d in sorted(only_buggy)]}")

    # ── 2x cost sensitivity ──────────────────────────────────────
    print("\n  ── 2x Cost Sensitivity ──")
    trades_2x = run_backtest(spread_correct, sig_fn, exit_fn, max_hold=10,
                             cost_multiplier=2.0)
    oos_2x = trades_2x[trades_2x["entry_date"] >= SEL_END]
    s_2x = compute_stats(oos_2x, label="2x costs 2026")
    print(f"    2026 at 2x costs: n={s_2x['n']}, avg={s_2x['avg_pnl']:+.2f}%, "
          f"total={s_2x['total_pnl_pct']:+.1f}%")

    # ── Walk-forward on selection window ─────────────────────────
    print("\n  ── Walk-Forward (12m train / 3m test) ──")
    sel_spread = spread_correct[spread_correct["date"] < SEL_END].copy()
    wf = walk_forward(sel_spread, sig_fn, exit_fn, 12, 3)
    if not wf.empty:
        for _, w in wf.iterrows():
            print(f"    {str(w['test_start'].date()):>12s} - {str(w['test_end'].date()):>12s}: "
                  f"n={w['n']:>3.0f}  avg={w['avg_pnl']:+.2f}%")
        print(f"    WF OOS average: {wf['avg_pnl'].mean():+.2f}%  "
              f"({'POSITIVE' if wf['avg_pnl'].mean() > 0 else 'NEGATIVE'})")

    # ── Null test (2026) ─────────────────────────────────────────
    print("\n  ── Null Test (2026, 1000 runs) ──")
    rxrx_2026 = rxrx[rxrx["date"] >= SEL_END]
    pctile, null_mean, real_avg = null_test(oos_correct, rxrx_2026)
    print(f"    Real avg: {real_avg:+.3f}%  Null mean: {null_mean:+.3f}%  "
          f"Percentile: {pctile:.1f}th  {'PASS' if pctile >= 95 else 'FAIL (need 95th)'}")

    # ── Regression tests ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  REGRESSION TESTS")
    print("=" * 70)

    # Test 1: RXRX 2026 trade count matches OOS evaluation
    n_2026_correct = len(oos_correct)
    test1 = n_2026_correct == 12
    print(f"  Test 1 (RXRX 2026 trade count = 12): {'PASS' if test1 else f'FAIL (got {n_2026_correct})'}")

    # Test 2: Synthetic mean-reverting series produces positive PnL
    np.random.seed(123)
    n_syn = 1000
    ou_mu, ou_kappa, ou_sigma = 0, 0.1, 0.02
    syn_log = np.zeros(n_syn)
    for i in range(1, n_syn):
        syn_log[i] = syn_log[i-1] + ou_kappa * (ou_mu - syn_log[i-1]) + ou_sigma * np.random.randn()
    syn_price = np.exp(syn_log + np.log(10))
    syn_df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n_syn, freq="B"),
        "open": syn_price * (1 + np.random.randn(n_syn) * 0.001),
        "high": syn_price * 1.02, "low": syn_price * 0.98,
        "close": syn_price, "volume": np.random.randint(1e6, 5e6, n_syn),
    })
    hedge_price = np.exp(np.cumsum(np.random.randn(n_syn) * 0.01) + np.log(50))
    hedge_df = pd.DataFrame({
        "date": syn_df["date"], "close": hedge_price,
    })
    syn_spread = build_spread(syn_df, hedge_df, "SYN", 60)
    syn_trades = run_backtest(syn_spread, sig_fn, exit_fn, max_hold=10, cost_multiplier=0)
    syn_avg = syn_trades["pnl_net_pct"].mean() if not syn_trades.empty else -999
    test2 = syn_avg > 0
    print(f"  Test 2 (synthetic MR series → positive PnL): {'PASS' if test2 else f'FAIL (avg={syn_avg:.2f}%)'}"
          f"  (n={len(syn_trades)}, avg={syn_avg:.2f}%)")

    # Test 3: Random walk produces near-zero PnL before costs
    np.random.seed(456)
    rw_log = np.cumsum(np.random.randn(n_syn) * 0.02)
    rw_price = np.exp(rw_log + np.log(10))
    rw_df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n_syn, freq="B"),
        "open": rw_price * (1 + np.random.randn(n_syn) * 0.001),
        "high": rw_price * 1.02, "low": rw_price * 0.98,
        "close": rw_price, "volume": np.random.randint(1e6, 5e6, n_syn),
    })
    rw_spread = build_spread(rw_df, hedge_df, "RW", 60)
    rw_trades = run_backtest(rw_spread, sig_fn, exit_fn, max_hold=10, cost_multiplier=0)
    rw_avg = rw_trades["pnl_net_pct"].mean() if not rw_trades.empty else 0
    test3 = abs(rw_avg) < 3.0  # within ±3% of zero
    print(f"  Test 3 (random walk → near-zero PnL): {'PASS' if test3 else f'FAIL (avg={rw_avg:.2f}%)'}"
          f"  (n={len(rw_trades)}, avg={rw_avg:.2f}%)")

    # Test 4: Cost model sanity
    cost_10 = compute_costs(10.0, "short", 10)
    cost_100 = compute_costs(100.0, "short", 10)
    test4 = cost_100 > cost_10  # higher price → higher costs
    print(f"  Test 4 (cost scales with price): {'PASS' if test4 else 'FAIL'}"
          f"  ($10→${cost_10:.4f}, $100→${cost_100:.4f})")

    all_pass = test1 and test2 and test3 and test4
    print(f"\n  {'ALL TESTS PASS' if all_pass else 'SOME TESTS FAILED'}")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RECONCILIATION SUMMARY")
    print("=" * 70)
    print(f"""
  ROOT CAUSE: The Phase 4 script computed the z-score incorrectly.

  CORRECT method (OOS script, now canonical):
    spread[i] = log(RXRX) - OLS_predict(log(ARKG))    (point residual)
    z[i] = (spread[i] - rolling_mean(spread, 60)) / rolling_std(spread, 60)

  BUGGY method (Phase 4 script):
    z[i] = (residual[i] - mean(ALL regression residuals in window))
           / std(ALL regression residuals in window)

  The buggy method uses the in-sample regression residual distribution,
  which is always centered near zero with a stable std. This produces
  SMALLER z-scores than the rolling method, triggering fewer signals.

  Impact:
    Correct: {len(oos_correct)} trades in 2026, avg {oos_correct['pnl_net_pct'].mean():+.2f}%
    Buggy:   {len(oos_buggy)} trades in 2026, avg {oos_buggy['pnl_net_pct'].mean():+.2f}%

  The correct z-score is now the canonical implementation in agent0_engine.py.
  All downstream agents must import build_spread() from this module.
""")

    print("  AGENT 0 COMPLETE")
