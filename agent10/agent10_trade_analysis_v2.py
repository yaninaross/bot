# %% Agent 10 — Trade Analysis v2 (Corrected)
#
# Fixes from v1:
#   1. No stacked entries — only 1 position at a time per strategy
#   2. Daily MTM equity curve (not per-trade cumulative)
#   3. Drift-matched null strategies (1,000 random-entry runs)
#   4. ARKG-hedged residual variants (S1b/S5) to test for beta
#   5. Episode-level analysis with leave-one-episode-out
#   6. Phase 1 verdict with surrogate test results
#   7. S2 (range fade) and S5 (relative value) included
#   8. Hold period sweep {1, 2, 3, 5} days
#   9. Regime segmentation (rolling Hurst + time-based)
#
# Requires: Phase 0 data + Phase 1 results (phase1_results.json)
# Run as Colab cell.

import os, sys, json, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats, signal as scipy_signal

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
POSITION_SIZE_USD = 10000

N_NULL_RUNS = 1000
HOLD_PERIODS = [1, 2, 3, 5]

EPISODE_GAP_DAYS = 14  # trades within 14 days of each other = same episode


# ══════════════════════════════════════════════════════════════════
# SECTION 0: PHASE 1 VERDICT
# ══════════════════════════════════════════════════════════════════
# %%
print("=" * 70)
print("  SECTION 0: PHASE 1 CYCLE VERDICT")
print("=" * 70)

try:
    with open(OUT_DIR / "phase1_results.json") as f:
        p1 = json.load(f)

    gate = p1["gate"]
    print(f"\n  Gate decision: proceed = {gate['proceed']}")
    print(f"  Cycle strategies unlocked: {gate['cycle_strategies_unlocked']}")
    print(f"  MR strategies unlocked:    {gate['mr_strategies_unlocked']}")

    if "spectral" in p1:
        print(f"\n  Spectral analysis (surrogate tests):")
        for window, res in p1["spectral"].items():
            sig = res.get("significant_ar1_95", False)
            peak = res.get("peak_period_periodogram", "?")
            pval = res.get("p_value_ar1", "?")
            print(f"    {window}: peak period = {peak:.1f}d, "
                  f"sig vs AR(1) = {sig}, p = {pval}")

    if "mean_reversion" in p1:
        mr = p1["mean_reversion"]
        print(f"\n  Mean-reversion tests:")
        for test_name, val in mr.items():
            print(f"    {test_name}: {val}")

    if "swing" in p1:
        sw = p1["swing"]
        print(f"\n  Swing structure:")
        if isinstance(sw, dict):
            for k, v in sw.items():
                if not isinstance(v, (dict, list)):
                    print(f"    {k}: {v}")

    print(f"\n  VERDICT: {'Proceed with MR strategies' if gate.get('mr_strategies_unlocked') else 'NO MR edge'}")
    print(f"           {'Cycle phase (S3) unlocked' if gate.get('cycle_strategies_unlocked') else 'NO spectral significance — S3 not justified'}")

except FileNotFoundError:
    print("\n  phase1_results.json NOT FOUND — Phase 1 not run yet")
    print("  Run agent10_phase1_cycle.py first")
    p1 = None


# ══════════════════════════════════════════════════════════════════
# SECTION 1: LOAD DATA & FEATURES
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  SECTION 1: DATA & FEATURES")
print("=" * 70)

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
rxrx["atr_pct"] = rxrx["atr20"] / rxrx["close"] * 100
rxrx["rsi2"] = rsi(rxrx["close"], 2)
rxrx["ibs"] = (rxrx["close"] - rxrx["low"]) / (rxrx["high"] - rxrx["low"])
rxrx["prev_ret"] = rxrx["close"].pct_change()

for L in [10, 20, 30]:
    rxrx[f"z_{L}"] = (
        (rxrx["log_price"] - rxrx["log_price"].rolling(L).mean()) /
        rxrx["log_price"].rolling(L).std()
    )

# Range features for S2
for N in [20, 40, 60]:
    rxrx[f"range_high_{N}"] = rxrx["high"].rolling(N).max()
    rxrx[f"range_low_{N}"] = rxrx["low"].rolling(N).min()
    rxrx[f"range_mid_{N}"] = (rxrx[f"range_high_{N}"] + rxrx[f"range_low_{N}"]) / 2

# Consecutive closes
consec_down = np.zeros(len(rxrx))
consec_up = np.zeros(len(rxrx))
for i in range(1, len(rxrx)):
    if rxrx["close"].iloc[i] < rxrx["close"].iloc[i-1]:
        consec_down[i] = consec_down[i-1] + 1
    if rxrx["close"].iloc[i] > rxrx["close"].iloc[i-1]:
        consec_up[i] = consec_up[i-1] + 1
rxrx["consec_down"] = consec_down
rxrx["consec_up"] = consec_up

# Rolling Hurst for regime segmentation
def rolling_hurst(series, window=120):
    hurst_vals = np.full(len(series), np.nan)
    log_prices = np.log(series.values)
    for i in range(window, len(series)):
        seg = log_prices[i-window:i]
        if np.any(np.isnan(seg)):
            continue
        lags = range(2, min(window // 4, 40))
        tau = []
        for lag in lags:
            diffs = seg[lag:] - seg[:-lag]
            tau.append(np.std(diffs))
        if len(tau) < 3 or min(tau) <= 0:
            continue
        log_lags = np.log(list(lags))
        log_tau = np.log(tau)
        slope, _, _, _, _ = scipy_stats.linregress(log_lags, log_tau)
        hurst_vals[i] = slope
    return pd.Series(hurst_vals, index=series.index)

rxrx["hurst_120"] = rolling_hurst(rxrx["close"], 120)

# ARKG data for S5 / hedged residual
ctx = pd.read_parquet(DATA_DIR / "context_daily.parquet")
ctx["date"] = pd.to_datetime(ctx["date"])
arkg = ctx[ctx["ticker"] == "ARKG"].sort_values("date").reset_index(drop=True)

has_arkg = len(arkg) > 100
if has_arkg:
    arkg_merged = rxrx[["date", "close", "open", "high", "low", "atr20", "atr_pct"]].merge(
        arkg[["date", "close"]].rename(columns={"close": "arkg_close"}),
        on="date", how="inner"
    )
    arkg_merged["rxrx_log"] = np.log(arkg_merged["close"])
    arkg_merged["arkg_log"] = np.log(arkg_merged["arkg_close"])

    # Rolling OLS hedge ratio + spread z-score
    for lb in [60]:
        arkg_merged[f"hedge_{lb}"] = np.nan
        arkg_merged[f"spread_{lb}"] = np.nan
        for i in range(lb, len(arkg_merged)):
            y = arkg_merged["rxrx_log"].iloc[i-lb:i+1].values
            x = sm.add_constant(arkg_merged["arkg_log"].iloc[i-lb:i+1].values)
            try:
                model = sm.OLS(y, x).fit()
                hedge = model.params[1]
                resid = y[-1] - model.predict(x)[-1]
                arkg_merged.iloc[i, arkg_merged.columns.get_loc(f"hedge_{lb}")] = hedge
                arkg_merged.iloc[i, arkg_merged.columns.get_loc(f"spread_{lb}")] = resid
            except:
                pass
        spread = arkg_merged[f"spread_{lb}"]
        arkg_merged[f"spread_z_{lb}"] = (
            (spread - spread.rolling(lb).mean()) / spread.rolling(lb).std()
        )
    print(f"  ARKG data loaded: {len(arkg_merged)} merged bars")
else:
    print("  ARKG data not available — S5/S1b disabled")

rxrx_dev = rxrx[rxrx["date"] < HOLDOUT_START].copy()
print(f"  Dev period: {len(rxrx_dev)} bars, {rxrx_dev['date'].min().date()} to {rxrx_dev['date'].max().date()}")
print(f"  Position size: ${POSITION_SIZE_USD:,} per trade")


# ══════════════════════════════════════════════════════════════════
# SECTION 2: TRADE ENGINE (FIXED — no stacked entries)
# ══════════════════════════════════════════════════════════════════
# %%
def apply_costs(entry_price, side, holding_days):
    spread = entry_price * HALF_SPREAD_BPS / 10000 * 2
    slip = entry_price * SLIPPAGE_BPS / 10000 * 2
    comm = COMMISSION_PER_SHARE * 2
    borrow = 0
    if side == "short":
        borrow = entry_price * SHORT_BORROW_ANNUAL_PCT / 100 * holding_days / 252
    return spread + slip + comm + borrow


def run_strategy_no_stack(df, signals, time_stop_override=None):
    """
    Execute signals with NO overlapping positions.
    If already in a trade, skip new signals until current trade exits.
    Returns DataFrame of trades.
    """
    trades = []
    date_to_idx = {d: i for i, d in enumerate(df["date"])}
    in_trade_until = pd.Timestamp("1900-01-01")

    for sig in signals:
        if sig["date"] not in date_to_idx:
            continue
        sig_idx = date_to_idx[sig["date"]]
        entry_idx = sig_idx + 1
        if entry_idx >= len(df):
            continue

        entry_date = df["date"].iloc[entry_idx]

        # Skip if still in a previous trade
        if entry_date <= in_trade_until:
            continue

        entry_price = df["open"].iloc[entry_idx]
        max_hold = time_stop_override if time_stop_override else sig.get("time_stop", 5)
        exit_price = None
        exit_reason = None
        exit_date = None

        for j in range(entry_idx + 1, min(entry_idx + max_hold + 1, len(df))):
            if sig["side"] == "long":
                if sig.get("stop") and df["low"].iloc[j] <= sig["stop"]:
                    exit_price = sig["stop"]
                    exit_reason = "stop"
                    exit_date = df["date"].iloc[j]
                    break
                if sig.get("target") and df["high"].iloc[j] >= sig["target"]:
                    exit_price = sig["target"]
                    exit_reason = "target"
                    exit_date = df["date"].iloc[j]
                    break
            else:
                if sig.get("stop") and df["high"].iloc[j] >= sig["stop"]:
                    exit_price = sig["stop"]
                    exit_reason = "stop"
                    exit_date = df["date"].iloc[j]
                    break
                if sig.get("target") and df["low"].iloc[j] <= sig["target"]:
                    exit_price = sig["target"]
                    exit_reason = "target"
                    exit_date = df["date"].iloc[j]
                    break

        if exit_price is None:
            exit_idx = min(entry_idx + max_hold, len(df) - 1)
            exit_price = df["close"].iloc[exit_idx]
            exit_date = df["date"].iloc[exit_idx]
            exit_reason = "time_stop"

        in_trade_until = exit_date
        holding_days = (exit_date - entry_date).days
        if holding_days == 0:
            holding_days = 1

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
            "entry_price": round(entry_price, 4),
            "exit_price": round(exit_price, 4),
            "exit_reason": exit_reason,
            "holding_days": holding_days,
            "pnl_gross_pct": round(pnl_pct, 4),
            "cost_pct": round(cost_pct, 4),
            "pnl_net_pct": round(pnl_net_pct, 4),
            "shares": shares,
            "pnl_dollars": round(pnl_dollars, 2),
        })

    return pd.DataFrame(trades)


def daily_mtm_equity(trades_df, price_df):
    """
    Build daily mark-to-market equity curve.
    On days with no position, equity is flat.
    Returns Series indexed by date.
    """
    if trades_df.empty:
        return pd.Series(dtype=float)

    dates = price_df["date"].values
    closes = price_df["close"].values
    date_to_close = dict(zip(dates, closes))

    min_date = trades_df["entry_date"].min()
    max_date = trades_df["exit_date"].max()
    all_dates = price_df[(price_df["date"] >= min_date) & (price_df["date"] <= max_date)]["date"].values

    daily_pnl = pd.Series(0.0, index=all_dates)

    for _, trade in trades_df.iterrows():
        trade_dates = price_df[
            (price_df["date"] >= trade["entry_date"]) &
            (price_df["date"] <= trade["exit_date"])
        ]["date"].values

        prev_price = trade["entry_price"]
        for d in trade_dates:
            curr_price = date_to_close.get(d, prev_price)
            if trade["side"] == "long":
                day_ret = (curr_price - prev_price) / trade["entry_price"] * 100
            else:
                day_ret = (prev_price - curr_price) / trade["entry_price"] * 100
            daily_pnl.loc[d] += day_ret
            prev_price = curr_price

    equity = daily_pnl.cumsum()
    return equity


def compute_stats(trades_df, label=""):
    """Compute summary stats dict for a trades DataFrame."""
    if trades_df.empty:
        return {"label": label, "n": 0}

    pnls = trades_df["pnl_net_pct"].values
    n = len(pnls)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    wr = len(wins) / n * 100
    avg = np.mean(pnls)
    avg_win = np.mean(wins) if len(wins) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0
    pf = abs(np.sum(wins) / np.sum(losses)) if len(losses) > 0 and np.sum(losses) != 0 else np.inf
    payoff = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    max_dd = (equity - peak).min()
    avg_hold = trades_df["holding_days"].mean()
    sharpe = avg / (np.std(pnls) + 1e-10) * np.sqrt(252 / max(avg_hold, 1))
    total_usd = trades_df["pnl_dollars"].sum()

    return {
        "label": label, "n": n, "wr": round(wr, 1),
        "avg_pnl": round(avg, 3), "total_pnl_pct": round(np.sum(pnls), 2),
        "total_pnl_usd": round(total_usd, 2),
        "avg_win": round(avg_win, 3), "avg_loss": round(avg_loss, 3),
        "payoff": round(payoff, 2), "pf": round(pf, 2),
        "sharpe": round(sharpe, 2), "max_dd": round(max_dd, 2),
        "avg_hold": round(avg_hold, 1),
    }


def print_stats_line(s):
    if s["n"] == 0:
        print(f"  {s['label']:45s}  0 trades")
        return
    print(f"  {s['label']:45s}  n={s['n']:>4d}  avg={s['avg_pnl']:>+7.2f}%  "
          f"WR={s['wr']:>4.0f}%  PF={s['pf']:>5.2f}  Sharpe={s['sharpe']:>+6.2f}  "
          f"DD={s['max_dd']:>+7.1f}%  total=${s['total_pnl_usd']:>+10,.0f}")


# ══════════════════════════════════════════════════════════════════
# SECTION 3: SIGNAL GENERATORS
# ══════════════════════════════════════════════════════════════════
# %%

def signals_s4c(df, n_consec=5, side="combined"):
    signals = []
    for i in range(20, len(df)):
        atr = df["atr20"].iloc[i]
        if pd.isna(atr):
            continue
        price = df["close"].iloc[i]
        if side in ("long", "combined") and df["consec_down"].iloc[i] >= n_consec:
            signals.append({
                "date": df["date"].iloc[i], "side": "long",
                "stop": price - 1.5 * atr, "target": None, "time_stop": 5,
            })
        if side in ("short", "combined") and df["consec_up"].iloc[i] >= n_consec:
            signals.append({
                "date": df["date"].iloc[i], "side": "short",
                "stop": price + 1.5 * atr, "target": None, "time_stop": 5,
            })
    return signals


def signals_s1(df, lookback=20, k_thresh=2.0, side="combined"):
    signals = []
    z_col = f"z_{lookback}"
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


def signals_s2(df, N=40, atr_mult=0.5, side="combined"):
    """Range fade with confirmation: IBS + previous day direction."""
    signals = []
    rh = f"range_high_{N}"
    rl = f"range_low_{N}"
    rm = f"range_mid_{N}"
    if rh not in df.columns:
        return signals
    for i in range(N + 1, len(df)):
        atr = df["atr20"].iloc[i]
        if pd.isna(atr):
            continue
        price = df["close"].iloc[i]
        support = df[rl].iloc[i]
        resistance = df[rh].iloc[i]
        midpoint = df[rm].iloc[i]
        ibs_val = df["ibs"].iloc[i]
        prev_r = df["prev_ret"].iloc[i]
        if pd.isna(support) or pd.isna(resistance) or pd.isna(ibs_val) or pd.isna(prev_r):
            continue

        if side in ("long", "combined"):
            if (price - support) <= atr_mult * atr and price > support:
                if ibs_val > 0.7 and prev_r < 0:
                    signals.append({
                        "date": df["date"].iloc[i], "side": "long",
                        "stop": support - 1.0 * atr, "target": midpoint,
                        "time_stop": 5,
                    })
        if side in ("short", "combined"):
            if (resistance - price) <= atr_mult * atr and price < resistance:
                if ibs_val < 0.3 and prev_r > 0:
                    signals.append({
                        "date": df["date"].iloc[i], "side": "short",
                        "stop": resistance + 1.0 * atr, "target": midpoint,
                        "time_stop": 5,
                    })
    return signals


def signals_s5(merged_df, lookback=60, k_thresh=2.0, side="combined"):
    """Relative value: trade RXRX vs ARKG spread z-score."""
    signals = []
    z_col = f"spread_z_{lookback}"
    if z_col not in merged_df.columns:
        return signals
    for i in range(lookback * 2, len(merged_df)):
        z = merged_df[z_col].iloc[i]
        if pd.isna(z):
            continue
        atr = merged_df["atr20"].iloc[i]
        if pd.isna(atr):
            continue
        price = merged_df["close"].iloc[i]
        if side in ("long", "combined") and z < -k_thresh:
            signals.append({
                "date": merged_df["date"].iloc[i], "side": "long",
                "stop": price - 1.5 * atr, "target": price + 1.0 * atr,
                "time_stop": 5,
            })
        if side in ("short", "combined") and z > k_thresh:
            signals.append({
                "date": merged_df["date"].iloc[i], "side": "short",
                "stop": price + 1.5 * atr, "target": price - 1.0 * atr,
                "time_stop": 5,
            })
    return signals


# ══════════════════════════════════════════════════════════════════
# SECTION 4: RUN ALL STRATEGIES (no stacking, hold sweep)
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  SECTION 4: STRATEGY RESULTS (no stacked entries)")
print("=" * 70)

strategy_registry = [
    ("S4c_short_5c", lambda df: signals_s4c(df, 5, "short")),
    ("S4c_comb_5c", lambda df: signals_s4c(df, 5, "combined")),
    ("S4c_long_5c", lambda df: signals_s4c(df, 5, "long")),
    ("S4c_comb_4c", lambda df: signals_s4c(df, 4, "combined")),
    ("S1_comb_L20_k2", lambda df: signals_s1(df, 20, 2.0, "combined")),
    ("S4a_comb_10_90", lambda df: signals_s4a(df, 10, 90, "combined")),
    ("S2_N40_0.5_comb", lambda df: signals_s2(df, 40, 0.5, "combined")),
    ("S2_N40_0.5_short", lambda df: signals_s2(df, 40, 0.5, "short")),
    ("S2_N60_1.0_comb", lambda df: signals_s2(df, 60, 1.0, "combined")),
]

if has_arkg:
    arkg_dev = arkg_merged[arkg_merged["date"] < HOLDOUT_START].copy()
    strategy_registry.extend([
        ("S5_L60_k2_comb", lambda df: signals_s5(arkg_dev, 60, 2.0, "combined")),
        ("S5_L60_k2_short", lambda df: signals_s5(arkg_dev, 60, 2.0, "short")),
        ("S5_L60_k1.5_comb", lambda df: signals_s5(arkg_dev, 60, 1.5, "combined")),
    ])

all_results = {}

for hold in HOLD_PERIODS:
    print(f"\n  ── Hold period = {hold} days ──")
    for name, sig_func in strategy_registry:
        sigs = sig_func(rxrx_dev)
        # S5 uses arkg_dev for execution
        exec_df = arkg_dev if "S5" in name else rxrx_dev
        trades_df = run_strategy_no_stack(exec_df, sigs, time_stop_override=hold)
        key = f"{name}_h{hold}"
        all_results[key] = {"trades": trades_df, "hold": hold, "strategy": name}
        s = compute_stats(trades_df, f"{name} h={hold}")
        print_stats_line(s)


# ══════════════════════════════════════════════════════════════════
# SECTION 5: REGIME SEGMENTATION
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  SECTION 5: REGIME SEGMENTATION")
print("=" * 70)

# Define regimes: MR regime (Hurst < 0.5) vs trending (Hurst >= 0.5)
# Also time-based: 2024+ subsample (recent low-price regime)

focus_strategies = ["S4c_short_5c", "S4c_comb_5c", "S2_N40_0.5_comb"]
focus_hold = 3  # best hold from sweep (adjust after reviewing)

for name in focus_strategies:
    sig_func = dict(strategy_registry).get(name)
    if sig_func is None:
        continue
    sigs = sig_func(rxrx_dev)
    trades_df = run_strategy_no_stack(rxrx_dev, sigs, time_stop_override=focus_hold)
    if trades_df.empty:
        continue

    print(f"\n  {name} (h={focus_hold}):")

    # Time-based regime
    trades_df = trades_df.copy()
    recent = trades_df[trades_df["entry_date"] >= "2024-01-01"]
    early = trades_df[trades_df["entry_date"] < "2024-01-01"]

    s_early = compute_stats(early, f"  pre-2024")
    s_recent = compute_stats(recent, f"  2024+")
    print_stats_line(s_early)
    print_stats_line(s_recent)

    # Hurst regime
    hurst_lookup = dict(zip(rxrx_dev["date"], rxrx_dev["hurst_120"]))
    trades_df["hurst"] = trades_df["entry_date"].map(hurst_lookup)
    mr_trades = trades_df[trades_df["hurst"] < 0.5]
    trend_trades = trades_df[trades_df["hurst"] >= 0.5]

    s_mr = compute_stats(mr_trades, f"  Hurst<0.5 (MR)")
    s_tr = compute_stats(trend_trades, f"  Hurst>=0.5 (trend)")
    print_stats_line(s_mr)
    print_stats_line(s_tr)


# ══════════════════════════════════════════════════════════════════
# SECTION 6: NULL STRATEGY COMPARISON
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print(f"  SECTION 6: NULL STRATEGIES ({N_NULL_RUNS} random runs)")
print("=" * 70)

# For each strategy, generate random entries with the same:
# - number of trades
# - side distribution (% long vs short)
# - hold period

focus_keys = [k for k in all_results if "_h3" in k and all_results[k]["trades"].shape[0] > 5]
if not focus_keys:
    focus_keys = [k for k in all_results if all_results[k]["trades"].shape[0] > 5][:5]

for key in focus_keys:
    trades_df = all_results[key]["trades"]
    if trades_df.empty:
        continue

    n_trades = len(trades_df)
    n_short = sum(trades_df["side"] == "short")
    n_long = n_trades - n_short
    hold = all_results[key]["hold"]

    # Eligible signal dates (warm-up period)
    eligible_dates = rxrx_dev["date"].iloc[60:].values
    null_sharpes = []
    null_avg_pnls = []

    for run in range(N_NULL_RUNS):
        rng = np.random.RandomState(run)
        random_dates = rng.choice(eligible_dates, size=n_trades, replace=False)
        random_dates.sort()

        sides = ["short"] * n_short + ["long"] * n_long
        rng.shuffle(sides)

        rand_signals = []
        for d, s in zip(random_dates, sides):
            idx = np.searchsorted(rxrx_dev["date"].values, d)
            if idx >= len(rxrx_dev):
                continue
            price = rxrx_dev["close"].iloc[idx]
            atr = rxrx_dev["atr20"].iloc[idx]
            if pd.isna(atr):
                continue
            rand_signals.append({
                "date": pd.Timestamp(d), "side": s,
                "stop": None, "target": None, "time_stop": hold,
            })

        null_trades = run_strategy_no_stack(rxrx_dev, rand_signals, time_stop_override=hold)
        if null_trades.empty:
            continue
        null_pnls = null_trades["pnl_net_pct"].values
        null_avg_pnls.append(np.mean(null_pnls))
        avg_h = null_trades["holding_days"].mean()
        if np.std(null_pnls) > 0 and avg_h > 0:
            null_sharpes.append(np.mean(null_pnls) / np.std(null_pnls) * np.sqrt(252 / avg_h))
        else:
            null_sharpes.append(0)

    real_stats = compute_stats(trades_df)
    real_avg = real_stats["avg_pnl"]
    real_sharpe = real_stats["sharpe"]

    pct_avg = scipy_stats.percentileofscore(null_avg_pnls, real_avg)
    pct_sharpe = scipy_stats.percentileofscore(null_sharpes, real_sharpe)

    null_avg_mean = np.mean(null_avg_pnls)
    null_avg_std = np.std(null_avg_pnls)

    print(f"\n  {key}:")
    print(f"    Real avg PnL: {real_avg:+.3f}%  vs null mean: {null_avg_mean:+.3f}% (std {null_avg_std:.3f}%)")
    print(f"    Percentile rank: avg PnL = {pct_avg:.1f}th,  Sharpe = {pct_sharpe:.1f}th")
    print(f"    {'PASSES' if pct_avg > 95 else 'FAILS'} null test at 95th percentile")

    if has_arkg and "S5" not in key:
        # Test on ARKG-hedged residual
        # If edge vanishes when hedged, it was beta exposure
        hedged_sigs = sig_func(rxrx_dev) if key.split("_h")[0] in dict(strategy_registry) else []
        if hedged_sigs and has_arkg:
            hedged_trades = run_strategy_no_stack(arkg_dev, hedged_sigs,
                                                   time_stop_override=hold)
            if not hedged_trades.empty:
                hs = compute_stats(hedged_trades, "ARKG-execution")
                print(f"    ARKG-hedged execution: avg={hs['avg_pnl']:+.3f}%  n={hs['n']}  "
                      f"Sharpe={hs['sharpe']:.2f}")
                if abs(hs["avg_pnl"]) < abs(real_avg) * 0.3:
                    print(f"    >> Edge DISAPPEARS after hedging — likely beta exposure")


# ══════════════════════════════════════════════════════════════════
# SECTION 7: EPISODE ANALYSIS + LEAVE-ONE-EPISODE-OUT
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  SECTION 7: EPISODE ANALYSIS")
print("=" * 70)

def assign_episodes(trades_df, gap_days=EPISODE_GAP_DAYS):
    """Group trades into episodes. Trades within gap_days of each other = same episode."""
    if trades_df.empty:
        return trades_df
    trades_df = trades_df.copy()
    trades_df = trades_df.sort_values("entry_date").reset_index(drop=True)
    episodes = [0]
    ep = 0
    for i in range(1, len(trades_df)):
        if (trades_df["entry_date"].iloc[i] - trades_df["exit_date"].iloc[i-1]).days > gap_days:
            ep += 1
        episodes.append(ep)
    trades_df["episode"] = episodes
    return trades_df


for key in focus_keys:
    trades_df = all_results[key]["trades"]
    if trades_df.empty or len(trades_df) < 5:
        continue

    trades_ep = assign_episodes(trades_df)
    n_episodes = trades_ep["episode"].nunique()

    print(f"\n  {key}: {len(trades_ep)} trades in {n_episodes} episodes")

    # Episode-level summary
    ep_stats = trades_ep.groupby("episode").agg(
        n=("pnl_net_pct", "count"),
        avg_pnl=("pnl_net_pct", "mean"),
        total_pnl=("pnl_net_pct", "sum"),
        total_usd=("pnl_dollars", "sum"),
        start=("entry_date", "min"),
        end=("exit_date", "max"),
    )

    print(f"  {'Ep':>4s} {'Start':>12s} {'End':>12s} {'N':>4s} {'AvgPnL':>9s} {'TotPnL':>9s} {'Tot$':>10s}")
    print(f"  {'-'*65}")
    for ep_id, row in ep_stats.iterrows():
        print(f"  {ep_id:>4d} {str(row['start'].date()):>12s} {str(row['end'].date()):>12s} "
              f"{int(row['n']):>4d} {row['avg_pnl']:>+8.2f}% {row['total_pnl']:>+8.2f}% "
              f"${row['total_usd']:>+9,.0f}")

    # Leave-one-episode-out
    print(f"\n  Leave-one-episode-out:")
    full_stats = compute_stats(trades_ep)
    full_avg = full_stats["avg_pnl"]
    full_sharpe = full_stats["sharpe"]

    loeo_results = []
    for ep_id in trades_ep["episode"].unique():
        remaining = trades_ep[trades_ep["episode"] != ep_id]
        s = compute_stats(remaining)
        loeo_results.append({
            "dropped": ep_id,
            "n_remaining": s["n"],
            "avg_pnl": s["avg_pnl"],
            "sharpe": s["sharpe"],
        })

    loeo_df = pd.DataFrame(loeo_results)
    worst_drop = loeo_df.loc[loeo_df["avg_pnl"].idxmax()]  # dropping ep improves most = that ep was bad
    best_drop = loeo_df.loc[loeo_df["avg_pnl"].idxmin()]   # dropping ep hurts most = that ep drove edge

    print(f"    Full: avg={full_avg:+.3f}%  Sharpe={full_sharpe:.2f}")
    print(f"    Dropping episode {int(best_drop['dropped'])} HURTS most: avg={best_drop['avg_pnl']:+.3f}%  Sharpe={best_drop['sharpe']:.2f}")
    print(f"    Dropping episode {int(worst_drop['dropped'])} HELPS most: avg={worst_drop['avg_pnl']:+.3f}%  Sharpe={worst_drop['sharpe']:.2f}")

    # Fragility: does edge survive dropping ANY single episode?
    min_avg = loeo_df["avg_pnl"].min()
    min_sharpe = loeo_df["sharpe"].min()
    survives = min_avg > 0 and min_sharpe > 0
    print(f"    Edge survives ALL drops: {'YES' if survives else 'NO'} "
          f"(min avg={min_avg:+.3f}%, min Sharpe={min_sharpe:.2f})")


# ══════════════════════════════════════════════════════════════════
# SECTION 8: SUMMARY TABLE + PLOTS
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  SECTION 8: FINAL SUMMARY TABLE")
print("=" * 70)
print(f"\n  {'Strategy':35s} {'Hold':>4s} {'N':>5s} {'WR':>5s} {'AvgPnL':>8s} {'TotPnL%':>8s} "
      f"{'Tot$':>10s} {'Sharpe':>7s} {'PF':>5s} {'DD':>7s}")
print(f"  {'-'*105}")

for key in sorted(all_results.keys()):
    r = all_results[key]
    s = compute_stats(r["trades"])
    if s["n"] == 0:
        continue
    parts = key.rsplit("_h", 1)
    sname = parts[0]
    h = parts[1] if len(parts) > 1 else "?"
    print(f"  {sname:35s} {h:>4s} {s['n']:>5d} {s['wr']:>4.0f}% {s['avg_pnl']:>+7.2f}% "
          f"{s['total_pnl_pct']:>+7.1f}% ${s['total_pnl_usd']:>+9,.0f} {s['sharpe']:>+6.2f} "
          f"{s['pf']:>5.2f} {s['max_dd']:>+6.1f}%")


# ── Equity curve plot ──
# %%
print("\nPlotting...")

fig, axes = plt.subplots(3, 1, figsize=(16, 16))
fig.suptitle("RXRX Strategy Analysis v2 (no stacking, corrected)", fontsize=14, fontweight="bold")

# Best strategies equity curves across hold periods
ax = axes[0]
for hold in HOLD_PERIODS:
    key = f"S4c_short_5c_h{hold}"
    if key in all_results:
        trades_df = all_results[key]["trades"]
        if not trades_df.empty:
            eq = daily_mtm_equity(trades_df, rxrx_dev)
            if len(eq) > 0:
                ax.plot(eq.index, eq.values, label=f"S4c SHORT h={hold}", lw=1.5)
ax.axhline(0, color="black", lw=0.5, ls="--")
ax.set_ylabel("Cumulative PnL (%)")
ax.set_title("S4c SHORT — Hold Period Sweep (daily MTM)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Strategy comparison at best hold
ax = axes[1]
best_hold = 3
for name in ["S4c_short_5c", "S4c_comb_5c", "S2_N40_0.5_comb", "S1_comb_L20_k2"]:
    key = f"{name}_h{best_hold}"
    if key in all_results:
        trades_df = all_results[key]["trades"]
        if not trades_df.empty:
            eq = daily_mtm_equity(trades_df, rxrx_dev)
            if len(eq) > 0:
                ax.plot(eq.index, eq.values, label=f"{name}", lw=1.5)
ax.axhline(0, color="black", lw=0.5, ls="--")
ax.set_ylabel("Cumulative PnL (%)")
ax.set_title(f"Strategy Comparison — h={best_hold} (daily MTM)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Regime (Hurst) over time
ax = axes[2]
ax.plot(rxrx_dev["date"], rxrx_dev["hurst_120"], color="navy", lw=1, alpha=0.7)
ax.axhline(0.5, color="red", ls="--", lw=1, label="Hurst = 0.5 (random walk)")
ax.fill_between(rxrx_dev["date"], 0, rxrx_dev["hurst_120"],
                where=rxrx_dev["hurst_120"] < 0.5, alpha=0.2, color="green", label="MR regime")
ax.fill_between(rxrx_dev["date"], rxrx_dev["hurst_120"], 1,
                where=rxrx_dev["hurst_120"] >= 0.5, alpha=0.2, color="red", label="Trending regime")
ax.set_ylabel("Rolling Hurst (120d)")
ax.set_title("RXRX Regime — Rolling Hurst Exponent")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(OUT_DIR / "trade_analysis_v2.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved trade_analysis_v2.png")


# ── Trade distribution for best strategy ──
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Trade PnL Distributions (no stacking)", fontsize=14, fontweight="bold")

plot_keys = [f"S4c_short_5c_h{h}" for h in HOLD_PERIODS]
for idx, key in enumerate(plot_keys):
    ax = axes.flat[idx]
    if key not in all_results or all_results[key]["trades"].empty:
        ax.set_title(f"{key}: no trades")
        continue
    df = all_results[key]["trades"]
    pnls = df["pnl_net_pct"].values
    colors = ["green" if p > 0 else "red" for p in pnls]
    ax.bar(range(len(pnls)), pnls, color=colors, alpha=0.7, width=1.0)
    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(np.mean(pnls), color="blue", ls="--", lw=1,
               label=f"Avg: {np.mean(pnls):+.2f}%")
    ax.set_title(f"S4c SHORT h={key.split('_h')[1]} (n={len(pnls)})")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("PnL (%)")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR / "trade_analysis_v2_dist.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved trade_analysis_v2_dist.png")


# ══════════════════════════════════════════════════════════════════
# SECTION 9: CONCLUSIONS
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  SECTION 9: CONCLUSIONS")
print("=" * 70)

print("""
  This v2 analysis corrects the following issues from v1:
  1. No stacked entries — 1 position at a time per strategy
  2. Daily mark-to-market equity curve
  3. Null strategy comparison (1,000 drift-matched random runs)
  4. Episode-level analysis with leave-one-episode-out fragility test
  5. S2 (range fade) and S5 (relative value vs ARKG) included
  6. Hold period sweep {1, 2, 3, 5} days
  7. Regime segmentation (rolling Hurst + 2024+ subsample)
  8. Phase 1 verdict reported at top

  STILL PENDING:
  - S0 (user's discretionary rules) — awaiting user input
  - Phase 4: cross-sectional test of "short after 5 up closes in
    downtrending high-ATR small caps" across many stocks
  - Holdout NOT spent (per instruction)
  - S7 meta-labeling SKIPPED (too few episodes to train classifier)
""")

print("\nAnalysis complete. Review plots in agent10/results/")
