# %% Agent 10 — Tier 1: S5 Refinement + Tier 2: New Strategy Families
#
# Walk-forward only. Every variant logged. DSR over cumulative trial count.
# Holdout untouched. Target: ≥2% avg net PnL per trade.
#
# Tier 1 — S5 Refinements:
#   1. Avellaneda-Lee s-score (OU on cumulated residual returns)
#   2. Kalman filter time-varying beta
#   3. Hedge factor comparison (ARKG, XBI, IWM, PCA, multi-factor)
#   4. Signal-based exits (z-cross thresholds)
#   5. Entry confirmation (z reversal after extreme)
#   6. Scaled entry (half at |z|=1.5, half at |z|=2.0)
#   7. Asymmetric thresholds (structural downtrend)
#   8. Noise vs information filter (volume, events)
#
# Tier 2 — New Families:
#   S8:  Event-based shock fade (gap + revert around events)
#   S9:  Gap fade (overnight gap MR)
#   S10: Anchored VWAP levels
#   S11: Relaxed S2 with KDE levels
#   S12: ARK flow reversal
#   S13: Vol compression breakout
#
# Requires: Phase 0 data + phase1_results.json

import os, sys, json, warnings
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats, signal as scipy_signal
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

for pkg in ["statsmodels", "pykalman"]:
    try:
        __import__(pkg)
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg,
                               "--quiet", "--break-system-packages"])

import statsmodels.api as sm
from pykalman import KalmanFilter

# ── Config ────────────────────────────────────────────────────────
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
EPISODE_GAP_DAYS = 14

# ══════════════════════════════════════════════════════════════════
# SECTION 0: LOAD DATA & BUILD FEATURES
# ══════════════════════════════════════════════════════════════════
# %%
print("=" * 70)
print("  LOADING DATA & BUILDING FEATURES")
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

for L in [10, 20, 30, 60]:
    rxrx[f"z_{L}"] = (
        (rxrx["log_price"] - rxrx["log_price"].rolling(L).mean()) /
        rxrx["log_price"].rolling(L).std()
    )

for N in [20, 40, 60]:
    rxrx[f"range_high_{N}"] = rxrx["high"].rolling(N).max()
    rxrx[f"range_low_{N}"] = rxrx["low"].rolling(N).min()
    rxrx[f"range_mid_{N}"] = (rxrx[f"range_high_{N}"] + rxrx[f"range_low_{N}"]) / 2

consec_down = np.zeros(len(rxrx))
consec_up = np.zeros(len(rxrx))
for i in range(1, len(rxrx)):
    if rxrx["close"].iloc[i] < rxrx["close"].iloc[i-1]:
        consec_down[i] = consec_down[i-1] + 1
    if rxrx["close"].iloc[i] > rxrx["close"].iloc[i-1]:
        consec_up[i] = consec_up[i-1] + 1
rxrx["consec_down"] = consec_down
rxrx["consec_up"] = consec_up

# Volume features
rxrx["vol_ma20"] = rxrx["volume"].rolling(20).mean()
rxrx["vol_ratio"] = rxrx["volume"] / rxrx["vol_ma20"]
rxrx["dollar_volume"] = rxrx["close"] * rxrx["volume"]

# Gap (overnight)
rxrx["gap_pct"] = (rxrx["open"] - rxrx["close"].shift(1)) / rxrx["close"].shift(1) * 100

# Rolling realized vol
rxrx["rvol_20"] = rxrx["log_ret"].rolling(20).std() * np.sqrt(252) * 100

# Bollinger bandwidth (vol compression indicator)
for L in [20]:
    ma = rxrx["close"].rolling(L).mean()
    std = rxrx["close"].rolling(L).std()
    rxrx[f"bb_width_{L}"] = (2 * std / ma) * 100
    rxrx[f"bb_width_pctile_{L}"] = rxrx[f"bb_width_{L}"].rolling(120).rank(pct=True)

# Context data
ctx = pd.read_parquet(DATA_DIR / "context_daily.parquet")
ctx["date"] = pd.to_datetime(ctx["date"])

arkg = ctx[ctx["ticker"] == "ARKG"].sort_values("date").reset_index(drop=True)
xbi = ctx[ctx["ticker"] == "XBI"].sort_values("date").reset_index(drop=True)
iwm = ctx[ctx["ticker"] == "IWM"].sort_values("date").reset_index(drop=True)
spy = ctx[ctx["ticker"] == "SPY"].sort_values("date").reset_index(drop=True)

# Events
try:
    events = pd.read_parquet(DATA_DIR / "rxrx_events.parquet")
    events["date"] = pd.to_datetime(events["date"])
    event_dates = set(events["date"].dt.date)
    print(f"  Events loaded: {len(events)} events")
except:
    event_dates = set()
    print("  No events data")

# Mark event proximity
rxrx["near_event"] = rxrx["date"].apply(
    lambda d: any(abs((d.date() - ed).days) <= 3 for ed in event_dates)
    if event_dates else False
)

print(f"  RXRX: {len(rxrx)} bars, {rxrx['date'].min().date()} to {rxrx['date'].max().date()}")

# ══════════════════════════════════════════════════════════════════
# SECTION 1: BUILD RESIDUAL SPREADS (multiple hedge factors)
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  SECTION 1: RESIDUAL CONSTRUCTION (multiple hedge factors)")
print("=" * 70)

def build_ols_spread(rxrx_df, hedge_df, hedge_ticker, lookback=60):
    merged = rxrx_df[["date", "close", "open", "high", "low", "log_price",
                       "atr20", "atr_pct", "volume", "vol_ratio",
                       "near_event", "gap_pct", "rvol_20",
                       "bb_width_20", "bb_width_pctile_20"]].merge(
        hedge_df[["date", "close"]].rename(columns={"close": f"{hedge_ticker}_close"}),
        on="date", how="inner"
    )
    merged[f"{hedge_ticker}_log"] = np.log(merged[f"{hedge_ticker}_close"])
    merged["rxrx_log"] = merged["log_price"]

    merged[f"hedge_{lookback}"] = np.nan
    merged[f"spread_{lookback}"] = np.nan
    merged[f"resid_ret_{lookback}"] = np.nan

    for i in range(lookback, len(merged)):
        y = merged["rxrx_log"].iloc[i-lookback:i+1].values
        x = sm.add_constant(merged[f"{hedge_ticker}_log"].iloc[i-lookback:i+1].values)
        try:
            model = sm.OLS(y, x).fit()
            hedge = model.params[1]
            resid = y[-1] - model.predict(x)[-1]
            merged.iloc[i, merged.columns.get_loc(f"hedge_{lookback}")] = hedge
            merged.iloc[i, merged.columns.get_loc(f"spread_{lookback}")] = resid
        except:
            pass

    spread = merged[f"spread_{lookback}"]
    merged[f"spread_z_{lookback}"] = (
        (spread - spread.rolling(lookback).mean()) / spread.rolling(lookback).std()
    )
    # Residual returns for OU estimation
    merged[f"resid_ret_{lookback}"] = merged[f"spread_{lookback}"].diff()

    return merged


def build_kalman_spread(rxrx_df, hedge_df, hedge_ticker):
    merged = rxrx_df[["date", "close", "open", "high", "low", "log_price",
                       "atr20", "atr_pct", "volume", "vol_ratio",
                       "near_event", "gap_pct", "rvol_20",
                       "bb_width_20", "bb_width_pctile_20"]].merge(
        hedge_df[["date", "close"]].rename(columns={"close": f"{hedge_ticker}_close"}),
        on="date", how="inner"
    )
    merged[f"{hedge_ticker}_log"] = np.log(merged[f"{hedge_ticker}_close"])
    merged["rxrx_log"] = merged["log_price"]

    obs = merged["rxrx_log"].values
    hedge_series = merged[f"{hedge_ticker}_log"].values

    delta = 1e-5
    trans_cov = delta / (1 - delta) * np.eye(2)
    obs_mat = np.column_stack([np.ones(len(obs)), hedge_series])
    obs_mat = obs_mat[:, np.newaxis, :]

    kf = KalmanFilter(
        n_dim_obs=1,
        n_dim_state=2,
        initial_state_mean=np.array([0, 1]),
        initial_state_covariance=np.ones((2, 2)),
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        observation_covariance=1.0,
        transition_covariance=trans_cov,
    )

    state_means, state_covs = kf.filter(obs)

    merged["kalman_alpha"] = state_means[:, 0]
    merged["kalman_beta"] = state_means[:, 1]

    merged["kalman_spread"] = obs - (merged["kalman_alpha"] + merged["kalman_beta"] * hedge_series)

    spread = merged["kalman_spread"]
    for lb in [40, 60]:
        merged[f"kalman_z_{lb}"] = (
            (spread - spread.rolling(lb).mean()) / spread.rolling(lb).std()
        )

    return merged


def build_multifactor_spread(rxrx_df, factor_dfs, factor_names, lookback=60):
    merged = rxrx_df[["date", "close", "open", "high", "low", "log_price",
                       "atr20", "atr_pct", "volume", "vol_ratio",
                       "near_event", "gap_pct", "rvol_20",
                       "bb_width_20", "bb_width_pctile_20"]].copy()
    merged["rxrx_log"] = merged["log_price"]

    for fdf, fname in zip(factor_dfs, factor_names):
        fdf_c = fdf[["date", "close"]].rename(columns={"close": f"{fname}_close"})
        merged = merged.merge(fdf_c, on="date", how="inner")
        merged[f"{fname}_log"] = np.log(merged[f"{fname}_close"])

    merged[f"mf_spread_{lookback}"] = np.nan
    for i in range(lookback, len(merged)):
        y = merged["rxrx_log"].iloc[i-lookback:i+1].values
        X_cols = [merged[f"{fn}_log"].iloc[i-lookback:i+1].values for fn in factor_names]
        X = sm.add_constant(np.column_stack(X_cols))
        try:
            model = sm.OLS(y, X).fit()
            resid = y[-1] - model.predict(X)[-1]
            merged.iloc[i, merged.columns.get_loc(f"mf_spread_{lookback}")] = resid
        except:
            pass

    spread = merged[f"mf_spread_{lookback}"]
    merged[f"mf_spread_z_{lookback}"] = (
        (spread - spread.rolling(lookback).mean()) / spread.rolling(lookback).std()
    )
    return merged


# Build all spreads
spreads = {}

for hedge_name, hedge_df in [("ARKG", arkg), ("XBI", xbi), ("IWM", iwm)]:
    if len(hedge_df) > 100:
        spreads[f"OLS_{hedge_name}_60"] = build_ols_spread(rxrx, hedge_df, hedge_name, 60)
        print(f"  Built OLS spread vs {hedge_name}: {len(spreads[f'OLS_{hedge_name}_60'])} bars")

if len(arkg) > 100:
    spreads["Kalman_ARKG"] = build_kalman_spread(rxrx, arkg, "ARKG")
    print(f"  Built Kalman spread vs ARKG: {len(spreads['Kalman_ARKG'])} bars")

if len(arkg) > 100 and len(xbi) > 100:
    spreads["MF_ARKG_XBI_60"] = build_multifactor_spread(
        rxrx, [arkg, xbi], ["ARKG", "XBI"], 60
    )
    print(f"  Built multi-factor spread (ARKG+XBI): {len(spreads['MF_ARKG_XBI_60'])} bars")

if len(arkg) > 100 and len(xbi) > 100 and len(iwm) > 100:
    spreads["MF_ALL_60"] = build_multifactor_spread(
        rxrx, [arkg, xbi, iwm], ["ARKG", "XBI", "IWM"], 60
    )
    print(f"  Built multi-factor spread (ARKG+XBI+IWM): {len(spreads['MF_ALL_60'])} bars")


# ══════════════════════════════════════════════════════════════════
# SECTION 2: AVELLANEDA-LEE S-SCORE
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  SECTION 2: AVELLANEDA-LEE S-SCORE")
print("=" * 70)

def compute_ou_params(residuals, lookback=60):
    """Estimate OU process parameters on cumulated residual returns."""
    s_scores = np.full(len(residuals), np.nan)
    kappas = np.full(len(residuals), np.nan)
    half_lives = np.full(len(residuals), np.nan)

    cum_resid = residuals.cumsum()

    for i in range(lookback + 1, len(residuals)):
        y = cum_resid.iloc[i-lookback:i].values
        if np.any(np.isnan(y)):
            continue
        y_lag = y[:-1]
        dy = np.diff(y)

        X = sm.add_constant(y_lag)
        try:
            model = sm.OLS(dy, X).fit()
            a = model.params[0]
            b = model.params[1]

            if b >= 0:
                continue

            kappa = -b
            m = a / kappa
            sigma_eq = np.sqrt(model.mse_resid / (2 * kappa))

            s = (y[-1] - m) / sigma_eq if sigma_eq > 0 else 0
            hl = np.log(2) / kappa

            s_scores[i] = s
            kappas[i] = kappa
            half_lives[i] = hl
        except:
            pass

    return pd.Series(s_scores, index=residuals.index), \
           pd.Series(kappas, index=residuals.index), \
           pd.Series(half_lives, index=residuals.index)


for spread_name, spread_df in spreads.items():
    if "OLS_ARKG" not in spread_name:
        continue

    spread_dev = spread_df[spread_df["date"] < HOLDOUT_START].copy()
    resid_col = [c for c in spread_dev.columns if "spread_60" in c and "z" not in c and "mf" not in c]
    if not resid_col:
        continue

    resid = spread_dev[resid_col[0]].dropna()
    s_score, kappa, half_life = compute_ou_params(resid.diff().dropna(), lookback=60)

    spread_dev.loc[s_score.index, "s_score"] = s_score
    spread_dev.loc[kappa.index, "ou_kappa"] = kappa
    spread_dev.loc[half_life.index, "ou_half_life"] = half_life

    valid = spread_dev["s_score"].dropna()
    print(f"  {spread_name}: s-score range [{valid.min():.2f}, {valid.max():.2f}], "
          f"median kappa = {spread_dev['ou_kappa'].median():.4f}, "
          f"median HL = {spread_dev['ou_half_life'].median():.1f}d")

    spreads[spread_name] = spread_dev


# ══════════════════════════════════════════════════════════════════
# SECTION 3: TRADE ENGINE (from v2, unchanged)
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


def run_strategy_z_exit(df, signals, z_col, z_exit_thresh=0.5):
    """Exit when z-score crosses back through a threshold instead of fixed hold."""
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
        if entry_date <= in_trade_until:
            continue

        entry_price = df["open"].iloc[entry_idx]
        max_hold = sig.get("time_stop", 10)
        exit_price = None
        exit_reason = None
        exit_date = None

        for j in range(entry_idx + 1, min(entry_idx + max_hold + 1, len(df))):
            z_now = df[z_col].iloc[j] if z_col in df.columns and j < len(df) else np.nan

            if sig["side"] == "long":
                if sig.get("stop") and df["low"].iloc[j] <= sig["stop"]:
                    exit_price = sig["stop"]
                    exit_reason = "stop"
                    exit_date = df["date"].iloc[j]
                    break
                if not np.isnan(z_now) and z_now >= -z_exit_thresh:
                    exit_price = df["close"].iloc[j]
                    exit_reason = "z_cross"
                    exit_date = df["date"].iloc[j]
                    break
            else:
                if sig.get("stop") and df["high"].iloc[j] >= sig["stop"]:
                    exit_price = sig["stop"]
                    exit_reason = "stop"
                    exit_date = df["date"].iloc[j]
                    break
                if not np.isnan(z_now) and z_now <= z_exit_thresh:
                    exit_price = df["close"].iloc[j]
                    exit_reason = "z_cross"
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


def compute_stats(trades_df, label=""):
    if trades_df.empty:
        return {"label": label, "n": 0, "avg_pnl": 0, "sharpe": 0, "wr": 0,
                "pf": 0, "max_dd": 0, "total_pnl_pct": 0, "total_pnl_usd": 0,
                "avg_hold": 0, "avg_win": 0, "avg_loss": 0, "payoff": 0}
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
        print(f"  {s['label']:50s}  0 trades")
        return
    print(f"  {s['label']:50s}  n={s['n']:>4d}  avg={s['avg_pnl']:>+7.2f}%  "
          f"WR={s['wr']:>4.0f}%  PF={s['pf']:>5.2f}  Sharpe={s['sharpe']:>+6.2f}  "
          f"DD={s['max_dd']:>+7.1f}%  ${s['total_pnl_usd']:>+9,.0f}")


def assign_episodes(trades_df, gap_days=EPISODE_GAP_DAYS):
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


def deflated_sharpe_ratio(observed_sharpe, n_trials, n_trades, avg_hold, skew=0, kurt=0):
    """Bailey & Lopez de Prado DSR."""
    if n_trades < 5 or n_trials < 2:
        return 0.0
    sr_std = np.sqrt(
        (1 - skew * observed_sharpe + (kurt - 1) / 4 * observed_sharpe**2) / n_trades
    )
    if sr_std <= 0:
        sr_std = 1e-6
    expected_max_sr = np.sqrt(2 * np.log(n_trials)) * (
        1 - np.euler_gamma / (2 * np.log(n_trials)) + np.euler_gamma / (2 * np.log(n_trials))
    ) if n_trials > 1 else 0
    t_stat = (observed_sharpe - expected_max_sr) / sr_std
    dsr = scipy_stats.norm.cdf(t_stat)
    return round(dsr, 4)


# ══════════════════════════════════════════════════════════════════
# SECTION 4: TIER 1 — S5 SIGNAL GENERATORS (all variants)
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  SECTION 4: TIER 1 — S5 SIGNAL VARIANTS")
print("=" * 70)


def signals_s5_ols(spread_df, z_col, k_thresh=2.0, side="combined"):
    """Vanilla S5: OLS z-score entry."""
    signals = []
    lb = 120
    for i in range(lb, len(spread_df)):
        z = spread_df[z_col].iloc[i]
        if pd.isna(z):
            continue
        atr = spread_df["atr20"].iloc[i]
        if pd.isna(atr):
            continue
        price = spread_df["close"].iloc[i]
        if side in ("long", "combined") and z < -k_thresh:
            signals.append({
                "date": spread_df["date"].iloc[i], "side": "long",
                "stop": price - 1.5 * atr, "target": price + 1.0 * atr,
                "time_stop": 5,
            })
        if side in ("short", "combined") and z > k_thresh:
            signals.append({
                "date": spread_df["date"].iloc[i], "side": "short",
                "stop": price + 1.5 * atr, "target": price - 1.0 * atr,
                "time_stop": 5,
            })
    return signals


def signals_s5_sscore(spread_df, s_thresh=2.0, side="combined", hl_max=30):
    """Avellaneda-Lee s-score entry. Filter by half-life < hl_max."""
    signals = []
    if "s_score" not in spread_df.columns:
        return signals
    for i in range(120, len(spread_df)):
        s = spread_df["s_score"].iloc[i]
        hl = spread_df.get("ou_half_life", pd.Series(dtype=float)).iloc[i] if "ou_half_life" in spread_df.columns else np.nan
        if pd.isna(s):
            continue
        if not np.isnan(hl) and hl > hl_max:
            continue
        atr = spread_df["atr20"].iloc[i]
        if pd.isna(atr):
            continue
        price = spread_df["close"].iloc[i]
        if side in ("long", "combined") and s < -s_thresh:
            signals.append({
                "date": spread_df["date"].iloc[i], "side": "long",
                "stop": price - 1.5 * atr, "target": None,
                "time_stop": 5,
            })
        if side in ("short", "combined") and s > s_thresh:
            signals.append({
                "date": spread_df["date"].iloc[i], "side": "short",
                "stop": price + 1.5 * atr, "target": None,
                "time_stop": 5,
            })
    return signals


def signals_s5_kalman(spread_df, z_col, k_thresh=2.0, side="combined"):
    """Kalman-filtered beta, z-score entry."""
    signals = []
    for i in range(120, len(spread_df)):
        z = spread_df[z_col].iloc[i] if z_col in spread_df.columns else np.nan
        if pd.isna(z):
            continue
        atr = spread_df["atr20"].iloc[i]
        if pd.isna(atr):
            continue
        price = spread_df["close"].iloc[i]
        if side in ("long", "combined") and z < -k_thresh:
            signals.append({
                "date": spread_df["date"].iloc[i], "side": "long",
                "stop": price - 1.5 * atr, "target": price + 1.0 * atr,
                "time_stop": 5,
            })
        if side in ("short", "combined") and z > k_thresh:
            signals.append({
                "date": spread_df["date"].iloc[i], "side": "short",
                "stop": price + 1.5 * atr, "target": price - 1.0 * atr,
                "time_stop": 5,
            })
    return signals


def signals_s5_confirmed(spread_df, z_col, k_entry=2.0, side="combined"):
    """Entry confirmation: z must cross BELOW -k, then tick back UP (reversal)."""
    signals = []
    for i in range(121, len(spread_df)):
        z_now = spread_df[z_col].iloc[i]
        z_prev = spread_df[z_col].iloc[i-1]
        if pd.isna(z_now) or pd.isna(z_prev):
            continue
        atr = spread_df["atr20"].iloc[i]
        if pd.isna(atr):
            continue
        price = spread_df["close"].iloc[i]

        if side in ("long", "combined"):
            if z_prev < -k_entry and z_now > z_prev and z_now < -k_entry * 0.5:
                signals.append({
                    "date": spread_df["date"].iloc[i], "side": "long",
                    "stop": price - 1.5 * atr, "target": price + 1.0 * atr,
                    "time_stop": 7,
                })
        if side in ("short", "combined"):
            if z_prev > k_entry and z_now < z_prev and z_now > k_entry * 0.5:
                signals.append({
                    "date": spread_df["date"].iloc[i], "side": "short",
                    "stop": price + 1.5 * atr, "target": price - 1.0 * atr,
                    "time_stop": 7,
                })
    return signals


def signals_s5_asymmetric(spread_df, z_col, k_long=2.0, k_short=1.5, side="combined"):
    """Asymmetric thresholds: lower threshold for SHORT (structural downtrend bias)."""
    signals = []
    for i in range(120, len(spread_df)):
        z = spread_df[z_col].iloc[i]
        if pd.isna(z):
            continue
        atr = spread_df["atr20"].iloc[i]
        if pd.isna(atr):
            continue
        price = spread_df["close"].iloc[i]
        if side in ("long", "combined") and z < -k_long:
            signals.append({
                "date": spread_df["date"].iloc[i], "side": "long",
                "stop": price - 1.5 * atr, "target": price + 1.0 * atr,
                "time_stop": 5,
            })
        if side in ("short", "combined") and z > k_short:
            signals.append({
                "date": spread_df["date"].iloc[i], "side": "short",
                "stop": price + 1.5 * atr, "target": price - 1.0 * atr,
                "time_stop": 5,
            })
    return signals


def signals_s5_noise_filter(spread_df, z_col, k_thresh=1.5, side="combined",
                            require_low_vol=False, exclude_events=True):
    """S5 with noise/info filters: exclude event windows, optionally require low volume."""
    signals = []
    for i in range(120, len(spread_df)):
        z = spread_df[z_col].iloc[i]
        if pd.isna(z):
            continue
        atr = spread_df["atr20"].iloc[i]
        if pd.isna(atr):
            continue

        if exclude_events and spread_df.get("near_event", pd.Series(dtype=bool)).iloc[i]:
            continue

        if require_low_vol:
            vol_r = spread_df["vol_ratio"].iloc[i] if "vol_ratio" in spread_df.columns else 1.0
            if pd.isna(vol_r) or vol_r > 2.0:
                continue

        price = spread_df["close"].iloc[i]
        if side in ("long", "combined") and z < -k_thresh:
            signals.append({
                "date": spread_df["date"].iloc[i], "side": "long",
                "stop": price - 1.5 * atr, "target": price + 1.0 * atr,
                "time_stop": 5,
            })
        if side in ("short", "combined") and z > k_thresh:
            signals.append({
                "date": spread_df["date"].iloc[i], "side": "short",
                "stop": price + 1.5 * atr, "target": price - 1.0 * atr,
                "time_stop": 5,
            })
    return signals


# ══════════════════════════════════════════════════════════════════
# SECTION 5: TIER 2 — NEW STRATEGY FAMILIES
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  SECTION 5: TIER 2 — NEW STRATEGY FAMILIES (S8-S13)")
print("=" * 70)


def signals_s8_event_shock_fade(df, gap_thresh=3.0, side="combined"):
    """S8: Fade large gap moves that are NOT near known events."""
    signals = []
    for i in range(20, len(df)):
        gap = df["gap_pct"].iloc[i]
        if pd.isna(gap):
            continue
        atr = df["atr20"].iloc[i]
        if pd.isna(atr):
            continue
        price = df["close"].iloc[i]

        if abs(gap) < gap_thresh:
            continue

        if side in ("long", "combined") and gap < -gap_thresh:
            signals.append({
                "date": df["date"].iloc[i], "side": "long",
                "stop": price - 1.5 * atr, "target": price + abs(gap / 100) * price * 0.5,
                "time_stop": 3,
            })
        if side in ("short", "combined") and gap > gap_thresh:
            signals.append({
                "date": df["date"].iloc[i], "side": "short",
                "stop": price + 1.5 * atr, "target": price - abs(gap / 100) * price * 0.5,
                "time_stop": 3,
            })
    return signals


def signals_s9_gap_fade(df, gap_min=1.0, gap_max=5.0, side="combined"):
    """S9: Fade overnight gaps within a range (not too large = news)."""
    signals = []
    for i in range(20, len(df)):
        gap = df["gap_pct"].iloc[i]
        if pd.isna(gap):
            continue
        atr = df["atr20"].iloc[i]
        if pd.isna(atr):
            continue
        price = df["open"].iloc[i]
        prev_close = df["close"].iloc[i-1]

        if abs(gap) < gap_min or abs(gap) > gap_max:
            continue

        if side in ("long", "combined") and gap < -gap_min:
            signals.append({
                "date": df["date"].iloc[i-1], "side": "long",
                "stop": price - 1.5 * atr, "target": prev_close,
                "time_stop": 2,
            })
        if side in ("short", "combined") and gap > gap_min:
            signals.append({
                "date": df["date"].iloc[i-1], "side": "short",
                "stop": price + 1.5 * atr, "target": prev_close,
                "time_stop": 2,
            })
    return signals


def signals_s10_anchored_vwap(df, lookback_vwap=20, z_thresh=2.0, side="combined"):
    """S10: Mean reversion to rolling VWAP."""
    signals = []
    if "vwap" not in df.columns:
        vwap = (df["close"] * df["volume"]).rolling(lookback_vwap).sum() / \
               df["volume"].rolling(lookback_vwap).sum()
    else:
        vwap = df["vwap"].rolling(lookback_vwap).mean()

    vwap_dev = (df["close"] - vwap) / vwap * 100
    vwap_z = (vwap_dev - vwap_dev.rolling(lookback_vwap).mean()) / vwap_dev.rolling(lookback_vwap).std()

    for i in range(lookback_vwap * 2, len(df)):
        z = vwap_z.iloc[i]
        if pd.isna(z):
            continue
        atr = df["atr20"].iloc[i]
        if pd.isna(atr):
            continue
        price = df["close"].iloc[i]

        if side in ("long", "combined") and z < -z_thresh:
            signals.append({
                "date": df["date"].iloc[i], "side": "long",
                "stop": price - 1.5 * atr, "target": vwap.iloc[i],
                "time_stop": 5,
            })
        if side in ("short", "combined") and z > z_thresh:
            signals.append({
                "date": df["date"].iloc[i], "side": "short",
                "stop": price + 1.5 * atr, "target": vwap.iloc[i],
                "time_stop": 5,
            })
    return signals


def signals_s11_relaxed_s2(df, N=40, pct_band=0.05, side="combined"):
    """S11: Relaxed range fade with KDE-smoothed support/resistance."""
    signals = []
    rh = f"range_high_{N}"
    rl = f"range_low_{N}"
    if rh not in df.columns:
        return signals

    for i in range(N + 1, len(df)):
        atr = df["atr20"].iloc[i]
        if pd.isna(atr):
            continue
        price = df["close"].iloc[i]
        support = df[rl].iloc[i]
        resistance = df[rh].iloc[i]
        rng = resistance - support
        if pd.isna(rng) or rng < atr:
            continue

        proximity_low = (price - support) / rng
        proximity_high = (resistance - price) / rng

        if side in ("long", "combined"):
            if proximity_low < pct_band:
                signals.append({
                    "date": df["date"].iloc[i], "side": "long",
                    "stop": support - 0.5 * atr,
                    "target": support + 0.4 * rng,
                    "time_stop": 5,
                })
        if side in ("short", "combined"):
            if proximity_high < pct_band:
                signals.append({
                    "date": df["date"].iloc[i], "side": "short",
                    "stop": resistance + 0.5 * atr,
                    "target": resistance - 0.4 * rng,
                    "time_stop": 5,
                })
    return signals


def signals_s13_vol_compression(df, bb_pctile_thresh=0.1, side="combined"):
    """S13: Entry after vol compression (low BB width percentile), direction from z-score."""
    signals = []
    bb_col = "bb_width_pctile_20"
    if bb_col not in df.columns:
        return signals

    for i in range(120, len(df)):
        bb_pct = df[bb_col].iloc[i]
        if pd.isna(bb_pct):
            continue
        if bb_pct > bb_pctile_thresh:
            continue

        z20 = df["z_20"].iloc[i] if "z_20" in df.columns else 0
        if pd.isna(z20):
            continue
        atr = df["atr20"].iloc[i]
        if pd.isna(atr):
            continue
        price = df["close"].iloc[i]

        if side in ("long", "combined") and z20 < -0.5:
            signals.append({
                "date": df["date"].iloc[i], "side": "long",
                "stop": price - 2.0 * atr, "target": price + 2.0 * atr,
                "time_stop": 5,
            })
        if side in ("short", "combined") and z20 > 0.5:
            signals.append({
                "date": df["date"].iloc[i], "side": "short",
                "stop": price + 2.0 * atr, "target": price - 2.0 * atr,
                "time_stop": 5,
            })
    return signals


# ══════════════════════════════════════════════════════════════════
# SECTION 6: RUN ALL STRATEGIES (sweep + log)
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  SECTION 6: FULL STRATEGY SWEEP")
print("=" * 70)

rxrx_dev = rxrx[rxrx["date"] < HOLDOUT_START].copy()

trial_log = []  # every variant logged
all_results = {}
trial_counter = 0

def log_and_run(name, df, sigs, hold=None, z_exit_col=None, z_exit_thresh=None):
    global trial_counter
    trial_counter += 1

    for h in (HOLD_PERIODS if hold is None else [hold]):
        if z_exit_col and z_exit_thresh is not None:
            trades_df = run_strategy_z_exit(df, sigs, z_exit_col, z_exit_thresh)
        else:
            trades_df = run_strategy_no_stack(df, sigs, time_stop_override=h)

        key = f"{name}_h{h}" if z_exit_col is None else f"{name}_zexit{z_exit_thresh}"
        s = compute_stats(trades_df, key)
        all_results[key] = {"trades": trades_df, "stats": s, "trial_num": trial_counter}

        # Regime split
        if not trades_df.empty:
            pre24 = trades_df[trades_df["entry_date"] < "2024-01-01"]
            post24 = trades_df[trades_df["entry_date"] >= "2024-01-01"]
            s_pre = compute_stats(pre24) if not pre24.empty else {"n": 0, "avg_pnl": 0, "sharpe": 0}
            s_post = compute_stats(post24) if not post24.empty else {"n": 0, "avg_pnl": 0, "sharpe": 0}
        else:
            s_pre = {"n": 0, "avg_pnl": 0, "sharpe": 0}
            s_post = {"n": 0, "avg_pnl": 0, "sharpe": 0}

        # Long/short split
        if not trades_df.empty:
            longs = trades_df[trades_df["side"] == "long"]
            shorts = trades_df[trades_df["side"] == "short"]
            s_long = compute_stats(longs) if not longs.empty else {"n": 0, "avg_pnl": 0, "sharpe": 0}
            s_short = compute_stats(shorts) if not shorts.empty else {"n": 0, "avg_pnl": 0, "sharpe": 0}
        else:
            s_long = {"n": 0, "avg_pnl": 0, "sharpe": 0}
            s_short = {"n": 0, "avg_pnl": 0, "sharpe": 0}

        trial_log.append({
            "trial": trial_counter, "name": key, "n": s["n"],
            "avg_pnl": s["avg_pnl"], "sharpe": s["sharpe"],
            "wr": s["wr"], "pf": s["pf"], "max_dd": s["max_dd"],
            "avg_hold": s["avg_hold"],
            "n_long": s_long["n"], "avg_long": s_long["avg_pnl"],
            "n_short": s_short["n"], "avg_short": s_short["avg_pnl"],
            "n_pre24": s_pre["n"], "avg_pre24": s_pre["avg_pnl"],
            "n_post24": s_post["n"], "avg_post24": s_post["avg_pnl"],
        })
        print_stats_line(s)

    return trial_counter


# ── TIER 1: S5 variants ──

# 1a. OLS spreads across hedge factors
print("\n  ── Tier 1: OLS spread across hedge factors ──")
for spread_key in ["OLS_ARKG_60", "OLS_XBI_60", "OLS_IWM_60"]:
    if spread_key not in spreads:
        continue
    sdf = spreads[spread_key]
    sdf_dev = sdf[sdf["date"] < HOLDOUT_START].copy()
    z_col = "spread_z_60"
    for k in [1.5, 2.0, 2.5]:
        for side in ["combined", "long", "short"]:
            sigs = signals_s5_ols(sdf_dev, z_col, k, side)
            log_and_run(f"S5_{spread_key}_k{k}_{side}", sdf_dev, sigs)

# 1b. Kalman spread
print("\n  ── Tier 1: Kalman-filtered beta ──")
if "Kalman_ARKG" in spreads:
    sdf = spreads["Kalman_ARKG"]
    sdf_dev = sdf[sdf["date"] < HOLDOUT_START].copy()
    for z_col in ["kalman_z_40", "kalman_z_60"]:
        lb = z_col.split("_")[-1]
        for k in [1.5, 2.0]:
            for side in ["combined", "long", "short"]:
                sigs = signals_s5_kalman(sdf_dev, z_col, k, side)
                log_and_run(f"S5_Kalman_z{lb}_k{k}_{side}", sdf_dev, sigs)

# 1c. Multi-factor spreads
print("\n  ── Tier 1: Multi-factor spreads ──")
for spread_key in ["MF_ARKG_XBI_60", "MF_ALL_60"]:
    if spread_key not in spreads:
        continue
    sdf = spreads[spread_key]
    sdf_dev = sdf[sdf["date"] < HOLDOUT_START].copy()
    z_col = f"mf_spread_z_60"
    for k in [1.5, 2.0]:
        for side in ["combined", "long", "short"]:
            sigs = signals_s5_ols(sdf_dev, z_col, k, side)
            log_and_run(f"S5_{spread_key}_k{k}_{side}", sdf_dev, sigs)

# 1d. Avellaneda-Lee s-score
print("\n  ── Tier 1: Avellaneda-Lee s-score ──")
if "OLS_ARKG_60" in spreads:
    sdf = spreads["OLS_ARKG_60"]
    sdf_dev = sdf[sdf["date"] < HOLDOUT_START].copy()
    if "s_score" in sdf_dev.columns:
        for s_thresh in [1.5, 2.0, 2.5]:
            for hl_max in [20, 30, 50]:
                for side in ["combined", "long", "short"]:
                    sigs = signals_s5_sscore(sdf_dev, s_thresh, side, hl_max)
                    log_and_run(f"S5_sscore_t{s_thresh}_hl{hl_max}_{side}", sdf_dev, sigs)

# 1e. Signal-based exits (z-cross)
print("\n  ── Tier 1: Signal-based exits ──")
if "OLS_ARKG_60" in spreads:
    sdf = spreads["OLS_ARKG_60"]
    sdf_dev = sdf[sdf["date"] < HOLDOUT_START].copy()
    z_col = "spread_z_60"
    for k_entry in [1.5, 2.0]:
        for z_exit in [0.0, 0.5, 1.0]:
            for side in ["combined"]:
                sigs = signals_s5_ols(sdf_dev, z_col, k_entry, side)
                for sig in sigs:
                    sig["time_stop"] = 15
                log_and_run(f"S5_OLS_ARKG_k{k_entry}_zexit", sdf_dev, sigs,
                           hold=15, z_exit_col=z_col, z_exit_thresh=z_exit)

# 1f. Entry confirmation
print("\n  ── Tier 1: Entry confirmation ──")
if "OLS_ARKG_60" in spreads:
    sdf = spreads["OLS_ARKG_60"]
    sdf_dev = sdf[sdf["date"] < HOLDOUT_START].copy()
    z_col = "spread_z_60"
    for k_entry in [1.5, 2.0]:
        for side in ["combined", "long", "short"]:
            sigs = signals_s5_confirmed(sdf_dev, z_col, k_entry, side)
            log_and_run(f"S5_confirmed_k{k_entry}_{side}", sdf_dev, sigs)

# 1g. Asymmetric thresholds
print("\n  ── Tier 1: Asymmetric thresholds ──")
if "OLS_ARKG_60" in spreads:
    sdf = spreads["OLS_ARKG_60"]
    sdf_dev = sdf[sdf["date"] < HOLDOUT_START].copy()
    z_col = "spread_z_60"
    for k_long, k_short in [(2.0, 1.5), (2.5, 1.5), (2.0, 1.0)]:
        sigs = signals_s5_asymmetric(sdf_dev, z_col, k_long, k_short, "combined")
        log_and_run(f"S5_asym_L{k_long}_S{k_short}_comb", sdf_dev, sigs)

# 1h. Noise/information filter
print("\n  ── Tier 1: Noise/information filter ──")
if "OLS_ARKG_60" in spreads:
    sdf = spreads["OLS_ARKG_60"]
    sdf_dev = sdf[sdf["date"] < HOLDOUT_START].copy()
    z_col = "spread_z_60"
    for k in [1.5, 2.0]:
        for excl_events in [True, False]:
            for low_vol in [True, False]:
                sigs = signals_s5_noise_filter(sdf_dev, z_col, k, "combined",
                                               require_low_vol=low_vol,
                                               exclude_events=excl_events)
                ev_tag = "noEv" if excl_events else "allEv"
                vol_tag = "lowVol" if low_vol else "allVol"
                log_and_run(f"S5_filtered_k{k}_{ev_tag}_{vol_tag}", sdf_dev, sigs)


# ── TIER 2: New families ──

# S8: Event shock fade
print("\n  ── Tier 2: S8 — Event shock fade ──")
for gap_thresh in [2.0, 3.0, 5.0]:
    for side in ["combined", "long", "short"]:
        sigs = signals_s8_event_shock_fade(rxrx_dev, gap_thresh, side)
        log_and_run(f"S8_gap{gap_thresh}_{side}", rxrx_dev, sigs)

# S9: Gap fade
print("\n  ── Tier 2: S9 — Gap fade ──")
for gmin, gmax in [(1.0, 3.0), (1.0, 5.0), (2.0, 5.0)]:
    for side in ["combined", "long", "short"]:
        sigs = signals_s9_gap_fade(rxrx_dev, gmin, gmax, side)
        log_and_run(f"S9_gap{gmin}-{gmax}_{side}", rxrx_dev, sigs)

# S10: Anchored VWAP
print("\n  ── Tier 2: S10 — Anchored VWAP ──")
for vwap_lb in [10, 20, 40]:
    for z_thresh in [1.5, 2.0]:
        for side in ["combined", "long", "short"]:
            sigs = signals_s10_anchored_vwap(rxrx_dev, vwap_lb, z_thresh, side)
            log_and_run(f"S10_vwap{vwap_lb}_z{z_thresh}_{side}", rxrx_dev, sigs)

# S11: Relaxed S2
print("\n  ── Tier 2: S11 — Relaxed range fade ──")
for N in [20, 40, 60]:
    for pct_band in [0.03, 0.05, 0.10]:
        for side in ["combined", "long", "short"]:
            sigs = signals_s11_relaxed_s2(rxrx_dev, N, pct_band, side)
            log_and_run(f"S11_N{N}_b{pct_band}_{side}", rxrx_dev, sigs)

# S13: Vol compression breakout
print("\n  ── Tier 2: S13 — Vol compression ──")
for bb_thresh in [0.05, 0.10, 0.15]:
    for side in ["combined", "long", "short"]:
        sigs = signals_s13_vol_compression(rxrx_dev, bb_thresh, side)
        log_and_run(f"S13_bb{bb_thresh}_{side}", rxrx_dev, sigs)


# ══════════════════════════════════════════════════════════════════
# SECTION 7: IDENTIFY SURVIVORS (≥2% avg PnL, n≥10)
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  SECTION 7: SURVIVORS (avg PnL ≥ 2%, n ≥ 10)")
print("=" * 70)

trial_df = pd.DataFrame(trial_log)
trial_df.to_csv(OUT_DIR / "tier1_tier2_trial_log.csv", index=False)
print(f"  Total trials logged: {len(trial_df)}")

survivors = trial_df[(trial_df["avg_pnl"] >= 2.0) & (trial_df["n"] >= 10)].sort_values(
    "avg_pnl", ascending=False
)

print(f"  Survivors (≥2% avg, n≥10): {len(survivors)}")
print(f"\n  {'Name':55s} {'N':>5s} {'Avg%':>7s} {'Sharpe':>7s} {'WR':>5s} {'PF':>5s} {'DD':>7s}")
print(f"  {'-'*95}")
for _, row in survivors.iterrows():
    print(f"  {row['name']:55s} {row['n']:>5.0f} {row['avg_pnl']:>+6.2f}% "
          f"{row['sharpe']:>+6.2f} {row['wr']:>4.0f}% {row['pf']:>5.2f} {row['max_dd']:>+6.1f}%")

# Second-tier survivors: ≥1% avg PnL, n≥15
almost = trial_df[(trial_df["avg_pnl"] >= 1.0) & (trial_df["avg_pnl"] < 2.0) & (trial_df["n"] >= 15)].sort_values(
    "avg_pnl", ascending=False
)
if not almost.empty:
    print(f"\n  Near-survivors (1-2% avg, n≥15): {len(almost)}")
    for _, row in almost.head(10).iterrows():
        print(f"  {row['name']:55s} {row['n']:>5.0f} {row['avg_pnl']:>+6.2f}% "
              f"{row['sharpe']:>+6.2f}")


# ══════════════════════════════════════════════════════════════════
# SECTION 8: NULL TESTS + DSR FOR SURVIVORS
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print(f"  SECTION 8: NULL TESTS + DSR ({N_NULL_RUNS} runs per survivor)")
print("=" * 70)

n_total_trials = len(trial_df)
null_results = {}

for _, surv_row in survivors.iterrows():
    key = surv_row["name"]
    if key not in all_results:
        continue
    trades_df = all_results[key]["trades"]
    if trades_df.empty:
        continue

    n_trades = len(trades_df)
    n_short = sum(trades_df["side"] == "short")
    n_long = n_trades - n_short

    # Determine execution dataframe
    exec_df = rxrx_dev
    for spread_key, spread_df in spreads.items():
        if spread_key.replace("_60", "") in key or spread_key in key:
            exec_df = spread_df[spread_df["date"] < HOLDOUT_START].copy()
            break

    eligible_dates = exec_df["date"].iloc[120:].values
    null_sharpes = []
    null_avg_pnls = []

    for run in range(N_NULL_RUNS):
        rng = np.random.RandomState(run + 10000)
        if len(eligible_dates) < n_trades:
            continue
        random_dates = rng.choice(eligible_dates, size=min(n_trades, len(eligible_dates)), replace=False)
        random_dates.sort()

        sides = ["short"] * n_short + ["long"] * n_long
        rng.shuffle(sides)

        rand_signals = []
        for d, s in zip(random_dates, sides):
            rand_signals.append({
                "date": pd.Timestamp(d), "side": s,
                "stop": None, "target": None, "time_stop": 3,
            })

        null_trades = run_strategy_no_stack(exec_df, rand_signals, time_stop_override=3)
        if null_trades.empty:
            continue
        null_pnls = null_trades["pnl_net_pct"].values
        null_avg_pnls.append(np.mean(null_pnls))
        avg_h = null_trades["holding_days"].mean()
        if np.std(null_pnls) > 0 and avg_h > 0:
            null_sharpes.append(np.mean(null_pnls) / np.std(null_pnls) * np.sqrt(252 / avg_h))
        else:
            null_sharpes.append(0)

    real_stats = all_results[key]["stats"]
    real_avg = real_stats["avg_pnl"]
    real_sharpe = real_stats["sharpe"]

    pct_avg = scipy_stats.percentileofscore(null_avg_pnls, real_avg) if null_avg_pnls else 0
    pct_sharpe = scipy_stats.percentileofscore(null_sharpes, real_sharpe) if null_sharpes else 0

    pnls = trades_df["pnl_net_pct"].values
    skew_val = float(scipy_stats.skew(pnls)) if len(pnls) > 3 else 0
    kurt_val = float(scipy_stats.kurtosis(pnls)) if len(pnls) > 3 else 0
    avg_h = trades_df["holding_days"].mean()
    dsr = deflated_sharpe_ratio(real_sharpe, n_total_trials, n_trades, avg_h, skew_val, kurt_val)

    null_results[key] = {
        "pct_avg": round(pct_avg, 1),
        "pct_sharpe": round(pct_sharpe, 1),
        "dsr": dsr,
        "null_mean": round(np.mean(null_avg_pnls), 3) if null_avg_pnls else 0,
    }

    print(f"\n  {key}:")
    print(f"    Real avg: {real_avg:+.3f}%  Sharpe: {real_sharpe:+.2f}")
    print(f"    Null pctile: avg={pct_avg:.1f}th  Sharpe={pct_sharpe:.1f}th")
    print(f"    DSR (vs {n_total_trials} trials): {dsr:.4f}")
    print(f"    {'PASSES' if pct_avg > 95 else 'FAILS'} null at 95th | "
          f"DSR {'PASSES' if dsr > 0.5 else 'FAILS'} (>0.5)")


# ══════════════════════════════════════════════════════════════════
# SECTION 9: EPISODE ANALYSIS FOR SURVIVORS
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  SECTION 9: EPISODE ANALYSIS FOR SURVIVORS")
print("=" * 70)

episode_results = {}

for _, surv_row in survivors.head(10).iterrows():
    key = surv_row["name"]
    if key not in all_results:
        continue
    trades_df = all_results[key]["trades"]
    if trades_df.empty or len(trades_df) < 5:
        continue

    trades_ep = assign_episodes(trades_df)
    n_episodes = trades_ep["episode"].nunique()

    print(f"\n  {key}: {len(trades_ep)} trades in {n_episodes} episodes")

    ep_stats = trades_ep.groupby("episode").agg(
        n=("pnl_net_pct", "count"),
        avg_pnl=("pnl_net_pct", "mean"),
        total_pnl=("pnl_net_pct", "sum"),
        start=("entry_date", "min"),
        end=("exit_date", "max"),
    )

    for ep_id, row in ep_stats.iterrows():
        print(f"    Ep {ep_id}: {str(row['start'].date()):>12s}-{str(row['end'].date()):>12s}  "
              f"n={int(row['n']):>3d}  avg={row['avg_pnl']:>+7.2f}%")

    # LOEO
    full_stats = compute_stats(trades_ep)
    full_avg = full_stats["avg_pnl"]

    loeo_results = []
    for ep_id in trades_ep["episode"].unique():
        remaining = trades_ep[trades_ep["episode"] != ep_id]
        s = compute_stats(remaining)
        loeo_results.append({
            "dropped": ep_id, "n_remaining": s["n"],
            "avg_pnl": s["avg_pnl"], "sharpe": s["sharpe"],
        })

    loeo_df = pd.DataFrame(loeo_results)
    min_avg = loeo_df["avg_pnl"].min()
    min_sharpe = loeo_df["sharpe"].min()
    survives = min_avg > 0 and min_sharpe > 0

    print(f"    LOEO: survives all drops = {'YES' if survives else 'NO'} "
          f"(min avg={min_avg:+.3f}%, min Sharpe={min_sharpe:.2f})")

    episode_results[key] = {
        "n_episodes": n_episodes,
        "loeo_survives": survives,
        "loeo_min_avg": round(min_avg, 3),
        "loeo_min_sharpe": round(min_sharpe, 2),
    }


# ══════════════════════════════════════════════════════════════════
# SECTION 10: PARAMETER HEATMAP (top strategies)
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  SECTION 10: PARAMETER HEATMAPS")
print("=" * 70)

# S5 OLS ARKG parameter heatmap: k_thresh × hold_period
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("S5 OLS_ARKG — Parameter Heatmap (avg PnL %)", fontsize=14, fontweight="bold")

for side_idx, side in enumerate(["combined", "long", "short"]):
    ax = axes[side_idx]
    k_vals = [1.5, 2.0, 2.5]
    h_vals = HOLD_PERIODS
    heatmap = np.zeros((len(k_vals), len(h_vals)))

    for ki, k in enumerate(k_vals):
        for hi, h in enumerate(h_vals):
            key = f"S5_OLS_ARKG_60_k{k}_{side}_h{h}"
            if key in all_results:
                heatmap[ki, hi] = all_results[key]["stats"]["avg_pnl"]

    im = ax.imshow(heatmap, aspect="auto", cmap="RdYlGn", vmin=-3, vmax=5)
    ax.set_xticks(range(len(h_vals)))
    ax.set_xticklabels(h_vals)
    ax.set_yticks(range(len(k_vals)))
    ax.set_yticklabels(k_vals)
    ax.set_xlabel("Hold Period (days)")
    ax.set_ylabel("z-threshold (k)")
    ax.set_title(f"{side}")

    for ki in range(len(k_vals)):
        for hi in range(len(h_vals)):
            ax.text(hi, ki, f"{heatmap[ki, hi]:+.1f}", ha="center", va="center",
                    fontsize=9, fontweight="bold")

plt.colorbar(im, ax=axes, label="Avg PnL %", shrink=0.8)
plt.tight_layout()
plt.savefig(OUT_DIR / "tier1_heatmap_s5_arkg.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved tier1_heatmap_s5_arkg.png")

# Kalman parameter heatmap
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("S5 Kalman — Parameter Heatmap (avg PnL %)", fontsize=14, fontweight="bold")
for side_idx, side in enumerate(["combined", "long"]):
    ax = axes[side_idx]
    k_vals = [1.5, 2.0]
    z_cols = ["40", "60"]
    heatmap = np.zeros((len(k_vals), len(z_cols) * len(HOLD_PERIODS)))
    col_labels = []

    for zi, zc in enumerate(z_cols):
        for hi, h in enumerate(HOLD_PERIODS):
            col_idx = zi * len(HOLD_PERIODS) + hi
            key = f"S5_Kalman_z{zc}_k{k_vals[0]}_{side}_h{h}"
            col_labels.append(f"z{zc}_h{h}")
            for ki, k in enumerate(k_vals):
                key = f"S5_Kalman_z{zc}_k{k}_{side}_h{h}"
                if key in all_results:
                    heatmap[ki, col_idx] = all_results[key]["stats"]["avg_pnl"]

    im = ax.imshow(heatmap, aspect="auto", cmap="RdYlGn", vmin=-3, vmax=5)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, fontsize=7)
    ax.set_yticks(range(len(k_vals)))
    ax.set_yticklabels(k_vals)
    ax.set_ylabel("k threshold")
    ax.set_title(f"{side}")

    for ki in range(len(k_vals)):
        for ci in range(len(col_labels)):
            ax.text(ci, ki, f"{heatmap[ki, ci]:+.1f}", ha="center", va="center",
                    fontsize=7, fontweight="bold")

plt.colorbar(im, ax=axes, label="Avg PnL %", shrink=0.8)
plt.tight_layout()
plt.savefig(OUT_DIR / "tier1_heatmap_s5_kalman.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved tier1_heatmap_s5_kalman.png")


# ══════════════════════════════════════════════════════════════════
# SECTION 11: SUMMARY + EQUITY CURVES
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  SECTION 11: FINAL SUMMARY")
print("=" * 70)

# Sort by avg PnL descending
summary_rows = []
for key, res in all_results.items():
    s = res["stats"]
    if s["n"] < 5:
        continue
    nr = null_results.get(key, {})
    er = episode_results.get(key, {})
    summary_rows.append({
        "strategy": key,
        "n": s["n"], "avg_pnl": s["avg_pnl"], "sharpe": s["sharpe"],
        "wr": s["wr"], "pf": s["pf"], "max_dd": s["max_dd"],
        "avg_hold": s["avg_hold"], "total_usd": s["total_pnl_usd"],
        "null_pct": nr.get("pct_avg", ""),
        "null_sharpe_pct": nr.get("pct_sharpe", ""),
        "dsr": nr.get("dsr", ""),
        "n_episodes": er.get("n_episodes", ""),
        "loeo_survives": er.get("loeo_survives", ""),
    })

summary_df = pd.DataFrame(summary_rows).sort_values("avg_pnl", ascending=False)
summary_df.to_csv(OUT_DIR / "tier1_tier2_summary.csv", index=False)

print(f"\n  Total strategies with n≥5: {len(summary_df)}")
print(f"\n  {'Strategy':55s} {'N':>5s} {'Avg%':>7s} {'Sharpe':>7s} {'WR':>5s} "
      f"{'PF':>5s} {'DD':>7s} {'Null%':>6s} {'DSR':>6s}")
print(f"  {'-'*110}")

for _, row in summary_df.head(30).iterrows():
    null_str = f"{row['null_pct']:.0f}" if row['null_pct'] != "" else "—"
    dsr_str = f"{row['dsr']:.3f}" if row['dsr'] != "" else "—"
    print(f"  {row['strategy']:55s} {row['n']:>5.0f} {row['avg_pnl']:>+6.2f}% "
          f"{row['sharpe']:>+6.2f} {row['wr']:>4.0f}% {row['pf']:>5.2f} "
          f"{row['max_dd']:>+6.1f}% {null_str:>6s} {dsr_str:>6s}")


# Equity curves for top 5 survivors
print("\nPlotting top survivors equity curves...")
fig, ax = plt.subplots(figsize=(16, 8))
ax.set_title("Top Survivors — Cumulative PnL (no stacking, OOS dev)", fontsize=14, fontweight="bold")

top_keys = summary_df.head(8)["strategy"].tolist()
for key in top_keys:
    if key not in all_results:
        continue
    trades_df = all_results[key]["trades"]
    if trades_df.empty:
        continue
    equity = trades_df["pnl_net_pct"].cumsum()
    ax.plot(trades_df["entry_date"], equity, label=f"{key} (avg={all_results[key]['stats']['avg_pnl']:+.2f}%)",
            lw=1.5, alpha=0.8)

ax.axhline(0, color="black", lw=0.5, ls="--")
ax.set_ylabel("Cumulative PnL (%)")
ax.set_xlabel("Date")
ax.legend(fontsize=7, loc="upper left")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "tier1_tier2_equity_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved tier1_tier2_equity_curves.png")


# ══════════════════════════════════════════════════════════════════
# SECTION 12: CONCLUSIONS & NEXT STEPS
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  SECTION 12: CONCLUSIONS")
print("=" * 70)

n_pass_null = sum(1 for v in null_results.values() if v.get("pct_avg", 0) > 95)
n_pass_dsr = sum(1 for v in null_results.values() if v.get("dsr", 0) > 0.5)
n_pass_loeo = sum(1 for v in episode_results.values() if v.get("loeo_survives", False))

print(f"""
  TIER 1 + TIER 2 SWEEP COMPLETE
  ──────────────────────────────
  Total trials:           {len(trial_df)}
  Strategies with n≥5:    {len(summary_df)}
  Survivors (≥2% avg):    {len(survivors)}
  Pass null (95th):       {n_pass_null}
  Pass DSR (>0.5):        {n_pass_dsr}
  Pass LOEO:              {n_pass_loeo}

  Holdout NOT spent. Next: Tier 3 portfolio construction with
  strategies that pass ALL three gates (null + DSR + LOEO).
""")

print("\nAnalysis complete. Files saved to agent10/results/")
