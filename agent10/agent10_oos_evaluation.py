# %% Agent 10 — Out-of-Sample Evaluation (2026)
#
# Protocol:
#   1. Selection on 2021-04-16 to 2025-12-31 (no 2026 data)
#   2. Pre-register max 1 config per family (max 5 total)
#   3. Freeze configs, THEN load 2026 data
#   4. Evaluate on Jan-Mar 2026 ("semi-seen") + Apr+ ("clean holdout")
#   5. Peer basket generalization
#
# All exits capped at 10 trading days.
# Sharpe from daily MTM equity including flat days.

import os, sys, json, warnings, hashlib
from pathlib import Path
from datetime import datetime, date
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

for pkg in ["statsmodels", "pykalman"]:
    try:
        __import__(pkg)
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg,
                               "--quiet", "--break-system-packages"])

import statsmodels.api as sm

# ── Config ────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data" if "__file__" in dir() else Path("agent10/data")
OUT_DIR = Path(__file__).parent / "results" if "__file__" in dir() else Path("agent10/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

HALF_SPREAD_BPS = 10
SLIPPAGE_BPS = 7.5
COMMISSION_PER_SHARE = 0.005
SHORT_BORROW_ANNUAL_PCT = 5.0
POSITION_SIZE_USD = 10000

MAX_HOLD = 10  # hard cap
N_NULL_RUNS = 1000
N_BOOTSTRAP = 5000

SELECTION_END = "2026-01-01"
SEMI_SEEN_END = "2026-04-01"

PEER_TICKERS = ["SDGR", "ABCL", "ABSI", "BEAM", "CRSP", "NTLA", "TXG", "PACB", "TWST"]

# ══════════════════════════════════════════════════════════════════
# SECTION 0: LOAD SELECTION-PERIOD DATA ONLY (no 2026)
# ══════════════════════════════════════════════════════════════════
# %%
print("=" * 70)
print("  SECTION 0: LOAD SELECTION DATA (pre-2026 only)")
print("=" * 70)

rxrx_all = pd.read_parquet(DATA_DIR / "rxrx_daily.parquet")
rxrx_all["date"] = pd.to_datetime(rxrx_all["date"])
rxrx_all = rxrx_all.sort_values("date").reset_index(drop=True)

ctx = pd.read_parquet(DATA_DIR / "context_daily.parquet")
ctx["date"] = pd.to_datetime(ctx["date"])

# SELECTION ONLY — no 2026 data loaded yet
rxrx_sel = rxrx_all[rxrx_all["date"] < SELECTION_END].copy()
print(f"  Selection: {len(rxrx_sel)} bars, {rxrx_sel['date'].min().date()} to {rxrx_sel['date'].max().date()}")

# ══════════════════════════════════════════════════════════════════
# SECTION 1: FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════
# %%

def add_features(df):
    df = df.copy()
    df["log_price"] = np.log(df["close"])
    df["log_ret"] = df["log_price"].diff()
    df["pct_ret"] = df["close"].pct_change()

    high, low, prev_close = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    df["atr20"] = tr.rolling(20).mean()
    df["atr_pct"] = df["atr20"] / df["close"] * 100

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(2).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(2).mean()
    rs = gain / loss
    df["rsi2"] = 100 - (100 / (1 + rs))

    df["ibs"] = (df["close"] - df["low"]) / (df["high"] - df["low"])
    df["gap_pct"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1) * 100
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma20"]

    for L in [10, 20, 30, 60]:
        df[f"z_{L}"] = (df["log_price"] - df["log_price"].rolling(L).mean()) / df["log_price"].rolling(L).std()

    for N in [20, 40, 60]:
        df[f"range_high_{N}"] = df["high"].rolling(N).max()
        df[f"range_low_{N}"] = df["low"].rolling(N).min()
        df[f"range_mid_{N}"] = (df[f"range_high_{N}"] + df[f"range_low_{N}"]) / 2

    consec_down = np.zeros(len(df))
    consec_up = np.zeros(len(df))
    for i in range(1, len(df)):
        if df["close"].iloc[i] < df["close"].iloc[i-1]:
            consec_down[i] = consec_down[i-1] + 1
        if df["close"].iloc[i] > df["close"].iloc[i-1]:
            consec_up[i] = consec_up[i-1] + 1
    df["consec_down"] = consec_down
    df["consec_up"] = consec_up

    ma = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    df["bb_width_20"] = (2 * std / ma) * 100
    df["bb_width_pctile_20"] = df["bb_width_20"].rolling(120).rank(pct=True)

    return df

rxrx_sel = add_features(rxrx_sel)

# Context ETFs
arkg_all = ctx[ctx["ticker"] == "ARKG"].sort_values("date").reset_index(drop=True)
xbi_all = ctx[ctx["ticker"] == "XBI"].sort_values("date").reset_index(drop=True)
iwm_all = ctx[ctx["ticker"] == "IWM"].sort_values("date").reset_index(drop=True)

arkg_sel = arkg_all[arkg_all["date"] < SELECTION_END].copy()
xbi_sel = xbi_all[xbi_all["date"] < SELECTION_END].copy()
iwm_sel = iwm_all[iwm_all["date"] < SELECTION_END].copy()

print(f"  ARKG: {len(arkg_sel)} bars | XBI: {len(xbi_sel)} bars | IWM: {len(iwm_sel)} bars")


# ══════════════════════════════════════════════════════════════════
# SECTION 2: SPREAD CONSTRUCTION
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  SECTION 2: SPREAD CONSTRUCTION (selection window)")
print("=" * 70)

def build_ols_spread(target_df, hedge_df, hedge_name, lookback=60):
    merged = target_df[["date", "close", "open", "high", "low", "log_price",
                         "atr20", "atr_pct", "volume"]].merge(
        hedge_df[["date", "close"]].rename(columns={"close": f"{hedge_name}_close"}),
        on="date", how="inner"
    )
    merged[f"{hedge_name}_log"] = np.log(merged[f"{hedge_name}_close"])
    merged["target_log"] = merged["log_price"]

    merged[f"spread_{lookback}"] = np.nan
    merged[f"hedge_{lookback}"] = np.nan

    for i in range(lookback, len(merged)):
        y = merged["target_log"].iloc[i-lookback:i+1].values
        x = sm.add_constant(merged[f"{hedge_name}_log"].iloc[i-lookback:i+1].values)
        try:
            model = sm.OLS(y, x).fit()
            merged.iloc[i, merged.columns.get_loc(f"hedge_{lookback}")] = model.params[1]
            merged.iloc[i, merged.columns.get_loc(f"spread_{lookback}")] = y[-1] - model.predict(x)[-1]
        except:
            pass

    spread = merged[f"spread_{lookback}"]
    merged[f"spread_z_{lookback}"] = (spread - spread.rolling(lookback).mean()) / spread.rolling(lookback).std()

    # Avellaneda-Lee s-score
    cum_resid = spread.diff().cumsum()
    merged["s_score"] = np.nan
    merged["ou_kappa"] = np.nan
    for i in range(lookback + 1, len(merged)):
        seg = cum_resid.iloc[i-lookback:i].values
        if np.any(np.isnan(seg)):
            continue
        dy = np.diff(seg)
        y_lag = seg[:-1]
        X = sm.add_constant(y_lag)
        try:
            model = sm.OLS(dy, X).fit()
            b = model.params[1]
            if b >= 0:
                continue
            kappa = -b
            m = model.params[0] / kappa
            sigma_eq = np.sqrt(model.mse_resid / (2 * kappa))
            if sigma_eq > 0:
                merged.iloc[i, merged.columns.get_loc("s_score")] = (seg[-1] - m) / sigma_eq
                merged.iloc[i, merged.columns.get_loc("ou_kappa")] = kappa
        except:
            pass

    return merged

spreads_sel = {}
for name, hdf in [("ARKG", arkg_sel), ("XBI", xbi_sel), ("IWM", iwm_sel)]:
    sp = build_ols_spread(rxrx_sel, hdf, name, 60)
    spreads_sel[name] = sp
    valid_z = sp["spread_z_60"].dropna()
    print(f"  {name}: {len(sp)} bars, z-range [{valid_z.min():.2f}, {valid_z.max():.2f}]")


# ══════════════════════════════════════════════════════════════════
# SECTION 3: TRADE ENGINE (10-day cap, fixed Sharpe)
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


def run_strategy(df, signals, max_hold=MAX_HOLD, z_exit_col=None, z_exit_thresh=None):
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
        exit_price = None
        exit_reason = None
        exit_date = None

        for j in range(entry_idx + 1, min(entry_idx + max_hold + 1, len(df))):
            # Stop loss
            if sig["side"] == "long":
                if sig.get("stop") and df["low"].iloc[j] <= sig["stop"]:
                    exit_price, exit_reason = sig["stop"], "stop"
                    exit_date = df["date"].iloc[j]
                    break
            else:
                if sig.get("stop") and df["high"].iloc[j] >= sig["stop"]:
                    exit_price, exit_reason = sig["stop"], "stop"
                    exit_date = df["date"].iloc[j]
                    break

            # z-exit
            if z_exit_col and z_exit_thresh is not None and z_exit_col in df.columns:
                z_now = df[z_exit_col].iloc[j]
                if not np.isnan(z_now):
                    if sig["side"] == "long" and z_now >= -z_exit_thresh:
                        exit_price, exit_reason = df["close"].iloc[j], "z_cross"
                        exit_date = df["date"].iloc[j]
                        break
                    elif sig["side"] == "short" and z_now <= z_exit_thresh:
                        exit_price, exit_reason = df["close"].iloc[j], "z_cross"
                        exit_date = df["date"].iloc[j]
                        break

            # Target
            if sig["side"] == "long" and sig.get("target") and df["high"].iloc[j] >= sig["target"]:
                exit_price, exit_reason = sig["target"], "target"
                exit_date = df["date"].iloc[j]
                break
            if sig["side"] == "short" and sig.get("target") and df["low"].iloc[j] <= sig["target"]:
                exit_price, exit_reason = sig["target"], "target"
                exit_date = df["date"].iloc[j]
                break

        if exit_price is None:
            exit_idx = min(entry_idx + max_hold, len(df) - 1)
            exit_price = df["close"].iloc[exit_idx]
            exit_date = df["date"].iloc[exit_idx]
            exit_reason = "time_stop"

        in_trade_until = exit_date
        holding_days = max((exit_date - entry_date).days, 1)

        pnl_pct = ((exit_price - entry_price) / entry_price * 100) if sig["side"] == "long" \
                  else ((entry_price - exit_price) / entry_price * 100)
        cost_pct = apply_costs(entry_price, sig["side"], holding_days) / entry_price * 100
        pnl_net = pnl_pct - cost_pct
        shares = int(POSITION_SIZE_USD / entry_price)

        trades.append({
            "entry_date": entry_date, "exit_date": exit_date,
            "side": sig["side"], "entry_price": round(entry_price, 4),
            "exit_price": round(exit_price, 4), "exit_reason": exit_reason,
            "holding_days": holding_days,
            "pnl_gross_pct": round(pnl_pct, 4), "cost_pct": round(cost_pct, 4),
            "pnl_net_pct": round(pnl_net, 4),
            "shares": shares, "pnl_dollars": round(pnl_net / 100 * shares * entry_price, 2),
        })

    return pd.DataFrame(trades)


def daily_mtm_sharpe(trades_df, price_df):
    """Sharpe from daily MTM equity INCLUDING flat days."""
    if trades_df.empty:
        return 0.0
    dates = price_df["date"].values
    closes = dict(zip(price_df["date"], price_df["close"]))
    min_d = trades_df["entry_date"].min()
    max_d = trades_df["exit_date"].max()
    all_dates = price_df[(price_df["date"] >= min_d) & (price_df["date"] <= max_d)]["date"].values

    daily_pnl = pd.Series(0.0, index=all_dates)
    for _, t in trades_df.iterrows():
        td = price_df[(price_df["date"] >= t["entry_date"]) & (price_df["date"] <= t["exit_date"])]["date"].values
        prev = t["entry_price"]
        for d in td:
            curr = closes.get(d, prev)
            ret = ((curr - prev) / t["entry_price"] * 100) if t["side"] == "long" \
                  else ((prev - curr) / t["entry_price"] * 100)
            daily_pnl.loc[d] += ret
            prev = curr

    if daily_pnl.std() == 0:
        return 0.0
    return float(daily_pnl.mean() / daily_pnl.std() * np.sqrt(252))


def compute_stats(trades_df, price_df=None, label=""):
    if trades_df.empty:
        return {"label": label, "n": 0, "avg_pnl": 0, "median_pnl": 0, "sharpe": 0,
                "wr": 0, "pf": 0, "max_dd": 0, "total_pnl_pct": 0, "total_pnl_usd": 0,
                "avg_hold": 0, "pnl_per_day": 0}
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
    max_dd = (equity - peak).min()
    avg_hold = trades_df["holding_days"].mean()
    pnl_per_day = avg / avg_hold if avg_hold > 0 else 0

    if price_df is not None:
        sharpe = daily_mtm_sharpe(trades_df, price_df)
    else:
        sharpe = avg / (np.std(pnls) + 1e-10) * np.sqrt(252 / max(avg_hold, 1))

    return {
        "label": label, "n": n, "avg_pnl": round(avg, 3), "median_pnl": round(med, 3),
        "sharpe": round(sharpe, 2), "wr": round(wr, 1),
        "pf": round(pf, 2), "max_dd": round(max_dd, 2),
        "total_pnl_pct": round(np.sum(pnls), 2),
        "total_pnl_usd": round(trades_df["pnl_dollars"].sum(), 2),
        "avg_hold": round(avg_hold, 1), "pnl_per_day": round(pnl_per_day, 3),
    }


# ══════════════════════════════════════════════════════════════════
# SECTION 4: SIGNAL GENERATORS (all families)
# ══════════════════════════════════════════════════════════════════
# %%

def signals_s5_short(spread_df, z_col, k_thresh=2.0):
    signals = []
    for i in range(120, len(spread_df)):
        z = spread_df[z_col].iloc[i]
        if pd.isna(z):
            continue
        atr = spread_df["atr20"].iloc[i]
        if pd.isna(atr):
            continue
        price = spread_df["close"].iloc[i]
        if z > k_thresh:
            signals.append({
                "date": spread_df["date"].iloc[i], "side": "short",
                "stop": price + 1.5 * atr, "target": None, "time_stop": MAX_HOLD,
            })
    return signals


def signals_s5_zexit(spread_df, z_col, k_thresh=2.0, side="combined"):
    signals = []
    for i in range(120, len(spread_df)):
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
                "stop": price - 1.5 * atr, "target": None, "time_stop": MAX_HOLD,
            })
        if side in ("short", "combined") and z > k_thresh:
            signals.append({
                "date": spread_df["date"].iloc[i], "side": "short",
                "stop": price + 1.5 * atr, "target": None, "time_stop": MAX_HOLD,
            })
    return signals


def signals_s5_sscore_short(spread_df, s_thresh=2.0):
    signals = []
    if "s_score" not in spread_df.columns:
        return signals
    for i in range(120, len(spread_df)):
        s = spread_df["s_score"].iloc[i]
        if pd.isna(s):
            continue
        atr = spread_df["atr20"].iloc[i]
        if pd.isna(atr):
            continue
        price = spread_df["close"].iloc[i]
        if s > s_thresh:
            signals.append({
                "date": spread_df["date"].iloc[i], "side": "short",
                "stop": price + 1.5 * atr, "target": None, "time_stop": MAX_HOLD,
            })
    return signals


def signals_s5_confirmed_short(spread_df, z_col, k_entry=2.0):
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
        if z_prev > k_entry and z_now < z_prev and z_now > k_entry * 0.5:
            signals.append({
                "date": spread_df["date"].iloc[i], "side": "short",
                "stop": price + 1.5 * atr, "target": None, "time_stop": MAX_HOLD,
            })
    return signals


def signals_s10_vwap_short(df, lookback_vwap=40, z_thresh=2.0):
    signals = []
    vwap = (df["close"] * df["volume"]).rolling(lookback_vwap).sum() / df["volume"].rolling(lookback_vwap).sum()
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
        if z > z_thresh:
            signals.append({
                "date": df["date"].iloc[i], "side": "short",
                "stop": price + 1.5 * atr, "target": vwap.iloc[i], "time_stop": MAX_HOLD,
            })
    return signals


def signals_s11_short(df, N=40, pct_band=0.05):
    signals = []
    rh, rl = f"range_high_{N}", f"range_low_{N}"
    if rh not in df.columns:
        return signals
    for i in range(N + 1, len(df)):
        atr = df["atr20"].iloc[i]
        if pd.isna(atr):
            continue
        price = df["close"].iloc[i]
        resistance = df[rh].iloc[i]
        support = df[rl].iloc[i]
        rng = resistance - support
        if pd.isna(rng) or rng < atr:
            continue
        proximity_high = (resistance - price) / rng
        if proximity_high < pct_band:
            signals.append({
                "date": df["date"].iloc[i], "side": "short",
                "stop": resistance + 0.5 * atr, "target": resistance - 0.4 * rng,
                "time_stop": MAX_HOLD,
            })
    return signals


# ══════════════════════════════════════════════════════════════════
# SECTION 5: SELECTION (2021-2025 only, 10-day cap)
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  SECTION 5: SELECTION (2021-2025, max 10-day hold)")
print("=" * 70)

z_col = "spread_z_60"
candidates = []

# Family 1: S5 short, OLS vs XBI/ARKG/IWM, z-exit at 0.5 or time stop
for hedge_name in ["XBI", "ARKG", "IWM"]:
    sp = spreads_sel[hedge_name]
    for k in [1.5, 2.0]:
        sigs = signals_s5_short(sp, z_col, k)
        trades = run_strategy(sp, sigs, z_exit_col=z_col, z_exit_thresh=0.5)
        s = compute_stats(trades, sp, f"S5_short_{hedge_name}_k{k}")
        candidates.append({"family": "S5_short", "config": f"{hedge_name}_k{k}", "hedge": hedge_name,
                           "k": k, "stats": s, "trades": trades, "type": "short_zexit05"})
        print(f"  {s['label']:45s}  n={s['n']:>3d}  avg={s['avg_pnl']:>+7.2f}%  Sharpe={s['sharpe']:>+6.2f}  hold={s['avg_hold']:.1f}d")

# Family 2: S5 z-exit, L+S vs ARKG, k in {1.5,2.0}, z-exit in {0,0.5,1.0}
sp_arkg = spreads_sel["ARKG"]
for k in [1.5, 2.0]:
    for z_exit in [0.0, 0.5, 1.0]:
        sigs = signals_s5_zexit(sp_arkg, z_col, k, "combined")
        trades = run_strategy(sp_arkg, sigs, z_exit_col=z_col, z_exit_thresh=z_exit)
        s = compute_stats(trades, sp_arkg, f"S5_zexit_ARKG_k{k}_ze{z_exit}")
        candidates.append({"family": "S5_zexit", "config": f"ARKG_k{k}_ze{z_exit}", "hedge": "ARKG",
                           "k": k, "z_exit": z_exit, "stats": s, "trades": trades, "type": "zexit"})
        print(f"  {s['label']:45s}  n={s['n']:>3d}  avg={s['avg_pnl']:>+7.2f}%  Sharpe={s['sharpe']:>+6.2f}  hold={s['avg_hold']:.1f}d")

# Family 3: S5 Avellaneda-Lee s-score short
for t in [1.5, 2.0]:
    sigs = signals_s5_sscore_short(sp_arkg, t)
    trades = run_strategy(sp_arkg, sigs)
    s = compute_stats(trades, sp_arkg, f"S5_sscore_short_t{t}")
    candidates.append({"family": "S5_sscore", "config": f"ARKG_t{t}", "hedge": "ARKG",
                       "t": t, "stats": s, "trades": trades, "type": "sscore"})
    print(f"  {s['label']:45s}  n={s['n']:>3d}  avg={s['avg_pnl']:>+7.2f}%  Sharpe={s['sharpe']:>+6.2f}  hold={s['avg_hold']:.1f}d")

# Family 4: S5 confirmed-entry short
for k in [1.5, 2.0]:
    sigs = signals_s5_confirmed_short(sp_arkg, z_col, k)
    trades = run_strategy(sp_arkg, sigs)
    s = compute_stats(trades, sp_arkg, f"S5_confirmed_short_k{k}")
    candidates.append({"family": "S5_confirmed", "config": f"ARKG_k{k}", "hedge": "ARKG",
                       "k": k, "stats": s, "trades": trades, "type": "confirmed"})
    print(f"  {s['label']:45s}  n={s['n']:>3d}  avg={s['avg_pnl']:>+7.2f}%  Sharpe={s['sharpe']:>+6.2f}  hold={s['avg_hold']:.1f}d")

# Family 5a: S10 VWAP short
sigs = signals_s10_vwap_short(rxrx_sel, 40, 2.0)
trades = run_strategy(rxrx_sel, sigs)
s = compute_stats(trades, rxrx_sel, "S10_vwap40_z2_short")
candidates.append({"family": "S10_vwap", "config": "vwap40_z2", "hedge": None,
                   "stats": s, "trades": trades, "type": "vwap"})
print(f"  {s['label']:45s}  n={s['n']:>3d}  avg={s['avg_pnl']:>+7.2f}%  Sharpe={s['sharpe']:>+6.2f}  hold={s['avg_hold']:.1f}d")

# Family 5b: S11 range fade short
for N in [40, 60]:
    for b in [0.03, 0.05]:
        sigs = signals_s11_short(rxrx_sel, N, b)
        trades = run_strategy(rxrx_sel, sigs)
        s = compute_stats(trades, rxrx_sel, f"S11_N{N}_b{b}_short")
        candidates.append({"family": "S11_range", "config": f"N{N}_b{b}", "hedge": None,
                           "N": N, "b": b, "stats": s, "trades": trades, "type": "range"})
        print(f"  {s['label']:45s}  n={s['n']:>3d}  avg={s['avg_pnl']:>+7.2f}%  Sharpe={s['sharpe']:>+6.2f}  hold={s['avg_hold']:.1f}d")

# Also report 2024-2025 subsample for top candidates
print("\n  ── 2024-2025 subsample ──")
for c in candidates:
    tr = c["trades"]
    if tr.empty:
        continue
    sub = tr[(tr["entry_date"] >= "2024-01-01") & (tr["entry_date"] < SELECTION_END)]
    if not sub.empty and len(sub) >= 3:
        s_sub = compute_stats(sub, label=f"  2024-25: {c['stats']['label']}")
        print(f"  {c['stats']['label']:45s}  2024-25: n={s_sub['n']:>3d}  avg={s_sub['avg_pnl']:>+7.2f}%")


# ══════════════════════════════════════════════════════════════════
# SECTION 6: PRE-REGISTRATION (max 1 per family, max 5 total)
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  SECTION 6: PRE-REGISTRATION")
print("=" * 70)

# Selection criteria: avg >= 2%, n >= 15, hold <= 10d
# Rank by total net PnL within qualifying candidates
# Pick best per family

qualifying = [c for c in candidates
              if c["stats"]["avg_pnl"] >= 2.0
              and c["stats"]["n"] >= 15
              and c["stats"]["avg_hold"] <= 10]

# Rank by total PnL within each family
family_best = {}
for c in qualifying:
    fam = c["family"]
    if fam not in family_best or c["stats"]["total_pnl_pct"] > family_best[fam]["stats"]["total_pnl_pct"]:
        family_best[fam] = c

# If fewer than 5 qualifying, also include best from families that nearly qualify (avg >= 1.5%)
if len(family_best) < 5:
    near_qualifying = [c for c in candidates
                       if c["stats"]["avg_pnl"] >= 1.5
                       and c["stats"]["n"] >= 10
                       and c["stats"]["avg_hold"] <= 10
                       and c["family"] not in family_best]
    for c in near_qualifying:
        fam = c["family"]
        if fam not in family_best or c["stats"]["total_pnl_pct"] > family_best[fam]["stats"]["total_pnl_pct"]:
            family_best[fam] = c

pre_registered = list(family_best.values())[:5]

# Build pre-registration document
prereg_timestamp = datetime.utcnow().isoformat() + "Z"
prereg = {
    "timestamp": prereg_timestamp,
    "protocol_version": "1.0",
    "selection_window": "2021-04-16 to 2025-12-31",
    "test_window": "2026-01-01 to latest available",
    "max_hold_days": MAX_HOLD,
    "position_size_usd": POSITION_SIZE_USD,
    "cost_model": {
        "half_spread_bps": HALF_SPREAD_BPS,
        "slippage_bps": SLIPPAGE_BPS,
        "commission_per_share": COMMISSION_PER_SHARE,
        "short_borrow_annual_pct": SHORT_BORROW_ANNUAL_PCT,
    },
    "decision_rule": {
        "min_2026_avg_pnl_pct": 1.5,
        "must_be_in_selection_ci": True,
        "must_have_positive_total_2026_pnl": True,
        "must_have_positive_pooled_peer_avg": True,
        "rxrx_specific_if_peers_fail": "trade at reduced size",
    },
    "configurations": [],
}

for c in pre_registered:
    s = c["stats"]
    config_entry = {
        "family": c["family"],
        "config_name": c["config"],
        "label": s["label"],
        "hedge_factor": c.get("hedge", "none"),
        "side": "short" if "short" in c["type"] or "sscore" in c["type"] else "combined",
        "z_entry_threshold": c.get("k", c.get("t", None)),
        "z_exit_threshold": c.get("z_exit", 0.5 if "short" in c["type"] else None),
        "max_hold_days": MAX_HOLD,
        "stop_loss": "1.5 ATR",
        "selection_stats": {
            "n_trades": s["n"],
            "avg_pnl_pct": s["avg_pnl"],
            "total_pnl_pct": s["total_pnl_pct"],
            "sharpe_daily_mtm": s["sharpe"],
            "win_rate": s["wr"],
            "avg_hold_days": s["avg_hold"],
        },
    }
    prereg["configurations"].append(config_entry)
    print(f"  PRE-REGISTERED: {s['label']}")
    print(f"    n={s['n']}  avg={s['avg_pnl']:+.2f}%  total={s['total_pnl_pct']:+.1f}%  "
          f"Sharpe={s['sharpe']:+.2f}  hold={s['avg_hold']:.1f}d")

# Save pre-registration file
prereg_path = OUT_DIR / "pre_registration.json"
with open(prereg_path, "w") as f:
    json.dump(prereg, f, indent=2, default=str)
print(f"\n  Pre-registration saved: {prereg_path}")
print(f"  Timestamp: {prereg_timestamp}")
print(f"  Configs registered: {len(prereg['configurations'])}")


# ══════════════════════════════════════════════════════════════════
# SECTION 7: LOAD 2026 DATA (NOW permitted)
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  SECTION 7: LOADING 2026 DATA")
print("=" * 70)

rxrx_2026 = rxrx_all[rxrx_all["date"] >= SELECTION_END].copy()
rxrx_2026 = add_features(rxrx_2026)

# We need features built on full history for continuity
rxrx_full = add_features(rxrx_all)
rxrx_2026_full = rxrx_full[rxrx_full["date"] >= SELECTION_END].copy()

print(f"  2026 data: {len(rxrx_2026)} bars, {rxrx_2026['date'].min().date()} to {rxrx_2026['date'].max().date()}")

semi_seen = rxrx_2026_full[rxrx_2026_full["date"] < SEMI_SEEN_END]
clean_holdout = rxrx_2026_full[rxrx_2026_full["date"] >= SEMI_SEEN_END]
print(f"  Semi-seen (Jan-Mar 2026): {len(semi_seen)} bars")
print(f"  Clean holdout (Apr+ 2026): {len(clean_holdout)} bars")

# Build full-history spreads (features need lookback into pre-2026)
spreads_full = {}
for name, hdf in [("ARKG", arkg_all), ("XBI", xbi_all), ("IWM", iwm_all)]:
    hdf = hdf.copy()
    spreads_full[name] = build_ols_spread(rxrx_full, hdf, name, 60)
    print(f"  Full spread {name}: {len(spreads_full[name])} bars")

# Buy-and-hold / short-and-hold benchmarks
bh_start = rxrx_2026_full["open"].iloc[0] if not rxrx_2026_full.empty else 0
bh_end = rxrx_2026_full["close"].iloc[-1] if not rxrx_2026_full.empty else 0
bh_return = (bh_end - bh_start) / bh_start * 100 if bh_start > 0 else 0
sh_return = -bh_return
print(f"\n  Benchmarks (2026 full):")
print(f"    Buy-and-hold:   {bh_return:+.2f}% (${bh_start:.2f} -> ${bh_end:.2f})")
print(f"    Short-and-hold: {sh_return:+.2f}%")


# ══════════════════════════════════════════════════════════════════
# SECTION 8: 2026 EVALUATION (frozen configs)
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  SECTION 8: 2026 EVALUATION (frozen configurations)")
print("=" * 70)

def run_config_on_period(config, period_df, full_spread_dfs, full_rxrx_df):
    """Run a pre-registered config on a specific period."""
    c = config
    fam = c["family"]
    hedge = c.get("hedge")

    if hedge and hedge in full_spread_dfs:
        exec_df = full_spread_dfs[hedge]
        exec_period = exec_df[exec_df["date"].isin(period_df["date"])]
        # Need full history for signal generation (lookback)
        exec_full = exec_df.copy()
    else:
        exec_period = period_df
        exec_full = full_rxrx_df

    z_col = "spread_z_60"

    if fam == "S5_short":
        k = c.get("k", 2.0)
        sigs = signals_s5_short(exec_full, z_col, k)
        trades = run_strategy(exec_full, sigs, z_exit_col=z_col, z_exit_thresh=0.5)
    elif fam == "S5_zexit":
        k = c.get("k", 1.5)
        z_exit = c.get("z_exit", 0.5)
        sigs = signals_s5_zexit(exec_full, z_col, k, "combined")
        trades = run_strategy(exec_full, sigs, z_exit_col=z_col, z_exit_thresh=z_exit)
    elif fam == "S5_sscore":
        t = c.get("t", 1.5)
        sigs = signals_s5_sscore_short(exec_full, t)
        trades = run_strategy(exec_full, sigs)
    elif fam == "S5_confirmed":
        k = c.get("k", 2.0)
        sigs = signals_s5_confirmed_short(exec_full, z_col, k)
        trades = run_strategy(exec_full, sigs)
    elif fam == "S10_vwap":
        sigs = signals_s10_vwap_short(full_rxrx_df, 40, 2.0)
        trades = run_strategy(full_rxrx_df, sigs)
    elif fam == "S11_range":
        N = c.get("N", 40)
        b = c.get("b", 0.03)
        sigs = signals_s11_short(full_rxrx_df, N, b)
        trades = run_strategy(full_rxrx_df, sigs)
    else:
        trades = pd.DataFrame()

    # Filter to period
    if not trades.empty:
        trades = trades[trades["entry_date"].isin(period_df["date"])].copy()

    return trades


segments = {
    "Semi-seen (Jan-Mar 2026)": rxrx_2026_full[rxrx_2026_full["date"] < SEMI_SEEN_END],
    "Clean holdout (Apr+ 2026)": rxrx_2026_full[rxrx_2026_full["date"] >= SEMI_SEEN_END],
    "Full 2026": rxrx_2026_full,
}

eval_results = {}

for seg_name, seg_df in segments.items():
    print(f"\n  ━━━ {seg_name} ({len(seg_df)} bars) ━━━")

    for c in pre_registered:
        label = c["stats"]["label"]
        trades = run_config_on_period(c, seg_df, spreads_full, rxrx_full)
        s = compute_stats(trades, seg_df, label)

        eval_results[(seg_name, label)] = {"stats": s, "trades": trades}

        if s["n"] == 0:
            print(f"    {label:45s}  0 trades")
            continue

        print(f"    {label:45s}  n={s['n']:>3d}  avg={s['avg_pnl']:>+7.2f}%  "
              f"med={s['median_pnl']:>+7.2f}%  total={s['total_pnl_pct']:>+7.1f}%  "
              f"${s['total_pnl_usd']:>+8,.0f}  WR={s['wr']:.0f}%  hold={s['avg_hold']:.1f}d  "
              f"PnL/day={s['pnl_per_day']:+.2f}%  DD={s['max_dd']:+.1f}%  Sharpe={s['sharpe']:+.2f}")

        # Full trade list
        if not trades.empty:
            print(f"    {'Entry':>12s} {'Exit':>12s} {'Side':>6s} {'Days':>4s} {'Net%':>8s} {'$':>9s} {'Reason':>8s}")
            for _, t in trades.iterrows():
                print(f"    {str(t['entry_date'].date()):>12s} {str(t['exit_date'].date()):>12s} "
                      f"{t['side']:>6s} {t['holding_days']:>4d} {t['pnl_net_pct']:>+7.2f}% "
                      f"${t['pnl_dollars']:>+8,.0f} {t['exit_reason']:>8s}")


# ══════════════════════════════════════════════════════════════════
# SECTION 9: BOOTSTRAP CI + NULL TESTS
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  SECTION 9: BOOTSTRAP CI + NULL TESTS")
print("=" * 70)

for c in pre_registered:
    label = c["stats"]["label"]
    sel_trades = c["trades"]
    if sel_trades.empty:
        continue

    sel_pnls = sel_trades["pnl_net_pct"].values
    n_sel = len(sel_pnls)

    # Bootstrap 90% CI of selection-period avg PnL
    boot_means = []
    for _ in range(N_BOOTSTRAP):
        sample = np.random.choice(sel_pnls, size=n_sel, replace=True)
        boot_means.append(np.mean(sample))
    ci_5 = np.percentile(boot_means, 5)
    ci_95 = np.percentile(boot_means, 95)

    print(f"\n  {label}:")
    print(f"    Selection avg PnL: {np.mean(sel_pnls):+.3f}%  90% CI: [{ci_5:+.3f}%, {ci_95:+.3f}%]")

    # Check 2026 results
    for seg_name in ["Semi-seen (Jan-Mar 2026)", "Clean holdout (Apr+ 2026)", "Full 2026"]:
        key = (seg_name, label)
        if key not in eval_results:
            continue
        er = eval_results[key]
        s26 = er["stats"]
        if s26["n"] == 0:
            print(f"    {seg_name}: 0 trades")
            continue

        in_ci = ci_5 <= s26["avg_pnl"] <= ci_95
        print(f"    {seg_name}: avg={s26['avg_pnl']:+.3f}%  n={s26['n']}  "
              f"{'IN CI' if in_ci else 'OUTSIDE CI'} [{ci_5:+.2f}, {ci_95:+.2f}]")

    # Null test on full 2026
    full_26_key = ("Full 2026", label)
    if full_26_key in eval_results:
        trades_26 = eval_results[full_26_key]["trades"]
        if not trades_26.empty:
            n_26 = len(trades_26)
            n_short_26 = sum(trades_26["side"] == "short")

            eligible = rxrx_2026_full["date"].iloc[20:].values
            null_avgs = []
            for run in range(N_NULL_RUNS):
                rng = np.random.RandomState(run + 20000)
                if len(eligible) < n_26:
                    continue
                rd = rng.choice(eligible, size=min(n_26, len(eligible)), replace=False)
                rd.sort()
                rs = [{"date": pd.Timestamp(d), "side": "short", "stop": None, "target": None, "time_stop": MAX_HOLD}
                      for d in rd]
                nt = run_strategy(rxrx_2026_full, rs)
                if not nt.empty:
                    null_avgs.append(np.mean(nt["pnl_net_pct"].values))

            real_avg = eval_results[full_26_key]["stats"]["avg_pnl"]
            pct = scipy_stats.percentileofscore(null_avgs, real_avg) if null_avgs else 0
            null_mean = np.mean(null_avgs) if null_avgs else 0
            print(f"    2026 Null test: real={real_avg:+.3f}% vs null_mean={null_mean:+.3f}%  "
                  f"percentile={pct:.1f}th  {'PASSES' if pct > 90 else 'FAILS'}")


# ══════════════════════════════════════════════════════════════════
# SECTION 10: PEER BASKET EVALUATION
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  SECTION 10: PEER BASKET (2026)")
print("=" * 70)

peer_results = defaultdict(list)
peer_available = []

for ticker in PEER_TICKERS:
    peer_path = DATA_DIR / f"peer_{ticker}.parquet"
    if not peer_path.exists():
        print(f"  {ticker}: data file not found, skipping")
        continue

    peer_df = pd.read_parquet(peer_path)
    peer_df["date"] = pd.to_datetime(peer_df["date"])
    peer_df = peer_df.sort_values("date").reset_index(drop=True)

    if len(peer_df) < 200:
        print(f"  {ticker}: only {len(peer_df)} bars, skipping")
        continue

    peer_df = add_features(peer_df)
    peer_available.append(ticker)

    for c in pre_registered:
        fam = c["family"]
        hedge = c.get("hedge")

        if fam in ["S10_vwap", "S11_range"]:
            # These use raw price, not spread
            peer_2026 = peer_df[peer_df["date"] >= SELECTION_END].copy()
            if fam == "S10_vwap":
                sigs = signals_s10_vwap_short(peer_df, 40, 2.0)
            else:
                sigs = signals_s11_short(peer_df, c.get("N", 40), c.get("b", 0.03))
            trades = run_strategy(peer_df, sigs)
            if not trades.empty:
                trades = trades[trades["entry_date"] >= SELECTION_END]
        elif hedge:
            hedge_df = {"XBI": xbi_all, "ARKG": arkg_all, "IWM": iwm_all}.get(hedge)
            if hedge_df is None:
                continue
            peer_spread = build_ols_spread(peer_df, hedge_df, hedge, 60)
            z_col = "spread_z_60"
            peer_2026_sp = peer_spread[peer_spread["date"] >= SELECTION_END]

            if fam == "S5_short":
                sigs = signals_s5_short(peer_spread, z_col, c.get("k", 2.0))
                trades = run_strategy(peer_spread, sigs, z_exit_col=z_col, z_exit_thresh=0.5)
            elif fam == "S5_zexit":
                sigs = signals_s5_zexit(peer_spread, z_col, c.get("k", 1.5), "combined")
                trades = run_strategy(peer_spread, sigs, z_exit_col=z_col, z_exit_thresh=c.get("z_exit", 0.5))
            elif fam == "S5_sscore":
                sigs = signals_s5_sscore_short(peer_spread, c.get("t", 1.5))
                trades = run_strategy(peer_spread, sigs)
            elif fam == "S5_confirmed":
                sigs = signals_s5_confirmed_short(peer_spread, z_col, c.get("k", 2.0))
                trades = run_strategy(peer_spread, sigs)
            else:
                trades = pd.DataFrame()

            if not trades.empty:
                trades = trades[trades["entry_date"] >= SELECTION_END]
        else:
            continue

        if not trades.empty and len(trades) > 0:
            s = compute_stats(trades, label=f"{ticker}")
            peer_results[(fam, c["config"])].append({
                "ticker": ticker, "n": s["n"], "avg_pnl": s["avg_pnl"],
                "total_pnl_pct": s["total_pnl_pct"], "total_pnl_usd": s["total_pnl_usd"],
                "sharpe": s["sharpe"], "wr": s["wr"],
            })

print(f"\n  Peers with data: {len(peer_available)} ({', '.join(peer_available)})")

for c in pre_registered:
    label = c["stats"]["label"]
    fam = c["family"]
    cfg = c["config"]
    key = (fam, cfg)

    results = peer_results.get(key, [])
    if not results:
        print(f"\n  {label}: no peer trades")
        continue

    print(f"\n  {label} — Peer basket 2026:")
    total_n = 0
    total_pnl_pcts = []
    for r in results:
        print(f"    {r['ticker']:>6s}  n={r['n']:>3d}  avg={r['avg_pnl']:>+7.2f}%  "
              f"total={r['total_pnl_pct']:>+7.1f}%  ${r['total_pnl_usd']:>+8,.0f}")
        total_n += r["n"]
        total_pnl_pcts.extend([r["avg_pnl"]] * r["n"])

    pooled_avg = np.mean(total_pnl_pcts) if total_pnl_pcts else 0
    print(f"    {'POOLED':>6s}  n={total_n:>3d}  avg={pooled_avg:>+7.2f}%")


# ══════════════════════════════════════════════════════════════════
# SECTION 11: DEPLOYMENT DECISION
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  SECTION 11: DEPLOYMENT DECISION")
print("=" * 70)

for c in pre_registered:
    label = c["stats"]["label"]
    fam = c["family"]
    cfg = c["config"]

    # Get 2026 stats
    full_key = ("Full 2026", label)
    s26 = eval_results.get(full_key, {}).get("stats", {"n": 0, "avg_pnl": 0, "total_pnl_pct": 0})

    # Selection CI
    sel_pnls = c["trades"]["pnl_net_pct"].values if not c["trades"].empty else []
    if len(sel_pnls) > 0:
        boot_means = [np.mean(np.random.choice(sel_pnls, size=len(sel_pnls), replace=True))
                      for _ in range(N_BOOTSTRAP)]
        ci_5, ci_95 = np.percentile(boot_means, 5), np.percentile(boot_means, 95)
    else:
        ci_5, ci_95 = 0, 0

    # Peer pooled avg
    peer_data = peer_results.get((fam, cfg), [])
    peer_pnls = []
    for r in peer_data:
        peer_pnls.extend([r["avg_pnl"]] * r["n"])
    pooled_peer = np.mean(peer_pnls) if peer_pnls else 0

    # Decision rules
    rule1 = s26["avg_pnl"] >= 1.5  # avg >= 1.5%
    rule2 = ci_5 <= s26["avg_pnl"] <= ci_95 if s26["n"] > 0 else False  # in CI
    rule3 = s26["total_pnl_pct"] > 0 if s26["n"] > 0 else False  # positive total
    rule4 = pooled_peer > 0 if peer_pnls else False  # positive peer avg

    deployable = rule1 and rule2 and rule3 and rule4
    rxrx_specific = rule1 and rule2 and rule3 and not rule4

    status = "DEPLOYABLE" if deployable else ("RXRX-SPECIFIC (reduced size)" if rxrx_specific else "REJECT")

    print(f"\n  {label}:")
    print(f"    Rule 1 (2026 avg >= 1.5%):     {'PASS' if rule1 else 'FAIL'} ({s26['avg_pnl']:+.2f}%)")
    print(f"    Rule 2 (in selection CI):       {'PASS' if rule2 else 'FAIL'} ({s26['avg_pnl']:+.2f}% in [{ci_5:+.2f}, {ci_95:+.2f}])")
    print(f"    Rule 3 (positive total 2026):   {'PASS' if rule3 else 'FAIL'} ({s26['total_pnl_pct']:+.1f}%)")
    print(f"    Rule 4 (positive peer avg):     {'PASS' if rule4 else 'FAIL'} ({pooled_peer:+.2f}%)")
    print(f"    >>> {status}")


# ══════════════════════════════════════════════════════════════════
# SECTION 12: EQUITY CURVES + FINAL REPORT
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  SECTION 12: EQUITY CURVES + PLOTS")
print("=" * 70)

fig, axes = plt.subplots(2, 1, figsize=(16, 12))
fig.suptitle("Agent 10 — 2026 Out-of-Sample Evaluation", fontsize=14, fontweight="bold")

# Top: 2026 equity curves
ax = axes[0]
for c in pre_registered:
    label = c["stats"]["label"]
    full_key = ("Full 2026", label)
    if full_key not in eval_results:
        continue
    trades = eval_results[full_key]["trades"]
    if trades.empty:
        continue
    equity = trades["pnl_net_pct"].cumsum()
    short_label = label.replace("S5_", "").replace("S10_", "").replace("S11_", "")
    ax.plot(trades["entry_date"], equity,
            label=f"{short_label} (avg={eval_results[full_key]['stats']['avg_pnl']:+.2f}%)", lw=2)

# Buy-and-hold benchmark
if not rxrx_2026_full.empty:
    bh_eq = (rxrx_2026_full["close"] / rxrx_2026_full["open"].iloc[0] - 1) * 100
    ax.plot(rxrx_2026_full["date"], bh_eq, "k--", alpha=0.5, lw=1, label=f"Buy&Hold ({bh_return:+.1f}%)")
    sh_eq = -(rxrx_2026_full["close"] / rxrx_2026_full["open"].iloc[0] - 1) * 100
    ax.plot(rxrx_2026_full["date"], sh_eq, "k:", alpha=0.5, lw=1, label=f"Short&Hold ({sh_return:+.1f}%)")

ax.axhline(0, color="black", lw=0.5, ls="--")
ax.axvline(pd.Timestamp(SEMI_SEEN_END), color="red", ls="--", alpha=0.5, label="Holdout boundary")
ax.set_ylabel("Cumulative PnL (%)")
ax.set_title("2026 Out-of-Sample: Pre-Registered Strategies")
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, alpha=0.3)

# Bottom: Selection period equity curves (for comparison)
ax = axes[1]
for c in pre_registered:
    label = c["stats"]["label"]
    trades = c["trades"]
    if trades.empty:
        continue
    equity = trades["pnl_net_pct"].cumsum()
    short_label = label.replace("S5_", "").replace("S10_", "").replace("S11_", "")
    ax.plot(trades["entry_date"], equity,
            label=f"{short_label} (avg={c['stats']['avg_pnl']:+.2f}%)", lw=1.5, alpha=0.7)

ax.axhline(0, color="black", lw=0.5, ls="--")
ax.set_ylabel("Cumulative PnL (%)")
ax.set_title("Selection Period (2021-2025): Pre-Registered Strategies")
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "oos_2026_equity_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved oos_2026_equity_curves.png")

print("\n" + "=" * 70)
print("  EVALUATION COMPLETE")
print("=" * 70)
print(f"  Pre-registration: {prereg_path}")
print(f"  Equity curves: {OUT_DIR / 'oos_2026_equity_curves.png'}")
print(f"  Holdout NOT re-used for parameter selection.")
