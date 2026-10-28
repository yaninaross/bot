# %% Agent 10 — Phase 2: Strategy Families S1-S7
# Implements each strategy with parameter grid.
# Evaluates LONG, SHORT, and COMBINED separately.
#
# Requires: Phase 0 data + Phase 1 results (gate decision)
# Outputs: trials_log.json, strategy comparison plots
#
# Run as Colab cell. Config vars at top.
# S0 (baseline) is a placeholder until user provides their rules.

import os, sys, json, warnings, itertools
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal, stats

warnings.filterwarnings("ignore", category=FutureWarning)

for pkg in ["pywt", "statsmodels"]:
    try:
        __import__(pkg)
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg,
                               "--quiet", "--break-system-packages"])

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# ── Config ─────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data" if "__file__" in dir() else Path("agent10/data")
OUT_DIR = Path(__file__).parent / "results" if "__file__" in dir() else Path("agent10/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HOLDOUT_START = "2026-04-01"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Cost model (RXRX is low single digits — costs are material)
HALF_SPREAD_BPS = 10    # half-spread in bps
SLIPPAGE_BPS = 7.5      # slippage in bps
COMMISSION_PER_SHARE = 0.005  # typical IBKR
SHORT_BORROW_ANNUAL_PCT = 5.0  # mid-estimate, sensitivity at 3% and 10%

# ── Load Data ──────────────────────────────────────────────────────
# %%
print("Loading data...")
rxrx = pd.read_parquet(DATA_DIR / "rxrx_daily.parquet")
rxrx["date"] = pd.to_datetime(rxrx["date"])
rxrx = rxrx.sort_values("date").reset_index(drop=True)
rxrx_dev = rxrx[rxrx["date"] < HOLDOUT_START].copy()

ctx = pd.read_parquet(DATA_DIR / "context_daily.parquet")
ctx["date"] = pd.to_datetime(ctx["date"])

# Load events for event filter
try:
    events = pd.read_parquet(DATA_DIR / "rxrx_events.parquet")
    events["date"] = pd.to_datetime(events["date"])
    earnings_dates = events[events["type"].str.contains("earn", case=False, na=False)]["date"].values
except:
    earnings_dates = np.array([])
    print("  No events data — event filter disabled")

# Load Phase 1 gate
with open(OUT_DIR / "phase1_results.json") as f:
    p1_results = json.load(f)

gate = p1_results["gate"]
print(f"Phase 1 gate: proceed={gate['proceed']}, "
      f"cycle={gate['cycle_strategies_unlocked']}, "
      f"MR={gate['mr_strategies_unlocked']}")

if not gate["proceed"]:
    print("GATE FAILED — Phase 2 cannot proceed. Review Phase 1 results.")
    sys.exit(1)

# Precompute features
rxrx_dev["log_price"] = np.log(rxrx_dev["close"])
rxrx_dev["log_ret"] = rxrx_dev["log_price"].diff()
rxrx_dev["pct_ret"] = rxrx_dev["close"].pct_change()

def compute_atr(df, period=20):
    high = df["high"]
    low = df["low"]
    close = df["close"].shift(1)
    tr = pd.concat([high - low, (high - close).abs(), (low - close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

rxrx_dev["atr20"] = compute_atr(rxrx_dev, 20)
rxrx_dev["atr_pct"] = rxrx_dev["atr20"] / rxrx_dev["close"] * 100

print(f"Development set: {len(rxrx_dev)} bars, "
      f"{rxrx_dev['date'].min().date()} to {rxrx_dev['date'].max().date()}")

# ── Backtest Engine ────────────────────────────────────────────────
# %%
class Trade:
    def __init__(self, entry_date, entry_price, side, stop, target, time_stop_days,
                 strategy, params_str):
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.side = side  # 'long' or 'short'
        self.stop = stop
        self.target = target
        self.time_stop_days = time_stop_days
        self.strategy = strategy
        self.params_str = params_str
        self.exit_date = None
        self.exit_price = None
        self.exit_reason = None
        self.pnl_gross = None
        self.pnl_net = None
        self.holding_days = None


def apply_costs(entry_price, exit_price, side, holding_days):
    """Apply realistic transaction costs."""
    spread_cost = entry_price * HALF_SPREAD_BPS / 10000 * 2  # entry + exit
    slip_cost = entry_price * SLIPPAGE_BPS / 10000 * 2
    comm_cost = COMMISSION_PER_SHARE * 2
    borrow_cost = 0
    if side == "short":
        borrow_cost = entry_price * SHORT_BORROW_ANNUAL_PCT / 100 * holding_days / 252
    total_cost = spread_cost + slip_cost + comm_cost + borrow_cost
    return total_cost


def execute_trades(df, signals, max_hold=7):
    """
    Execute trades from signal list.
    signals: list of dicts with keys: date, side, stop, target, time_stop,
             strategy, params
    Returns list of Trade objects.
    """
    trades = []
    date_to_idx = {d: i for i, d in enumerate(df["date"])}

    for sig in signals:
        sig_date = sig["date"]
        if sig_date not in date_to_idx:
            continue
        sig_idx = date_to_idx[sig_date]

        # Entry at next day's open (no lookahead)
        entry_idx = sig_idx + 1
        if entry_idx >= len(df):
            continue

        entry_date = df["date"].iloc[entry_idx]
        entry_price = df["open"].iloc[entry_idx]

        trade = Trade(
            entry_date=entry_date,
            entry_price=entry_price,
            side=sig["side"],
            stop=sig.get("stop"),
            target=sig.get("target"),
            time_stop_days=sig.get("time_stop", max_hold),
            strategy=sig["strategy"],
            params_str=sig.get("params", ""),
        )

        # Walk forward to find exit
        hold_limit = min(sig.get("time_stop", max_hold), max_hold)
        for j in range(entry_idx + 1, min(entry_idx + hold_limit + 1, len(df))):
            bar_high = df["high"].iloc[j]
            bar_low = df["low"].iloc[j]
            bar_close = df["close"].iloc[j]

            if trade.side == "long":
                if trade.stop and bar_low <= trade.stop:
                    trade.exit_price = trade.stop
                    trade.exit_reason = "stop"
                    trade.exit_date = df["date"].iloc[j]
                    break
                if trade.target and bar_high >= trade.target:
                    trade.exit_price = trade.target
                    trade.exit_reason = "target"
                    trade.exit_date = df["date"].iloc[j]
                    break
            else:  # short
                if trade.stop and bar_high >= trade.stop:
                    trade.exit_price = trade.stop
                    trade.exit_reason = "stop"
                    trade.exit_date = df["date"].iloc[j]
                    break
                if trade.target and bar_low <= trade.target:
                    trade.exit_price = trade.target
                    trade.exit_reason = "target"
                    trade.exit_date = df["date"].iloc[j]
                    break

        # Time stop
        if trade.exit_date is None:
            exit_idx = min(entry_idx + hold_limit, len(df) - 1)
            trade.exit_date = df["date"].iloc[exit_idx]
            trade.exit_price = df["close"].iloc[exit_idx]
            trade.exit_reason = "time_stop"

        # PnL
        if trade.side == "long":
            trade.pnl_gross = (trade.exit_price - trade.entry_price) / trade.entry_price * 100
        else:
            trade.pnl_gross = (trade.entry_price - trade.exit_price) / trade.entry_price * 100

        trade.holding_days = (trade.exit_date - trade.entry_date).days
        cost = apply_costs(trade.entry_price, trade.exit_price, trade.side, trade.holding_days)
        trade.pnl_net = trade.pnl_gross - (cost / trade.entry_price * 100)

        trades.append(trade)

    return trades


def summarize_trades(trades, label=""):
    """Compute strategy metrics from trade list."""
    if not trades:
        return {"label": label, "n_trades": 0}

    pnls = [t.pnl_net for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    holds = [t.holding_days for t in trades]

    n = len(pnls)
    wr = len(wins) / n * 100
    avg_pnl = np.mean(pnls)
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    pf = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else np.inf

    # R-multiples (using avg loss as 1R)
    avg_r_win = avg_win / abs(avg_loss) if avg_loss != 0 else np.inf
    avg_r_loss = 1.0
    expectancy = (wr/100 * avg_r_win) - ((1 - wr/100) * avg_r_loss)

    # Equity curve for Sharpe/DD
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    max_dd = dd.min()

    # Annualized metrics (assume ~252 trading days)
    total_days = (trades[-1].exit_date - trades[0].entry_date).days
    years = total_days / 365.25 if total_days > 0 else 1
    cagr = (1 + sum(pnls)/100) ** (1/years) - 1 if years > 0 else 0
    sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252 / np.mean(holds)) if np.std(pnls) > 0 and holds else 0
    sortino_denom = np.std([p for p in pnls if p < 0]) if losses else 1e-10
    sortino = np.mean(pnls) / sortino_denom * np.sqrt(252 / np.mean(holds)) if sortino_denom > 0 else 0
    calmar = cagr / abs(max_dd) * 100 if max_dd != 0 else 0
    exposure = sum(holds) / total_days * 100 if total_days > 0 else 0

    return {
        "label": label,
        "n_trades": n,
        "win_rate": round(wr, 1),
        "avg_pnl_pct": round(avg_pnl, 3),
        "avg_win_pct": round(avg_win, 3),
        "avg_loss_pct": round(avg_loss, 3),
        "profit_factor": round(pf, 2),
        "avg_r_win": round(avg_r_win, 2),
        "expectancy_r": round(expectancy, 3),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd_pct": round(max_dd, 2),
        "cagr_pct": round(cagr * 100, 2),
        "calmar": round(calmar, 2),
        "exposure_pct": round(exposure, 1),
        "avg_hold_days": round(np.mean(holds), 1),
        "total_pnl_pct": round(sum(pnls), 2),
    }


# ── Event Filter ───────────────────────────────────────────────────
def is_near_event(dt, event_dates, days_before=3):
    """Check if date is within N days before an event."""
    if len(event_dates) == 0:
        return False
    for ed in event_dates:
        ed = pd.Timestamp(ed)
        diff = (ed - dt).days
        if 0 <= diff <= days_before:
            return True
    return False


# ── Trials Log ─────────────────────────────────────────────────────
trials_log = []
trial_counter = 0

def log_trial(strategy, params, side, metrics, notes=""):
    global trial_counter
    trial_counter += 1
    entry = {
        "trial_id": trial_counter,
        "strategy": strategy,
        "params": params,
        "side": side,
        "notes": notes,
        **metrics,
    }
    trials_log.append(entry)
    return entry


# ══════════════════════════════════════════════════════════════════
# S1: Z-SCORE REVERSION
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 60)
print("S1: Z-SCORE REVERSION")
print("=" * 60)

s1_results = []

for lookback in [10, 20, 30]:
    for k_thresh in [1.5, 2.0, 2.5]:
        rxrx_dev[f"z_{lookback}"] = (
            (rxrx_dev["log_price"] - rxrx_dev["log_price"].rolling(lookback).mean()) /
            rxrx_dev["log_price"].rolling(lookback).std()
        )

        for side in ["long", "short", "combined"]:
            signals = []
            for i in range(lookback + 1, len(rxrx_dev)):
                z = rxrx_dev[f"z_{lookback}"].iloc[i]
                if pd.isna(z):
                    continue

                atr = rxrx_dev["atr20"].iloc[i]
                if pd.isna(atr):
                    continue
                price = rxrx_dev["close"].iloc[i]

                if side in ("long", "combined") and z < -k_thresh:
                    signals.append({
                        "date": rxrx_dev["date"].iloc[i],
                        "side": "long",
                        "stop": price - 1.5 * atr,
                        "target": price + (rxrx_dev["log_price"].rolling(lookback).mean().iloc[i] -
                                           rxrx_dev["log_price"].iloc[i]) * price,
                        "time_stop": 7,
                        "strategy": "S1",
                        "params": f"L={lookback},k={k_thresh}",
                    })
                if side in ("short", "combined") and z > k_thresh:
                    signals.append({
                        "date": rxrx_dev["date"].iloc[i],
                        "side": "short",
                        "stop": price + 1.5 * atr,
                        "target": price - (rxrx_dev["log_price"].iloc[i] -
                                           rxrx_dev["log_price"].rolling(lookback).mean().iloc[i]) * price,
                        "time_stop": 7,
                        "strategy": "S1",
                        "params": f"L={lookback},k={k_thresh}",
                    })

            trades = execute_trades(rxrx_dev, signals)
            metrics = summarize_trades(trades, f"S1_L{lookback}_k{k_thresh}_{side}")
            entry = log_trial("S1", f"L={lookback},k={k_thresh}", side, metrics)
            s1_results.append(entry)

            if metrics["n_trades"] > 0:
                print(f"  S1 L={lookback} k={k_thresh} {side:8s}: "
                      f"n={metrics['n_trades']:3d}  avg={metrics['avg_pnl_pct']:+.2f}%  "
                      f"WR={metrics['win_rate']:.0f}%  Sharpe={metrics['sharpe']:.2f}")


# ══════════════════════════════════════════════════════════════════
# S1b: Z-SCORE ON ARKG-HEDGED RESIDUAL
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 60)
print("S1b: Z-SCORE ON ARKG-HEDGED RESIDUAL")
print("=" * 60)

arkg = ctx[ctx["ticker"] == "ARKG"].sort_values("date").reset_index(drop=True)
arkg["arkg_log"] = np.log(arkg["close"])

merged = rxrx_dev[["date", "log_price", "close", "open", "high", "low", "atr20"]].merge(
    arkg[["date", "arkg_log"]], on="date", how="inner"
)

# Rolling regression residual
reg_window = 60
merged["resid"] = np.nan
for i in range(reg_window, len(merged)):
    y = merged["log_price"].iloc[i-reg_window:i+1].values
    x = sm.add_constant(merged["arkg_log"].iloc[i-reg_window:i+1].values)
    try:
        model = sm.OLS(y, x).fit()
        merged.iloc[i, merged.columns.get_loc("resid")] = model.resid[-1]
    except:
        pass

for lookback in [10, 20, 30]:
    merged[f"z_resid_{lookback}"] = (
        (merged["resid"] - merged["resid"].rolling(lookback).mean()) /
        merged["resid"].rolling(lookback).std()
    )

    for k_thresh in [1.5, 2.0, 2.5]:
        for side in ["long", "short", "combined"]:
            signals = []
            for i in range(reg_window + lookback, len(merged)):
                z = merged[f"z_resid_{lookback}"].iloc[i]
                if pd.isna(z):
                    continue
                atr = merged["atr20"].iloc[i]
                if pd.isna(atr):
                    continue
                price = merged["close"].iloc[i]

                if side in ("long", "combined") and z < -k_thresh:
                    signals.append({
                        "date": merged["date"].iloc[i],
                        "side": "long",
                        "stop": price - 1.5 * atr,
                        "target": price + 1.0 * atr,
                        "time_stop": 7,
                        "strategy": "S1b",
                        "params": f"L={lookback},k={k_thresh}",
                    })
                if side in ("short", "combined") and z > k_thresh:
                    signals.append({
                        "date": merged["date"].iloc[i],
                        "side": "short",
                        "stop": price + 1.5 * atr,
                        "target": price - 1.0 * atr,
                        "time_stop": 7,
                        "strategy": "S1b",
                        "params": f"L={lookback},k={k_thresh}",
                    })

            trades = execute_trades(merged, signals)
            metrics = summarize_trades(trades, f"S1b_L{lookback}_k{k_thresh}_{side}")
            log_trial("S1b", f"L={lookback},k={k_thresh}", side, metrics)

            if metrics["n_trades"] > 0:
                print(f"  S1b L={lookback} k={k_thresh} {side:8s}: "
                      f"n={metrics['n_trades']:3d}  avg={metrics['avg_pnl_pct']:+.2f}%  "
                      f"WR={metrics['win_rate']:.0f}%  Sharpe={metrics['sharpe']:.2f}")


# ══════════════════════════════════════════════════════════════════
# S2: RANGE FADE
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 60)
print("S2: RANGE FADE")
print("=" * 60)

for N in [20, 40, 60]:
    rxrx_dev[f"range_high_{N}"] = rxrx_dev["high"].rolling(N).max()
    rxrx_dev[f"range_low_{N}"] = rxrx_dev["low"].rolling(N).min()
    rxrx_dev[f"range_mid_{N}"] = (rxrx_dev[f"range_high_{N}"] + rxrx_dev[f"range_low_{N}"]) / 2
    rxrx_dev["ibs"] = (rxrx_dev["close"] - rxrx_dev["low"]) / (rxrx_dev["high"] - rxrx_dev["low"])
    rxrx_dev["prev_ret"] = rxrx_dev["close"].pct_change()

    for atr_mult in [0.5, 1.0]:
        for side in ["long", "short", "combined"]:
            signals = []
            for i in range(N + 1, len(rxrx_dev)):
                atr = rxrx_dev["atr20"].iloc[i]
                if pd.isna(atr):
                    continue
                price = rxrx_dev["close"].iloc[i]
                support = rxrx_dev[f"range_low_{N}"].iloc[i]
                resistance = rxrx_dev[f"range_high_{N}"].iloc[i]
                midpoint = rxrx_dev[f"range_mid_{N}"].iloc[i]
                ibs = rxrx_dev["ibs"].iloc[i]
                prev_ret_val = rxrx_dev["prev_ret"].iloc[i]

                if pd.isna(support) or pd.isna(resistance):
                    continue

                # Long: price near support + confirmation (IBS > 0.7 after down day)
                if side in ("long", "combined"):
                    if (price - support) <= atr_mult * atr and price > support:
                        if ibs > 0.7 and prev_ret_val < 0:
                            signals.append({
                                "date": rxrx_dev["date"].iloc[i],
                                "side": "long",
                                "stop": support - 1.0 * atr,
                                "target": midpoint,
                                "time_stop": 7,
                                "strategy": "S2",
                                "params": f"N={N},atr_x={atr_mult}",
                            })

                # Short: price near resistance + confirmation (IBS < 0.3 after up day)
                if side in ("short", "combined"):
                    if (resistance - price) <= atr_mult * atr and price < resistance:
                        if ibs < 0.3 and prev_ret_val > 0:
                            signals.append({
                                "date": rxrx_dev["date"].iloc[i],
                                "side": "short",
                                "stop": resistance + 1.0 * atr,
                                "target": midpoint,
                                "time_stop": 7,
                                "strategy": "S2",
                                "params": f"N={N},atr_x={atr_mult}",
                            })

            trades = execute_trades(rxrx_dev, signals)
            metrics = summarize_trades(trades, f"S2_N{N}_atr{atr_mult}_{side}")
            log_trial("S2", f"N={N},atr_x={atr_mult}", side, metrics)

            if metrics["n_trades"] > 0:
                print(f"  S2 N={N} atr×{atr_mult} {side:8s}: "
                      f"n={metrics['n_trades']:3d}  avg={metrics['avg_pnl_pct']:+.2f}%  "
                      f"WR={metrics['win_rate']:.0f}%  Sharpe={metrics['sharpe']:.2f}")


# ══════════════════════════════════════════════════════════════════
# S3: CYCLE PHASE (only if spectral gate passed)
# ══════════════════════════════════════════════════════════════════
# %%
if gate["cycle_strategies_unlocked"]:
    print("\n" + "=" * 60)
    print("S3: CYCLE PHASE (GATE PASSED)")
    print("=" * 60)

    # Get dominant period from Phase 1
    periods_found = [v["peak_period_periodogram"] for v in p1_results["spectral"].values()
                     if v["significant_ar1_95"]]
    if periods_found:
        dominant_period = np.median(periods_found)
    else:
        dominant_period = 30  # fallback

    print(f"  Dominant period from Phase 1: {dominant_period:.0f} days")

    for period in [dominant_period, 25, 30, 35]:
        # Bandpass filter
        low_freq = 1.0 / (period * 1.4)
        high_freq = 1.0 / (period * 0.7)
        b, a = signal.butter(3, [low_freq, high_freq], btype="band", fs=1.0)

        log_price = rxrx_dev["log_price"].values
        log_price_clean = log_price.copy()
        log_price_clean[np.isnan(log_price_clean)] = 0

        if len(log_price_clean) < 50:
            continue

        try:
            filtered = signal.filtfilt(b, a, log_price_clean)
        except:
            continue

        # Hilbert transform for instantaneous phase
        analytic = signal.hilbert(filtered)
        phase = np.angle(analytic)

        for side in ["long", "short", "combined"]:
            signals = []
            for i in range(int(period) + 10, len(rxrx_dev)):
                p = phase[i]
                atr = rxrx_dev["atr20"].iloc[i]
                if pd.isna(atr):
                    continue
                price = rxrx_dev["close"].iloc[i]

                # Enter long near phase trough (phase ~ -π)
                if side in ("long", "combined"):
                    if p < -2.5 or p > 2.5:  # near -π/+π (trough)
                        signals.append({
                            "date": rxrx_dev["date"].iloc[i],
                            "side": "long",
                            "stop": price - 1.5 * atr,
                            "target": price + 1.5 * atr,
                            "time_stop": int(period / 2),
                            "strategy": "S3",
                            "params": f"T={period:.0f}",
                        })

                # Enter short near phase peak (phase ~ 0)
                if side in ("short", "combined"):
                    if -0.5 < p < 0.5:  # near 0 (peak)
                        signals.append({
                            "date": rxrx_dev["date"].iloc[i],
                            "side": "short",
                            "stop": price + 1.5 * atr,
                            "target": price - 1.5 * atr,
                            "time_stop": int(period / 2),
                            "strategy": "S3",
                            "params": f"T={period:.0f}",
                        })

            trades = execute_trades(rxrx_dev, signals, max_hold=int(period/2))
            metrics = summarize_trades(trades, f"S3_T{period:.0f}_{side}")
            log_trial("S3", f"T={period:.0f}", side, metrics)

            if metrics["n_trades"] > 0:
                print(f"  S3 T={period:.0f} {side:8s}: "
                      f"n={metrics['n_trades']:3d}  avg={metrics['avg_pnl_pct']:+.2f}%  "
                      f"WR={metrics['win_rate']:.0f}%  Sharpe={metrics['sharpe']:.2f}")
else:
    print("\nS3 SKIPPED — Cycle strategies not unlocked by Phase 1 gate")


# ══════════════════════════════════════════════════════════════════
# S4: SHORT-HORIZON REVERSAL
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 60)
print("S4: SHORT-HORIZON REVERSAL")
print("=" * 60)

# RSI(2)
def rsi(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

rxrx_dev["rsi2"] = rsi(rxrx_dev["close"], 2)
rxrx_dev["rsi3"] = rsi(rxrx_dev["close"], 3)
rxrx_dev["rsi14"] = rsi(rxrx_dev["close"], 14)

# IBS
rxrx_dev["ibs"] = (rxrx_dev["close"] - rxrx_dev["low"]) / (rxrx_dev["high"] - rxrx_dev["low"])

# Consecutive closes
rxrx_dev["consec_down"] = 0
rxrx_dev["consec_up"] = 0
for i in range(1, len(rxrx_dev)):
    if rxrx_dev["close"].iloc[i] < rxrx_dev["close"].iloc[i-1]:
        rxrx_dev.iloc[i, rxrx_dev.columns.get_loc("consec_down")] = \
            rxrx_dev["consec_down"].iloc[i-1] + 1
    if rxrx_dev["close"].iloc[i] > rxrx_dev["close"].iloc[i-1]:
        rxrx_dev.iloc[i, rxrx_dev.columns.get_loc("consec_up")] = \
            rxrx_dev["consec_up"].iloc[i-1] + 1

# ConnorsRSI = (RSI(3) + RSI_streak(2) + PctRank(100)) / 3
streak = rxrx_dev["consec_up"] - rxrx_dev["consec_down"]
rxrx_dev["rsi_streak"] = rsi(streak, 2)
rxrx_dev["pct_rank_100"] = rxrx_dev["pct_ret"].rolling(100).apply(
    lambda x: stats.percentileofscore(x, x.iloc[-1]), raw=False
)
rxrx_dev["connors_rsi"] = (rxrx_dev["rsi3"] + rxrx_dev["rsi_streak"] + rxrx_dev["pct_rank_100"]) / 3

# S4a: RSI(2) extremes
for rsi_low, rsi_high in [(10, 90), (5, 95), (15, 85)]:
    for side in ["long", "short", "combined"]:
        signals = []
        for i in range(20, len(rxrx_dev)):
            r2 = rxrx_dev["rsi2"].iloc[i]
            if pd.isna(r2):
                continue
            atr = rxrx_dev["atr20"].iloc[i]
            if pd.isna(atr):
                continue
            price = rxrx_dev["close"].iloc[i]

            if side in ("long", "combined") and r2 < rsi_low:
                signals.append({
                    "date": rxrx_dev["date"].iloc[i], "side": "long",
                    "stop": price - 1.5 * atr, "target": None,
                    "time_stop": 5, "strategy": "S4a",
                    "params": f"RSI2<{rsi_low}",
                })
            if side in ("short", "combined") and r2 > rsi_high:
                signals.append({
                    "date": rxrx_dev["date"].iloc[i], "side": "short",
                    "stop": price + 1.5 * atr, "target": None,
                    "time_stop": 5, "strategy": "S4a",
                    "params": f"RSI2>{rsi_high}",
                })

        trades = execute_trades(rxrx_dev, signals, max_hold=5)
        metrics = summarize_trades(trades, f"S4a_RSI{rsi_low}_{rsi_high}_{side}")
        log_trial("S4a", f"rsi_low={rsi_low},rsi_high={rsi_high}", side, metrics)

        if metrics["n_trades"] > 0:
            print(f"  S4a RSI2 {rsi_low}/{rsi_high} {side:8s}: "
                  f"n={metrics['n_trades']:3d}  avg={metrics['avg_pnl_pct']:+.2f}%  "
                  f"WR={metrics['win_rate']:.0f}%  Sharpe={metrics['sharpe']:.2f}")

# S4b: IBS extremes
for ibs_low, ibs_high in [(0.1, 0.9), (0.15, 0.85), (0.2, 0.8)]:
    for side in ["long", "short", "combined"]:
        signals = []
        for i in range(20, len(rxrx_dev)):
            ibs_val = rxrx_dev["ibs"].iloc[i]
            if pd.isna(ibs_val):
                continue
            atr = rxrx_dev["atr20"].iloc[i]
            if pd.isna(atr):
                continue
            price = rxrx_dev["close"].iloc[i]

            if side in ("long", "combined") and ibs_val < ibs_low:
                signals.append({
                    "date": rxrx_dev["date"].iloc[i], "side": "long",
                    "stop": price - 1.5 * atr, "target": None,
                    "time_stop": 5, "strategy": "S4b",
                    "params": f"IBS<{ibs_low}",
                })
            if side in ("short", "combined") and ibs_val > ibs_high:
                signals.append({
                    "date": rxrx_dev["date"].iloc[i], "side": "short",
                    "stop": price + 1.5 * atr, "target": None,
                    "time_stop": 5, "strategy": "S4b",
                    "params": f"IBS>{ibs_high}",
                })

        trades = execute_trades(rxrx_dev, signals, max_hold=5)
        metrics = summarize_trades(trades, f"S4b_IBS{ibs_low}_{ibs_high}_{side}")
        log_trial("S4b", f"ibs_low={ibs_low},ibs_high={ibs_high}", side, metrics)

        if metrics["n_trades"] > 0:
            print(f"  S4b IBS {ibs_low}/{ibs_high} {side:8s}: "
                  f"n={metrics['n_trades']:3d}  avg={metrics['avg_pnl_pct']:+.2f}%  "
                  f"WR={metrics['win_rate']:.0f}%  Sharpe={metrics['sharpe']:.2f}")

# S4c: Consecutive closes
for n_consec in [3, 4, 5]:
    for side in ["long", "short", "combined"]:
        signals = []
        for i in range(20, len(rxrx_dev)):
            atr = rxrx_dev["atr20"].iloc[i]
            if pd.isna(atr):
                continue
            price = rxrx_dev["close"].iloc[i]

            if side in ("long", "combined") and rxrx_dev["consec_down"].iloc[i] >= n_consec:
                signals.append({
                    "date": rxrx_dev["date"].iloc[i], "side": "long",
                    "stop": price - 1.5 * atr, "target": None,
                    "time_stop": 5, "strategy": "S4c",
                    "params": f"consec_down>={n_consec}",
                })
            if side in ("short", "combined") and rxrx_dev["consec_up"].iloc[i] >= n_consec:
                signals.append({
                    "date": rxrx_dev["date"].iloc[i], "side": "short",
                    "stop": price + 1.5 * atr, "target": None,
                    "time_stop": 5, "strategy": "S4c",
                    "params": f"consec_up>={n_consec}",
                })

        trades = execute_trades(rxrx_dev, signals, max_hold=5)
        metrics = summarize_trades(trades, f"S4c_consec{n_consec}_{side}")
        log_trial("S4c", f"n_consec={n_consec}", side, metrics)

        if metrics["n_trades"] > 0:
            print(f"  S4c consec={n_consec} {side:8s}: "
                  f"n={metrics['n_trades']:3d}  avg={metrics['avg_pnl_pct']:+.2f}%  "
                  f"WR={metrics['win_rate']:.0f}%  Sharpe={metrics['sharpe']:.2f}")

# S4d: ConnorsRSI
for crsi_low, crsi_high in [(10, 90), (15, 85), (5, 95)]:
    for side in ["long", "short", "combined"]:
        signals = []
        for i in range(110, len(rxrx_dev)):
            crsi = rxrx_dev["connors_rsi"].iloc[i]
            if pd.isna(crsi):
                continue
            atr = rxrx_dev["atr20"].iloc[i]
            if pd.isna(atr):
                continue
            price = rxrx_dev["close"].iloc[i]

            if side in ("long", "combined") and crsi < crsi_low:
                signals.append({
                    "date": rxrx_dev["date"].iloc[i], "side": "long",
                    "stop": price - 1.5 * atr, "target": None,
                    "time_stop": 5, "strategy": "S4d",
                    "params": f"CRSI<{crsi_low}",
                })
            if side in ("short", "combined") and crsi > crsi_high:
                signals.append({
                    "date": rxrx_dev["date"].iloc[i], "side": "short",
                    "stop": price + 1.5 * atr, "target": None,
                    "time_stop": 5, "strategy": "S4d",
                    "params": f"CRSI>{crsi_high}",
                })

        trades = execute_trades(rxrx_dev, signals, max_hold=5)
        metrics = summarize_trades(trades, f"S4d_CRSI{crsi_low}_{crsi_high}_{side}")
        log_trial("S4d", f"crsi_low={crsi_low},crsi_high={crsi_high}", side, metrics)

        if metrics["n_trades"] > 0:
            print(f"  S4d CRSI {crsi_low}/{crsi_high} {side:8s}: "
                  f"n={metrics['n_trades']:3d}  avg={metrics['avg_pnl_pct']:+.2f}%  "
                  f"WR={metrics['win_rate']:.0f}%  Sharpe={metrics['sharpe']:.2f}")


# ══════════════════════════════════════════════════════════════════
# S5: RELATIVE VALUE (RXRX vs ARKG)
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 60)
print("S5: RELATIVE VALUE (RXRX vs ARKG)")
print("=" * 60)

if not arkg.empty:
    arkg_merged = rxrx_dev[["date", "close", "open", "high", "low", "atr20"]].merge(
        arkg[["date", "close"]].rename(columns={"close": "arkg_close"}),
        on="date", how="inner"
    )

    # Kalman filter hedge ratio
    from statsmodels.regression.rolling import RollingOLS

    arkg_merged["rxrx_log"] = np.log(arkg_merged["close"])
    arkg_merged["arkg_log"] = np.log(arkg_merged["arkg_close"])

    # Rolling OLS for hedge ratio
    for lookback in [30, 60, 90]:
        arkg_merged[f"hedge_{lookback}"] = np.nan
        arkg_merged[f"spread_{lookback}"] = np.nan
        arkg_merged[f"spread_z_{lookback}"] = np.nan

        for i in range(lookback, len(arkg_merged)):
            y = arkg_merged["rxrx_log"].iloc[i-lookback:i+1].values
            x = sm.add_constant(arkg_merged["arkg_log"].iloc[i-lookback:i+1].values)
            try:
                model = sm.OLS(y, x).fit()
                hedge = model.params[1]
                resid = y[-1] - model.predict(x)[-1]
                arkg_merged.iloc[i, arkg_merged.columns.get_loc(f"hedge_{lookback}")] = hedge
                arkg_merged.iloc[i, arkg_merged.columns.get_loc(f"spread_{lookback}")] = resid
            except:
                pass

        spread = arkg_merged[f"spread_{lookback}"]
        arkg_merged[f"spread_z_{lookback}"] = (
            (spread - spread.rolling(lookback).mean()) / spread.rolling(lookback).std()
        )

        for k_thresh in [1.5, 2.0]:
            for side in ["long", "short", "combined"]:
                signals = []
                for i in range(lookback * 2, len(arkg_merged)):
                    z = arkg_merged[f"spread_z_{lookback}"].iloc[i]
                    if pd.isna(z):
                        continue
                    atr = arkg_merged["atr20"].iloc[i]
                    if pd.isna(atr):
                        continue
                    price = arkg_merged["close"].iloc[i]

                    if side in ("long", "combined") and z < -k_thresh:
                        signals.append({
                            "date": arkg_merged["date"].iloc[i], "side": "long",
                            "stop": price - 1.5 * atr, "target": price + 1.0 * atr,
                            "time_stop": 7, "strategy": "S5",
                            "params": f"L={lookback},k={k_thresh}",
                        })
                    if side in ("short", "combined") and z > k_thresh:
                        signals.append({
                            "date": arkg_merged["date"].iloc[i], "side": "short",
                            "stop": price + 1.5 * atr, "target": price - 1.0 * atr,
                            "time_stop": 7, "strategy": "S5",
                            "params": f"L={lookback},k={k_thresh}",
                        })

                trades = execute_trades(arkg_merged, signals)
                metrics = summarize_trades(trades, f"S5_L{lookback}_k{k_thresh}_{side}")
                log_trial("S5", f"L={lookback},k={k_thresh}", side, metrics)

                if metrics["n_trades"] > 0:
                    print(f"  S5 L={lookback} k={k_thresh} {side:8s}: "
                          f"n={metrics['n_trades']:3d}  avg={metrics['avg_pnl_pct']:+.2f}%  "
                          f"WR={metrics['win_rate']:.0f}%  Sharpe={metrics['sharpe']:.2f}")
else:
    print("  ARKG data not available — S5 skipped")


# ══════════════════════════════════════════════════════════════════
# S6: REGIME-GATED COMPOSITE
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 60)
print("S6: REGIME-GATED COMPOSITE")
print("=" * 60)

# Rolling Hurst for regime detection
def hurst_dfa_fast(returns, min_window=10, max_window=None):
    x = np.cumsum(returns - returns.mean())
    n = len(x)
    if max_window is None:
        max_window = n // 4
    windows = np.unique(np.logspace(np.log10(min_window), np.log10(max_window), 12).astype(int))
    flucts = []
    for w in windows:
        n_segments = n // w
        if n_segments < 2:
            continue
        f2 = []
        for seg in range(n_segments):
            segment = x[seg*w:(seg+1)*w]
            t = np.arange(w)
            coeffs = np.polyfit(t, segment, 1)
            trend = np.polyval(coeffs, t)
            f2.append(np.mean((segment - trend)**2))
        flucts.append((w, np.sqrt(np.mean(f2))))
    if len(flucts) < 3:
        return np.nan
    windows_used = [f[0] for f in flucts]
    f_values = [f[1] for f in flucts]
    coeffs = np.polyfit(np.log(windows_used), np.log(f_values), 1)
    return coeffs[0]

# Precompute rolling Hurst and ADF
hurst_window = 120
rxrx_dev["rolling_hurst"] = np.nan
rxrx_dev["rolling_adf_pval"] = np.nan
returns = rxrx_dev["log_ret"].values

for i in range(hurst_window + 1, len(rxrx_dev)):
    ret_window = returns[i-hurst_window:i]
    ret_window = ret_window[~np.isnan(ret_window)]
    if len(ret_window) > 30:
        rxrx_dev.iloc[i, rxrx_dev.columns.get_loc("rolling_hurst")] = hurst_dfa_fast(ret_window)
        try:
            _, pval, _, _, _, _ = adfuller(rxrx_dev["close"].iloc[i-hurst_window:i])
            rxrx_dev.iloc[i, rxrx_dev.columns.get_loc("rolling_adf_pval")] = pval
        except:
            pass

# Regime: mean-reversion when Hurst < 0.5 AND ADF < 0.05
rxrx_dev["regime_mr"] = (
    (rxrx_dev["rolling_hurst"] < 0.5) &
    (rxrx_dev["rolling_adf_pval"] < 0.10)
).astype(int)

print(f"  MR regime bars: {rxrx_dev['regime_mr'].sum()} / {len(rxrx_dev)} "
      f"({rxrx_dev['regime_mr'].mean()*100:.1f}%)")

# Breakout detection
rxrx_dev["range_high_40"] = rxrx_dev["high"].rolling(40).max()
rxrx_dev["range_low_40"] = rxrx_dev["low"].rolling(40).min()
rxrx_dev["vol_ratio"] = rxrx_dev["volume"] / rxrx_dev["volume"].rolling(20).mean()

# Best MR signals from S1 and S4 gated by regime
for base_strategy in ["S1_gated", "S4a_gated"]:
    for side in ["long", "short", "combined"]:
        signals = []
        for i in range(hurst_window + 20, len(rxrx_dev)):
            if rxrx_dev["regime_mr"].iloc[i] == 0:
                continue

            # Breakout override: if close outside range on high volume, skip
            atr = rxrx_dev["atr20"].iloc[i]
            if pd.isna(atr):
                continue
            price = rxrx_dev["close"].iloc[i]
            rh = rxrx_dev["range_high_40"].iloc[i]
            rl = rxrx_dev["range_low_40"].iloc[i]

            if not pd.isna(rh) and not pd.isna(rl):
                if (price > rh + atr or price < rl - atr) and rxrx_dev["vol_ratio"].iloc[i] > 2.0:
                    continue

            if base_strategy == "S1_gated":
                z20 = rxrx_dev.get("z_20")
                if z20 is not None:
                    z = z20.iloc[i]
                else:
                    continue
                if pd.isna(z):
                    continue
                if side in ("long", "combined") and z < -2.0:
                    signals.append({
                        "date": rxrx_dev["date"].iloc[i], "side": "long",
                        "stop": price - 1.5 * atr, "target": price + 1.0 * atr,
                        "time_stop": 7, "strategy": "S6",
                        "params": f"S1_gated,k=2.0",
                    })
                if side in ("short", "combined") and z > 2.0:
                    signals.append({
                        "date": rxrx_dev["date"].iloc[i], "side": "short",
                        "stop": price + 1.5 * atr, "target": price - 1.0 * atr,
                        "time_stop": 7, "strategy": "S6",
                        "params": f"S1_gated,k=2.0",
                    })

            elif base_strategy == "S4a_gated":
                r2 = rxrx_dev["rsi2"].iloc[i]
                if pd.isna(r2):
                    continue
                if side in ("long", "combined") and r2 < 10:
                    signals.append({
                        "date": rxrx_dev["date"].iloc[i], "side": "long",
                        "stop": price - 1.5 * atr, "target": None,
                        "time_stop": 5, "strategy": "S6",
                        "params": f"S4a_gated,RSI2<10",
                    })
                if side in ("short", "combined") and r2 > 90:
                    signals.append({
                        "date": rxrx_dev["date"].iloc[i], "side": "short",
                        "stop": price + 1.5 * atr, "target": None,
                        "time_stop": 5, "strategy": "S6",
                        "params": f"S4a_gated,RSI2>90",
                    })

        trades = execute_trades(rxrx_dev, signals, max_hold=7)
        metrics = summarize_trades(trades, f"S6_{base_strategy}_{side}")
        log_trial("S6", base_strategy, side, metrics)

        if metrics["n_trades"] > 0:
            print(f"  S6 {base_strategy} {side:8s}: "
                  f"n={metrics['n_trades']:3d}  avg={metrics['avg_pnl_pct']:+.2f}%  "
                  f"WR={metrics['win_rate']:.0f}%  Sharpe={metrics['sharpe']:.2f}")


# ══════════════════════════════════════════════════════════════════
# SUMMARY TABLE & PLOTS
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 60)
print("PHASE 2 SUMMARY")
print("=" * 60)

# Save trials log
with open(OUT_DIR / "trials_log.json", "w") as f:
    json.dump(trials_log, f, indent=2, default=str)
print(f"Saved trials_log.json ({len(trials_log)} trials)")

# Summary table
df_trials = pd.DataFrame(trials_log)
df_valid = df_trials[df_trials["n_trades"] >= 10].copy()

if not df_valid.empty:
    df_valid = df_valid.sort_values("sharpe", ascending=False)

    print(f"\nTop 20 strategies (n>=10 trades, sorted by Sharpe):")
    print(f"{'Label':40s} {'N':>4s} {'AvgPnL':>8s} {'WR':>5s} {'Sharpe':>7s} {'PF':>6s} {'MaxDD':>7s}")
    print("-" * 80)
    for _, row in df_valid.head(20).iterrows():
        print(f"{row['label']:40s} {row['n_trades']:4d} "
              f"{row['avg_pnl_pct']:+7.2f}% {row['win_rate']:4.0f}% "
              f"{row['sharpe']:7.3f} {row['profit_factor']:6.2f} {row['max_dd_pct']:+6.1f}%")

    # Plot: Sharpe comparison
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    top = df_valid.head(25)
    ax = axes[0]
    colors = ["steelblue" if s > 0 else "salmon" for s in top["sharpe"]]
    ax.barh(range(len(top)), top["sharpe"], color=colors, alpha=0.8)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["label"], fontsize=7)
    ax.set_xlabel("Sharpe Ratio")
    ax.set_title("Phase 2 — Top 25 Strategies by Sharpe (n >= 10 trades)")
    ax.axvline(0, color="black", lw=0.5)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()

    # Plot: PnL vs trade count
    ax = axes[1]
    for strat in df_valid["strategy"].unique():
        subset = df_valid[df_valid["strategy"] == strat]
        ax.scatter(subset["n_trades"], subset["avg_pnl_pct"],
                   label=strat, alpha=0.6, s=50)
    ax.axhline(0, color="red", ls="--", alpha=0.5)
    ax.set_xlabel("Trade Count")
    ax.set_ylabel("Avg PnL (%)")
    ax.set_title("Avg PnL vs Trade Count (all strategies)")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "p2_strategy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved p2_strategy_comparison.png")
else:
    print("\nNo strategies with >= 10 trades found")

print(f"\nTotal trials logged: {len(trials_log)}")
print("Phase 2 complete. Proceed to Phase 3 for walk-forward validation.")
print("\nNOTE: S0 (baseline) and S7 (meta-labeling) require additional input.")
print("S0: provide your discretionary rules for RXRX.")
print("S7: will be built on top of the best rule-based strategy from Phase 3.")
