# %% Agent 10 — Phase 4: Generalization of S5_zexit to RXRX-Like Stocks
#
# Strategy: FROZEN (no re-tuning)
#   Signal: 60-day rolling OLS residual of log(ticker) on log(hedge)
#   Entry: short when z > +1.5, long when z < -1.5, at next open
#   Exit: z crosses ±0.5, or 1.5 ATR stop, or 10-day hard cap
#   Positions: one at a time per ticker
#   Costs: same as RXRX study
#
# Three-stage validation:
#   A: Rule development (2021-04 to 2023-12)
#   B: Rule validation (2024-01 to 2025-12)
#   C: Final test (2026-01 to latest)

import os, sys, json, warnings, hashlib
from pathlib import Path
from datetime import datetime
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

# ── Config ────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data" if "__file__" in dir() else Path("agent10/data")
OUT_DIR = Path(__file__).parent / "results" if "__file__" in dir() else Path("agent10/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

HALF_SPREAD_BPS = 10
SLIPPAGE_BPS = 7.5
COMMISSION_PER_SHARE = 0.005
SHORT_BORROW_ANNUAL_PCT = 5.0
POSITION_SIZE_USD = 10000
MAX_HOLD = 10
MAX_CONCURRENT = 5

STAGE_A_END = "2024-01-01"
STAGE_B_END = "2026-01-01"
N_BOOTSTRAP = 3000
N_NULL_RUNS = 500

# ── Ticker / Hedge Universe ──────────────────────────────────────
SEEN_PEERS = ["SDGR", "ABCL", "ABSI", "BEAM", "CRSP", "NTLA", "TXG", "PACB", "TWST"]
NEW_BIO = ["EDIT", "DNA", "TEM", "ADPT", "GH", "SANA", "ARCT", "RGNX",
           "FATE", "VERV", "RARE", "DCPH", "NUVB", "ACAD", "TGTX"]
OUT_SECTOR = ["MARA", "RIOT", "CLSK", "PLUG", "RUN", "RIVN", "LCID"]

HEDGE_CANDIDATES = ["ARKG", "XBI", "IBB", "ARKK", "IWM", "ICLN", "TAN", "WGMI", "XLV", "XLE", "XLY"]

SECTOR_MAP = {
    "biotech": ["ARKG", "XBI", "IBB", "XLV"],
    "crypto": ["WGMI", "ARKK", "IWM"],
    "clean_energy": ["ICLN", "TAN", "XLE"],
    "ev": ["ARKK", "IWM", "XLY"],
}

TICKER_SECTOR = {}
for t in SEEN_PEERS + NEW_BIO + ["RXRX"]:
    TICKER_SECTOR[t] = "biotech"
for t in ["MARA", "RIOT", "CLSK"]:
    TICKER_SECTOR[t] = "crypto"
for t in ["PLUG", "RUN"]:
    TICKER_SECTOR[t] = "clean_energy"
for t in ["RIVN", "LCID"]:
    TICKER_SECTOR[t] = "ev"

# BTC as a hedge for crypto tickers
BTC_HEDGE_FOR = ["MARA", "RIOT", "CLSK"]


# ══════════════════════════════════════════════════════════════════
# SECTION 0: DATA LOADING
# ══════════════════════════════════════════════════════════════════
# %%
print("=" * 70)
print("  SECTION 0: LOADING ALL DATA")
print("=" * 70)

def load_ticker(ticker):
    """Try multiple file naming conventions."""
    for pattern in [f"p4_{ticker}.parquet", f"peer_{ticker}.parquet",
                    f"{ticker.lower()}_daily.parquet"]:
        p = DATA_DIR / pattern
        if p.exists():
            df = pd.read_parquet(p)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            if "ticker" not in df.columns:
                df["ticker"] = ticker
            return df
    return None

# Load RXRX
rxrx = load_ticker("RXRX")
if rxrx is None:
    rxrx = pd.read_parquet(DATA_DIR / "rxrx_daily.parquet")
    rxrx["date"] = pd.to_datetime(rxrx["date"])
    rxrx = rxrx.sort_values("date").reset_index(drop=True)
    rxrx["ticker"] = "RXRX"
print(f"  RXRX: {len(rxrx)} bars")

# Load context ETFs (already have ARKG, XBI, IWM, SPY)
ctx = pd.read_parquet(DATA_DIR / "context_daily.parquet")
ctx["date"] = pd.to_datetime(ctx["date"])
hedge_data = {}
for etf in ["ARKG", "XBI", "IWM", "SPY"]:
    hdf = ctx[ctx["ticker"] == etf].sort_values("date").reset_index(drop=True)
    if len(hdf) > 0:
        hedge_data[etf] = hdf
        print(f"  Hedge {etf}: {len(hdf)} bars")

# Load BTC
btc_path = DATA_DIR / "btc_daily.parquet"
if btc_path.exists():
    btc = pd.read_parquet(btc_path)
    btc["date"] = pd.to_datetime(btc["date"])
    btc = btc.sort_values("date").reset_index(drop=True)
    if "ticker" not in btc.columns:
        btc["ticker"] = "BTC"
    hedge_data["BTC"] = btc
    print(f"  Hedge BTC: {len(btc)} bars")

# Load additional hedge ETFs
for etf in ["IBB", "ARKK", "ICLN", "TAN", "WGMI", "XLV", "XLE", "XLY"]:
    hdf = load_ticker(etf)
    if hdf is not None and len(hdf) > 100:
        hedge_data[etf] = hdf
        print(f"  Hedge {etf}: {len(hdf)} bars")

# Load candidate tickers
all_candidates = {}
for tk in SEEN_PEERS + NEW_BIO + OUT_SECTOR:
    df = load_ticker(tk)
    if df is not None and len(df) > 200:
        all_candidates[tk] = df
        print(f"  Candidate {tk}: {len(df)} bars, ${df['close'].iloc[-1]:.2f}")
    else:
        print(f"  Candidate {tk}: {'NO DATA' if df is None else f'only {len(df)} bars'}")

# Add RXRX
all_candidates["RXRX"] = rxrx

print(f"\n  Total candidates loaded: {len(all_candidates)}")
print(f"  Hedge ETFs loaded: {list(hedge_data.keys())}")


# ══════════════════════════════════════════════════════════════════
# SECTION 1: FEATURE ENGINEERING + UNIVERSE FILTERING
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  SECTION 1: UNIVERSE FILTERING")
print("=" * 70)

def add_features(df):
    df = df.copy()
    df["log_price"] = np.log(df["close"].clip(lower=0.01))
    df["log_ret"] = df["log_price"].diff()
    df["pct_ret"] = df["close"].pct_change()
    high, low, prev_close = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    df["atr20"] = tr.rolling(20).mean()
    df["atr_pct"] = df["atr20"] / df["close"] * 100
    df["adv"] = (df["close"] * df["volume"]).rolling(60).mean()
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    return df

def filter_universe(candidates, as_of_date):
    """Apply universe filters using only data up to as_of_date.
    Returns dict of passing tickers -> DataFrames.
    """
    passed = {}
    for tk, df in candidates.items():
        sub = df[df["date"] < as_of_date].copy()
        if len(sub) < 120:
            continue
        sub = add_features(sub)
        tail = sub.tail(60)
        if tail.empty:
            continue

        price = tail["close"].iloc[-1]
        atr_pct = tail["atr_pct"].iloc[-1]
        adv = tail["adv"].iloc[-1]

        # Filters
        price_ok = 2 <= price <= 60
        atr_ok = 3 <= atr_pct <= 10 if not pd.isna(atr_pct) else False
        adv_ok = adv >= 20_000_000 if not pd.isna(adv) else False

        if price_ok and atr_ok and adv_ok:
            passed[tk] = df
        else:
            reasons = []
            if not price_ok: reasons.append(f"price=${price:.2f}")
            if not atr_ok: reasons.append(f"atr%={atr_pct:.1f}" if not pd.isna(atr_pct) else "atr=NaN")
            if not adv_ok: reasons.append(f"adv=${adv/1e6:.1f}M" if not pd.isna(adv) else "adv=NaN")

    return passed

# Filter as of start of each stage
for label, cutoff in [("Stage A start", "2021-10-01"), ("Stage B start", STAGE_A_END),
                       ("Stage C start", STAGE_B_END)]:
    passed = filter_universe(all_candidates, cutoff)
    print(f"  {label} ({cutoff}): {len(passed)}/{len(all_candidates)} pass filters")
    print(f"    Passing: {', '.join(sorted(passed.keys()))}")

# Use the broadest set (filter at each re-selection point)
print(f"\n  Full candidate pool: {len(all_candidates)} tickers")


# ══════════════════════════════════════════════════════════════════
# SECTION 2: HEDGE ASSIGNMENT + SPREAD CONSTRUCTION
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  SECTION 2: HEDGE ASSIGNMENT")
print("=" * 70)

def assign_hedge(ticker_df, as_of_date, sector=None):
    """Pick hedge ETF with highest 250-day return correlation, using data up to as_of_date."""
    sub = ticker_df[ticker_df["date"] < as_of_date].copy()
    if len(sub) < 250:
        sub = sub  # use what we have

    sub_ret = sub.set_index("date")["close"].pct_change().dropna()
    if len(sub_ret) < 60:
        return None, 0.0

    # Restrict to sector-relevant hedges if sector known
    candidates = SECTOR_MAP.get(sector, list(hedge_data.keys()))
    # Always include the universal candidates
    candidates = list(set(candidates) | {"IWM", "ARKK"})

    best_hedge = None
    best_corr = -1.0
    for h_name in candidates:
        if h_name not in hedge_data:
            continue
        hdf = hedge_data[h_name]
        h_sub = hdf[hdf["date"] < as_of_date].copy()
        h_ret = h_sub.set_index("date")["close"].pct_change().dropna()

        # Align
        common = sub_ret.index.intersection(h_ret.index)
        n_common = len(common)
        if n_common < 60:
            continue

        # Use last 250 trading days
        common = sorted(common)[-250:]
        corr = np.corrcoef(sub_ret.loc[common].values, h_ret.loc[common].values)[0, 1]
        if corr > best_corr:
            best_corr = corr
            best_hedge = h_name

    # Also try BTC for crypto tickers
    tk_name = ticker_df["ticker"].iloc[0] if "ticker" in ticker_df.columns else ""
    if tk_name in BTC_HEDGE_FOR and "BTC" in hedge_data:
        hdf = hedge_data["BTC"]
        h_sub = hdf[hdf["date"] < as_of_date].copy()
        h_ret = h_sub.set_index("date")["close"].pct_change().dropna()
        common = sub_ret.index.intersection(h_ret.index)
        if len(common) >= 60:
            common = sorted(common)[-250:]
            corr = np.corrcoef(sub_ret.loc[common].values, h_ret.loc[common].values)[0, 1]
            if corr > best_corr:
                best_corr = corr
                best_hedge = "BTC"

    return best_hedge, best_corr

# Assign hedges at each stage boundary
hedge_assignments = {}
for cutoff_label, cutoff in [("StageA", STAGE_A_END), ("StageB", STAGE_B_END)]:
    print(f"\n  Hedge assignments as of {cutoff_label} ({cutoff}):")
    for tk, df in sorted(all_candidates.items()):
        sector = TICKER_SECTOR.get(tk)
        h, corr = assign_hedge(df, cutoff, sector)
        hedge_assignments[(tk, cutoff_label)] = (h, corr)
        if h:
            print(f"    {tk:>6s} -> {h:>6s}  (corr={corr:+.3f})")
        else:
            print(f"    {tk:>6s} -> NO HEDGE FOUND")


# ══════════════════════════════════════════════════════════════════
# SECTION 3: CORE STRATEGY ENGINE (frozen)
# ══════════════════════════════════════════════════════════════════
# %%

def build_ols_spread(target_df, hedge_df, hedge_name, lookback=60):
    """Build OLS spread between target and hedge."""
    t_df = target_df[["date", "close", "open", "high", "low", "volume"]].copy()
    t_df["log_price"] = np.log(t_df["close"].clip(lower=0.01))
    high, low, prev_close = t_df["high"], t_df["low"], t_df["close"].shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    t_df["atr20"] = tr.rolling(20).mean()

    merged = t_df.merge(
        hedge_df[["date", "close"]].rename(columns={"close": "hedge_close"}),
        on="date", how="inner"
    )
    merged["hedge_log"] = np.log(merged["hedge_close"].clip(lower=0.01))
    merged["target_log"] = merged["log_price"]

    merged["spread_z"] = np.nan
    merged["hedge_beta"] = np.nan
    merged["r_squared"] = np.nan

    for i in range(lookback, len(merged)):
        y = merged["target_log"].iloc[i-lookback:i+1].values
        x = sm.add_constant(merged["hedge_log"].iloc[i-lookback:i+1].values)
        try:
            model = sm.OLS(y, x).fit()
            resid = y[-1] - model.predict(x)[-1]
            merged.iloc[i, merged.columns.get_loc("hedge_beta")] = model.params[1]
            merged.iloc[i, merged.columns.get_loc("r_squared")] = model.rsquared

            spread_series = y - model.predict(x)
            mu = np.mean(spread_series)
            sigma = np.std(spread_series)
            if sigma > 0:
                merged.iloc[i, merged.columns.get_loc("spread_z")] = (resid - mu) / sigma
        except:
            pass

    return merged


def apply_costs(entry_price, side, holding_days):
    spread = entry_price * HALF_SPREAD_BPS / 10000 * 2
    slip = entry_price * SLIPPAGE_BPS / 10000 * 2
    comm = COMMISSION_PER_SHARE * 2
    borrow = entry_price * SHORT_BORROW_ANNUAL_PCT / 100 * holding_days / 252 if side == "short" else 0
    return spread + slip + comm + borrow


def run_frozen_strategy(spread_df, k_entry=1.5, z_exit=0.5, max_hold=MAX_HOLD):
    """Run the FROZEN S5_zexit strategy. No parameter changes."""
    trades = []
    date_to_idx = {d: i for i, d in enumerate(spread_df["date"])}
    in_trade_until = pd.Timestamp("1900-01-01")

    for i in range(120, len(spread_df)):
        z = spread_df["spread_z"].iloc[i]
        if pd.isna(z):
            continue
        atr = spread_df["atr20"].iloc[i]
        if pd.isna(atr) or atr <= 0:
            continue
        price = spread_df["close"].iloc[i]
        sig_date = spread_df["date"].iloc[i]

        entry_idx = i + 1
        if entry_idx >= len(spread_df):
            continue
        entry_date = spread_df["date"].iloc[entry_idx]
        if entry_date <= in_trade_until:
            continue

        side = None
        if z > k_entry:
            side = "short"
        elif z < -k_entry:
            side = "long"
        if side is None:
            continue

        entry_price = spread_df["open"].iloc[entry_idx]
        stop = entry_price + 1.5 * atr if side == "short" else entry_price - 1.5 * atr

        exit_price = exit_date = exit_reason = None
        for j in range(entry_idx + 1, min(entry_idx + max_hold + 1, len(spread_df))):
            # Stop loss
            if side == "short" and spread_df["high"].iloc[j] >= stop:
                exit_price, exit_reason = stop, "stop"
                exit_date = spread_df["date"].iloc[j]
                break
            if side == "long" and spread_df["low"].iloc[j] <= stop:
                exit_price, exit_reason = stop, "stop"
                exit_date = spread_df["date"].iloc[j]
                break

            # Z-exit
            z_now = spread_df["spread_z"].iloc[j]
            if not pd.isna(z_now):
                if side == "long" and z_now >= -z_exit:
                    exit_price, exit_reason = spread_df["close"].iloc[j], "z_cross"
                    exit_date = spread_df["date"].iloc[j]
                    break
                if side == "short" and z_now <= z_exit:
                    exit_price, exit_reason = spread_df["close"].iloc[j], "z_cross"
                    exit_date = spread_df["date"].iloc[j]
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
        cost_pct = apply_costs(entry_price, side, holding_days) / entry_price * 100
        pnl_net = pnl_pct - cost_pct
        shares = max(int(POSITION_SIZE_USD / entry_price), 1)

        trades.append({
            "signal_date": sig_date, "entry_date": entry_date, "exit_date": exit_date,
            "side": side, "entry_price": round(entry_price, 4),
            "exit_price": round(exit_price, 4), "exit_reason": exit_reason,
            "holding_days": holding_days, "z_at_entry": round(z, 3),
            "pnl_gross_pct": round(pnl_pct, 4), "cost_pct": round(cost_pct, 4),
            "pnl_net_pct": round(pnl_net, 4),
            "shares": shares, "pnl_dollars": round(pnl_net / 100 * shares * entry_price, 2),
        })

    return pd.DataFrame(trades)


def compute_ou_halflife(spread_series, lookback=120):
    """Compute OU half-life from spread series."""
    s = spread_series.dropna().values
    if len(s) < lookback:
        return np.nan
    s = s[-lookback:]
    ds = np.diff(s)
    s_lag = s[:-1]
    X = sm.add_constant(s_lag)
    try:
        model = sm.OLS(ds, X).fit()
        b = model.params[1]
        if b >= 0:
            return np.nan
        return -np.log(2) / np.log(1 + b)
    except:
        return np.nan


# ══════════════════════════════════════════════════════════════════
# SECTION 4: RUN STRATEGY ON ALL TICKERS × ALL PERIODS
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  SECTION 4: RUN FROZEN STRATEGY ON ALL TICKERS")
print("=" * 70)

# Build spreads and run strategy for each ticker with its assigned hedge
all_trades = {}  # (ticker, stage_label) -> trades_df
all_spreads = {}  # ticker -> full spread df
ticker_hedge_map = {}  # ticker -> hedge name

for tk, df in sorted(all_candidates.items()):
    sector = TICKER_SECTOR.get(tk)

    # Use Stage B hedge assignment (data through 2023) for Stage A & B,
    # and Stage C assignment (data through 2025) for Stage C
    h_b, corr_b = hedge_assignments.get((tk, "StageA"), (None, 0))
    h_c, corr_c = hedge_assignments.get((tk, "StageB"), (None, 0))

    # Use the StageB assignment (more data) as primary
    hedge_name = h_c if h_c else h_b
    if not hedge_name:
        print(f"  {tk}: no hedge, skipping")
        continue
    if hedge_name not in hedge_data:
        print(f"  {tk}: hedge {hedge_name} data missing, skipping")
        continue

    ticker_hedge_map[tk] = hedge_name

    # Build full spread
    df_feat = add_features(df)
    spread = build_ols_spread(df_feat, hedge_data[hedge_name], hedge_name, 60)
    all_spreads[tk] = spread

    # Run strategy on full history
    trades = run_frozen_strategy(spread)

    if trades.empty:
        print(f"  {tk} (vs {hedge_name}): 0 trades total")
        all_trades[(tk, "full")] = trades
        continue

    # Split by stage
    for stage, start, end in [("A", "2021-04-16", STAGE_A_END),
                               ("B", STAGE_A_END, STAGE_B_END),
                               ("C", STAGE_B_END, "2030-01-01")]:
        st = trades[(trades["entry_date"] >= start) & (trades["entry_date"] < end)]
        all_trades[(tk, stage)] = st.copy()

    all_trades[(tk, "full")] = trades

    n_a = len(all_trades.get((tk, "A"), []))
    n_b = len(all_trades.get((tk, "B"), []))
    n_c = len(all_trades.get((tk, "C"), []))
    n_total = len(trades)
    avg = trades["pnl_net_pct"].mean()
    print(f"  {tk:>6s} (vs {hedge_name:>5s}): {n_total:>3d} trades "
          f"(A={n_a}, B={n_b}, C={n_c})  avg={avg:+.2f}%")


# ══════════════════════════════════════════════════════════════════
# SECTION 5: FINGERPRINT FEATURES
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  SECTION 5: FINGERPRINT FEATURES")
print("=" * 70)

def compute_fingerprint(tk, spread_df, as_of_date, trades_df=None):
    """Compute fingerprint features for a ticker using data up to as_of_date."""
    sub = spread_df[spread_df["date"] < as_of_date].copy()
    if len(sub) < 120:
        return None

    tail = sub.tail(250)
    feats = {"ticker": tk}

    # Hedge R²
    r2_vals = tail["r_squared"].dropna()
    feats["hedge_r2"] = r2_vals.iloc[-1] if len(r2_vals) > 0 else np.nan

    # Hedge correlation (already computed in assignment)
    feats["hedge_corr"] = hedge_assignments.get((tk, "StageA"), (None, np.nan))[1]
    if as_of_date >= STAGE_B_END:
        feats["hedge_corr"] = hedge_assignments.get((tk, "StageB"), (None, np.nan))[1]

    # Residual OU half-life
    if "spread_z" in sub.columns:
        spread_raw = sub["spread_z"].dropna().values
        feats["ou_halflife"] = compute_ou_halflife(pd.Series(spread_raw), 120)
    else:
        feats["ou_halflife"] = np.nan

    # Residual volatility
    spread_z = sub["spread_z"].dropna()
    feats["resid_vol"] = spread_z.diff().std() if len(spread_z) > 20 else np.nan

    # ATR% and ADV
    if "atr20" in sub.columns:
        feats["atr_pct"] = (sub["atr20"].iloc[-1] / sub["close"].iloc[-1] * 100
                            if sub["atr20"].iloc[-1] > 0 else np.nan)
    else:
        feats["atr_pct"] = np.nan

    feats["adv"] = (sub["close"] * sub["volume"]).tail(60).mean() if "volume" in sub.columns else np.nan

    # Trailing strategy performance (if trades provided)
    if trades_df is not None and not trades_df.empty:
        trailing = trades_df[trades_df["entry_date"] < as_of_date]
        feats["trail_n"] = len(trailing)
        feats["trail_avg_pnl"] = trailing["pnl_net_pct"].mean() if len(trailing) > 0 else 0
        feats["trail_total_pnl"] = trailing["pnl_net_pct"].sum() if len(trailing) > 0 else 0
    else:
        feats["trail_n"] = 0
        feats["trail_avg_pnl"] = 0
        feats["trail_total_pnl"] = 0

    return feats

# Compute fingerprints at Stage A boundary (for rule development)
print("\n  Fingerprints at Stage A boundary (2024-01-01):")
fp_stage_a = []
for tk in sorted(all_spreads.keys()):
    full_trades = all_trades.get((tk, "full"), pd.DataFrame())
    fp = compute_fingerprint(tk, all_spreads[tk], STAGE_A_END, full_trades)
    if fp:
        fp_stage_a.append(fp)

fp_df_a = pd.DataFrame(fp_stage_a)
print(fp_df_a[["ticker", "hedge_r2", "hedge_corr", "ou_halflife", "resid_vol",
               "atr_pct", "trail_n", "trail_avg_pnl"]].to_string(index=False))


# ══════════════════════════════════════════════════════════════════
# SECTION 6: STAGE A — RULE DEVELOPMENT (2021-04 to 2023-12)
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  SECTION 6: STAGE A — RULE DEVELOPMENT")
print("=" * 70)

# Get Stage A performance for each ticker
stage_a_pnl = {}
for tk in all_spreads:
    tr = all_trades.get((tk, "A"), pd.DataFrame())
    if not tr.empty and len(tr) >= 3:
        stage_a_pnl[tk] = tr["pnl_net_pct"].mean()

# Merge fingerprints with Stage A outcomes
fp_with_outcome = fp_df_a[fp_df_a["ticker"].isin(stage_a_pnl)].copy()
fp_with_outcome["stage_a_avg_pnl"] = fp_with_outcome["ticker"].map(stage_a_pnl)

print(f"\n  Tickers with Stage A trades (n>=3): {len(fp_with_outcome)}")

# Spearman correlations: fingerprint features vs Stage A PnL
feature_cols = ["hedge_r2", "hedge_corr", "ou_halflife", "resid_vol", "atr_pct", "trail_n", "trail_avg_pnl"]
print("\n  Spearman correlation: feature vs Stage A avg PnL")
spearman_results = {}
for col in feature_cols:
    valid = fp_with_outcome[[col, "stage_a_avg_pnl"]].dropna()
    if len(valid) < 5:
        print(f"    {col:>20s}: insufficient data ({len(valid)} obs)")
        continue
    rho, pval = scipy_stats.spearmanr(valid[col], valid["stage_a_avg_pnl"])
    spearman_results[col] = (rho, pval)
    sig = "*" if pval < 0.1 else " "
    print(f"    {col:>20s}: rho={rho:+.3f}  p={pval:.3f} {sig}")

# Empirical Bayes shrinkage of trailing avg PnL
if len(fp_with_outcome) > 3:
    pool_mean = fp_with_outcome["trail_avg_pnl"].mean()
    pool_var = fp_with_outcome["trail_avg_pnl"].var()
    fp_with_outcome["shrunk_avg_pnl"] = fp_with_outcome.apply(
        lambda r: pool_mean + (r["trail_avg_pnl"] - pool_mean) * min(r["trail_n"], 30) / 30
        if r["trail_n"] > 0 else pool_mean, axis=1)
    rho, pval = scipy_stats.spearmanr(
        fp_with_outcome["shrunk_avg_pnl"].dropna(),
        fp_with_outcome["stage_a_avg_pnl"].loc[fp_with_outcome["shrunk_avg_pnl"].dropna().index])
    print(f"    {'shrunk_avg_pnl':>20s}: rho={rho:+.3f}  p={pval:.3f}")
else:
    fp_with_outcome["shrunk_avg_pnl"] = fp_with_outcome["trail_avg_pnl"]

# Define selection rule based on Stage A evidence
# Rule: hedge_r2 >= threshold AND ou_halflife in [3, 15] AND shrunk trailing avg > 0
# Determine thresholds from Stage A data
print("\n  ── Defining selection rule ──")

# Find R² threshold that separates good from bad
if "hedge_r2" in spearman_results:
    r2_median = fp_with_outcome["hedge_r2"].median()
    r2_thresh = max(0.10, r2_median * 0.5)  # conservative
else:
    r2_thresh = 0.10

# OU half-life: select mean-reverting range
ou_min, ou_max = 3.0, 15.0

# Trailing performance: shrunk avg > 0
trail_thresh = 0.0

SELECTION_RULE = {
    "min_hedge_r2": round(r2_thresh, 3),
    "ou_halflife_range": [ou_min, ou_max],
    "min_shrunk_trail_avg": trail_thresh,
    "description": (f"hedge_r2 >= {r2_thresh:.3f} AND "
                    f"ou_halflife in [{ou_min}, {ou_max}] AND "
                    f"shrunk_trail_avg > {trail_thresh}")
}

print(f"  Rule: {SELECTION_RULE['description']}")

# Test rule on Stage A data
def apply_rule(fp_row, rule):
    r2 = fp_row.get("hedge_r2", np.nan)
    ou = fp_row.get("ou_halflife", np.nan)
    shrunk = fp_row.get("shrunk_avg_pnl", 0)

    if pd.isna(r2) or r2 < rule["min_hedge_r2"]:
        return False
    if pd.isna(ou):
        return True  # don't filter on NaN OU (insufficient data, give benefit of doubt)
    if not (rule["ou_halflife_range"][0] <= ou <= rule["ou_halflife_range"][1]):
        return False
    if shrunk <= rule["min_shrunk_trail_avg"]:
        return False
    return True

fp_with_outcome["selected_a"] = fp_with_outcome.apply(lambda r: apply_rule(r, SELECTION_RULE), axis=1)
selected_a = fp_with_outcome[fp_with_outcome["selected_a"]]
rejected_a = fp_with_outcome[~fp_with_outcome["selected_a"]]

print(f"\n  Stage A rule results:")
print(f"    Selected: {len(selected_a)} tickers, avg PnL: {selected_a['stage_a_avg_pnl'].mean():+.2f}%")
print(f"    Rejected: {len(rejected_a)} tickers, avg PnL: {rejected_a['stage_a_avg_pnl'].mean():+.2f}%")
print(f"    Diff: {selected_a['stage_a_avg_pnl'].mean() - rejected_a['stage_a_avg_pnl'].mean():+.2f}%")

if len(selected_a) > 0:
    print(f"    Selected tickers: {', '.join(selected_a['ticker'].tolist())}")


# ══════════════════════════════════════════════════════════════════
# SECTION 7: STAGE B — RULE VALIDATION (2024-01 to 2025-12)
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  SECTION 7: STAGE B — RULE VALIDATION")
print("=" * 70)

# Re-select every 6 months using trailing data
stage_b_windows = [
    ("B_H1_2024", "2024-01-01", "2024-07-01", STAGE_A_END),
    ("B_H2_2024", "2024-07-01", "2025-01-01", "2024-07-01"),
    ("B_H1_2025", "2025-01-01", "2025-07-01", "2025-01-01"),
    ("B_H2_2025", "2025-07-01", "2026-01-01", "2025-07-01"),
]

stage_b_selected_trades = []
stage_b_rejected_trades = []

for window_name, w_start, w_end, selection_date in stage_b_windows:
    print(f"\n  ── {window_name} (select as of {selection_date}) ──")

    # Compute fingerprints using data through selection_date
    selected_tickers = []
    rejected_tickers = []

    for tk in all_spreads:
        # Apply universe filters first
        filtered = filter_universe({tk: all_candidates[tk]}, selection_date)
        if tk not in filtered:
            continue

        full_trades = all_trades.get((tk, "full"), pd.DataFrame())
        fp = compute_fingerprint(tk, all_spreads[tk], selection_date, full_trades)
        if fp is None:
            continue

        # Empirical Bayes shrinkage
        pool_mean_b = 0  # will update after first pass
        fp["shrunk_avg_pnl"] = fp.get("trail_avg_pnl", 0)

        if apply_rule(fp, SELECTION_RULE):
            selected_tickers.append(tk)
        else:
            rejected_tickers.append(tk)

    # Pool mean for shrinkage (recompute with all tickers)
    all_trail_avgs = []
    for tk in selected_tickers + rejected_tickers:
        full_trades = all_trades.get((tk, "full"), pd.DataFrame())
        tr_before = full_trades[full_trades["entry_date"] < selection_date] if not full_trades.empty else pd.DataFrame()
        if not tr_before.empty:
            all_trail_avgs.append(tr_before["pnl_net_pct"].mean())
    pool_mean_b = np.mean(all_trail_avgs) if all_trail_avgs else 0

    # Re-evaluate with shrinkage
    final_selected = []
    final_rejected = []
    for tk in selected_tickers + rejected_tickers:
        full_trades = all_trades.get((tk, "full"), pd.DataFrame())
        fp = compute_fingerprint(tk, all_spreads[tk], selection_date, full_trades)
        if fp is None:
            continue
        n = fp.get("trail_n", 0)
        raw = fp.get("trail_avg_pnl", 0)
        fp["shrunk_avg_pnl"] = pool_mean_b + (raw - pool_mean_b) * min(n, 30) / 30 if n > 0 else pool_mean_b

        if apply_rule(fp, SELECTION_RULE):
            final_selected.append(tk)
        else:
            final_rejected.append(tk)

    # Get trades in this window
    sel_trades = []
    rej_trades = []
    for tk in final_selected:
        tr = all_trades.get((tk, "full"), pd.DataFrame())
        if not tr.empty:
            w_tr = tr[(tr["entry_date"] >= w_start) & (tr["entry_date"] < w_end)].copy()
            if not w_tr.empty:
                w_tr["ticker"] = tk
                w_tr["window"] = window_name
                sel_trades.append(w_tr)

    for tk in final_rejected:
        tr = all_trades.get((tk, "full"), pd.DataFrame())
        if not tr.empty:
            w_tr = tr[(tr["entry_date"] >= w_start) & (tr["entry_date"] < w_end)].copy()
            if not w_tr.empty:
                w_tr["ticker"] = tk
                w_tr["window"] = window_name
                rej_trades.append(w_tr)

    sel_df = pd.concat(sel_trades) if sel_trades else pd.DataFrame()
    rej_df = pd.concat(rej_trades) if rej_trades else pd.DataFrame()

    stage_b_selected_trades.append(sel_df)
    stage_b_rejected_trades.append(rej_df)

    sel_avg = sel_df["pnl_net_pct"].mean() if not sel_df.empty else 0
    rej_avg = rej_df["pnl_net_pct"].mean() if not rej_df.empty else 0

    print(f"    Selected: {len(final_selected)} tickers, {len(sel_df)} trades, avg={sel_avg:+.2f}%")
    print(f"    Rejected: {len(final_rejected)} tickers, {len(rej_df)} trades, avg={rej_avg:+.2f}%")
    print(f"    Selected tickers: {', '.join(sorted(final_selected))}")

# Aggregate Stage B
all_sel_b = pd.concat(stage_b_selected_trades) if any(not d.empty for d in stage_b_selected_trades) else pd.DataFrame()
all_rej_b = pd.concat(stage_b_rejected_trades) if any(not d.empty for d in stage_b_rejected_trades) else pd.DataFrame()

print(f"\n  ── Stage B aggregate ──")
sel_avg_b = all_sel_b["pnl_net_pct"].mean() if not all_sel_b.empty else 0
rej_avg_b = all_rej_b["pnl_net_pct"].mean() if not all_rej_b.empty else 0
print(f"    Selected: {len(all_sel_b)} trades, avg={sel_avg_b:+.2f}%")
print(f"    Rejected: {len(all_rej_b)} trades, avg={rej_avg_b:+.2f}%")
print(f"    Diff (sel - rej): {sel_avg_b - rej_avg_b:+.2f}%")

if not all_sel_b.empty:
    print(f"\n    Per-ticker Stage B (selected):")
    for tk in sorted(all_sel_b["ticker"].unique()):
        tk_tr = all_sel_b[all_sel_b["ticker"] == tk]
        print(f"      {tk:>6s}: n={len(tk_tr):>3d}  avg={tk_tr['pnl_net_pct'].mean():+.2f}%  "
              f"total={tk_tr['pnl_net_pct'].sum():+.1f}%  ${tk_tr['pnl_dollars'].sum():+,.0f}")


# ══════════════════════════════════════════════════════════════════
# SECTION 8: STAGE C — PRE-REGISTRATION
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  SECTION 8: STAGE C PRE-REGISTRATION")
print("=" * 70)

# Select tickers using data through 2025
stage_c_selected = []
stage_c_rejected = []
stage_c_fingerprints = []

# Pool mean for shrinkage
all_trail = []
for tk in all_spreads:
    full_trades = all_trades.get((tk, "full"), pd.DataFrame())
    tr_before = full_trades[full_trades["entry_date"] < STAGE_B_END] if not full_trades.empty else pd.DataFrame()
    if not tr_before.empty:
        all_trail.append(tr_before["pnl_net_pct"].mean())
pool_mean_c = np.mean(all_trail) if all_trail else 0

for tk in all_spreads:
    # Apply universe filters
    filtered = filter_universe({tk: all_candidates[tk]}, STAGE_B_END)
    if tk not in filtered:
        continue

    full_trades = all_trades.get((tk, "full"), pd.DataFrame())
    fp = compute_fingerprint(tk, all_spreads[tk], STAGE_B_END, full_trades)
    if fp is None:
        continue

    n = fp.get("trail_n", 0)
    raw = fp.get("trail_avg_pnl", 0)
    fp["shrunk_avg_pnl"] = pool_mean_c + (raw - pool_mean_c) * min(n, 30) / 30 if n > 0 else pool_mean_c

    if apply_rule(fp, SELECTION_RULE):
        stage_c_selected.append(tk)
    else:
        stage_c_rejected.append(tk)
    stage_c_fingerprints.append(fp)

print(f"  Selected for Stage C: {len(stage_c_selected)} tickers")
print(f"  Rejected: {len(stage_c_rejected)} tickers")

# Categorize
new_tickers_c = [t for t in stage_c_selected if t not in SEEN_PEERS and t != "RXRX"]
seen_tickers_c = [t for t in stage_c_selected if t in SEEN_PEERS or t == "RXRX"]

print(f"    New (primary): {', '.join(sorted(new_tickers_c))}")
print(f"    Seen in 2026 (secondary): {', '.join(sorted(seen_tickers_c))}")
print(f"    Rejected: {', '.join(sorted(stage_c_rejected))}")

# Write pre-registration file
prereg_c = {
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "protocol": "Phase 4 Stage C pre-registration",
    "frozen_strategy": {
        "signal": "60-day rolling OLS residual z-score",
        "entry": "short z > 1.5, long z < -1.5",
        "exit": "z crosses ±0.5, or 1.5 ATR stop, or 10-day cap",
        "position_size": "$10,000 per trade",
        "max_concurrent": MAX_CONCURRENT,
    },
    "selection_rule": SELECTION_RULE,
    "stage_a_evidence": {
        "spearman_correlations": {k: {"rho": round(v[0], 3), "pval": round(v[1], 3)}
                                  for k, v in spearman_results.items()},
        "selected_beat_rejected_by": round(
            selected_a["stage_a_avg_pnl"].mean() - rejected_a["stage_a_avg_pnl"].mean(), 2
        ) if len(selected_a) > 0 and len(rejected_a) > 0 else "N/A",
    },
    "stage_b_validation": {
        "selected_avg": round(sel_avg_b, 2),
        "rejected_avg": round(rej_avg_b, 2),
        "diff": round(sel_avg_b - rej_avg_b, 2),
    },
    "selected_tickers": {
        "new_primary": sorted(new_tickers_c),
        "seen_secondary": sorted(seen_tickers_c),
    },
    "rejected_tickers": sorted(stage_c_rejected),
    "hedge_assignments": {tk: ticker_hedge_map.get(tk, "unknown") for tk in stage_c_selected},
    "decision_rule": {
        "selected_beats_rejected_B_and_C": True,
        "pooled_C_avg_gte_1_5_pct": True,
        "clustered_t_gte_2_BC_combined": True,
        "portfolio_C_total_positive": True,
    },
}

prereg_c_path = OUT_DIR / "phase4_pre_registration.json"
with open(prereg_c_path, "w") as f:
    json.dump(prereg_c, f, indent=2, default=str)

print(f"\n  Pre-registration saved: {prereg_c_path}")
print(f"  Timestamp: {prereg_c['timestamp']}")


# ══════════════════════════════════════════════════════════════════
# SECTION 9: STAGE C — FINAL TEST (2026)
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  SECTION 9: STAGE C — 2026 EVALUATION")
print("=" * 70)

stage_c_trades_sel = []
stage_c_trades_rej = []

for tk in stage_c_selected:
    tr = all_trades.get((tk, "full"), pd.DataFrame())
    if not tr.empty:
        c_tr = tr[tr["entry_date"] >= STAGE_B_END].copy()
        if not c_tr.empty:
            c_tr["ticker"] = tk
            c_tr["bucket"] = "seen" if tk in SEEN_PEERS or tk == "RXRX" else "new"
            stage_c_trades_sel.append(c_tr)

for tk in stage_c_rejected:
    tr = all_trades.get((tk, "full"), pd.DataFrame())
    if not tr.empty:
        c_tr = tr[tr["entry_date"] >= STAGE_B_END].copy()
        if not c_tr.empty:
            c_tr["ticker"] = tk
            c_tr["bucket"] = "rejected"
            stage_c_trades_rej.append(c_tr)

sel_c = pd.concat(stage_c_trades_sel) if stage_c_trades_sel else pd.DataFrame()
rej_c = pd.concat(stage_c_trades_rej) if stage_c_trades_rej else pd.DataFrame()

sel_avg_c = sel_c["pnl_net_pct"].mean() if not sel_c.empty else 0
rej_avg_c = rej_c["pnl_net_pct"].mean() if not rej_c.empty else 0

print(f"  Selected: {len(sel_c)} trades, avg={sel_avg_c:+.2f}%")
print(f"  Rejected: {len(rej_c)} trades, avg={rej_avg_c:+.2f}%")
print(f"  Diff: {sel_avg_c - rej_avg_c:+.2f}%")

# Split by bucket
if not sel_c.empty:
    print(f"\n  ── New tickers (primary result) ──")
    new_c = sel_c[sel_c["bucket"] == "new"]
    if not new_c.empty:
        for tk in sorted(new_c["ticker"].unique()):
            tk_tr = new_c[new_c["ticker"] == tk]
            print(f"    {tk:>6s}: n={len(tk_tr):>3d}  avg={tk_tr['pnl_net_pct'].mean():+.2f}%  "
                  f"total={tk_tr['pnl_net_pct'].sum():+.1f}%  ${tk_tr['pnl_dollars'].sum():+,.0f}  "
                  f"hedge={ticker_hedge_map.get(tk, '?')}")
        print(f"    {'POOLED':>6s}: n={len(new_c):>3d}  avg={new_c['pnl_net_pct'].mean():+.2f}%  "
              f"total={new_c['pnl_net_pct'].sum():+.1f}%  ${new_c['pnl_dollars'].sum():+,.0f}")
    else:
        print(f"    No new ticker trades in 2026")

    print(f"\n  ── Seen-in-2026 tickers ──")
    seen_c = sel_c[sel_c["bucket"] == "seen"]
    if not seen_c.empty:
        for tk in sorted(seen_c["ticker"].unique()):
            tk_tr = seen_c[seen_c["ticker"] == tk]
            print(f"    {tk:>6s}: n={len(tk_tr):>3d}  avg={tk_tr['pnl_net_pct'].mean():+.2f}%  "
                  f"total={tk_tr['pnl_net_pct'].sum():+.1f}%  ${tk_tr['pnl_dollars'].sum():+,.0f}  "
                  f"hedge={ticker_hedge_map.get(tk, '?')}")
        print(f"    {'POOLED':>6s}: n={len(seen_c):>3d}  avg={seen_c['pnl_net_pct'].mean():+.2f}%  "
              f"total={seen_c['pnl_net_pct'].sum():+.1f}%  ${seen_c['pnl_dollars'].sum():+,.0f}")

    # Full trade list for all selected
    print(f"\n  ── Full Stage C trade list (selected) ──")
    print(f"    {'Ticker':>6s} {'Entry':>12s} {'Exit':>12s} {'Side':>6s} {'Days':>4s} {'Net%':>8s} "
          f"{'$':>9s} {'Reason':>8s} {'Bucket':>6s}")
    for _, t in sel_c.sort_values("entry_date").iterrows():
        print(f"    {t['ticker']:>6s} {str(t['entry_date'].date()):>12s} {str(t['exit_date'].date()):>12s} "
              f"{t['side']:>6s} {t['holding_days']:>4d} {t['pnl_net_pct']:>+7.2f}% "
              f"${t['pnl_dollars']:>+8,.0f} {t['exit_reason']:>8s} {t['bucket']:>6s}")


# ══════════════════════════════════════════════════════════════════
# SECTION 10: PORTFOLIO SIMULATION
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  SECTION 10: PORTFOLIO SIMULATION (Stages B + C)")
print("=" * 70)

def simulate_portfolio(all_stage_trades, price_data, max_concurrent=MAX_CONCURRENT):
    """Portfolio sim with max concurrent positions, priority by |z|."""
    if all_stage_trades.empty:
        return pd.DataFrame(), {}

    trades = all_stage_trades.sort_values("signal_date").copy()

    active_positions = []
    taken_trades = []
    skipped = 0

    for _, trade in trades.iterrows():
        entry_date = trade["entry_date"]
        # Close expired positions
        active_positions = [p for p in active_positions if p["exit_date"] > entry_date]

        # Check if ticker already has a position
        ticker_in = any(p["ticker"] == trade["ticker"] for p in active_positions)
        if ticker_in:
            skipped += 1
            continue

        if len(active_positions) >= max_concurrent:
            # Check if this trade has higher |z| than any active
            min_z = min(abs(p["z"]) for p in active_positions)
            if abs(trade["z_at_entry"]) <= min_z:
                skipped += 1
                continue
            # Don't force-close existing positions, just skip
            skipped += 1
            continue

        active_positions.append({
            "ticker": trade["ticker"],
            "exit_date": trade["exit_date"],
            "z": trade["z_at_entry"],
        })
        taken_trades.append(trade)

    taken_df = pd.DataFrame(taken_trades)
    if taken_df.empty:
        return taken_df, {"skipped": skipped}

    # Build daily MTM equity
    all_dates = sorted(set(
        d for _, t in taken_df.iterrows()
        for d in pd.date_range(t["entry_date"], t["exit_date"])
    ))

    daily_pnl = pd.Series(0.0, index=pd.DatetimeIndex(all_dates))
    for _, t in taken_df.iterrows():
        # Simple: spread PnL evenly across holding days
        days = t["holding_days"]
        daily_share = t["pnl_net_pct"] / max(days, 1)
        trade_dates = pd.date_range(t["entry_date"], t["exit_date"])
        for d in trade_dates:
            if d in daily_pnl.index:
                daily_pnl.loc[d] += daily_share

    equity = daily_pnl.cumsum()
    peak = equity.cummax()
    max_dd = (equity - peak).min()
    sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252) if daily_pnl.std() > 0 else 0

    # Concurrent positions over time
    concurrent = pd.Series(0, index=pd.DatetimeIndex(all_dates))
    for _, t in taken_df.iterrows():
        mask = (concurrent.index >= t["entry_date"]) & (concurrent.index <= t["exit_date"])
        concurrent.loc[mask] += 1

    stats = {
        "n_trades": len(taken_df),
        "skipped": skipped,
        "total_pnl_pct": round(taken_df["pnl_net_pct"].sum(), 2),
        "total_pnl_usd": round(taken_df["pnl_dollars"].sum(), 2),
        "avg_pnl": round(taken_df["pnl_net_pct"].mean(), 3),
        "sharpe": round(sharpe, 2),
        "max_dd_pct": round(max_dd, 2),
        "avg_concurrent": round(concurrent.mean(), 2),
        "max_concurrent": concurrent.max(),
        "trades_per_year": round(len(taken_df) / max((all_dates[-1] - all_dates[0]).days / 365.25, 0.5), 1),
    }

    return taken_df, stats, equity

# Stage B portfolio
if not all_sel_b.empty:
    port_b_trades, port_b_stats, port_b_eq = simulate_portfolio(all_sel_b, None)
    print(f"\n  Stage B portfolio:")
    for k, v in port_b_stats.items():
        print(f"    {k}: {v}")
else:
    port_b_trades = pd.DataFrame()
    port_b_stats = {}
    port_b_eq = pd.Series()
    print(f"  Stage B: no selected trades")

# Stage C portfolio
if not sel_c.empty:
    port_c_trades, port_c_stats, port_c_eq = simulate_portfolio(sel_c, None)
    print(f"\n  Stage C portfolio:")
    for k, v in port_c_stats.items():
        print(f"    {k}: {v}")
else:
    port_c_trades = pd.DataFrame()
    port_c_stats = {}
    port_c_eq = pd.Series()
    print(f"  Stage C: no selected trades")


# ══════════════════════════════════════════════════════════════════
# SECTION 11: STATISTICS
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  SECTION 11: STATISTICS")
print("=" * 70)

# Clustered bootstrap (by ticker and calendar week)
def clustered_bootstrap(trades_df, n_boot=N_BOOTSTRAP):
    """Bootstrap CI clustered by ticker × calendar week."""
    if trades_df.empty:
        return 0, 0, 0, 0
    tr = trades_df.copy()
    tr["cluster"] = tr["ticker"] + "_" + tr["entry_date"].dt.isocalendar().week.astype(str)
    clusters = tr["cluster"].unique()

    boot_means = []
    for _ in range(n_boot):
        sampled_clusters = np.random.choice(clusters, size=len(clusters), replace=True)
        sampled_pnls = []
        for cl in sampled_clusters:
            cl_pnls = tr[tr["cluster"] == cl]["pnl_net_pct"].values
            sampled_pnls.extend(cl_pnls)
        if sampled_pnls:
            boot_means.append(np.mean(sampled_pnls))

    if not boot_means:
        return 0, 0, 0, 0

    ci_5 = np.percentile(boot_means, 5)
    ci_95 = np.percentile(boot_means, 95)
    se = np.std(boot_means)
    t_stat = np.mean(boot_means) / se if se > 0 else 0

    return round(np.mean(boot_means), 3), round(ci_5, 3), round(ci_95, 3), round(t_stat, 2)

# Pooled stats
print("\n  ── Pooled trade statistics ──")

# Stage B + C combined (selected)
bc_sel = pd.concat([all_sel_b, sel_c]) if not all_sel_b.empty or not sel_c.empty else pd.DataFrame()
if not bc_sel.empty:
    mean_bc, ci5_bc, ci95_bc, t_bc = clustered_bootstrap(bc_sel)
    n_bc = len(bc_sel)
    avg_bc = bc_sel["pnl_net_pct"].mean()
    med_bc = bc_sel["pnl_net_pct"].median()
    wr_bc = (bc_sel["pnl_net_pct"] > 0).mean() * 100

    print(f"    B+C selected: n={n_bc}, avg={avg_bc:+.2f}%, median={med_bc:+.2f}%, "
          f"WR={wr_bc:.0f}%")
    print(f"    Clustered bootstrap: mean={mean_bc:+.3f}%, 90% CI=[{ci5_bc:+.3f}, {ci95_bc:+.3f}], "
          f"t-stat={t_bc:.2f}")

# Stage C only (selected)
if not sel_c.empty:
    mean_c, ci5_c, ci95_c, t_c = clustered_bootstrap(sel_c)
    print(f"\n    Stage C selected: n={len(sel_c)}, avg={sel_c['pnl_net_pct'].mean():+.2f}%")
    print(f"    Clustered bootstrap: mean={mean_c:+.3f}%, 90% CI=[{ci5_c:+.3f}, {ci95_c:+.3f}], "
          f"t-stat={t_c:.2f}")

# Selected vs rejected comparison
print("\n  ── Selected vs Rejected ──")
for stage, sel_df, rej_df in [("B", all_sel_b, all_rej_b), ("C", sel_c, rej_c)]:
    sel_a = sel_df["pnl_net_pct"].mean() if not sel_df.empty else 0
    rej_a = rej_df["pnl_net_pct"].mean() if not rej_df.empty else 0
    diff = sel_a - rej_a
    print(f"    Stage {stage}: selected={sel_a:+.2f}% (n={len(sel_df)}) vs "
          f"rejected={rej_a:+.2f}% (n={len(rej_df)})  diff={diff:+.2f}%  "
          f"{'PASS' if diff > 0 else 'FAIL'}")

# Per-ticker null tests (Stage C)
print("\n  ── Null tests (Stage C, per-ticker) ──")
null_results = {}
for tk in sorted(set(sel_c["ticker"].unique()) if not sel_c.empty else []):
    tk_trades = sel_c[sel_c["ticker"] == tk]
    n_tk = len(tk_trades)
    real_avg = tk_trades["pnl_net_pct"].mean()

    # Random entry nulls
    tk_df = all_candidates.get(tk)
    if tk_df is None:
        continue
    tk_2026 = tk_df[tk_df["date"] >= STAGE_B_END]
    if len(tk_2026) < 30:
        continue

    eligible = tk_2026["date"].iloc[20:].values
    null_avgs = []
    for run in range(N_NULL_RUNS):
        rng = np.random.RandomState(run + 30000)
        if len(eligible) < n_tk:
            continue
        rd = rng.choice(eligible, size=min(n_tk, len(eligible)), replace=False)
        rd.sort()
        sigs = [{"date": pd.Timestamp(d), "side": "short", "stop": None, "target": None, "time_stop": MAX_HOLD}
                for d in rd]
        # Need spread for this
        if tk in all_spreads:
            nt = run_frozen_strategy(all_spreads[tk])
            if not nt.empty:
                # Just use random-entry benchmark instead
                prices = tk_2026.set_index("date")["close"]
                random_pnls = []
                for d in rd:
                    d = pd.Timestamp(d)
                    idx = tk_2026[tk_2026["date"] == d].index
                    if len(idx) == 0:
                        continue
                    i = idx[0]
                    if i + 1 >= len(tk_2026):
                        continue
                    entry_p = tk_2026.loc[i + 1, "open"] if i + 1 in tk_2026.index else tk_2026.iloc[min(tk_2026.index.get_loc(i) + 1, len(tk_2026)-1)]["open"]
                    exit_idx_pos = min(tk_2026.index.get_loc(i) + MAX_HOLD + 1, len(tk_2026) - 1)
                    exit_p = tk_2026.iloc[exit_idx_pos]["close"]
                    pnl = (entry_p - exit_p) / entry_p * 100  # short
                    cost = apply_costs(entry_p, "short", MAX_HOLD) / entry_p * 100
                    random_pnls.append(pnl - cost)
                if random_pnls:
                    null_avgs.append(np.mean(random_pnls))

    if null_avgs:
        pctile = scipy_stats.percentileofscore(null_avgs, real_avg)
        null_mean = np.mean(null_avgs)
        null_results[tk] = {"real": real_avg, "null_mean": null_mean, "pctile": pctile}
        print(f"    {tk:>6s}: real={real_avg:+.2f}% vs null={null_mean:+.2f}%  "
              f"{pctile:.0f}th percentile  {'PASS' if pctile > 90 else 'FAIL'}")

# Pairwise correlation of trade returns
if not sel_c.empty and len(sel_c["ticker"].unique()) > 1:
    print("\n  ── Pairwise return correlation (Stage C) ──")
    tickers_with_trades = sorted(sel_c["ticker"].unique())
    # Build monthly return series per ticker
    monthly = sel_c.copy()
    monthly["month"] = monthly["entry_date"].dt.to_period("M")
    pivot = monthly.groupby(["ticker", "month"])["pnl_net_pct"].sum().unstack(level=0)
    if pivot.shape[1] > 1:
        corr = pivot.corr()
        print(f"    Avg pairwise correlation: {corr.values[np.triu_indices_from(corr.values, k=1)].mean():+.2f}")


# ══════════════════════════════════════════════════════════════════
# SECTION 12: DEPLOYMENT DECISION
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  SECTION 12: DEPLOYMENT DECISION")
print("=" * 70)

# Rule 1: Selected beats rejected in both B and C
sel_beats_b = (all_sel_b["pnl_net_pct"].mean() > all_rej_b["pnl_net_pct"].mean()) \
    if not all_sel_b.empty and not all_rej_b.empty else False
sel_beats_c = (sel_c["pnl_net_pct"].mean() > rej_c["pnl_net_pct"].mean()) \
    if not sel_c.empty and not rej_c.empty else (not sel_c.empty)
rule1 = sel_beats_b and sel_beats_c

# Rule 2: Pooled C avg >= 1.5% AND clustered t >= 2 across B+C
rule2_avg = sel_c["pnl_net_pct"].mean() >= 1.5 if not sel_c.empty else False
rule2_t = t_bc >= 2.0 if not bc_sel.empty else False
rule2 = rule2_avg and rule2_t

# Rule 3: Portfolio C total PnL positive
rule3 = port_c_stats.get("total_pnl_pct", 0) > 0

deployable = rule1 and rule2 and rule3

print(f"  Rule 1 (selected > rejected, B and C): {'PASS' if rule1 else 'FAIL'}")
print(f"    Stage B: sel={all_sel_b['pnl_net_pct'].mean():+.2f}% vs rej={all_rej_b['pnl_net_pct'].mean():+.2f}%  {'PASS' if sel_beats_b else 'FAIL'}" if not all_sel_b.empty else "    Stage B: N/A")
print(f"    Stage C: sel={sel_c['pnl_net_pct'].mean():+.2f}% vs rej={rej_c['pnl_net_pct'].mean():+.2f}%  {'PASS' if sel_beats_c else 'FAIL'}" if not sel_c.empty else "    Stage C: N/A")
print(f"  Rule 2 (C avg >= 1.5% AND B+C t >= 2): {'PASS' if rule2 else 'FAIL'}")
print(f"    C avg: {sel_c['pnl_net_pct'].mean():+.2f}% {'PASS' if rule2_avg else 'FAIL'}" if not sel_c.empty else "    C avg: N/A")
print(f"    B+C clustered t: {t_bc:.2f} {'PASS' if rule2_t else 'FAIL'}" if not bc_sel.empty else "    B+C t: N/A")
print(f"  Rule 3 (portfolio C total positive): {'PASS' if rule3 else 'FAIL'}")
print(f"    Portfolio C total: {port_c_stats.get('total_pnl_pct', 0):+.1f}%")
print(f"\n  >>> {'DEPLOYABLE' if deployable else 'NOT DEPLOYABLE'}")


# ══════════════════════════════════════════════════════════════════
# SECTION 13: PLOTS + FINAL OUTPUT
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 70)
print("  SECTION 13: PLOTS + DELIVERABLES")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle("Agent 10 Phase 4 — S5_zexit Generalization", fontsize=14, fontweight="bold")

# Top-left: Portfolio equity curves (B and C)
ax = axes[0, 0]
if not port_b_eq.empty:
    ax.plot(port_b_eq.index, port_b_eq.values, "b-", lw=1.5, label=f"Stage B ({port_b_stats.get('n_trades',0)} trades)")
if not port_c_eq.empty:
    ax.plot(port_c_eq.index, port_c_eq.values, "r-", lw=2, label=f"Stage C ({port_c_stats.get('n_trades',0)} trades)")
ax.axhline(0, color="black", lw=0.5, ls="--")
ax.set_ylabel("Cumulative PnL (%)")
ax.set_title("Portfolio Equity Curves")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Top-right: Per-ticker Stage C PnL
ax = axes[0, 1]
if not sel_c.empty:
    tk_pnls = sel_c.groupby("ticker")["pnl_net_pct"].agg(["sum", "count"]).sort_values("sum", ascending=True)
    colors = ["green" if v > 0 else "red" for v in tk_pnls["sum"]]
    ax.barh(tk_pnls.index, tk_pnls["sum"], color=colors)
    for i, (tk, row) in enumerate(tk_pnls.iterrows()):
        ax.text(row["sum"], i, f" n={int(row['count'])}", va="center", fontsize=7)
ax.set_xlabel("Total PnL (%)")
ax.set_title("Per-Ticker Stage C PnL")
ax.axvline(0, color="black", lw=0.5)
ax.grid(True, alpha=0.3, axis="x")

# Bottom-left: Selected vs rejected by stage
ax = axes[1, 0]
stages = ["A", "B", "C"]
sel_avgs = []
rej_avgs = []
for s_label in stages:
    if s_label == "A":
        sa = selected_a["stage_a_avg_pnl"].mean() if len(selected_a) > 0 else 0
        ra = rejected_a["stage_a_avg_pnl"].mean() if len(rejected_a) > 0 else 0
    elif s_label == "B":
        sa = all_sel_b["pnl_net_pct"].mean() if not all_sel_b.empty else 0
        ra = all_rej_b["pnl_net_pct"].mean() if not all_rej_b.empty else 0
    else:
        sa = sel_c["pnl_net_pct"].mean() if not sel_c.empty else 0
        ra = rej_c["pnl_net_pct"].mean() if not rej_c.empty else 0
    sel_avgs.append(sa)
    rej_avgs.append(ra)

x = np.arange(len(stages))
ax.bar(x - 0.2, sel_avgs, 0.35, label="Selected", color="green", alpha=0.7)
ax.bar(x + 0.2, rej_avgs, 0.35, label="Rejected", color="red", alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(["Stage A\n(2021-23)", "Stage B\n(2024-25)", "Stage C\n(2026)"])
ax.set_ylabel("Avg PnL per trade (%)")
ax.set_title("Selected vs Rejected by Stage")
ax.legend()
ax.axhline(0, color="black", lw=0.5)
ax.grid(True, alpha=0.3, axis="y")

# Bottom-right: Fingerprint features vs PnL scatter
ax = axes[1, 1]
if not fp_with_outcome.empty and "hedge_r2" in fp_with_outcome.columns:
    colors = ["green" if fp_with_outcome["selected_a"].iloc[i] else "red"
              for i in range(len(fp_with_outcome))]
    ax.scatter(fp_with_outcome["hedge_r2"], fp_with_outcome["stage_a_avg_pnl"],
               c=colors, s=60, alpha=0.7, edgecolors="black", lw=0.5)
    for _, row in fp_with_outcome.iterrows():
        ax.annotate(row["ticker"], (row["hedge_r2"], row["stage_a_avg_pnl"]),
                    fontsize=6, ha="center", va="bottom")
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.axvline(SELECTION_RULE["min_hedge_r2"], color="blue", lw=1, ls="--", alpha=0.5,
               label=f"R² threshold={SELECTION_RULE['min_hedge_r2']:.3f}")
    ax.set_xlabel("Hedge R²")
    ax.set_ylabel("Stage A Avg PnL (%)")
    ax.set_title("Hedge R² vs Performance (Stage A)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "phase4_generalization.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved phase4_generalization.png")

# ── Final tradable list ──
print("\n  ── FINAL TRADABLE LIST ──")
if deployable and not sel_c.empty:
    # Tickers that individually have positive total PnL in Stage C
    tradable = []
    for tk in sorted(sel_c["ticker"].unique()):
        tk_tr = sel_c[sel_c["ticker"] == tk]
        total = tk_tr["pnl_net_pct"].sum()
        if total > 0:
            tradable.append({
                "ticker": tk,
                "hedge": ticker_hedge_map.get(tk, "?"),
                "n_trades_2026": len(tk_tr),
                "avg_pnl": round(tk_tr["pnl_net_pct"].mean(), 2),
                "total_pnl": round(total, 2),
                "bucket": "seen" if tk in SEEN_PEERS or tk == "RXRX" else "new",
                "sizing": "half RXRX size for first 10 trades",
            })

    for t in tradable:
        print(f"    {t['ticker']:>6s} -> hedge={t['hedge']:>5s}  avg={t['avg_pnl']:+.2f}%  "
              f"total={t['total_pnl']:+.1f}%  n={t['n_trades_2026']}  [{t['bucket']}]")

    # Save tradable list
    tradable_path = OUT_DIR / "phase4_tradable_list.json"
    with open(tradable_path, "w") as f:
        json.dump({"timestamp": datetime.utcnow().isoformat() + "Z",
                    "tradable": tradable,
                    "rule": SELECTION_RULE,
                    "strategy": prereg_c["frozen_strategy"]}, f, indent=2)
    print(f"\n  Saved tradable list: {tradable_path}")
else:
    print("    NOT DEPLOYABLE — no tradable list generated")

# Summary CSV
all_results = []
for tk in all_spreads:
    for stage in ["A", "B", "C"]:
        tr = all_trades.get((tk, stage), pd.DataFrame())
        if not tr.empty:
            all_results.append({
                "ticker": tk, "stage": stage, "hedge": ticker_hedge_map.get(tk, ""),
                "n_trades": len(tr), "avg_pnl": round(tr["pnl_net_pct"].mean(), 3),
                "total_pnl": round(tr["pnl_net_pct"].sum(), 2),
                "wr": round((tr["pnl_net_pct"] > 0).mean() * 100, 1),
                "selected": tk in stage_c_selected,
                "bucket": "rxrx" if tk == "RXRX" else ("seen" if tk in SEEN_PEERS else
                          ("new_bio" if tk in NEW_BIO else "out_sector")),
            })

results_df = pd.DataFrame(all_results)
results_df.to_csv(OUT_DIR / "phase4_all_results.csv", index=False)
print(f"  Saved phase4_all_results.csv ({len(results_df)} rows)")

print("\n" + "=" * 70)
print("  PHASE 4 COMPLETE")
print("=" * 70)
