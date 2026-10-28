# %% Agent 4v2 -- Standalone Level-Bounce / Support-Resistance Strategy
#
# Builds STANDALONE strategies that trade RXRX bounces off support and
# rejections at resistance, using only price/volume data.
# Selection window ONLY: 2021-04-16 to 2025-12-31
# No S5_zexit references. No 2026 data.

import sys, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ── Config ──────────────────────────────────────────────────────────
SEL_END = "2026-01-01"
MAX_HOLD = 10
POSITION_SIZE = 10000
N_PERMUTATIONS = 1000
N_NULL_RUNS = 1000
SWING_WINDOWS = [10, 20]
VOLUME_PROFILE_LOOKBACK = 120
VOLUME_BIN_SIZE = 0.50
N_HVN_TOP = 3
ATR_PROX_MULT = 0.5        # within 0.5 ATR = "touching" a level
ATR_STOP_MULT = 1.0        # stop at level +/- 1 ATR
ATR_TARGET_MULT = 2.0      # target at +/- 2 ATR if no opposing level
ROUND_NUMBERS = [5, 10, 15, 20, 25, 30, 35, 40]
WF_TRAIN_MONTHS = 12
WF_TEST_MONTHS = 3

# ── Imports from engine ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent if "__file__" in dir() else Path("agent10")
sys.path.insert(0, str(BASE_DIR))
from agent0_engine import (run_backtest, compute_stats, compute_costs,
                           log_trial, add_features, null_test,
                           compute_sharpe, walk_forward)

DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════

rxrx_raw = pd.read_parquet(DATA_DIR / "rxrx_daily.parquet")
rxrx_raw["date"] = pd.to_datetime(rxrx_raw["date"])
rxrx_raw = rxrx_raw.sort_values("date").reset_index(drop=True)
rxrx_raw = add_features(rxrx_raw)

# Selection window only
rxrx = rxrx_raw[rxrx_raw["date"] < SEL_END].copy().reset_index(drop=True)
print(f"RXRX selection window: {len(rxrx)} bars, "
      f"{rxrx['date'].min().date()} to {rxrx['date'].max().date()}")
print(f"Price range: ${rxrx['close'].min():.2f} - ${rxrx['close'].max():.2f}")


# ═══════════════════════════════════════════════════════════════════
# LEVEL IDENTIFICATION (trailing only, no lookahead)
# ═══════════════════════════════════════════════════════════════════

def find_swing_lows(df, window):
    """Trailing swing lows: bar i is a swing low if close[i] is the min
    of close[i-window:i+1]. Only uses past data up to current bar."""
    lows = []
    for i in range(window, len(df)):
        segment = df["low"].iloc[i - window:i + 1].values
        if df["low"].iloc[i] == segment.min():
            lows.append((i, df["low"].iloc[i], df["date"].iloc[i]))
    return lows


def find_swing_highs(df, window):
    """Trailing swing highs: bar i is a swing high if close[i] is the max
    of close[i-window:i+1]. Only uses past data up to current bar."""
    highs = []
    for i in range(window, len(df)):
        segment = df["high"].iloc[i - window:i + 1].values
        if df["high"].iloc[i] == segment.max():
            highs.append((i, df["high"].iloc[i], df["date"].iloc[i]))
    return highs


def compute_trailing_hvn(df, idx, lookback=120, bin_size=0.50, n_top=3):
    """Volume-weighted price clusters from trailing data.
    Returns list of (price_center, volume_weight) for top-N HVN bins."""
    start = max(0, idx - lookback)
    segment = df.iloc[start:idx + 1]
    if len(segment) < 20:
        return []

    # Build volume profile using typical price * volume
    typical = (segment["high"] + segment["low"] + segment["close"]) / 3
    volumes = segment["volume"]

    price_min = typical.min()
    price_max = typical.max()
    bins = np.arange(price_min, price_max + bin_size, bin_size)
    if len(bins) < 2:
        return []

    vol_profile = np.zeros(len(bins) - 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    for p, v in zip(typical, volumes):
        bin_idx = int((p - price_min) / bin_size)
        bin_idx = min(bin_idx, len(vol_profile) - 1)
        vol_profile[bin_idx] += v

    # Top-N bins by volume
    top_idx = np.argsort(vol_profile)[-n_top:]
    return [(bin_centers[j], vol_profile[j]) for j in top_idx if vol_profile[j] > 0]


def get_round_numbers_near_price(price, distance_pct=50):
    """Return round numbers within distance_pct% of current price."""
    levels = []
    for r in ROUND_NUMBERS:
        if abs(r - price) / price * 100 < distance_pct:
            levels.append(r)
    return levels


def build_level_map(df):
    """Build trailing support/resistance levels for each bar.
    Returns dict: bar_idx -> {"support": [prices], "resistance": [prices],
                               "level_types": {price: type_string}}
    Only uses data up to and including bar_idx (no lookahead).
    """
    n = len(df)
    level_map = {}

    # Pre-compute all swing points
    all_swing_lows = {}
    all_swing_highs = {}
    for w in SWING_WINDOWS:
        for idx, price, date in find_swing_lows(df, w):
            if idx not in all_swing_lows:
                all_swing_lows[idx] = []
            all_swing_lows[idx].append((price, f"swing_low_{w}"))
        for idx, price, date in find_swing_highs(df, w):
            if idx not in all_swing_highs:
                all_swing_highs[idx] = []
            all_swing_highs[idx].append((price, f"swing_high_{w}"))

    for i in range(120, n):
        supports = []
        resistances = []
        level_types = {}
        current_price = df["close"].iloc[i]
        atr = df["atr20"].iloc[i]
        if pd.isna(atr) or atr <= 0:
            level_map[i] = {"support": [], "resistance": [], "level_types": {}}
            continue

        # 1. Recent swing lows -> support (trailing 120 bars)
        for j in range(max(0, i - 120), i):
            if j in all_swing_lows:
                for price, ltype in all_swing_lows[j]:
                    if price < current_price:
                        supports.append(price)
                        level_types[round(price, 2)] = ltype

        # 2. Recent swing highs -> resistance (trailing 120 bars)
        for j in range(max(0, i - 120), i):
            if j in all_swing_highs:
                for price, ltype in all_swing_highs[j]:
                    if price > current_price:
                        resistances.append(price)
                        level_types[round(price, 2)] = ltype

        # 3. Volume-weighted HVN levels
        hvns = compute_trailing_hvn(df, i, VOLUME_PROFILE_LOOKBACK,
                                    VOLUME_BIN_SIZE, N_HVN_TOP)
        for hvn_price, hvn_vol in hvns:
            if hvn_price < current_price - ATR_PROX_MULT * atr:
                supports.append(hvn_price)
                level_types[round(hvn_price, 2)] = "hvn"
            elif hvn_price > current_price + ATR_PROX_MULT * atr:
                resistances.append(hvn_price)
                level_types[round(hvn_price, 2)] = "hvn"

        # 4. Round numbers
        for r in get_round_numbers_near_price(current_price):
            if r < current_price - ATR_PROX_MULT * atr:
                supports.append(float(r))
                level_types[r] = "round"
            elif r > current_price + ATR_PROX_MULT * atr:
                resistances.append(float(r))
                level_types[r] = "round"

        # Deduplicate: merge levels within 0.25 ATR of each other
        supports = sorted(set([round(s, 2) for s in supports]), reverse=True)
        resistances = sorted(set([round(r, 2) for r in resistances]))

        # Keep closest 5 of each
        supports = supports[:5]
        resistances = resistances[:5]

        level_map[i] = {
            "support": supports,
            "resistance": resistances,
            "level_types": level_types,
        }

    return level_map


print("\nBuilding trailing S/R level map...")
level_map = build_level_map(rxrx)
print(f"Level map built for {len(level_map)} bars")

# Quick stats on level coverage
bars_with_support = sum(1 for v in level_map.values() if len(v["support"]) > 0)
bars_with_resistance = sum(1 for v in level_map.values() if len(v["resistance"]) > 0)
print(f"Bars with support levels: {bars_with_support}/{len(level_map)} "
      f"({bars_with_support/len(level_map)*100:.0f}%)")
print(f"Bars with resistance levels: {bars_with_resistance}/{len(level_map)} "
      f"({bars_with_resistance/len(level_map)*100:.0f}%)")


# ═══════════════════════════════════════════════════════════════════
# LEVEL-BOUNCE RATE ANALYSIS
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  LEVEL-BOUNCE RATE ANALYSIS")
print("=" * 70)


def analyze_level_touches(df, level_map, level_side="support"):
    """For each bar near a level, determine if price bounced or broke through.
    Returns list of dicts with touch info and forward returns."""
    touches = []
    for i in range(120, len(df) - 5):
        if i not in level_map:
            continue
        levels = level_map[i][level_side]
        if not levels:
            continue

        close_i = df["close"].iloc[i]
        low_i = df["low"].iloc[i]
        high_i = df["high"].iloc[i]
        atr = df["atr20"].iloc[i]
        if pd.isna(atr) or atr <= 0:
            continue

        for lvl in levels:
            proximity = abs(close_i - lvl) / atr

            if level_side == "support":
                # Touch = low came within 0.5 ATR of support
                if low_i <= lvl + ATR_PROX_MULT * atr and close_i > lvl:
                    # Check if bounced (next day close > level) or broke (below)
                    fwd_closes = {}
                    for d in [1, 3, 5]:
                        if i + d < len(df):
                            fwd_closes[d] = df["close"].iloc[i + d]
                    if not fwd_closes:
                        continue
                    bounced = fwd_closes.get(1, close_i) > lvl
                    fwd_rets = {d: (fwd_closes[d] - close_i) / close_i * 100
                                for d in fwd_closes}
                    ltype = level_map[i]["level_types"].get(round(lvl, 2), "unknown")
                    touches.append({
                        "bar_idx": i, "date": df["date"].iloc[i],
                        "level": lvl, "level_type": ltype,
                        "level_side": level_side,
                        "proximity_atr": round(proximity, 3),
                        "bounced": bounced, **{f"ret_{d}d": round(v, 3) for d, v in fwd_rets.items()},
                    })

            else:  # resistance
                # Touch = high came within 0.5 ATR of resistance
                if high_i >= lvl - ATR_PROX_MULT * atr and close_i < lvl:
                    fwd_closes = {}
                    for d in [1, 3, 5]:
                        if i + d < len(df):
                            fwd_closes[d] = df["close"].iloc[i + d]
                    if not fwd_closes:
                        continue
                    bounced = fwd_closes.get(1, close_i) < lvl  # rejected
                    fwd_rets = {d: (fwd_closes[d] - close_i) / close_i * 100
                                for d in fwd_closes}
                    ltype = level_map[i]["level_types"].get(round(lvl, 2), "unknown")
                    touches.append({
                        "bar_idx": i, "date": df["date"].iloc[i],
                        "level": lvl, "level_type": ltype,
                        "level_side": level_side,
                        "proximity_atr": round(proximity, 3),
                        "bounced": bounced, **{f"ret_{d}d": round(v, 3) for d, v in fwd_rets.items()},
                    })

    return pd.DataFrame(touches)


support_touches = analyze_level_touches(rxrx, level_map, "support")
resistance_touches = analyze_level_touches(rxrx, level_map, "resistance")

print(f"\nSupport touches: {len(support_touches)}")
print(f"Resistance touches: {len(resistance_touches)}")


def compute_bounce_rates(touches_df, label=""):
    """Compute bounce rate statistics by level type."""
    if touches_df.empty:
        print(f"  {label}: No touches found")
        return {}
    rates = {}
    for ltype in touches_df["level_type"].unique():
        subset = touches_df[touches_df["level_type"] == ltype]
        n = len(subset)
        bounce_rate = subset["bounced"].mean() * 100
        avg_ret_1d = subset["ret_1d"].mean() if "ret_1d" in subset.columns else np.nan
        avg_ret_3d = subset["ret_3d"].mean() if "ret_3d" in subset.columns else np.nan
        avg_ret_5d = subset["ret_5d"].mean() if "ret_5d" in subset.columns else np.nan
        rates[ltype] = {
            "n": n, "bounce_rate": round(bounce_rate, 1),
            "avg_ret_1d": round(avg_ret_1d, 3),
            "avg_ret_3d": round(avg_ret_3d, 3),
            "avg_ret_5d": round(avg_ret_5d, 3),
        }
        print(f"  {label} {ltype}: n={n}, bounce={bounce_rate:.1f}%, "
              f"ret_1d={avg_ret_1d:+.3f}%, ret_3d={avg_ret_3d:+.3f}%, "
              f"ret_5d={avg_ret_5d:+.3f}%")
    # Overall
    n_all = len(touches_df)
    br_all = touches_df["bounced"].mean() * 100
    avg1 = touches_df["ret_1d"].mean() if "ret_1d" in touches_df.columns else np.nan
    avg3 = touches_df["ret_3d"].mean() if "ret_3d" in touches_df.columns else np.nan
    avg5 = touches_df["ret_5d"].mean() if "ret_5d" in touches_df.columns else np.nan
    rates["ALL"] = {"n": n_all, "bounce_rate": round(br_all, 1),
                    "avg_ret_1d": round(avg1, 3), "avg_ret_3d": round(avg3, 3),
                    "avg_ret_5d": round(avg5, 3)}
    print(f"  {label} ALL: n={n_all}, bounce={br_all:.1f}%, "
          f"ret_1d={avg1:+.3f}%, ret_3d={avg3:+.3f}%, ret_5d={avg5:+.3f}%")
    return rates


print("\n--- Support Level Bounce Rates ---")
support_rates = compute_bounce_rates(support_touches, "SUPPORT")

print("\n--- Resistance Level Bounce Rates ---")
resistance_rates = compute_bounce_rates(resistance_touches, "RESISTANCE")


# ── Permutation test: compare bounce rates to random levels ─────────

def permutation_test_bounce_rate(df, touches_df, n_perms=1000, side="support"):
    """Compare real bounce rate to random-level null."""
    if touches_df.empty:
        return 0, 0, 0
    real_bounce_rate = touches_df["bounced"].mean()
    n_touches_target = len(touches_df)

    null_rates = []
    for perm in range(n_perms):
        rng = np.random.RandomState(perm + 50000)
        # Generate random price levels within the price range
        price_range = df["close"].max() - df["close"].min()
        random_levels = rng.uniform(df["close"].min(), df["close"].max(),
                                    size=20)
        bounces = 0
        total = 0
        for i in range(120, len(df) - 5):
            atr = df["atr20"].iloc[i]
            if pd.isna(atr) or atr <= 0:
                continue
            close_i = df["close"].iloc[i]
            for lvl in random_levels:
                if side == "support" and lvl < close_i:
                    if df["low"].iloc[i] <= lvl + ATR_PROX_MULT * atr and close_i > lvl:
                        total += 1
                        if i + 1 < len(df) and df["close"].iloc[i + 1] > lvl:
                            bounces += 1
                elif side == "resistance" and lvl > close_i:
                    if df["high"].iloc[i] >= lvl - ATR_PROX_MULT * atr and close_i < lvl:
                        total += 1
                        if i + 1 < len(df) and df["close"].iloc[i + 1] < lvl:
                            bounces += 1
        if total > 0:
            null_rates.append(bounces / total)

    if not null_rates:
        return 0, 0, 0
    from scipy.stats import percentileofscore
    pctile = percentileofscore(null_rates, real_bounce_rate)
    return round(pctile, 1), round(np.mean(null_rates) * 100, 1), round(real_bounce_rate * 100, 1)


print("\n--- Permutation Tests (bounce rate vs random levels) ---")
sup_pctile, sup_null_mean, sup_real = permutation_test_bounce_rate(
    rxrx, support_touches, N_PERMUTATIONS, "support")
print(f"  SUPPORT bounce: real={sup_real:.1f}%, null_mean={sup_null_mean:.1f}%, "
      f"percentile={sup_pctile:.1f}th")

res_pctile, res_null_mean, res_real = permutation_test_bounce_rate(
    rxrx, resistance_touches, N_PERMUTATIONS, "resistance")
print(f"  RESISTANCE bounce: real={res_real:.1f}%, null_mean={res_null_mean:.1f}%, "
      f"percentile={res_pctile:.1f}th")


# ═══════════════════════════════════════════════════════════════════
# STRATEGY 1: S/R BOUNCE LONG
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  STRATEGY 1: S/R BOUNCE LONG")
print("=" * 70)


def sr_bounce_long_signal(row, i, df):
    """Go long when price touches a support level."""
    if i not in level_map:
        return None
    levels = level_map[i]
    supports = levels["support"]
    resistances = levels["resistance"]
    if not supports:
        return None

    close_i = row["close"]
    low_i = row["low"]
    atr = row["atr20"]
    if pd.isna(atr) or atr <= 0:
        return None

    # Check if price touched support (low within 0.5 ATR of any support)
    for sup in supports:
        if low_i <= sup + ATR_PROX_MULT * atr and close_i > sup:
            # Entry signal: long
            stop = sup - ATR_STOP_MULT * atr
            # Target: next resistance or +2 ATR
            if resistances:
                target = resistances[0]  # closest resistance above
            else:
                target = close_i + ATR_TARGET_MULT * atr
            return {"side": "long", "stop": stop, "target": target}
    return None


trades_bounce_long = run_backtest(rxrx, sr_bounce_long_signal, None, MAX_HOLD,
                                  POSITION_SIZE)
if not trades_bounce_long.empty:
    trades_bounce_long = trades_bounce_long[
        trades_bounce_long["entry_date"] < SEL_END].copy()

stats_bl = compute_stats(trades_bounce_long, rxrx, "SR_Bounce_Long")
print(f"  n={stats_bl['n']}, avg_pnl={stats_bl['avg_pnl']:+.3f}%, "
      f"sharpe={stats_bl['sharpe']}, wr={stats_bl['wr']}%, "
      f"t={stats_bl['t_stat']:.2f}, pf={stats_bl['pf']}")
log_trial("agent4v2", "SR_Bounce_Long",
          {"prox": ATR_PROX_MULT, "stop": ATR_STOP_MULT, "target": ATR_TARGET_MULT},
          stats_bl)


# ═══════════════════════════════════════════════════════════════════
# STRATEGY 2: S/R BOUNCE SHORT
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  STRATEGY 2: S/R BOUNCE SHORT")
print("=" * 70)


def sr_bounce_short_signal(row, i, df):
    """Go short when price touches a resistance level."""
    if i not in level_map:
        return None
    levels = level_map[i]
    supports = levels["support"]
    resistances = levels["resistance"]
    if not resistances:
        return None

    close_i = row["close"]
    high_i = row["high"]
    atr = row["atr20"]
    if pd.isna(atr) or atr <= 0:
        return None

    for res in resistances:
        if high_i >= res - ATR_PROX_MULT * atr and close_i < res:
            stop = res + ATR_STOP_MULT * atr
            if supports:
                target = supports[0]  # closest support below
            else:
                target = close_i - ATR_TARGET_MULT * atr
            return {"side": "short", "stop": stop, "target": target}
    return None


trades_bounce_short = run_backtest(rxrx, sr_bounce_short_signal, None, MAX_HOLD,
                                   POSITION_SIZE)
if not trades_bounce_short.empty:
    trades_bounce_short = trades_bounce_short[
        trades_bounce_short["entry_date"] < SEL_END].copy()

stats_bs = compute_stats(trades_bounce_short, rxrx, "SR_Bounce_Short")
print(f"  n={stats_bs['n']}, avg_pnl={stats_bs['avg_pnl']:+.3f}%, "
      f"sharpe={stats_bs['sharpe']}, wr={stats_bs['wr']}%, "
      f"t={stats_bs['t_stat']:.2f}, pf={stats_bs['pf']}")
log_trial("agent4v2", "SR_Bounce_Short",
          {"prox": ATR_PROX_MULT, "stop": ATR_STOP_MULT, "target": ATR_TARGET_MULT},
          stats_bs)


# ═══════════════════════════════════════════════════════════════════
# STRATEGY 3: BREAKOUT
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  STRATEGY 3: BREAKOUT (momentum)")
print("=" * 70)


def breakout_signal(row, i, df):
    """Go long on close above resistance, short on close below support."""
    if i not in level_map:
        return None
    levels = level_map[i]
    supports = levels["support"]
    resistances = levels["resistance"]

    close_i = row["close"]
    atr = row["atr20"]
    if pd.isna(atr) or atr <= 0:
        return None

    # Breakout above resistance
    for res in resistances:
        if close_i > res:
            stop = res - 0.5 * atr
            target = close_i + ATR_TARGET_MULT * atr
            return {"side": "long", "stop": stop, "target": target}

    # Breakdown below support
    for sup in supports:
        if close_i < sup:
            stop = sup + 0.5 * atr
            target = close_i - ATR_TARGET_MULT * atr
            return {"side": "short", "stop": stop, "target": target}

    return None


trades_breakout = run_backtest(rxrx, breakout_signal, None, MAX_HOLD,
                               POSITION_SIZE)
if not trades_breakout.empty:
    trades_breakout = trades_breakout[
        trades_breakout["entry_date"] < SEL_END].copy()

stats_bo = compute_stats(trades_breakout, rxrx, "Breakout")
print(f"  n={stats_bo['n']}, avg_pnl={stats_bo['avg_pnl']:+.3f}%, "
      f"sharpe={stats_bo['sharpe']}, wr={stats_bo['wr']}%, "
      f"t={stats_bo['t_stat']:.2f}, pf={stats_bo['pf']}")
log_trial("agent4v2", "Breakout",
          {"prox": 0.5, "stop_mult": 0.5, "target": ATR_TARGET_MULT},
          stats_bo)


# ═══════════════════════════════════════════════════════════════════
# STRATEGY 4: RANGE-FADE (mean reversion within S/R band)
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  STRATEGY 4: RANGE-FADE")
print("=" * 70)


def range_fade_signal(row, i, df):
    """Fade moves toward S/R edges when price is between levels.
    Long near support, short near resistance, within a defined range."""
    if i not in level_map:
        return None
    levels = level_map[i]
    supports = levels["support"]
    resistances = levels["resistance"]
    if not supports or not resistances:
        return None

    close_i = row["close"]
    low_i = row["low"]
    high_i = row["high"]
    atr = row["atr20"]
    if pd.isna(atr) or atr <= 0:
        return None

    # Closest support and resistance
    nearest_sup = supports[0]
    nearest_res = resistances[0]
    range_width = nearest_res - nearest_sup

    # Need a meaningful range (at least 2 ATR wide)
    if range_width < 2 * atr:
        return None

    # Position within range (0 = at support, 1 = at resistance)
    range_pct = (close_i - nearest_sup) / range_width

    # Fade toward support (long when in lower 25% of range)
    if range_pct < 0.25 and low_i <= nearest_sup + ATR_PROX_MULT * atr:
        stop = nearest_sup - ATR_STOP_MULT * atr
        # Target: mid-range
        target = nearest_sup + range_width * 0.5
        return {"side": "long", "stop": stop, "target": target}

    # Fade toward resistance (short when in upper 75% of range)
    if range_pct > 0.75 and high_i >= nearest_res - ATR_PROX_MULT * atr:
        stop = nearest_res + ATR_STOP_MULT * atr
        target = nearest_res - range_width * 0.5
        return {"side": "short", "stop": stop, "target": target}

    return None


trades_range = run_backtest(rxrx, range_fade_signal, None, MAX_HOLD,
                            POSITION_SIZE)
if not trades_range.empty:
    trades_range = trades_range[
        trades_range["entry_date"] < SEL_END].copy()

stats_rf = compute_stats(trades_range, rxrx, "Range_Fade")
print(f"  n={stats_rf['n']}, avg_pnl={stats_rf['avg_pnl']:+.3f}%, "
      f"sharpe={stats_rf['sharpe']}, wr={stats_rf['wr']}%, "
      f"t={stats_rf['t_stat']:.2f}, pf={stats_rf['pf']}")
log_trial("agent4v2", "Range_Fade",
          {"range_min_atr": 2, "long_pct": 0.25, "short_pct": 0.75},
          stats_rf)


# ═══════════════════════════════════════════════════════════════════
# STRATEGY 5: COMBINED S/R (Long + Short bounce together)
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  STRATEGY 5: COMBINED S/R BOUNCE (Long + Short)")
print("=" * 70)


def combined_sr_signal(row, i, df):
    """Combined: long at support bounce, short at resistance rejection."""
    # Try long first
    sig = sr_bounce_long_signal(row, i, df)
    if sig:
        return sig
    # Then try short
    sig = sr_bounce_short_signal(row, i, df)
    if sig:
        return sig
    return None


trades_combined = run_backtest(rxrx, combined_sr_signal, None, MAX_HOLD,
                               POSITION_SIZE)
if not trades_combined.empty:
    trades_combined = trades_combined[
        trades_combined["entry_date"] < SEL_END].copy()

stats_comb = compute_stats(trades_combined, rxrx, "Combined_SR")
print(f"  n={stats_comb['n']}, avg_pnl={stats_comb['avg_pnl']:+.3f}%, "
      f"sharpe={stats_comb['sharpe']}, wr={stats_comb['wr']}%, "
      f"t={stats_comb['t_stat']:.2f}, pf={stats_comb['pf']}")
log_trial("agent4v2", "Combined_SR",
          {"prox": ATR_PROX_MULT, "stop": ATR_STOP_MULT},
          stats_comb)


# ═══════════════════════════════════════════════════════════════════
# STRATEGY 6: VOLUME-CONFIRMED BOUNCE
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  STRATEGY 6: VOLUME-CONFIRMED BOUNCE")
print("=" * 70)


def volume_confirmed_bounce_signal(row, i, df):
    """Like bounce long/short but requires above-average volume on the
    touch bar (confirms institutional interest at the level)."""
    if i not in level_map:
        return None
    levels = level_map[i]

    vol = row["volume"]
    vol_ma = row.get("vol_ma20", np.nan)
    if pd.isna(vol_ma) or vol_ma <= 0 or vol < 1.2 * vol_ma:
        return None  # Need 20% above average volume

    # Try long at support
    sig = sr_bounce_long_signal(row, i, df)
    if sig:
        return sig

    # Try short at resistance
    sig = sr_bounce_short_signal(row, i, df)
    if sig:
        return sig

    return None


trades_volcf = run_backtest(rxrx, volume_confirmed_bounce_signal, None, MAX_HOLD,
                            POSITION_SIZE)
if not trades_volcf.empty:
    trades_volcf = trades_volcf[
        trades_volcf["entry_date"] < SEL_END].copy()

stats_vc = compute_stats(trades_volcf, rxrx, "Vol_Confirmed_Bounce")
print(f"  n={stats_vc['n']}, avg_pnl={stats_vc['avg_pnl']:+.3f}%, "
      f"sharpe={stats_vc['sharpe']}, wr={stats_vc['wr']}%, "
      f"t={stats_vc['t_stat']:.2f}, pf={stats_vc['pf']}")
log_trial("agent4v2", "Vol_Confirmed_Bounce",
          {"vol_thresh": 1.2, "prox": ATR_PROX_MULT, "stop": ATR_STOP_MULT},
          stats_vc)


# ═══════════════════════════════════════════════════════════════════
# NULL TESTS (for any strategy with positive avg PnL)
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  NULL TESTS (1000 random entries)")
print("=" * 70)

all_strategies = {
    "SR_Bounce_Long": (trades_bounce_long, stats_bl),
    "SR_Bounce_Short": (trades_bounce_short, stats_bs),
    "Breakout": (trades_breakout, stats_bo),
    "Range_Fade": (trades_range, stats_rf),
    "Combined_SR": (trades_combined, stats_comb),
    "Vol_Confirmed": (trades_volcf, stats_vc),
}

null_results = {}
for name, (trades, st) in all_strategies.items():
    if st["n"] < 5:
        print(f"  {name}: SKIP (n={st['n']} < 5)")
        null_results[name] = (0, 0, 0)
        continue
    if st["avg_pnl"] > 0:
        pctile, null_mean, real_avg = null_test(trades, rxrx, N_NULL_RUNS, MAX_HOLD)
        null_results[name] = (pctile, null_mean, real_avg)
        status = "PASS" if pctile >= 95 else "FAIL"
        print(f"  {name}: real={real_avg:+.3f}%, null_mean={null_mean:+.3f}%, "
              f"pctile={pctile:.1f}th -> {status}")
    else:
        print(f"  {name}: SKIP (avg_pnl={st['avg_pnl']:+.3f}% <= 0)")
        null_results[name] = (0, 0, st["avg_pnl"])


# ═══════════════════════════════════════════════════════════════════
# WALK-FORWARD (for any strategy passing null test)
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  WALK-FORWARD (12m train / 3m test)")
print("=" * 70)

wf_results = {}
best_wf_name = None
best_wf_avg = -999

for name, (trades, st) in all_strategies.items():
    pctile = null_results.get(name, (0,))[0]
    if st["avg_pnl"] > 0 and st["n"] >= 30 and pctile >= 50:
        # Run walk-forward even if null pctile is moderate, to see stability
        signal_fns = {
            "SR_Bounce_Long": sr_bounce_long_signal,
            "SR_Bounce_Short": sr_bounce_short_signal,
            "Breakout": breakout_signal,
            "Range_Fade": range_fade_signal,
            "Combined_SR": combined_sr_signal,
            "Vol_Confirmed": volume_confirmed_bounce_signal,
        }
        sig_fn = signal_fns.get(name)
        if sig_fn is None:
            continue
        wf = walk_forward(rxrx, sig_fn, None, WF_TRAIN_MONTHS, WF_TEST_MONTHS,
                          MAX_HOLD, POSITION_SIZE)
        if not wf.empty:
            wf = wf[wf["test_end"] < pd.Timestamp(SEL_END)]
        if not wf.empty:
            wf_avg = wf["avg_pnl"].mean()
            wf_results[name] = wf
            print(f"\n  {name} Walk-Forward:")
            for _, w in wf.iterrows():
                print(f"    {str(w['test_start'].date()):>12s} - "
                      f"{str(w['test_end'].date()):>12s}: n={w['n']:>3.0f}, "
                      f"avg={w['avg_pnl']:+.3f}%")
            pos_folds = (wf["avg_pnl"] > 0).sum()
            print(f"    WF avg: {wf_avg:+.3f}%, "
                  f"positive folds: {pos_folds}/{len(wf)}")
            if wf_avg > best_wf_avg:
                best_wf_avg = wf_avg
                best_wf_name = name
        else:
            print(f"  {name}: no walk-forward folds in selection window")
    else:
        reason = []
        if st["avg_pnl"] <= 0:
            reason.append(f"avg_pnl={st['avg_pnl']:+.3f}")
        if st["n"] < 30:
            reason.append(f"n={st['n']}<30")
        if pctile < 50:
            reason.append(f"null_pctile={pctile:.0f}")
        print(f"  {name}: SKIP WF ({', '.join(reason)})")


# ═══════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  STRATEGY COMPARISON SUMMARY")
print("=" * 70)

summary_rows = []
print(f"\n  {'Strategy':<22s} {'n':>5s} {'avg_pnl':>8s} {'sharpe':>7s} "
      f"{'wr':>6s} {'pf':>6s} {'t_stat':>7s} {'null%':>6s} {'verdict':>10s}")
print("  " + "-" * 80)

for name, (trades, st) in all_strategies.items():
    pctile = null_results.get(name, (0,))[0]
    if st["avg_pnl"] > 0 and pctile >= 95 and st["n"] >= 30:
        verdict = "PASS"
    elif st["avg_pnl"] > 0 and st["n"] >= 30:
        verdict = "WEAK"
    elif st["n"] < 30:
        verdict = "LOW_N"
    else:
        verdict = "FAIL"
    print(f"  {name:<22s} {st['n']:>5d} {st['avg_pnl']:>+7.3f}% "
          f"{st['sharpe']:>7.2f} {st['wr']:>5.1f}% {st['pf']:>6.2f} "
          f"{st['t_stat']:>7.2f} {pctile:>5.1f}% {verdict:>10s}")
    summary_rows.append({
        "strategy": name, "n": st["n"], "avg_pnl": st["avg_pnl"],
        "sharpe": st["sharpe"], "wr": st["wr"], "pf": st["pf"],
        "t_stat": st["t_stat"], "null_pctile": pctile, "verdict": verdict,
    })

# Final determination
any_pass = any(r["verdict"] == "PASS" for r in summary_rows)
if any_pass:
    print("\n  RESULT: At least one S/R strategy passes null test")
else:
    print("\n  RESULT: Hypothesis tested and rejected on selection window")
    print("  No standalone S/R level strategy achieves statistical significance")


# ═══════════════════════════════════════════════════════════════════
# MATPLOTLIB FIGURE
# ═══════════════════════════════════════════════════════════════════

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle("Agent 4v2: Standalone S/R Level-Bounce Strategies (Selection Window)",
             fontsize=14, fontweight="bold")

# ── Panel A: RXRX price with S/R levels (last 500 days) ──────────
ax = axes[0, 0]
last_500_start = max(0, len(rxrx) - 500)
plot_df = rxrx.iloc[last_500_start:].copy()
ax.plot(plot_df["date"], plot_df["close"], color="black", linewidth=0.8,
        label="RXRX Close", zorder=3)

# Overlay some S/R levels (sample every 20 bars to avoid clutter)
sup_plotted = set()
res_plotted = set()
for i in range(last_500_start, len(rxrx), 20):
    if i not in level_map:
        continue
    for s in level_map[i]["support"][:2]:
        key = round(s, 1)
        if key not in sup_plotted:
            ax.axhline(s, color="green", alpha=0.15, linewidth=0.5)
            sup_plotted.add(key)
    for r in level_map[i]["resistance"][:2]:
        key = round(r, 1)
        if key not in res_plotted:
            ax.axhline(r, color="red", alpha=0.15, linewidth=0.5)
            res_plotted.add(key)

# Mark round numbers
for rn in ROUND_NUMBERS:
    if plot_df["close"].min() < rn < plot_df["close"].max():
        ax.axhline(rn, color="blue", alpha=0.3, linewidth=0.8, linestyle="--")

ax.set_title("A) RXRX Price with S/R Levels (last 500 days)")
ax.set_ylabel("Price ($)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.tick_params(axis="x", rotation=45)
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)

# ── Panel B: Bounce rates vs random-level null ────────────────────
ax = axes[0, 1]
bar_labels = []
bar_real = []
bar_null = []

if support_rates:
    all_sup = support_rates.get("ALL", {})
    bar_labels.append("Support")
    bar_real.append(all_sup.get("bounce_rate", 0))
    bar_null.append(sup_null_mean)

if resistance_rates:
    all_res = resistance_rates.get("ALL", {})
    bar_labels.append("Resistance")
    bar_real.append(all_res.get("bounce_rate", 0))
    bar_null.append(res_null_mean)

x_pos = np.arange(len(bar_labels))
width = 0.35
ax.bar(x_pos - width/2, bar_real, width, label="Real Levels", color="steelblue")
ax.bar(x_pos + width/2, bar_null, width, label="Random Levels (null)", color="gray")
ax.set_xticks(x_pos)
ax.set_xticklabels(bar_labels)
ax.set_ylabel("Bounce Rate (%)")
ax.set_title("B) Level Bounce Rate: Real vs Random Null")
ax.legend()
ax.grid(axis="y", alpha=0.3)

# Add p-value annotations
for j, (label, pct) in enumerate(zip(bar_labels, [sup_pctile, res_pctile])):
    ax.annotate(f"p={pct:.0f}th", (x_pos[j], max(bar_real[j], bar_null[j]) + 1),
                ha="center", fontsize=9)

# ── Panel C: Strategy equity curves ──────────────────────────────
ax = axes[1, 0]
colors = ["steelblue", "coral", "green", "purple", "orange", "brown"]
for idx, (name, (trades, st)) in enumerate(all_strategies.items()):
    if not trades.empty and st["n"] >= 5:
        equity = trades["pnl_net_pct"].cumsum().values
        ax.plot(range(len(equity)), equity, label=f"{name} (n={st['n']})",
                color=colors[idx % len(colors)], linewidth=1.0)

ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax.set_xlabel("Trade #")
ax.set_ylabel("Cumulative PnL (%)")
ax.set_title("C) Strategy Equity Curves (Selection Window)")
ax.legend(fontsize=7, loc="best")
ax.grid(alpha=0.3)

# ── Panel D: Walk-forward fold results ────────────────────────────
ax = axes[1, 1]
if wf_results:
    best_name = best_wf_name or list(wf_results.keys())[0]
    wf = wf_results[best_name]
    fold_labels = [f"{str(w['test_start'].date())[:7]}" for _, w in wf.iterrows()]
    fold_avgs = wf["avg_pnl"].values
    bar_colors = ["green" if v > 0 else "red" for v in fold_avgs]
    ax.bar(range(len(fold_avgs)), fold_avgs, color=bar_colors, alpha=0.7)
    ax.set_xticks(range(len(fold_labels)))
    ax.set_xticklabels(fold_labels, rotation=45, fontsize=7)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Avg PnL per Trade (%)")
    ax.set_title(f"D) Walk-Forward: {best_name}")
    ax.grid(axis="y", alpha=0.3)
else:
    ax.text(0.5, 0.5, "No strategy qualified\nfor walk-forward",
            transform=ax.transAxes, ha="center", va="center", fontsize=12)
    ax.set_title("D) Walk-Forward Results")

plt.tight_layout()
fig_path = OUT_DIR / "agent4v2_sr_levels.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nFigure saved to {fig_path}")


# ═══════════════════════════════════════════════════════════════════
# FINAL REPORT
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  AGENT 4v2 FINAL REPORT")
print("=" * 70)
print(f"""
  Level Identification:
    - Swing lows (N=10,20): trailing window, no lookahead
    - Swing highs (N=10,20): trailing window, no lookahead
    - Volume-weighted HVN: trailing 120-day, $0.50 bins, top-3
    - Round numbers: $5, $10, $15, $20, $25, $30

  Bounce Rate Analysis:
    Support: real={sup_real:.1f}% vs null={sup_null_mean:.1f}% (pctile={sup_pctile:.1f}th)
    Resistance: real={res_real:.1f}% vs null={res_null_mean:.1f}% (pctile={res_pctile:.1f}th)

  Strategy Results (Selection Window, pre-2026):
""")

for row in summary_rows:
    print(f"    {row['strategy']:<22s} n={row['n']:>4d} avg={row['avg_pnl']:>+7.3f}% "
          f"sharpe={row['sharpe']:>6.2f} null={row['null_pctile']:>5.1f}th "
          f"-> {row['verdict']}")

if any_pass:
    passing = [r for r in summary_rows if r["verdict"] == "PASS"]
    print(f"\n  {len(passing)} strategy(ies) PASS null test at 95th percentile.")
    if best_wf_name:
        print(f"  Best walk-forward: {best_wf_name} (avg={best_wf_avg:+.3f}%)")
else:
    print(f"\n  CONCLUSION: Hypothesis tested and rejected on selection window.")
    print("  Standalone S/R level-bounce strategies do NOT produce statistically")
    print("  significant edge on RXRX over the 2021-2025 period.")

print("\n  AGENT 4v2 COMPLETE")
