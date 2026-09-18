"""
Feature Investigation F11-F15: Historical patterns from daily data
=================================================================
Tests five features derived from v2_daily_data.json:
  F11: Historical Gap-Fade Rate (most important)
  F12: Prior Trend Direction (10-day linear regression slope)
  F13: Volume Spike Significance (PM vol vs historical daily vol)
  F14: Distance From Recent High (position in 20-day range)
  F15: Idiosyncratic vs Sector Gap (stock gap minus SPY gap)

Data: v2_features_full.csv (316 trades), v2_daily_data.json (253 tickers x ~29 bars),
      v3_minute_bars.json (SPY bars for F15).
"""

# ── Config ───────────────────────────────────────────────────────────────────
BASE_DIR = "os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../') + '/sentiment/scanner_optimization"
RESULTS_DIR = f"{BASE_DIR}/results"
OUTPUT_DIR  = f"{RESULTS_DIR}/features"

CSV_PATH    = f"{RESULTS_DIR}/v2_features_full.csv"
DAILY_PATH  = f"{RESULTS_DIR}/v2_daily_data.json"
MINUTE_PATH = f"{RESULTS_DIR}/v3_minute_bars.json"

GAP_THRESHOLD = 0.03        # 3% for historical gap detection
TREND_LOOKBACK = 10          # days for F12
VOL_LOOKBACK = 20            # days for F13
RANGE_LOOKBACK = 20          # days for F14

VALIDATION_REQUIRED = True
LIVE_TRADING_WEEKS_NEEDED = 4

# ── Imports ──────────────────────────────────────────────────────────────────
import json
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

warnings.filterwarnings("ignore", category=RuntimeWarning)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(CSV_PATH)
with open(DAILY_PATH) as f:
    daily_data = json.load(f)
with open(MINUTE_PATH) as f:
    minute_data = json.load(f)

print(f"  CSV: {len(df)} rows, pnl_t2_close non-null: {df.pnl_t2_close.notna().sum()}, "
      f"pnl_60m_nosl non-null: {df.pnl_60m_nosl.notna().sum()}")
print(f"  Daily data: {len(daily_data)} tickers")
print(f"  Minute data: {len(minute_data.get('ticker_bars', {}))} ticker-date combos, "
      f"{len(minute_data.get('spy_bars', {}))} SPY dates")

# ── Strategy assignment ──────────────────────────────────────────────────────
def assign_strategy(row):
    gp = row["gap_pct"]
    agp = row["abs_gap_pct"]
    d = row["direction"]
    earn = row.get("is_earnings", False)

    # Momentum first (highest gap threshold)
    if gp > 5.0 and d == "LONG":
        return "D1_momentum_long"
    if gp < -5.0 and d == "SHORT":
        return "D2_momentum_short"

    # SHORT fade (gap-up shorted)
    if d == "SHORT" and gp > 0:
        if agp >= 3.0:
            if earn:
                return "V1_short_earnings"
            return "B2_short_large"
        if agp >= 1.5:
            return "B1_short_small"
        return "B3_short_tiny"

    # LONG fade (gap-down bought)
    if d == "LONG" and gp < 0:
        if agp >= 3.0:
            if earn:
                return "C3_long_earnings"
            return "C1_long_large"
        return "C2_long_small"

    return "other"


df["strategy"] = df.apply(assign_strategy, axis=1)

# PnL column per strategy
def get_pnl(row):
    s = row["strategy"]
    if s.startswith("D1") or s.startswith("D2"):
        return row.get("pnl_60m_nosl", np.nan)
    v = row.get("pnl_t2_close", np.nan)
    if pd.isna(v):
        return row.get("pnl_60m_nosl", np.nan)  # fallback
    return v


df["pnl"] = df.apply(get_pnl, axis=1)

print(f"\nStrategy distribution:")
for s, g in df.groupby("strategy"):
    pnl_valid = g["pnl"].dropna()
    print(f"  {s:25s}  n={len(g):3d}  pnl_valid={len(pnl_valid):3d}  "
          f"avg={pnl_valid.mean():.4f}" if len(pnl_valid) > 0 else f"  {s:25s}  n={len(g):3d}  pnl_valid=0")

# ── Utility functions ────────────────────────────────────────────────────────
def compute_metrics(pnl_array):
    pnl = np.array([x for x in pnl_array if x is not None and np.isfinite(x)])
    n = len(pnl)
    if n < 5:
        return {"n": n, "note": "insufficient"}
    avg = float(np.mean(pnl))
    med = float(np.median(pnl))
    std = float(np.std(pnl, ddof=1))
    se = std / np.sqrt(n)
    trimmed = float(sp_stats.trim_mean(pnl, 0.1)) if n >= 10 else avg
    return {
        "n": n,
        "avg_pnl": round(avg, 4),
        "median_pnl": round(med, 4),
        "trimmed_mean": round(trimmed, 4),
        "std": round(std, 4),
        "sharpe": round(avg / std if std > 0 else 0, 4),
        "win_rate": round(float((pnl > 0).mean()), 4),
        "ci_95_low": round(avg - 1.96 * se, 4),
        "ci_95_high": round(avg + 1.96 * se, 4),
        "max_loss": round(float(np.min(pnl)), 4),
        "max_gain": round(float(np.max(pnl)), 4),
    }


def correlation_test(feature_vals, pnl_vals):
    """Pearson and Spearman correlation with p-values."""
    mask = np.isfinite(feature_vals) & np.isfinite(pnl_vals)
    f = feature_vals[mask]
    p = pnl_vals[mask]
    if len(f) < 10:
        return {"n": len(f), "note": "insufficient"}
    pr, pp = sp_stats.pearsonr(f, p)
    sr, sp_ = sp_stats.spearmanr(f, p)
    return {
        "n": int(len(f)),
        "pearson_r": round(float(pr), 4),
        "pearson_p": round(float(pp), 4),
        "spearman_r": round(float(sr), 4),
        "spearman_p": round(float(sp_), 4),
    }


def quartile_analysis(feature_vals, pnl_vals, feature_name="feature"):
    """Split into quartiles, report metrics per quartile."""
    mask = np.isfinite(feature_vals) & np.isfinite(pnl_vals)
    f = feature_vals[mask]
    p = pnl_vals[mask]
    if len(f) < 20:
        return {"note": f"insufficient data (n={len(f)})"}
    try:
        qs = pd.qcut(f, 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"], duplicates="drop")
    except ValueError:
        return {"note": "cannot form quartiles (too few unique values)"}
    result = {}
    for q in qs.unique():
        if pd.isna(q):
            continue
        qp = p[qs == q]
        result[str(q)] = compute_metrics(qp)
    return result


def threshold_sweep(feature_vals, pnl_vals, thresholds, direction="above"):
    """Sweep thresholds; split PASS (above/below threshold) vs FAIL."""
    mask = np.isfinite(feature_vals) & np.isfinite(pnl_vals)
    f = feature_vals[mask]
    p = pnl_vals[mask]
    results = {}
    for t in thresholds:
        if direction == "above":
            pass_mask = f >= t
        else:
            pass_mask = f <= t
        pass_pnl = p[pass_mask]
        fail_pnl = p[~pass_mask]
        results[str(t)] = {
            "pass": compute_metrics(pass_pnl),
            "fail": compute_metrics(fail_pnl),
            "pass_rate": round(float(pass_mask.mean()), 4) if len(f) > 0 else 0,
        }
    return results


def cross_strategy_check(df_with_feature, feature_col, pnl_col="pnl"):
    """Check sign consistency of correlation across strategies."""
    results = {}
    pos_count = 0
    neg_count = 0
    total = 0
    for strat, g in df_with_feature.groupby("strategy"):
        if strat == "other":
            continue
        fv = g[feature_col].values.astype(float)
        pv = g[pnl_col].values.astype(float)
        mask = np.isfinite(fv) & np.isfinite(pv)
        if mask.sum() < 10:
            continue
        sr, sp_ = sp_stats.spearmanr(fv[mask], pv[mask])
        results[strat] = {"spearman_r": round(float(sr), 4), "p": round(float(sp_), 4), "n": int(mask.sum())}
        total += 1
        if sr > 0:
            pos_count += 1
        else:
            neg_count += 1
    consistent = max(pos_count, neg_count)
    return {
        "per_strategy": results,
        "consistent_direction": consistent,
        "total_strategies": total,
        "is_consistent": consistent >= 6 if total >= 10 else consistent >= total * 0.6,
    }


def leave_one_out_stability(vals, metric_fn):
    """For small samples, leave-one-out stability check."""
    if len(vals) < 10:
        return {"note": "too few for LOO"}
    base = metric_fn(vals)
    loo_results = []
    for i in range(len(vals)):
        subset = np.concatenate([vals[:i], vals[i+1:]])
        loo_results.append(metric_fn(subset))
    return {
        "base": round(base, 4),
        "loo_min": round(min(loo_results), 4),
        "loo_max": round(max(loo_results), 4),
        "loo_std": round(float(np.std(loo_results)), 4),
        "loo_range_pct": round((max(loo_results) - min(loo_results)) / abs(base) * 100, 2) if base != 0 else None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# F11: HISTORICAL GAP-FADE RATE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("F11: Historical Gap-Fade Rate")
print("="*80)


def compute_historical_fade_rate(ticker, eval_date, gap_threshold=GAP_THRESHOLD):
    if ticker not in daily_data:
        return None
    bars = daily_data[ticker]
    sorted_dates = sorted(d for d in bars.keys() if d < eval_date)

    if len(sorted_dates) < 2:
        return {"gap_event_count": 0, "fade_rate_eod": None, "avg_fade_magnitude": None}

    gaps = []
    for i in range(1, len(sorted_dates)):
        prev_c = bars[sorted_dates[i - 1]]["close"]
        today = bars[sorted_dates[i]]
        if prev_c == 0:
            continue
        gap = (today["open"] - prev_c) / prev_c

        if abs(gap) > gap_threshold:
            if gap > 0:
                eod_fade = today["close"] < today["open"]
                fade_mag = (today["open"] - today["close"]) / today["open"] if today["open"] > 0 else 0
            else:
                eod_fade = today["close"] > today["open"]
                fade_mag = (today["close"] - today["open"]) / today["open"] if today["open"] > 0 else 0
            gaps.append({"gap": gap, "eod_fade": eod_fade, "fade_magnitude": fade_mag})

    if not gaps:
        return {"gap_event_count": 0, "fade_rate_eod": None, "avg_fade_magnitude": None}

    return {
        "gap_event_count": len(gaps),
        "fade_rate_eod": sum(g["eod_fade"] for g in gaps) / len(gaps),
        "avg_fade_magnitude": float(np.mean([g["fade_magnitude"] for g in gaps])),
    }


# Compute F11 for all rows
f11_results = []
for _, row in df.iterrows():
    r = compute_historical_fade_rate(row["ticker"], row["eval_date"])
    f11_results.append(r)

df["f11_fade_rate"] = [r["fade_rate_eod"] if r and r["fade_rate_eod"] is not None else np.nan for r in f11_results]
df["f11_gap_count"] = [r["gap_event_count"] if r else 0 for r in f11_results]
df["f11_avg_fade_mag"] = [r["avg_fade_magnitude"] if r and r["avg_fade_magnitude"] is not None else np.nan for r in f11_results]

print(f"  Fade rate computed: {df.f11_fade_rate.notna().sum()} / {len(df)} rows")
print(f"  Gap count range: {df.f11_gap_count.min()} - {df.f11_gap_count.max()}")
print(f"  Fade rate range: {df.f11_fade_rate.min():.3f} - {df.f11_fade_rate.max():.3f}")

# Correlation
f11_corr = correlation_test(df["f11_fade_rate"].values, df["pnl"].values)
print(f"  Correlation with PnL: Pearson r={f11_corr.get('pearson_r','N/A')}, "
      f"p={f11_corr.get('pearson_p','N/A')}")

# Quartile analysis
f11_quartiles = quartile_analysis(df["f11_fade_rate"].values, df["pnl"].values, "fade_rate_eod")
print(f"  Quartile analysis:")
for q, m in f11_quartiles.items():
    if isinstance(m, dict) and "avg_pnl" in m:
        print(f"    {q}: n={m['n']}, avg={m['avg_pnl']:.4f}, median={m['median_pnl']:.4f}, wr={m['win_rate']:.2%}")

# Threshold sweep
f11_thresholds = threshold_sweep(
    df["f11_fade_rate"].values, df["pnl"].values,
    [0.3, 0.4, 0.5, 0.6, 0.7, 0.8], direction="above"
)
print(f"\n  Threshold sweep (fade_rate >= X):")
for t, r in f11_thresholds.items():
    p = r["pass"]
    f = r["fail"]
    if isinstance(p, dict) and "avg_pnl" in p and isinstance(f, dict) and "avg_pnl" in f:
        print(f"    >={t}: PASS n={p['n']} avg={p['avg_pnl']:.4f} | "
              f"FAIL n={f['n']} avg={f['avg_pnl']:.4f} | rate={r['pass_rate']:.2%}")

# Min gap_event_count filter
print(f"\n  Minimum gap event count filter:")
for min_count in [1, 2, 3, 5]:
    mask = df["f11_gap_count"] >= min_count
    sub = df[mask]
    f_corr = correlation_test(sub["f11_fade_rate"].values, sub["pnl"].values)
    n_valid = sub["pnl"].dropna().shape[0]
    print(f"    min_count={min_count}: n={mask.sum()} (pnl_valid={n_valid}), "
          f"Pearson r={f_corr.get('pearson_r','N/A')}, p={f_corr.get('pearson_p','N/A')}")

# Cross-strategy
f11_cross = cross_strategy_check(df, "f11_fade_rate")
print(f"  Cross-strategy consistency: {f11_cross['consistent_direction']}/{f11_cross['total_strategies']}")

# LOO stability on overall avg PnL for high fade-rate group
high_fade = df[(df["f11_fade_rate"] >= 0.5) & df["pnl"].notna()]["pnl"].values
f11_loo = leave_one_out_stability(high_fade, np.mean) if len(high_fade) >= 10 else {"note": "insufficient"}
print(f"  LOO stability (fade_rate>=0.5): {f11_loo}")

# By direction
print(f"\n  By direction (SHORT trades, fade for gap-ups):")
short_mask = df["direction"] == "SHORT"
f11_short_corr = correlation_test(df.loc[short_mask, "f11_fade_rate"].values,
                                   df.loc[short_mask, "pnl"].values)
print(f"    SHORT: {f11_short_corr}")
long_mask = df["direction"] == "LONG"
f11_long_corr = correlation_test(df.loc[long_mask, "f11_fade_rate"].values,
                                  df.loc[long_mask, "pnl"].values)
print(f"    LONG: {f11_long_corr}")

# Save F11
f11_output = {
    "feature": "F11_historical_fade_rate",
    "description": "Fraction of prior 3%+ gaps that faded by EOD",
    "data_source": "v2_daily_data.json",
    "gap_threshold": GAP_THRESHOLD,
    "coverage": int(df.f11_fade_rate.notna().sum()),
    "total_rows": len(df),
    "correlation": f11_corr,
    "quartile_analysis": f11_quartiles,
    "threshold_sweep": f11_thresholds,
    "min_gap_count_sweep": {},
    "cross_strategy": f11_cross,
    "loo_stability_high_fade": f11_loo,
    "by_direction": {"SHORT": f11_short_corr, "LONG": f11_long_corr},
    "validation_required": VALIDATION_REQUIRED,
    "live_trading_weeks_needed": LIVE_TRADING_WEEKS_NEEDED,
}
for mc in [1, 2, 3, 5]:
    mask = df["f11_gap_count"] >= mc
    sub = df[mask]
    f11_output["min_gap_count_sweep"][str(mc)] = correlation_test(
        sub["f11_fade_rate"].values, sub["pnl"].values
    )

with open(f"{OUTPUT_DIR}/F11_historical_fade_rate_results.json", "w") as f:
    json.dump(f11_output, f, indent=2, default=str)
print("  Saved F11 results.")


# ══════════════════════════════════════════════════════════════════════════════
# F12: PRIOR TREND DIRECTION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("F12: Prior Trend Direction")
print("="*80)


def compute_prior_trend(ticker, eval_date, lookback=TREND_LOOKBACK):
    if ticker not in daily_data:
        return None
    bars = daily_data[ticker]
    prior_dates = sorted(d for d in bars.keys() if d < eval_date)[-lookback:]
    if len(prior_dates) < lookback:
        return None
    closes = [bars[d]["close"] for d in prior_dates]
    if closes[0] == 0:
        return None
    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(range(len(closes)), closes)
    return {
        "trend_slope": slope / closes[0],
        "trend_r2": r_value ** 2,
        "trend_direction": "up" if slope > 0 else "down",
    }


f12_results = []
for _, row in df.iterrows():
    r = compute_prior_trend(row["ticker"], row["eval_date"])
    f12_results.append(r)

df["f12_trend_slope"] = [r["trend_slope"] if r else np.nan for r in f12_results]
df["f12_trend_r2"] = [r["trend_r2"] if r else np.nan for r in f12_results]

print(f"  Trend computed: {df.f12_trend_slope.notna().sum()} / {len(df)} rows")
print(f"  Slope range: {df.f12_trend_slope.min():.6f} to {df.f12_trend_slope.max():.6f}")

# Overall correlation
f12_corr = correlation_test(df["f12_trend_slope"].values, df["pnl"].values)
print(f"  Correlation with PnL: {f12_corr}")

# Quartile analysis
f12_quartiles = quartile_analysis(df["f12_trend_slope"].values, df["pnl"].values, "trend_slope")
print(f"  Quartile analysis:")
for q, m in f12_quartiles.items():
    if isinstance(m, dict) and "avg_pnl" in m:
        print(f"    {q}: n={m['n']}, avg={m['avg_pnl']:.4f}, median={m['median_pnl']:.4f}")

# By direction: for SHORT, negative slope should predict better fade
print(f"\n  By direction:")
for direction in ["SHORT", "LONG"]:
    mask = df["direction"] == direction
    corr = correlation_test(df.loc[mask, "f12_trend_slope"].values, df.loc[mask, "pnl"].values)
    print(f"    {direction}: {corr}")

    # Quartile within direction
    q = quartile_analysis(df.loc[mask, "f12_trend_slope"].values, df.loc[mask, "pnl"].values)
    for qk, qv in q.items():
        if isinstance(qv, dict) and "avg_pnl" in qv:
            print(f"      {qk}: n={qv['n']}, avg={qv['avg_pnl']:.4f}")

# Interaction: trend direction vs gap direction
# For SHORT: gap > 0 (gap-up). Does prior downtrend (slope < 0) make fade more likely?
short_df = df[(df["direction"] == "SHORT") & df["f12_trend_slope"].notna() & df["pnl"].notna()]
if len(short_df) >= 10:
    down_trend = short_df[short_df["f12_trend_slope"] < 0]
    up_trend = short_df[short_df["f12_trend_slope"] >= 0]
    print(f"\n  SHORT + prior downtrend: n={len(down_trend)}, avg_pnl={down_trend.pnl.mean():.4f}")
    print(f"  SHORT + prior uptrend:   n={len(up_trend)}, avg_pnl={up_trend.pnl.mean():.4f}")

# Cross-strategy
f12_cross = cross_strategy_check(df, "f12_trend_slope")
print(f"  Cross-strategy consistency: {f12_cross['consistent_direction']}/{f12_cross['total_strategies']}")

# R2 as quality filter
f12_r2_corr = correlation_test(df["f12_trend_r2"].values, df["pnl"].values)
print(f"  R2 correlation with PnL: {f12_r2_corr}")

# Save
f12_output = {
    "feature": "F12_prior_trend_direction",
    "description": "10-day linear regression slope of closing prices, normalized by first close",
    "lookback": TREND_LOOKBACK,
    "coverage": int(df.f12_trend_slope.notna().sum()),
    "total_rows": len(df),
    "correlation_overall": f12_corr,
    "correlation_r2": f12_r2_corr,
    "quartile_analysis": f12_quartiles,
    "by_direction": {
        "SHORT": correlation_test(df.loc[df.direction == "SHORT", "f12_trend_slope"].values,
                                   df.loc[df.direction == "SHORT", "pnl"].values),
        "LONG": correlation_test(df.loc[df.direction == "LONG", "f12_trend_slope"].values,
                                  df.loc[df.direction == "LONG", "pnl"].values),
    },
    "cross_strategy": f12_cross,
    "validation_required": VALIDATION_REQUIRED,
    "live_trading_weeks_needed": LIVE_TRADING_WEEKS_NEEDED,
}
with open(f"{OUTPUT_DIR}/F12_prior_trend_results.json", "w") as f:
    json.dump(f12_output, f, indent=2, default=str)
print("  Saved F12 results.")


# ══════════════════════════════════════════════════════════════════════════════
# F13: VOLUME SPIKE SIGNIFICANCE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("F13: Volume Spike Significance")
print("="*80)


def compute_volume_spike(pm_vol, ticker, eval_date, lookback=VOL_LOOKBACK):
    if ticker not in daily_data or pd.isna(pm_vol) or pm_vol <= 0:
        return None
    bars = daily_data[ticker]
    prior_dates = sorted(d for d in bars.keys() if d < eval_date)[-lookback:]
    if not prior_dates:
        return None
    volumes = [bars[d]["volume"] for d in prior_dates if bars[d]["volume"] > 0]
    if not volumes:
        return None
    max_vol = max(volumes)
    avg_vol = float(np.mean(volumes))
    return {
        "vs_max_prior": pm_vol / max_vol if max_vol > 0 else None,
        "vs_avg_prior": pm_vol / avg_vol if avg_vol > 0 else None,
        "is_record_volume": pm_vol > max_vol,
        "n_prior_days": len(volumes),
    }


f13_results = []
for _, row in df.iterrows():
    r = compute_volume_spike(row["pm_vol"], row["ticker"], row["eval_date"])
    f13_results.append(r)

df["f13_vs_max"] = [r["vs_max_prior"] if r and r["vs_max_prior"] is not None else np.nan for r in f13_results]
df["f13_vs_avg"] = [r["vs_avg_prior"] if r and r["vs_avg_prior"] is not None else np.nan for r in f13_results]
df["f13_is_record"] = [r["is_record_volume"] if r else False for r in f13_results]

print(f"  Volume spike computed: {df.f13_vs_avg.notna().sum()} / {len(df)} rows")
print(f"  vs_avg range: {df.f13_vs_avg.min():.4f} to {df.f13_vs_avg.max():.4f}")
print(f"  Record volume trades: {df.f13_is_record.sum()}")

# Correlation
f13_corr_max = correlation_test(df["f13_vs_max"].values, df["pnl"].values)
f13_corr_avg = correlation_test(df["f13_vs_avg"].values, df["pnl"].values)
print(f"  vs_max correlation: {f13_corr_max}")
print(f"  vs_avg correlation: {f13_corr_avg}")

# Log transform for heavy-tailed distributions
df["f13_log_vs_avg"] = np.log1p(df["f13_vs_avg"])
f13_corr_log = correlation_test(df["f13_log_vs_avg"].values, df["pnl"].values)
print(f"  log(vs_avg) correlation: {f13_corr_log}")

# Quartile
f13_quartiles = quartile_analysis(df["f13_vs_avg"].values, df["pnl"].values, "vs_avg")
print(f"  Quartile analysis (vs_avg):")
for q, m in f13_quartiles.items():
    if isinstance(m, dict) and "avg_pnl" in m:
        print(f"    {q}: n={m['n']}, avg={m['avg_pnl']:.4f}, median={m['median_pnl']:.4f}")

# Record volume vs not
rec_pnl = df[df["f13_is_record"] & df["pnl"].notna()]["pnl"]
norec_pnl = df[~df["f13_is_record"] & df["pnl"].notna()]["pnl"]
print(f"\n  Record volume: n={len(rec_pnl)}, avg={rec_pnl.mean():.4f}" if len(rec_pnl) > 0 else "  Record volume: n=0")
print(f"  Non-record:    n={len(norec_pnl)}, avg={norec_pnl.mean():.4f}" if len(norec_pnl) > 0 else "  Non-record: n=0")

# Cross-strategy
f13_cross = cross_strategy_check(df, "f13_vs_avg")

# Save
f13_output = {
    "feature": "F13_volume_spike_significance",
    "description": "PM volume relative to prior 20-day daily volume (max and avg)",
    "lookback": VOL_LOOKBACK,
    "coverage": int(df.f13_vs_avg.notna().sum()),
    "total_rows": len(df),
    "correlation_vs_max": f13_corr_max,
    "correlation_vs_avg": f13_corr_avg,
    "correlation_log_vs_avg": f13_corr_log,
    "quartile_analysis_vs_avg": f13_quartiles,
    "record_volume": {
        "record": compute_metrics(rec_pnl.values),
        "non_record": compute_metrics(norec_pnl.values),
    },
    "cross_strategy": f13_cross,
    "validation_required": VALIDATION_REQUIRED,
    "live_trading_weeks_needed": LIVE_TRADING_WEEKS_NEEDED,
}
with open(f"{OUTPUT_DIR}/F13_volume_spike_results.json", "w") as f:
    json.dump(f13_output, f, indent=2, default=str)
print("  Saved F13 results.")


# ══════════════════════════════════════════════════════════════════════════════
# F14: DISTANCE FROM RECENT HIGH
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("F14: Distance From Recent High")
print("="*80)


def compute_distance_from_high(ticker, eval_date, lookback=RANGE_LOOKBACK):
    if ticker not in daily_data:
        return None
    bars = daily_data[ticker]
    prior_dates = sorted(d for d in bars.keys() if d < eval_date)[-lookback:]
    if not prior_dates:
        return None
    high_20d = max(bars[d]["high"] for d in prior_dates)
    low_20d = min(bars[d]["low"] for d in prior_dates)
    # Use open from eval_date as proxy for entry level before gap
    if eval_date in bars:
        open_price = bars[eval_date]["open"]
    else:
        return None
    rng = high_20d - low_20d
    position = (open_price - low_20d) / rng if rng > 0 else 0.5
    dist_above = (open_price - high_20d) / high_20d if high_20d > 0 else 0
    return {
        "position_in_20d_range": position,
        "distance_above_20d_high": dist_above,
        "is_new_20d_high": open_price > high_20d,
    }


f14_results = []
for _, row in df.iterrows():
    r = compute_distance_from_high(row["ticker"], row["eval_date"])
    f14_results.append(r)

df["f14_position"] = [r["position_in_20d_range"] if r else np.nan for r in f14_results]
df["f14_dist_above"] = [r["distance_above_20d_high"] if r else np.nan for r in f14_results]
df["f14_new_high"] = [r["is_new_20d_high"] if r else False for r in f14_results]

print(f"  F14 computed: {df.f14_position.notna().sum()} / {len(df)} rows")
print(f"  Position range: {df.f14_position.min():.4f} to {df.f14_position.max():.4f}")
print(f"  New 20d high trades: {df.f14_new_high.sum()}")

# Overall correlation
f14_corr_pos = correlation_test(df["f14_position"].values, df["pnl"].values)
f14_corr_dist = correlation_test(df["f14_dist_above"].values, df["pnl"].values)
print(f"  Position correlation: {f14_corr_pos}")
print(f"  Dist above high correlation: {f14_corr_dist}")

# Quartile analysis
f14_quartiles = quartile_analysis(df["f14_position"].values, df["pnl"].values, "position")
print(f"  Quartile analysis (position_in_range):")
for q, m in f14_quartiles.items():
    if isinstance(m, dict) and "avg_pnl" in m:
        print(f"    {q}: n={m['n']}, avg={m['avg_pnl']:.4f}, median={m['median_pnl']:.4f}")

# For SHORT: does new_20d_high predict stronger fade?
short_new_high = df[(df.direction == "SHORT") & df.f14_new_high & df.pnl.notna()]["pnl"]
short_no_high = df[(df.direction == "SHORT") & ~df.f14_new_high & df.pnl.notna()]["pnl"]
print(f"\n  SHORT + new 20d high: n={len(short_new_high)}, "
      f"avg={short_new_high.mean():.4f}" if len(short_new_high) > 0 else "  SHORT + new high: n=0")
print(f"  SHORT + no new high:  n={len(short_no_high)}, "
      f"avg={short_no_high.mean():.4f}" if len(short_no_high) > 0 else "  SHORT + no new high: n=0")

# By direction
print(f"\n  By direction:")
for direction in ["SHORT", "LONG"]:
    mask = df["direction"] == direction
    corr = correlation_test(df.loc[mask, "f14_position"].values, df.loc[mask, "pnl"].values)
    print(f"    {direction}: position corr = {corr}")

# Cross-strategy
f14_cross = cross_strategy_check(df, "f14_position")

# Save
f14_output = {
    "feature": "F14_distance_from_high",
    "description": "Open price position within prior 20-day high-low range",
    "lookback": RANGE_LOOKBACK,
    "coverage": int(df.f14_position.notna().sum()),
    "total_rows": len(df),
    "correlation_position": f14_corr_pos,
    "correlation_dist_above": f14_corr_dist,
    "quartile_analysis": f14_quartiles,
    "short_new_20d_high": {
        "new_high": compute_metrics(short_new_high.values) if len(short_new_high) > 0 else {"n": 0},
        "no_new_high": compute_metrics(short_no_high.values) if len(short_no_high) > 0 else {"n": 0},
    },
    "by_direction": {
        "SHORT": correlation_test(df.loc[df.direction == "SHORT", "f14_position"].values,
                                   df.loc[df.direction == "SHORT", "pnl"].values),
        "LONG": correlation_test(df.loc[df.direction == "LONG", "f14_position"].values,
                                  df.loc[df.direction == "LONG", "pnl"].values),
    },
    "cross_strategy": f14_cross,
    "validation_required": VALIDATION_REQUIRED,
    "live_trading_weeks_needed": LIVE_TRADING_WEEKS_NEEDED,
}
with open(f"{OUTPUT_DIR}/F14_distance_from_high_results.json", "w") as f:
    json.dump(f14_output, f, indent=2, default=str)
print("  Saved F14 results.")


# ══════════════════════════════════════════════════════════════════════════════
# F15: IDIOSYNCRATIC VS SECTOR GAP
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("F15: Idiosyncratic vs Sector Gap")
print("="*80)

# Build SPY daily opens/closes from minute bars
spy_bars = minute_data.get("spy_bars", {})
spy_daily = {}  # date -> {open_930, close_prev}

# For each SPY date, find 9:30 open and ~15:59 close (or last available bar)
for date_str, bars in spy_bars.items():
    # Open at 9:30
    open_930 = bars.get("09:30", {}).get("o", None)
    # Close: find last bar on or before 15:59
    close_bar = None
    for t in ["15:59", "15:58", "15:57", "15:55", "15:50", "15:45"]:
        if t in bars:
            close_bar = bars[t]["c"]
            break
    if close_bar is None:
        # Use the latest available bar
        sorted_times = sorted(bars.keys())
        if sorted_times:
            close_bar = bars[sorted_times[-1]]["c"]
    spy_daily[date_str] = {"open": open_930, "close": close_bar}

# Compute SPY gap for each eval_date
spy_dates_sorted = sorted(spy_daily.keys())
spy_gaps = {}  # date -> gap_pct
for i in range(1, len(spy_dates_sorted)):
    prev_date = spy_dates_sorted[i - 1]
    curr_date = spy_dates_sorted[i]
    prev_close = spy_daily[prev_date]["close"]
    curr_open = spy_daily[curr_date]["open"]
    if prev_close and curr_open and prev_close > 0:
        spy_gaps[curr_date] = (curr_open - prev_close) / prev_close * 100  # in percent
    else:
        spy_gaps[curr_date] = 0.0

print(f"  SPY gaps computed for {len(spy_gaps)} dates:")
for d, g in sorted(spy_gaps.items()):
    print(f"    {d}: {g:.4f}%")

# Compute idiosyncratic gap
df["f15_spy_gap_pct"] = df["eval_date"].map(spy_gaps)
df["f15_idio_gap"] = df["gap_pct"] - df["f15_spy_gap_pct"]

valid_f15 = df["f15_idio_gap"].notna().sum()
print(f"  Idiosyncratic gap computed: {valid_f15} / {len(df)} rows")

# Correlation
f15_corr = correlation_test(df["f15_idio_gap"].values, df["pnl"].values)
print(f"  Correlation with PnL: {f15_corr}")

# For SHORT specifically
short_mask = df["direction"] == "SHORT"
f15_short_corr = correlation_test(df.loc[short_mask, "f15_idio_gap"].values,
                                   df.loc[short_mask, "pnl"].values)
print(f"  SHORT correlation: {f15_short_corr}")

# abs(idio_gap) vs PnL
df["f15_abs_idio_gap"] = df["f15_idio_gap"].abs()
f15_abs_corr = correlation_test(df["f15_abs_idio_gap"].values, df["pnl"].values)
print(f"  abs(idio_gap) correlation: {f15_abs_corr}")

# Quartile analysis
f15_quartiles = quartile_analysis(df["f15_idio_gap"].values, df["pnl"].values, "idio_gap")
print(f"  Quartile analysis (idiosyncratic gap):")
for q, m in f15_quartiles.items():
    if isinstance(m, dict) and "avg_pnl" in m:
        print(f"    {q}: n={m['n']}, avg={m['avg_pnl']:.4f}, median={m['median_pnl']:.4f}")

# Cross-strategy
f15_cross = cross_strategy_check(df, "f15_idio_gap")

# For SHORT: higher idiosyncratic gap-up should predict stronger fade
# Threshold: idio_gap > 5%, > 10%, > 20%
short_df = df[short_mask & df["pnl"].notna() & df["f15_idio_gap"].notna()]
f15_short_thresh = threshold_sweep(
    short_df["f15_idio_gap"].values, short_df["pnl"].values,
    [3.0, 5.0, 10.0, 15.0, 20.0], direction="above"
)
print(f"\n  SHORT idio_gap threshold sweep:")
for t, r in f15_short_thresh.items():
    p = r["pass"]
    f_ = r["fail"]
    if isinstance(p, dict) and "avg_pnl" in p and isinstance(f_, dict) and "avg_pnl" in f_:
        print(f"    >={t}: PASS n={p['n']} avg={p['avg_pnl']:.4f} | "
              f"FAIL n={f_['n']} avg={f_['avg_pnl']:.4f}")

# Save
f15_output = {
    "feature": "F15_idiosyncratic_gap",
    "description": "Stock gap minus SPY gap (market-adjusted gap magnitude)",
    "spy_source": "v3_minute_bars.json spy_bars",
    "spy_gaps": {k: round(v, 4) for k, v in spy_gaps.items()},
    "coverage": int(valid_f15),
    "total_rows": len(df),
    "correlation_overall": f15_corr,
    "correlation_short": f15_short_corr,
    "correlation_abs": f15_abs_corr,
    "quartile_analysis": f15_quartiles,
    "short_threshold_sweep": f15_short_thresh,
    "cross_strategy": f15_cross,
    "validation_required": VALIDATION_REQUIRED,
    "live_trading_weeks_needed": LIVE_TRADING_WEEKS_NEEDED,
}
with open(f"{OUTPUT_DIR}/F15_idiosyncratic_gap_results.json", "w") as f:
    json.dump(f15_output, f, indent=2, default=str)
print("  Saved F15 results.")


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("F11-F15 COMBINED SUMMARY")
print("="*80)

features_summary = []
feature_configs = [
    ("F11", "historical_fade_rate", "f11_fade_rate", f11_corr),
    ("F12", "prior_trend_slope", "f12_trend_slope", f12_corr),
    ("F13", "volume_spike_vs_avg", "f13_vs_avg", f13_corr_avg),
    ("F14", "position_in_20d_range", "f14_position", f14_corr_pos),
    ("F15", "idiosyncratic_gap", "f15_idio_gap", f15_corr),
]

for fid, fname, fcol, fcorr in feature_configs:
    entry = {
        "feature_id": fid,
        "feature_name": fname,
        "coverage": int(df[fcol].notna().sum()),
        "pearson_r": fcorr.get("pearson_r"),
        "pearson_p": fcorr.get("pearson_p"),
        "spearman_r": fcorr.get("spearman_r"),
        "spearman_p": fcorr.get("spearman_p"),
        "abs_spearman": abs(fcorr["spearman_r"]) if fcorr.get("spearman_r") is not None else 0,
    }
    features_summary.append(entry)
    print(f"  {fid} {fname:30s}  cov={entry['coverage']:3d}  "
          f"rho={entry['spearman_r']:.4f} (p={entry['spearman_p']:.4f})" if entry["spearman_r"] is not None
          else f"  {fid} {fname:30s}  insufficient data")

# Rank by absolute Spearman
features_summary.sort(key=lambda x: x["abs_spearman"], reverse=True)
print(f"\n  Ranked by |Spearman rho|:")
for i, fs in enumerate(features_summary):
    sig = "***" if (fs["spearman_p"] or 1) < 0.01 else "**" if (fs["spearman_p"] or 1) < 0.05 else "*" if (fs["spearman_p"] or 1) < 0.1 else ""
    print(f"    {i+1}. {fs['feature_id']} {fs['feature_name']:30s}  "
          f"|rho|={fs['abs_spearman']:.4f}  p={fs['spearman_p']:.4f} {sig}")

# Direction-specific rankings
print(f"\n  SHORT-specific rankings:")
short_rankings = []
for fid, fname, fcol, _ in feature_configs:
    mask = df["direction"] == "SHORT"
    corr = correlation_test(df.loc[mask, fcol].values, df.loc[mask, "pnl"].values)
    short_rankings.append({
        "feature_id": fid, "feature_name": fname,
        "spearman_r": corr.get("spearman_r"), "spearman_p": corr.get("spearman_p"),
        "abs_spearman": abs(corr["spearman_r"]) if corr.get("spearman_r") is not None else 0,
    })
short_rankings.sort(key=lambda x: x["abs_spearman"], reverse=True)
for i, sr in enumerate(short_rankings):
    sig = "***" if (sr["spearman_p"] or 1) < 0.01 else "**" if (sr["spearman_p"] or 1) < 0.05 else "*" if (sr["spearman_p"] or 1) < 0.1 else ""
    print(f"    {i+1}. {sr['feature_id']} {sr['feature_name']:30s}  "
          f"rho={sr['spearman_r']:.4f}  p={sr['spearman_p']:.4f} {sig}")

print(f"\n  LONG-specific rankings:")
long_rankings = []
for fid, fname, fcol, _ in feature_configs:
    mask = df["direction"] == "LONG"
    corr = correlation_test(df.loc[mask, fcol].values, df.loc[mask, "pnl"].values)
    long_rankings.append({
        "feature_id": fid, "feature_name": fname,
        "spearman_r": corr.get("spearman_r"), "spearman_p": corr.get("spearman_p"),
        "abs_spearman": abs(corr["spearman_r"]) if corr.get("spearman_r") is not None else 0,
    })
long_rankings.sort(key=lambda x: x["abs_spearman"], reverse=True)
for i, lr in enumerate(long_rankings):
    sig = "***" if (lr["spearman_p"] or 1) < 0.01 else "**" if (lr["spearman_p"] or 1) < 0.05 else "*" if (lr["spearman_p"] or 1) < 0.1 else ""
    print(f"    {i+1}. {lr['feature_id']} {lr['feature_name']:30s}  "
          f"rho={lr['spearman_r']:.4f}  p={lr['spearman_p']:.4f} {sig}")

combined_summary = {
    "features_tested": [fid for fid, _, _, _ in feature_configs],
    "overall_rankings": features_summary,
    "short_rankings": short_rankings,
    "long_rankings": long_rankings,
    "sample_sizes": {
        "total_trades": len(df),
        "pnl_valid": int(df.pnl.notna().sum()),
        "pnl_t2_close_valid": int(df.pnl_t2_close.notna().sum()),
        "dates": sorted(df.eval_date.unique().tolist()),
    },
    "validation_required": VALIDATION_REQUIRED,
    "live_trading_weeks_needed": LIVE_TRADING_WEEKS_NEEDED,
    "warning": "Only 4 trading days of data. All results require out-of-sample validation.",
}
with open(f"{OUTPUT_DIR}/F11_F15_summary.json", "w") as f:
    json.dump(combined_summary, f, indent=2, default=str)
print("\n  Saved combined summary.")


# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("Generating charts...")
print("="*80)

fig, axes = plt.subplots(3, 2, figsize=(16, 18))
fig.suptitle("F11-F15 Feature Investigation\n(316 trades, 4 days, Sep 8-11 2026)",
             fontsize=14, fontweight="bold", y=0.98)

# ── F11: Fade Rate vs PnL scatter + quartile bars ──
ax = axes[0, 0]
mask = df.f11_fade_rate.notna() & df.pnl.notna()
colors = ["#2196F3" if d == "LONG" else "#F44336" for d in df.loc[mask, "direction"]]
ax.scatter(df.loc[mask, "f11_fade_rate"], df.loc[mask, "pnl"], c=colors, alpha=0.5, s=20, edgecolors="none")
ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
if f11_corr.get("spearman_r") is not None:
    ax.set_title(f"F11: Fade Rate vs PnL\nrho={f11_corr['spearman_r']:.3f}, p={f11_corr['spearman_p']:.3f}", fontsize=11)
else:
    ax.set_title("F11: Fade Rate vs PnL", fontsize=11)
ax.set_xlabel("Historical Fade Rate (EOD)")
ax.set_ylabel("PnL (%)")
# Add quartile bar overlay
if isinstance(f11_quartiles, dict) and "Q1_low" in f11_quartiles:
    q_names = ["Q1_low", "Q2", "Q3", "Q4_high"]
    q_avgs = [f11_quartiles[q]["avg_pnl"] for q in q_names if q in f11_quartiles and "avg_pnl" in f11_quartiles[q]]
    q_ns = [f11_quartiles[q]["n"] for q in q_names if q in f11_quartiles and "n" in f11_quartiles[q]]
    if q_avgs:
        ax2 = ax.twinx()
        bar_colors = ["#4CAF50" if v > 0 else "#FF5722" for v in q_avgs]
        ax2.bar(range(len(q_avgs)), q_avgs, alpha=0.3, color=bar_colors, width=0.6)
        ax2.set_ylabel("Quartile Avg PnL (%)", fontsize=8)
        ax2.tick_params(labelsize=8)

# ── F12: Trend Slope vs PnL by direction ──
ax = axes[0, 1]
for direction, color, marker in [("SHORT", "#F44336", "v"), ("LONG", "#2196F3", "^")]:
    mask = (df.direction == direction) & df.f12_trend_slope.notna() & df.pnl.notna()
    ax.scatter(df.loc[mask, "f12_trend_slope"] * 100, df.loc[mask, "pnl"],
               c=color, alpha=0.5, s=20, marker=marker, label=direction, edgecolors="none")
ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
ax.set_title(f"F12: Prior 10d Trend Slope vs PnL\nrho={f12_corr.get('spearman_r', 0):.3f}", fontsize=11)
ax.set_xlabel("Trend Slope (% per day, normalized)")
ax.set_ylabel("PnL (%)")
ax.legend(fontsize=9)

# ── F13: Volume Spike vs PnL ──
ax = axes[1, 0]
mask = df.f13_vs_avg.notna() & df.pnl.notna()
# Clip extreme volume ratios for visualization
vs_avg_clipped = df.loc[mask, "f13_vs_avg"].clip(upper=50)
colors = ["#2196F3" if d == "LONG" else "#F44336" for d in df.loc[mask, "direction"]]
ax.scatter(vs_avg_clipped, df.loc[mask, "pnl"], c=colors, alpha=0.5, s=20, edgecolors="none")
ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax.axvline(1, color="orange", linewidth=0.5, linestyle=":", label="1x avg vol")
ax.set_title(f"F13: PM Vol / Avg Daily Vol vs PnL\nrho={f13_corr_avg.get('spearman_r', 0):.3f}", fontsize=11)
ax.set_xlabel("PM Vol / Avg 20d Daily Vol (capped at 50x)")
ax.set_ylabel("PnL (%)")

# ── F14: Position in Range vs PnL ──
ax = axes[1, 1]
for direction, color, marker in [("SHORT", "#F44336", "v"), ("LONG", "#2196F3", "^")]:
    mask = (df.direction == direction) & df.f14_position.notna() & df.pnl.notna()
    ax.scatter(df.loc[mask, "f14_position"], df.loc[mask, "pnl"],
               c=color, alpha=0.5, s=20, marker=marker, label=direction, edgecolors="none")
ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax.axvline(1.0, color="orange", linewidth=0.5, linestyle=":", label="20d high")
ax.set_title(f"F14: Position in 20d Range vs PnL\nrho={f14_corr_pos.get('spearman_r', 0):.3f}", fontsize=11)
ax.set_xlabel("Position in 20d Range (0=low, 1=high)")
ax.set_ylabel("PnL (%)")
ax.legend(fontsize=9)

# ── F15: Idiosyncratic Gap vs PnL ──
ax = axes[2, 0]
mask = df.f15_idio_gap.notna() & df.pnl.notna()
colors = ["#2196F3" if d == "LONG" else "#F44336" for d in df.loc[mask, "direction"]]
# Clip for viz
idio_clipped = df.loc[mask, "f15_idio_gap"].clip(-40, 125)
ax.scatter(idio_clipped, df.loc[mask, "pnl"], c=colors, alpha=0.5, s=20, edgecolors="none")
ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
ax.set_title(f"F15: Idiosyncratic Gap vs PnL\nrho={f15_corr.get('spearman_r', 0):.3f}", fontsize=11)
ax.set_xlabel("Idiosyncratic Gap (stock gap - SPY gap, %)")
ax.set_ylabel("PnL (%)")

# ── Summary ranking bar chart ──
ax = axes[2, 1]
fids = [fs["feature_id"] for fs in features_summary]
rhos = [abs(fs["spearman_r"]) if fs["spearman_r"] is not None else 0 for fs in features_summary]
pvals = [fs["spearman_p"] if fs["spearman_p"] is not None else 1.0 for fs in features_summary]
bar_colors = ["#4CAF50" if p < 0.05 else "#FFC107" if p < 0.1 else "#9E9E9E" for p in pvals]
bars = ax.barh(range(len(fids)), rhos, color=bar_colors, edgecolor="white")
ax.set_yticks(range(len(fids)))
ax.set_yticklabels([f"{fids[i]} ({features_summary[i]['feature_name'][:20]})" for i in range(len(fids))], fontsize=9)
ax.set_xlabel("|Spearman rho|")
ax.set_title("Feature Ranking by |Spearman rho|\nGreen=p<0.05, Yellow=p<0.1, Gray=p>=0.1", fontsize=11)
for i, (r, p) in enumerate(zip(rhos, pvals)):
    ax.text(r + 0.002, i, f"{r:.3f} (p={p:.3f})", va="center", fontsize=8)
ax.invert_yaxis()

plt.tight_layout(rect=[0, 0, 1, 0.96])
chart_path = f"{OUTPUT_DIR}/F11_F15_analysis.png"
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Chart saved to {chart_path}")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL VERDICT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("VERDICT")
print("="*80)
for fs in features_summary:
    sig = fs["spearman_p"] is not None and fs["spearman_p"] < 0.05
    strong = fs["abs_spearman"] > 0.15
    verdict = "PROMISING" if sig and strong else "WEAK SIGNAL" if sig else "NO SIGNAL"
    print(f"  {fs['feature_id']} {fs['feature_name']:30s}  |rho|={fs['abs_spearman']:.4f}  "
          f"p={fs['spearman_p']:.4f}  -> {verdict}")

print(f"\n  WARNING: Only 4 days of data (n=316 trades, {df.pnl.notna().sum()} with PnL).")
print(f"  ALL results require {LIVE_TRADING_WEEKS_NEEDED} weeks of live validation.")
print(f"\n  Output files in: {OUTPUT_DIR}/")

print("\nDone.")
