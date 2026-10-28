# %% Agent 10 — Phase 1: Cycle Characterization (Go/No-Go Gate)
# Determines whether RXRX has a statistically real, recurring oscillation.
#
# Sections:
#   1.1 Dominant period detection (spectral analysis + surrogates)
#   1.2 Swing structure (ZigZag, duration/amplitude distributions)
#   1.3 Mean-reversion strength (VR, Hurst, ADF, KPSS, OU half-life)
#   1.4 Level structure (KDE of swing extremes, reversal rates)
#   1.5 What drives the cycle (coherence, calendar, flow)
#
# Requires: Phase 0 data in agent10/data/
# Outputs: plots + phase1_results.json with gate decision
#
# Run as Colab cell. All pip installs handled inline.

import os, sys, json, warnings
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import signal, stats
from scipy.ndimage import uniform_filter1d

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Install optional deps ─────────────────────────────────────────
for pkg in ["PyWavelets", "statsmodels", "arch"]:
    try:
        __import__(pkg)
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg,
                               "--quiet", "--break-system-packages"])

import pywt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss

# ── Config ─────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data" if "__file__" in dir() else Path("agent10/data")
OUT_DIR = Path(__file__).parent / "results" if "__file__" in dir() else Path("agent10/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HOLDOUT_START = "2026-04-01"
N_SURROGATES = 1000
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

FIGSIZE = (14, 6)
DPI = 150

# ── Load Data ──────────────────────────────────────────────────────
# %%
print("Loading data...")
rxrx = pd.read_parquet(DATA_DIR / "rxrx_daily.parquet")
rxrx["date"] = pd.to_datetime(rxrx["date"])
rxrx = rxrx.sort_values("date").reset_index(drop=True)

# Enforce holdout
rxrx_dev = rxrx[rxrx["date"] < HOLDOUT_START].copy()
print(f"RXRX development set: {len(rxrx_dev)} bars, "
      f"{rxrx_dev['date'].min().date()} to {rxrx_dev['date'].max().date()}")
print(f"Holdout locked from {HOLDOUT_START} ({len(rxrx[rxrx['date'] >= HOLDOUT_START])} bars)")

# Context
ctx = pd.read_parquet(DATA_DIR / "context_daily.parquet")
ctx["date"] = pd.to_datetime(ctx["date"])

# Log returns and log price
rxrx_dev["log_price"] = np.log(rxrx_dev["close"])
rxrx_dev["log_ret"] = rxrx_dev["log_price"].diff()
rxrx_dev["pct_ret"] = rxrx_dev["close"].pct_change()

results = {}  # collect all Phase 1 results

# ══════════════════════════════════════════════════════════════════
# 1.1 DOMINANT PERIOD DETECTION
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 60)
print("1.1 DOMINANT PERIOD DETECTION")
print("=" * 60)

def detrend_rolling_mean(series, window):
    """Deviation from rolling mean."""
    rm = series.rolling(window, center=False, min_periods=window).mean()
    return (series - rm).dropna()

def detrend_hp(series, lamb=6.25):
    """HP filter (Ravn-Uhlig lambda for daily data)."""
    cycle, trend = sm.tsa.filters.hpfilter(series.dropna(), lamb=lamb)
    return cycle

def detrend_arkg_residual(rxrx_df, arkg_df):
    """Residual from rolling regression on ARKG (idiosyncratic component)."""
    merged = rxrx_df[["date", "log_price"]].merge(
        arkg_df[["date", "close"]].rename(columns={"close": "arkg_close"}),
        on="date", how="inner"
    )
    merged["arkg_log"] = np.log(merged["arkg_close"])
    residuals = []
    window = 60
    for i in range(window, len(merged)):
        y = merged["log_price"].iloc[i-window:i+1].values
        x = sm.add_constant(merged["arkg_log"].iloc[i-window:i+1].values)
        try:
            model = sm.OLS(y, x).fit()
            residuals.append(model.resid[-1])
        except:
            residuals.append(np.nan)
    res = pd.Series(residuals, index=merged.index[window:])
    return res.dropna()


# Detrend three ways
detrended = {}
for w in [20, 40, 60]:
    detrended[f"rolling_{w}d"] = detrend_rolling_mean(rxrx_dev["log_price"], w)

detrended["hp_filter"] = detrend_hp(rxrx_dev["log_price"])

arkg = ctx[ctx["ticker"] == "ARKG"].sort_values("date").reset_index(drop=True)
if not arkg.empty:
    detrended["arkg_residual"] = detrend_arkg_residual(rxrx_dev, arkg)

print(f"Detrended series: {list(detrended.keys())}")

# Spectral analysis on each detrended series
def compute_periodogram(series):
    x = series.values
    x = x - x.mean()
    freqs, psd = signal.periodogram(x, fs=1.0, detrend="linear")
    periods = np.where(freqs > 0, 1.0 / freqs, np.inf)
    mask = (periods >= 5) & (periods <= 120)
    return freqs[mask], psd[mask], periods[mask]

def compute_lomb_scargle(series, t=None):
    x = series.values
    x = x - x.mean()
    if t is None:
        t = np.arange(len(x), dtype=float)
    freqs = np.linspace(1/120, 1/5, 500)
    power = signal.lombscargle(t, x, freqs * 2 * np.pi, normalize=True)
    periods = 1.0 / freqs
    return freqs, power, periods

def compute_wavelet(series, scales=None):
    x = series.values
    x = (x - x.mean()) / (x.std() + 1e-10)
    if scales is None:
        scales = np.arange(5, 80)
    coeffs, freqs = pywt.cwt(x, scales, "cmor1.5-1.0")
    power = np.abs(coeffs) ** 2
    periods = 1.0 / freqs
    return power, periods, scales


# Generate surrogates for significance testing
def generate_ar1_surrogate(series, n_surrogates=N_SURROGATES):
    """AR(1) red-noise surrogates."""
    x = series.values
    ar_coef = np.corrcoef(x[:-1], x[1:])[0, 1]
    std = np.std(x) * np.sqrt(1 - ar_coef**2)
    surrogates = np.zeros((n_surrogates, len(x)))
    for i in range(n_surrogates):
        surrogates[i, 0] = np.random.normal(0, np.std(x))
        for t in range(1, len(x)):
            surrogates[i, t] = ar_coef * surrogates[i, t-1] + np.random.normal(0, std)
    return surrogates

def generate_phase_randomized(series, n_surrogates=N_SURROGATES):
    """Phase-randomized surrogates (preserve power spectrum, destroy phase)."""
    x = series.values
    n = len(x)
    fft_x = np.fft.rfft(x)
    amp = np.abs(fft_x)
    surrogates = np.zeros((n_surrogates, n))
    for i in range(n_surrogates):
        rand_phase = np.random.uniform(0, 2*np.pi, len(fft_x))
        rand_phase[0] = 0
        if n % 2 == 0:
            rand_phase[-1] = 0
        fft_surr = amp * np.exp(1j * rand_phase)
        surrogates[i] = np.fft.irfft(fft_surr, n=n)
    return surrogates


spectral_results = {}

fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle("1.1 Spectral Analysis — Dominant Period Detection", fontsize=14, fontweight="bold")

for idx, (name, series) in enumerate(list(detrended.items())[:6]):
    if idx >= 6:
        break
    ax = axes.flat[idx]

    freqs_p, psd_p, periods_p = compute_periodogram(series)
    freqs_ls, power_ls, periods_ls = compute_lomb_scargle(series)

    # Peak detection
    peak_idx_p = np.argmax(psd_p)
    peak_period_p = periods_p[peak_idx_p]
    peak_idx_ls = np.argmax(power_ls)
    peak_period_ls = periods_ls[peak_idx_ls]

    # Surrogate test (AR1)
    surr = generate_ar1_surrogate(series, n_surrogates=N_SURROGATES)
    surr_peaks = []
    for s in surr:
        s_series = pd.Series(s)
        _, psd_s, periods_s = compute_periodogram(s_series)
        surr_peaks.append(np.max(psd_s))

    p95 = np.percentile(surr_peaks, 95)
    p99 = np.percentile(surr_peaks, 99)
    is_significant_95 = psd_p[peak_idx_p] > p95
    is_significant_99 = psd_p[peak_idx_p] > p99

    # Phase-randomized test
    surr_pr = generate_phase_randomized(series, n_surrogates=N_SURROGATES)
    surr_pr_peaks = []
    for s in surr_pr:
        s_series = pd.Series(s)
        _, psd_s, periods_s = compute_periodogram(s_series)
        surr_pr_peaks.append(np.max(psd_s))
    p95_pr = np.percentile(surr_pr_peaks, 95)

    spectral_results[name] = {
        "peak_period_periodogram": float(peak_period_p),
        "peak_period_lomb_scargle": float(peak_period_ls),
        "peak_power": float(psd_p[peak_idx_p]),
        "surr_ar1_p95": float(p95),
        "surr_ar1_p99": float(p99),
        "surr_phase_rand_p95": float(p95_pr),
        "significant_ar1_95": bool(is_significant_95),
        "significant_ar1_99": bool(is_significant_99),
        "significant_phase_rand_95": bool(psd_p[peak_idx_p] > p95_pr),
        "percentile_in_surrogates": float(np.mean(np.array(surr_peaks) < psd_p[peak_idx_p]) * 100),
    }

    ax.semilogy(periods_p, psd_p, "b-", lw=1.2, label="Periodogram")
    ax.axhline(p95, color="r", ls="--", alpha=0.7, label=f"AR1 95th: {p95:.4f}")
    ax.axhline(p99, color="r", ls=":", alpha=0.5, label=f"AR1 99th: {p99:.4f}")
    ax.axhline(p95_pr, color="orange", ls="--", alpha=0.7, label=f"PhaseRand 95th")
    ax.axvline(peak_period_p, color="g", ls="-", alpha=0.5,
               label=f"Peak: {peak_period_p:.0f}d")
    sig_str = "SIG" if is_significant_95 else "NOT SIG"
    ax.set_title(f"{name} — Peak {peak_period_p:.0f}d [{sig_str}]", fontsize=10)
    ax.set_xlabel("Period (days)")
    ax.set_ylabel("PSD")
    ax.set_xlim(5, 120)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "p1_spectral_analysis.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("Saved p1_spectral_analysis.png")

for name, r in spectral_results.items():
    sig = "✓" if r["significant_ar1_95"] else "✗"
    print(f"  {name:25s}  peak={r['peak_period_periodogram']:.0f}d  "
          f"percentile={r['percentile_in_surrogates']:.1f}%  [{sig}]")


# Wavelet scalogram
# %%
print("\nComputing wavelet scalogram...")
primary = detrended.get("rolling_40d", list(detrended.values())[0])
scales = np.arange(5, 80)
power, periods_w, _ = compute_wavelet(primary, scales)

fig, ax = plt.subplots(figsize=FIGSIZE)
dates_w = rxrx_dev["date"].iloc[-len(primary):]
T, S = np.meshgrid(np.arange(power.shape[1]), periods_w)
pcm = ax.pcolormesh(T, S, np.log10(power + 1e-10), cmap="jet", shading="auto")
ax.set_ylabel("Period (days)")
ax.set_xlabel("Time index")
ax.set_title("1.1 Continuous Wavelet Transform — RXRX Detrended (40d rolling)")
ax.set_ylim(5, 80)
ax.axhline(25, color="white", ls="--", alpha=0.7, label="25d")
ax.axhline(35, color="white", ls="--", alpha=0.7, label="35d")
ax.legend(loc="upper right")
plt.colorbar(pcm, ax=ax, label="log10(power)")
plt.tight_layout()
plt.savefig(OUT_DIR / "p1_wavelet_scalogram.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("Saved p1_wavelet_scalogram.png")


# ══════════════════════════════════════════════════════════════════
# 1.2 SWING STRUCTURE
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 60)
print("1.2 SWING STRUCTURE")
print("=" * 60)

def compute_atr(df, period=20):
    high = df["high"]
    low = df["low"]
    close = df["close"].shift(1)
    tr = pd.concat([high - low, (high - close).abs(), (low - close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

rxrx_dev["atr20"] = compute_atr(rxrx_dev, 20)

def zigzag(prices, threshold):
    """ATR-based ZigZag. Returns list of (index, price, type) tuples."""
    pivots = []
    last_pivot_price = prices.iloc[0]
    last_pivot_idx = 0
    last_pivot_type = None  # 'high' or 'low'

    for i in range(1, len(prices)):
        if last_pivot_type is None:
            if prices.iloc[i] > last_pivot_price + threshold.iloc[i]:
                pivots.append((last_pivot_idx, last_pivot_price, "low"))
                last_pivot_price = prices.iloc[i]
                last_pivot_idx = i
                last_pivot_type = "high"
            elif prices.iloc[i] < last_pivot_price - threshold.iloc[i]:
                pivots.append((last_pivot_idx, last_pivot_price, "high"))
                last_pivot_price = prices.iloc[i]
                last_pivot_idx = i
                last_pivot_type = "low"
        elif last_pivot_type == "high":
            if prices.iloc[i] > last_pivot_price:
                last_pivot_price = prices.iloc[i]
                last_pivot_idx = i
            elif prices.iloc[i] < last_pivot_price - threshold.iloc[i]:
                pivots.append((last_pivot_idx, last_pivot_price, "high"))
                last_pivot_price = prices.iloc[i]
                last_pivot_idx = i
                last_pivot_type = "low"
        elif last_pivot_type == "low":
            if prices.iloc[i] < last_pivot_price:
                last_pivot_price = prices.iloc[i]
                last_pivot_idx = i
            elif prices.iloc[i] > last_pivot_price + threshold.iloc[i]:
                pivots.append((last_pivot_idx, last_pivot_price, "low"))
                last_pivot_price = prices.iloc[i]
                last_pivot_idx = i
                last_pivot_type = "high"

    pivots.append((last_pivot_idx, last_pivot_price, last_pivot_type or "low"))
    return pivots


swing_stats = {}
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("1.2 Swing Structure — ZigZag Analysis", fontsize=14, fontweight="bold")

for mult_idx, atr_mult in enumerate([1.5, 2.0, 3.0]):
    threshold = rxrx_dev["atr20"] * atr_mult
    threshold = threshold.fillna(threshold.dropna().iloc[0])
    pivots = zigzag(rxrx_dev["close"], threshold)

    pivot_df = pd.DataFrame(pivots, columns=["idx", "price", "type"])
    pivot_df["date"] = rxrx_dev["date"].iloc[pivot_df["idx"]].values

    # Trough-to-trough and peak-to-peak durations
    troughs = pivot_df[pivot_df["type"] == "low"]
    peaks = pivot_df[pivot_df["type"] == "high"]

    t2t_dur = troughs["idx"].diff().dropna()
    p2p_dur = peaks["idx"].diff().dropna()
    all_dur = pivot_df["idx"].diff().dropna()  # half-cycle

    # Amplitude
    amps = []
    for i in range(1, len(pivot_df)):
        amp_pct = abs(pivot_df.iloc[i]["price"] - pivot_df.iloc[i-1]["price"]) / pivot_df.iloc[i-1]["price"] * 100
        amps.append(amp_pct)
    amps = np.array(amps)

    cv_dur_t2t = t2t_dur.std() / t2t_dur.mean() if len(t2t_dur) > 2 else np.nan
    cv_dur_p2p = p2p_dur.std() / p2p_dur.mean() if len(p2p_dur) > 2 else np.nan
    cv_amp = amps.std() / amps.mean() if len(amps) > 2 else np.nan

    swing_stats[f"atr_x{atr_mult}"] = {
        "n_pivots": len(pivots),
        "n_troughs": len(troughs),
        "n_peaks": len(peaks),
        "t2t_mean": float(t2t_dur.mean()) if len(t2t_dur) > 0 else None,
        "t2t_median": float(t2t_dur.median()) if len(t2t_dur) > 0 else None,
        "t2t_cv": float(cv_dur_t2t),
        "p2p_mean": float(p2p_dur.mean()) if len(p2p_dur) > 0 else None,
        "p2p_median": float(p2p_dur.median()) if len(p2p_dur) > 0 else None,
        "p2p_cv": float(cv_dur_p2p),
        "amp_mean_pct": float(amps.mean()),
        "amp_median_pct": float(np.median(amps)),
        "amp_cv": float(cv_amp),
        "half_cycle_mean": float(all_dur.mean()),
    }

    # Plot: duration histogram
    ax = axes[0, mult_idx]
    if len(t2t_dur) > 2:
        ax.hist(t2t_dur, bins=15, alpha=0.6, label=f"T2T (μ={t2t_dur.mean():.0f}d)", color="blue")
    if len(p2p_dur) > 2:
        ax.hist(p2p_dur, bins=15, alpha=0.6, label=f"P2P (μ={p2p_dur.mean():.0f}d)", color="red")
    ax.axvline(25, color="green", ls="--", alpha=0.5, label="25d")
    ax.axvline(35, color="green", ls=":", alpha=0.5, label="35d")
    ax.set_title(f"ATR×{atr_mult} — Swing Duration", fontsize=10)
    ax.set_xlabel("Days")
    ax.legend(fontsize=7)

    # Plot: amplitude histogram
    ax = axes[1, mult_idx]
    ax.hist(amps, bins=20, alpha=0.7, color="purple")
    ax.set_title(f"ATR×{atr_mult} — Swing Amplitude (CV={cv_amp:.2f})", fontsize=10)
    ax.set_xlabel("Amplitude (%)")
    ax.axvline(amps.mean(), color="red", ls="--", label=f"Mean: {amps.mean():.1f}%")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR / "p1_swing_structure.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("Saved p1_swing_structure.png")

for k, v in swing_stats.items():
    print(f"  {k}: T2T μ={v['t2t_mean']:.0f}d CV={v['t2t_cv']:.2f}, "
          f"Amp μ={v['amp_mean_pct']:.1f}% CV={v['amp_cv']:.2f}")

# Year-by-year breakdown
# %%
print("\nYear-by-year swing stats (ATR×2.0):")
threshold = rxrx_dev["atr20"] * 2.0
threshold = threshold.fillna(threshold.dropna().iloc[0])
pivots = zigzag(rxrx_dev["close"], threshold)
pivot_df = pd.DataFrame(pivots, columns=["idx", "price", "type"])
pivot_df["date"] = rxrx_dev["date"].iloc[pivot_df["idx"]].values
pivot_df["year"] = pivot_df["date"].dt.year

for yr in sorted(pivot_df["year"].unique()):
    yr_pivots = pivot_df[pivot_df["year"] == yr]
    yr_troughs = yr_pivots[yr_pivots["type"] == "low"]
    t2t = yr_troughs["idx"].diff().dropna()
    amps = []
    for i in range(1, len(yr_pivots)):
        amps.append(abs(yr_pivots.iloc[i]["price"] - yr_pivots.iloc[i-1]["price"]) / yr_pivots.iloc[i-1]["price"] * 100)
    print(f"  {yr}: {len(yr_pivots)} pivots, "
          f"T2T μ={t2t.mean():.0f}d" if len(t2t) > 0 else f"  {yr}: {len(yr_pivots)} pivots",
          f"  Amp μ={np.mean(amps):.1f}%" if amps else "")


# ══════════════════════════════════════════════════════════════════
# 1.3 MEAN-REVERSION STRENGTH
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 60)
print("1.3 MEAN-REVERSION STRENGTH")
print("=" * 60)

returns = rxrx_dev["log_ret"].dropna().values

# Lo-MacKinlay Variance Ratio
def variance_ratio(returns, k):
    n = len(returns)
    mu = returns.mean()
    var1 = np.sum((returns - mu)**2) / (n - 1)
    rk = np.array([np.sum(returns[i:i+k]) for i in range(n - k + 1)])
    vark = np.sum((rk - k * mu)**2) / (n - k)
    vr = vark / (k * var1)
    # Heteroscedasticity-robust z-stat
    delta = np.zeros(k - 1)
    for j in range(1, k):
        num = np.sum(((returns[j:] - mu)**2 * (returns[:-j] - mu)**2))
        den = (np.sum((returns - mu)**2))**2 / n
        delta[j-1] = num / den
    theta = 2 * np.sum(((np.arange(1, k) * 1.0 / k) - (np.arange(1, k) / k)**2)**2 * delta)
    z = (vr - 1) / np.sqrt(theta) if theta > 0 else 0
    return vr, z

vr_results = {}
horizons = list(range(2, 31))
vr_values = []
vr_zstats = []
for k in horizons:
    vr, z = variance_ratio(returns, k)
    vr_values.append(vr)
    vr_zstats.append(z)
    vr_results[k] = {"vr": float(vr), "z": float(z), "p_value": float(2 * (1 - stats.norm.cdf(abs(z))))}

print("Variance Ratios (VR < 1 = mean reversion):")
for k in [2, 5, 10, 15, 20, 25, 30]:
    r = vr_results[k]
    sig = "*" if abs(r["z"]) > 1.96 else ""
    print(f"  VR({k:2d}) = {r['vr']:.4f}  z={r['z']:+.2f}  p={r['p_value']:.4f} {sig}")

# Hurst exponent (DFA)
def hurst_dfa(series, min_window=10, max_window=None):
    x = np.cumsum(series - series.mean())
    n = len(x)
    if max_window is None:
        max_window = n // 4
    windows = np.unique(np.logspace(np.log10(min_window), np.log10(max_window), 20).astype(int))
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
        return np.nan, [], []
    windows_used = [f[0] for f in flucts]
    f_values = [f[1] for f in flucts]
    coeffs = np.polyfit(np.log(windows_used), np.log(f_values), 1)
    return coeffs[0], windows_used, f_values

hurst, dfa_windows, dfa_flucts = hurst_dfa(returns)
print(f"\nHurst exponent (DFA): {hurst:.4f}")
print(f"  < 0.5 = mean-reverting, = 0.5 random walk, > 0.5 trending")

# Rolling ADF and KPSS
def rolling_adf(prices, window):
    results = []
    for i in range(window, len(prices)):
        try:
            adf_stat, p_val, _, _, _, _ = adfuller(prices.iloc[i-window:i], maxlag=int(window/10))
            results.append({"idx": i, "adf_stat": adf_stat, "adf_pval": p_val})
        except:
            results.append({"idx": i, "adf_stat": np.nan, "adf_pval": np.nan})
    return pd.DataFrame(results)

def rolling_kpss_test(prices, window):
    results = []
    for i in range(window, len(prices)):
        try:
            stat, p_val, _, crit = kpss(prices.iloc[i-window:i], regression="c", nlags="auto")
            results.append({"idx": i, "kpss_stat": stat, "kpss_pval": p_val})
        except:
            results.append({"idx": i, "kpss_stat": np.nan, "kpss_pval": np.nan})
    return pd.DataFrame(results)

# Rolling OU half-life
def rolling_ou_halflife(prices, window):
    results = []
    for i in range(window, len(prices)):
        y = prices.iloc[i-window:i].values
        dy = np.diff(y)
        y_lag = y[:-1]
        y_lag_c = sm.add_constant(y_lag)
        try:
            model = sm.OLS(dy, y_lag_c).fit()
            theta = -model.params[1]
            half_life = np.log(2) / theta if theta > 0 else np.nan
            results.append({"idx": i, "half_life": half_life, "theta": theta})
        except:
            results.append({"idx": i, "half_life": np.nan, "theta": np.nan})
    return pd.DataFrame(results)

print("\nComputing rolling ADF (120d, 250d)...")
adf_120 = rolling_adf(rxrx_dev["close"], 120)
adf_250 = rolling_adf(rxrx_dev["close"], 250)

print("Computing rolling KPSS (120d, 250d)...")
kpss_120 = rolling_kpss_test(rxrx_dev["close"], 120)
kpss_250 = rolling_kpss_test(rxrx_dev["close"], 250)

print("Computing rolling OU half-life (120d)...")
ou_120 = rolling_ou_halflife(rxrx_dev["close"], 120)

# Rolling Hurst (120d)
print("Computing rolling Hurst (120d)...")
rolling_hurst = []
hurst_window = 120
for i in range(hurst_window, len(returns)):
    h, _, _ = hurst_dfa(returns[i-hurst_window:i])
    rolling_hurst.append({"idx": i + 1, "hurst": h})  # +1 because returns is diff'd
rolling_hurst_df = pd.DataFrame(rolling_hurst)

# Plot
fig, axes = plt.subplots(4, 1, figsize=(16, 18), sharex=True)
fig.suptitle("1.3 Mean-Reversion Strength — Rolling Tests", fontsize=14, fontweight="bold")

dates_plot = rxrx_dev["date"]

# VR plot
ax = axes[0]
ax.plot(horizons, vr_values, "b-o", markersize=3)
ax.axhline(1.0, color="red", ls="--", alpha=0.7, label="Random Walk")
ax.fill_between(horizons, [1 - 1.96/np.sqrt(len(returns)/k) for k in horizons],
                [1 + 1.96/np.sqrt(len(returns)/k) for k in horizons],
                alpha=0.15, color="red", label="95% CI (RW)")
ax.set_ylabel("Variance Ratio")
ax.set_title("Lo-MacKinlay Variance Ratio (VR < 1 = Mean Reversion)")
ax.legend()
ax.grid(True, alpha=0.3)

# Rolling Hurst
ax = axes[1]
if not rolling_hurst_df.empty:
    hurst_dates = dates_plot.iloc[rolling_hurst_df["idx"].values]
    ax.plot(hurst_dates, rolling_hurst_df["hurst"], "b-", lw=0.8)
    ax.axhline(0.5, color="red", ls="--", label="Random Walk (H=0.5)")
    ax.axhline(0.55, color="orange", ls=":", label="Trending (H>0.55)")
    ax.axhline(0.45, color="green", ls=":", label="Mean-Reverting (H<0.45)")
    ax.fill_between(hurst_dates, 0, 0.45, alpha=0.08, color="green")
    ax.fill_between(hurst_dates, 0.55, 1.0, alpha=0.08, color="orange")
ax.set_ylabel("Hurst (DFA)")
ax.set_title(f"Rolling Hurst Exponent (120d window) — Overall: {hurst:.3f}")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Rolling ADF
ax = axes[2]
if not adf_120.empty:
    adf_dates_120 = dates_plot.iloc[adf_120["idx"].values]
    ax.plot(adf_dates_120, adf_120["adf_pval"], "b-", lw=0.8, label="ADF p-val (120d)")
if not adf_250.empty:
    adf_dates_250 = dates_plot.iloc[adf_250["idx"].values]
    ax.plot(adf_dates_250, adf_250["adf_pval"], "r-", lw=0.8, label="ADF p-val (250d)")
ax.axhline(0.05, color="green", ls="--", alpha=0.7, label="p=0.05 (stationary)")
ax.set_ylabel("ADF p-value")
ax.set_title("Rolling ADF Test (low p-value = stationary / mean-reverting)")
ax.set_ylim(0, 1)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Rolling OU half-life
ax = axes[3]
if not ou_120.empty:
    ou_dates = dates_plot.iloc[ou_120["idx"].values]
    hl = ou_120["half_life"].clip(0, 100)
    ax.plot(ou_dates, hl, "purple", lw=0.8)
    ax.axhline(25, color="green", ls="--", alpha=0.5, label="25d")
    ax.axhline(35, color="green", ls=":", alpha=0.5, label="35d")
ax.set_ylabel("OU Half-Life (days)")
ax.set_title("Rolling Ornstein-Uhlenbeck Half-Life (120d window)")
ax.set_ylim(0, 80)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlabel("Date")

plt.tight_layout()
plt.savefig(OUT_DIR / "p1_mean_reversion.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("Saved p1_mean_reversion.png")


# ══════════════════════════════════════════════════════════════════
# 1.4 LEVEL STRUCTURE
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 60)
print("1.4 LEVEL STRUCTURE")
print("=" * 60)

# Collect swing highs and lows from ZigZag (ATR×2.0)
threshold = rxrx_dev["atr20"] * 2.0
threshold = threshold.fillna(threshold.dropna().iloc[0])
pivots = zigzag(rxrx_dev["close"], threshold)
pivot_df = pd.DataFrame(pivots, columns=["idx", "price", "type"])
pivot_df["date"] = rxrx_dev["date"].iloc[pivot_df["idx"]].values

swing_highs = pivot_df[pivot_df["type"] == "high"]["price"].values
swing_lows = pivot_df[pivot_df["type"] == "low"]["price"].values
all_pivots_price = pivot_df["price"].values

# KDE
from scipy.stats import gaussian_kde

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("1.4 Level Structure — Price Clusters & Reversal Rates", fontsize=14, fontweight="bold")

# KDE of all pivot prices
ax = axes[0, 0]
if len(all_pivots_price) > 3:
    kde = gaussian_kde(all_pivots_price, bw_method=0.15)
    x_kde = np.linspace(all_pivots_price.min() * 0.9, all_pivots_price.max() * 1.1, 500)
    ax.plot(x_kde, kde(x_kde), "b-", lw=2)
    ax.fill_between(x_kde, kde(x_kde), alpha=0.3)

    # Find peaks in KDE (price clusters)
    kde_vals = kde(x_kde)
    peaks_idx, props = signal.find_peaks(kde_vals, height=np.max(kde_vals) * 0.15, distance=10)
    cluster_prices = x_kde[peaks_idx]
    for cp in cluster_prices:
        ax.axvline(cp, color="red", ls="--", alpha=0.5)
    ax.set_title(f"KDE of Swing Extremes — {len(cluster_prices)} clusters")
else:
    ax.set_title("KDE — insufficient data")
ax.set_xlabel("Price ($)")
ax.set_ylabel("Density")
ax.grid(True, alpha=0.3)

# Reversal rate near clusters
ax = axes[0, 1]
if len(cluster_prices) > 0:
    atr_vals = rxrx_dev["atr20"].dropna()
    avg_atr = atr_vals.mean()

    reversal_rates = []
    for level in cluster_prices:
        approaches = 0
        reversals = 0
        for i in range(1, len(rxrx_dev) - 5):
            price = rxrx_dev["close"].iloc[i]
            if abs(price - level) <= avg_atr:
                approaches += 1
                future_prices = rxrx_dev["close"].iloc[i+1:i+6]
                if len(future_prices) == 0:
                    continue
                if price < level and future_prices.max() > level + avg_atr * 0.5:
                    pass  # broke through
                elif price > level and future_prices.min() < level - avg_atr * 0.5:
                    pass  # broke through
                else:
                    reversals += 1
        rate = reversals / approaches if approaches > 10 else np.nan
        reversal_rates.append({"level": level, "approaches": approaches,
                                "reversals": reversals, "rate": rate})

    rr_df = pd.DataFrame(reversal_rates)
    if not rr_df.empty and not rr_df["rate"].isna().all():
        ax.bar(range(len(rr_df)), rr_df["rate"], color="steelblue", alpha=0.7)
        ax.set_xticks(range(len(rr_df)))
        ax.set_xticklabels([f"${l:.1f}" for l in rr_df["level"]], rotation=45, fontsize=8)
        ax.axhline(0.5, color="red", ls="--", label="50% (random)")
        ax.set_ylabel("Reversal Rate")
        ax.set_title("Reversal Rate Within ±1 ATR of Cluster")
        ax.legend()
    else:
        ax.set_title("Insufficient approach data")
ax.grid(True, alpha=0.3)

# Price chart with ZigZag overlay
ax = axes[1, 0]
ax.plot(rxrx_dev["date"], rxrx_dev["close"], "gray", lw=0.5, alpha=0.7)
pivot_dates = rxrx_dev["date"].iloc[pivot_df["idx"]].values
ax.plot(pivot_dates, pivot_df["price"], "b-", lw=1.5, label="ZigZag")
highs_mask = pivot_df["type"] == "high"
lows_mask = pivot_df["type"] == "low"
ax.scatter(pivot_dates[highs_mask.values], pivot_df[highs_mask]["price"],
           color="red", s=30, zorder=5, label="Peaks")
ax.scatter(pivot_dates[lows_mask.values], pivot_df[lows_mask]["price"],
           color="green", s=30, zorder=5, label="Troughs")
if len(cluster_prices) > 0:
    for cp in cluster_prices:
        ax.axhline(cp, color="orange", ls=":", alpha=0.4)
ax.set_title("RXRX Price with ZigZag (ATR×2.0) + Level Clusters")
ax.set_ylabel("Price ($)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Level persistence
ax = axes[1, 1]
if len(cluster_prices) > 0:
    # For each cluster level, find when it was first identified and when it was invalidated
    level_info = []
    for level in cluster_prices:
        first_approach = None
        last_reversal = None
        n_tests = 0
        for i in range(20, len(rxrx_dev)):
            price = rxrx_dev["close"].iloc[i]
            if abs(price - level) <= avg_atr:
                if first_approach is None:
                    first_approach = rxrx_dev["date"].iloc[i]
                last_reversal = rxrx_dev["date"].iloc[i]
                n_tests += 1
        if first_approach and last_reversal:
            persistence = (last_reversal - first_approach).days
            level_info.append({"level": level, "persistence_days": persistence, "n_tests": n_tests})

    if level_info:
        li_df = pd.DataFrame(level_info)
        ax.barh(range(len(li_df)), li_df["persistence_days"], color="steelblue", alpha=0.7)
        ax.set_yticks(range(len(li_df)))
        ax.set_yticklabels([f"${l:.1f}" for l in li_df["level"]])
        ax.set_xlabel("Persistence (days)")
        ax.set_title("Level Persistence (first to last approach)")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "p1_level_structure.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("Saved p1_level_structure.png")


# ══════════════════════════════════════════════════════════════════
# 1.5 WHAT DRIVES THE CYCLE?
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 60)
print("1.5 WHAT DRIVES THE CYCLE?")
print("=" * 60)

# Coherence with ARKG and XBI
def compute_coherence(series1, series2, fs=1.0, nperseg=64):
    """Magnitude-squared coherence between two series."""
    min_len = min(len(series1), len(series2))
    s1 = series1[:min_len]
    s2 = series2[:min_len]
    freqs, coh = signal.coherence(s1, s2, fs=fs, nperseg=min(nperseg, min_len//4))
    periods = np.where(freqs > 0, 1.0 / freqs, np.inf)
    return freqs, coh, periods

# Get context returns
arkg_daily = ctx[ctx["ticker"] == "ARKG"].sort_values("date")
xbi_daily = ctx[ctx["ticker"] == "XBI"].sort_values("date")

rxrx_rets = rxrx_dev[["date", "log_ret"]].dropna()

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("1.5 What Drives the Cycle?", fontsize=14, fontweight="bold")

for plot_idx, (ticker, ctx_df) in enumerate([("ARKG", arkg_daily), ("XBI", xbi_daily)]):
    ctx_df = ctx_df.copy()
    ctx_df["log_ret"] = np.log(ctx_df["close"]).diff()

    merged = rxrx_rets.merge(ctx_df[["date", "log_ret"]].rename(columns={"log_ret": f"{ticker}_ret"}),
                             on="date", how="inner").dropna()

    if len(merged) < 100:
        continue

    # Cross-correlation
    ax = axes[0, plot_idx]
    max_lag = 30
    xcorr = [np.corrcoef(merged["log_ret"].values[max_lag:],
                          merged[f"{ticker}_ret"].values[max_lag-lag:len(merged)-lag])[0,1]
             if lag > 0 else np.corrcoef(merged["log_ret"].values[:len(merged)-max_lag],
                                          merged[f"{ticker}_ret"].values[max_lag:])[0,1]
             if lag < 0 else np.corrcoef(merged["log_ret"], merged[f"{ticker}_ret"])[0,1]
             for lag in range(-max_lag, max_lag+1)]
    lags = list(range(-max_lag, max_lag+1))
    ax.bar(lags, xcorr, color="steelblue", alpha=0.7)
    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(2/np.sqrt(len(merged)), color="red", ls="--", alpha=0.5, label="95% CI")
    ax.axhline(-2/np.sqrt(len(merged)), color="red", ls="--", alpha=0.5)
    ax.set_title(f"Cross-Correlation: RXRX vs {ticker}")
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Correlation")
    ax.legend(fontsize=8)

    # Coherence
    ax = axes[1, plot_idx]
    freqs, coh, periods = compute_coherence(
        merged["log_ret"].values, merged[f"{ticker}_ret"].values, nperseg=64
    )
    mask = (periods >= 5) & (periods <= 100)
    ax.plot(periods[mask], coh[mask], "b-", lw=1.5)
    ax.axvline(25, color="green", ls="--", alpha=0.5, label="25d")
    ax.axvline(35, color="green", ls=":", alpha=0.5, label="35d")
    ax.set_title(f"Coherence: RXRX vs {ticker}")
    ax.set_xlabel("Period (days)")
    ax.set_ylabel("Coherence")
    ax.set_xlim(5, 100)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "p1_cycle_drivers.png", dpi=DPI, bbox_inches="tight")
plt.close()
print("Saved p1_cycle_drivers.png")

# Calendar mechanics: test if swing turning points cluster near OPEX
# %%
print("\nCalendar mechanics test...")
pivot_dates_ts = pd.to_datetime(pivot_df["date"])

# Monthly OPEX: 3rd Friday of each month
from pandas.tseries.offsets import Week
opex_dates = []
for yr in range(2021, 2027):
    for mo in range(1, 13):
        try:
            first = pd.Timestamp(yr, mo, 1)
            # 3rd Friday: first Friday + 14 days
            first_fri = first + pd.offsets.Week(weekday=4)
            if first_fri.month != mo:
                first_fri = first_fri - pd.offsets.Week(1)
            third_fri = first_fri + pd.Timedelta(days=14)
            if third_fri.month == mo:
                opex_dates.append(third_fri)
        except:
            pass

opex_dates = pd.DatetimeIndex(opex_dates)

# Distance from each pivot to nearest OPEX
def nearest_distance(dates, reference_dates):
    distances = []
    for d in dates:
        diffs = np.abs((reference_dates - d).days)
        distances.append(diffs.min())
    return np.array(distances)

pivot_to_opex = nearest_distance(pivot_dates_ts, opex_dates)

# Permutation test: are pivots closer to OPEX than random dates?
n_perm = 1000
rand_distances = []
all_trading_dates = rxrx_dev["date"].values
for _ in range(n_perm):
    rand_idx = np.random.choice(len(all_trading_dates), len(pivot_dates_ts), replace=False)
    rand_dates = pd.DatetimeIndex(all_trading_dates[rand_idx])
    rand_distances.append(nearest_distance(rand_dates, opex_dates).mean())

actual_mean_dist = pivot_to_opex.mean()
p_opex = np.mean(np.array(rand_distances) <= actual_mean_dist)

print(f"  Pivot-to-OPEX mean distance: {actual_mean_dist:.1f} days")
print(f"  Random baseline mean: {np.mean(rand_distances):.1f} days")
print(f"  Permutation p-value: {p_opex:.4f}")
print(f"  {'SIGNIFICANT — pivots cluster near OPEX' if p_opex < 0.05 else 'Not significant — pivots do not cluster near OPEX'}")

# Month-end test
month_end_dates = pd.bdate_range(rxrx_dev["date"].min(), rxrx_dev["date"].max(), freq="BM")
pivot_to_me = nearest_distance(pivot_dates_ts, month_end_dates)
rand_me_distances = []
for _ in range(n_perm):
    rand_idx = np.random.choice(len(all_trading_dates), len(pivot_dates_ts), replace=False)
    rand_dates = pd.DatetimeIndex(all_trading_dates[rand_idx])
    rand_me_distances.append(nearest_distance(rand_dates, month_end_dates).mean())

p_month_end = np.mean(np.array(rand_me_distances) <= pivot_to_me.mean())
print(f"\n  Pivot-to-MonthEnd mean distance: {pivot_to_me.mean():.1f} days")
print(f"  Permutation p-value: {p_month_end:.4f}")


# ══════════════════════════════════════════════════════════════════
# PHASE 1 GATE DECISION
# ══════════════════════════════════════════════════════════════════
# %%
print("\n" + "=" * 60)
print("PHASE 1 GATE DECISION")
print("=" * 60)

# Collect gate metrics
any_spectral_significant = any(r["significant_ar1_95"] for r in spectral_results.values())

vr_mean_reversion = any(vr_results[k]["vr"] < 1 and abs(vr_results[k]["z"]) > 1.96
                        for k in [5, 10, 15, 20])

hurst_mean_reverting = hurst < 0.5

gate = {
    "spectral_significant": any_spectral_significant,
    "vr_mean_reversion": vr_mean_reversion,
    "hurst_value": float(hurst),
    "hurst_mean_reverting": hurst_mean_reverting,
    "cycle_strategies_unlocked": any_spectral_significant,
    "mr_strategies_unlocked": vr_mean_reversion or hurst_mean_reverting,
    "proceed": any_spectral_significant or vr_mean_reversion or hurst_mean_reverting,
}

print(f"\n  Spectral peak significant (>95th AR1)?  {'YES ✓' if any_spectral_significant else 'NO ✗'}")
print(f"  Variance ratio < 1 (significant)?        {'YES ✓' if vr_mean_reversion else 'NO ✗'}")
print(f"  Hurst < 0.5?                             {'YES ✓' if hurst_mean_reverting else 'NO ✗'} (H={hurst:.3f})")
print(f"\n  CYCLE strategies (S3) unlocked?           {'YES ✓' if gate['cycle_strategies_unlocked'] else 'NO ✗'}")
print(f"  MEAN-REVERSION strategies (S1,S2,S4-S7)?  {'YES ✓' if gate['mr_strategies_unlocked'] else 'NO ✗'}")

if gate["proceed"]:
    print(f"\n  ██ GATE PASSED — Proceed to Phase 2 ██")
else:
    print(f"\n  ██ GATE FAILED — RXRX does not exhibit tradeable mean reversion ██")
    print(f"  Report these results and STOP.")

# Save all results
results["spectral"] = spectral_results
results["swing_stats"] = swing_stats
results["variance_ratios"] = vr_results
results["hurst"] = float(hurst)
results["gate"] = gate
results["calendar"] = {
    "opex_mean_dist": float(actual_mean_dist),
    "opex_p_value": float(p_opex),
    "month_end_mean_dist": float(pivot_to_me.mean()),
    "month_end_p_value": float(p_month_end),
}

with open(OUT_DIR / "phase1_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved phase1_results.json")

print("\nPhase 1 complete.")
print("Review plots in agent10/results/ before proceeding to Phase 2.")
