"""
analyze_model_quality_bq.py — Model Prediction Quality Degradation Analysis
============================================================================
Queries BQ scoring logs to assess whether model prediction quality has
degraded, examining probability distributions, signal density, per-ticker
trajectories, and prob-to-outcome correlation using OHLCV data.

Run in Colab: paste as cell, runs immediately.
"""

import io
import gc
import ctypes
import warnings
import numpy as np
import pandas as pd
import pytz
from scipy import stats as sp_stats

warnings.filterwarnings('ignore')

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

from google.cloud import bigquery, storage as gcs_storage

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 200)

PST = pytz.timezone('America/Los_Angeles')
W = 120
BUCKET_NAME = os.environ.get("GCS_BUCKET", "")
OHLCV_DIR = "Data/Inference"
FOCUS_TICKERS = ['FCEL', 'LUNR', 'SMCI', 'RKLB', 'PL']
PNL_BARS = 30  # 30-min window for PnL computation

def _release_memory():
    gc.collect()
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

# ==============================================================================
# STEP 1: QUERY BQ SCORING LOGS
# ==============================================================================
print(f"{'='*W}")
print(f"  STEP 1: LOADING BQ SCORING LOGS")
print(f"{'='*W}")

bq_client = bigquery.Client()

query = """
SELECT score_date, ticker, experiment, prob, Threshold as threshold, Signal as signal,
       Filter_Status as filter_status, close, tpsl_mode, seed
FROM `{GCP_PROJECT}.scoring_logs.live_predictions`
WHERE score_date >= '2026-05-01'
ORDER BY score_date, ticker
"""

df_raw = bq_client.query(query).to_dataframe()
print(f"  Raw rows from BQ: {len(df_raw):,}")

# Convert score_date to string for dedup
df_raw['score_date'] = pd.to_datetime(df_raw['score_date']).dt.strftime('%Y-%m-%d')

# Deduplicate on (score_date, ticker), keep='first'
df = df_raw.drop_duplicates(subset=['score_date', 'ticker'], keep='first').copy()
print(f"  After dedup (score_date, ticker): {len(df):,} rows")
print(f"  Date range: {df['score_date'].min()} to {df['score_date'].max()}")
print(f"  Unique tickers: {df['ticker'].nunique()}")
print(f"  Unique experiments: {sorted(df['experiment'].unique())}")
print(f"  Signal distribution:\n{df['signal'].value_counts().to_string()}")


# ==============================================================================
# STEP 2: PROBABILITY DISTRIBUTION OVER TIME
# ==============================================================================
print(f"\n{'='*W}")
print(f"  STEP 2: PROBABILITY DISTRIBUTION OVER TIME")
print(f"{'='*W}")

# Daily stats — all signals
daily_all = df.groupby('score_date').agg(
    n_tickers=('ticker', 'nunique'),
    avg_prob=('prob', 'mean'),
    median_prob=('prob', 'median'),
    max_prob=('prob', 'max'),
    min_prob=('prob', 'min'),
    std_prob=('prob', 'std'),
).reset_index()

# Daily stats — GO signals only
go_df = df[df['signal'] == 'GO'].copy()
daily_go = go_df.groupby('score_date').agg(
    n_go=('ticker', 'nunique'),
    avg_prob_go=('prob', 'mean'),
    median_prob_go=('prob', 'median'),
    go_tickers=('ticker', lambda x: ', '.join(sorted(x.unique()))),
).reset_index()

# Merge
daily_merged = daily_all.merge(daily_go, on='score_date', how='left')
daily_merged['n_go'] = daily_merged['n_go'].fillna(0).astype(int)
daily_merged['avg_prob_go'] = daily_merged['avg_prob_go'].fillna(0)

print(f"\n  {'Date':<12} {'Tickers':>7} {'AvgProb':>8} {'MedProb':>8} {'MaxProb':>8} "
      f"{'StdProb':>8} {'#GO':>4} {'AvgProbGO':>10} {'GO Tickers'}")
print(f"  {'─'*12} {'─'*7} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*4} {'─'*10} {'─'*40}")

for _, r in daily_merged.iterrows():
    go_tix = r.get('go_tickers', '')
    if pd.isna(go_tix):
        go_tix = ''
    print(f"  {r['score_date']:<12} {r['n_tickers']:>7} {r['avg_prob']:>8.4f} "
          f"{r['median_prob']:>8.4f} {r['max_prob']:>8.4f} {r['std_prob']:>8.4f} "
          f"{r['n_go']:>4} {r['avg_prob_go']:>10.4f} {go_tix}")

# Trend analysis
dates = pd.to_datetime(daily_merged['score_date'])
day_nums = (dates - dates.min()).dt.days.values
avg_probs = daily_merged['avg_prob'].values

trend = "UNKNOWN"
trend_go = "UNKNOWN"
slope = 0
slope_go = 0
p_val = 1.0

if len(day_nums) > 3:
    slope, intercept, r_val, p_val, _ = sp_stats.linregress(day_nums, avg_probs)
    trend = ("DECLINING" if slope < -0.001 and p_val < 0.15 else
             "RISING" if slope > 0.001 and p_val < 0.15 else "STABLE")
    print(f"\n  TREND (avg_prob vs days): slope={slope*1000:+.3f}/1000d, "
          f"R²={r_val**2:.3f}, p={p_val:.3f} → {trend}")
else:
    print(f"\n  Not enough days for trend analysis")

# GO signal trend
go_counts = daily_merged['n_go'].values
if len(day_nums) > 3:
    slope_go, _, r_go, p_go, _ = sp_stats.linregress(day_nums, go_counts)
    trend_go = ("DECLINING" if slope_go < -0.05 and p_go < 0.15 else
                "RISING" if slope_go > 0.05 and p_go < 0.15 else "STABLE")
    print(f"  TREND (#GO/day vs days): slope={slope_go:+.3f}/day, "
          f"R²={r_go**2:.3f}, p={p_go:.3f} → {trend_go}")


# ==============================================================================
# STEP 3: PER-TICKER PROB TRAJECTORY
# ==============================================================================
print(f"\n{'='*W}")
print(f"  STEP 3: PER-TICKER PROBABILITY TRAJECTORY")
print(f"  Focus tickers: {', '.join(FOCUS_TICKERS)}")
print(f"{'='*W}")

for ticker in FOCUS_TICKERS:
    tdf = df[df['ticker'] == ticker].sort_values('score_date').copy()
    if tdf.empty:
        print(f"\n  {ticker}: NO DATA in scoring logs")
        continue

    n_days = len(tdf)
    n_go = (tdf['signal'] == 'GO').sum()
    n_no = (tdf['signal'] == 'NO').sum()
    avg_p = tdf['prob'].mean()
    first_p = tdf['prob'].iloc[0]
    last_p = tdf['prob'].iloc[-1]
    max_p = tdf['prob'].max()
    min_p = tdf['prob'].min()

    print(f"\n  {ticker} — {n_days} scoring days, {n_go} GO / {n_no} NO")
    print(f"  {'Date':<12} {'Prob':>6} {'Signal':>6} {'Threshold':>9} {'Filter':>10}")
    print(f"  {'─'*12} {'─'*6} {'─'*6} {'─'*9} {'─'*10}")

    for _, r in tdf.iterrows():
        filter_str = str(r['filter_status']) if pd.notna(r['filter_status']) else ''
        print(f"  {r['score_date']:<12} {r['prob']:>6.4f} {r['signal']:>6} "
              f"{r['threshold']:>9.3f} {filter_str:>10}")

    # Per-ticker trend
    if n_days >= 5:
        t_dates = pd.to_datetime(tdf['score_date'])
        t_days = (t_dates - t_dates.min()).dt.days.values
        t_probs = tdf['prob'].values
        t_slope, _, t_r, t_p, _ = sp_stats.linregress(t_days, t_probs)
        t_trend = ("DECLINING" if t_slope < -0.001 and t_p < 0.2 else
                   "RISING" if t_slope > 0.001 and t_p < 0.2 else "STABLE")
        print(f"  Trend: slope={t_slope*1000:+.3f}/1000d, p={t_p:.3f} → {t_trend}")
        print(f"  Range: {min_p:.4f} - {max_p:.4f}, First→Last: {first_p:.4f}→{last_p:.4f}")
    else:
        print(f"  Range: {min_p:.4f} - {max_p:.4f}, First→Last: {first_p:.4f}→{last_p:.4f}")

# All tickers summary
print(f"\n  ALL TICKERS — Avg Prob and GO Rate:")
print(f"  {'Ticker':<7} {'Days':>5} {'AvgProb':>8} {'MinProb':>8} {'MaxProb':>8} "
      f"{'#GO':>4} {'GO%':>6} {'Trend':>10}")
print(f"  {'─'*7} {'─'*5} {'─'*8} {'─'*8} {'─'*8} {'─'*4} {'─'*6} {'─'*10}")

ticker_summaries = []
for ticker in sorted(df['ticker'].unique()):
    tdf = df[df['ticker'] == ticker].sort_values('score_date')
    n = len(tdf)
    n_go = (tdf['signal'] == 'GO').sum()
    avg_p = tdf['prob'].mean()
    min_p = tdf['prob'].min()
    max_p = tdf['prob'].max()
    go_pct = n_go / n * 100 if n > 0 else 0

    trend_str = ''
    if n >= 5:
        t_dates = pd.to_datetime(tdf['score_date'])
        t_days = (t_dates - t_dates.min()).dt.days.values
        t_slope, _, _, t_p, _ = sp_stats.linregress(t_days, tdf['prob'].values)
        trend_str = ("DECLINING" if t_slope < -0.001 and t_p < 0.2 else
                     "RISING" if t_slope > 0.001 and t_p < 0.2 else "STABLE")

    ticker_summaries.append({
        'ticker': ticker, 'n': n, 'avg_prob': avg_p, 'min_prob': min_p,
        'max_prob': max_p, 'n_go': n_go, 'go_pct': go_pct, 'trend': trend_str
    })

    print(f"  {ticker:<7} {n:>5} {avg_p:>8.4f} {min_p:>8.4f} {max_p:>8.4f} "
          f"{n_go:>4} {go_pct:>5.1f}% {trend_str:>10}")


# ==============================================================================
# STEP 4: MAY vs JUNE SIGNAL DENSITY COMPARISON
# ==============================================================================
print(f"\n{'='*W}")
print(f"  STEP 4: SIGNAL DENSITY — MAY vs JUNE")
print(f"{'='*W}")

df['month'] = pd.to_datetime(df['score_date']).dt.to_period('M').astype(str)

for month in sorted(df['month'].unique()):
    mdf = df[df['month'] == month]
    mgo = mdf[mdf['signal'] == 'GO']
    n_days = mdf['score_date'].nunique()
    n_go = len(mgo)
    n_go_tickers = mgo['ticker'].nunique() if len(mgo) > 0 else 0
    avg_go_per_day = n_go / n_days if n_days > 0 else 0
    avg_prob_all = mdf['prob'].mean()
    avg_prob_go = mgo['prob'].mean() if len(mgo) > 0 else 0

    print(f"\n  {month}:")
    print(f"    Trading days scored: {n_days}")
    print(f"    Total GO signals:    {n_go}")
    print(f"    Unique GO tickers:   {n_go_tickers}")
    print(f"    Avg GO/day:          {avg_go_per_day:.1f}")
    print(f"    Avg prob (all):      {avg_prob_all:.4f}")
    print(f"    Avg prob (GO only):  {avg_prob_go:.4f}")

    if len(mgo) > 0:
        go_counts = mgo.groupby('ticker').size().sort_values(ascending=False)
        top_tickers = ', '.join(f"{t}({c})" for t, c in go_counts.head(10).items())
        print(f"    Top GO tickers:      {top_tickers}")

# Cross-month ticker overlap
months = sorted(df['month'].unique())
if len(months) >= 2:
    for i in range(len(months) - 1):
        m1, m2 = months[i], months[i + 1]
        go1 = set(df[(df['month'] == m1) & (df['signal'] == 'GO')]['ticker'].unique())
        go2 = set(df[(df['month'] == m2) & (df['signal'] == 'GO')]['ticker'].unique())
        overlap = go1 & go2
        only_m1 = go1 - go2
        only_m2 = go2 - go1
        print(f"\n  {m1} vs {m2} GO ticker overlap:")
        print(f"    Both months:    {sorted(overlap) if overlap else 'NONE'}")
        print(f"    Only {m1}: {sorted(only_m1) if only_m1 else 'NONE'}")
        print(f"    Only {m2}: {sorted(only_m2) if only_m2 else 'NONE'}")


# ==============================================================================
# STEP 5: PROB-TO-OUTCOME CORRELATION
# ==============================================================================
print(f"\n{'='*W}")
print(f"  STEP 5: PROB-TO-OUTCOME CORRELATION (GO signals only)")
print(f"  Loading OHLCV data to compute {PNL_BARS}-bar PnL after entry")
print(f"{'='*W}")

gcs_client = gcs_storage.Client()

go_signals = df[df['signal'] == 'GO'].copy()
go_signals['score_date_dt'] = pd.to_datetime(go_signals['score_date'])

if len(go_signals) == 0:
    print(f"\n  NO GO signals found — cannot compute PnL correlation")
else:
    pnl_results = []
    unique_go_tickers = sorted(go_signals['ticker'].unique())
    print(f"  GO signal tickers to process: {len(unique_go_tickers)}")

    TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

    for ticker in unique_go_tickers:
        file_ticker = TICKER_FILE_MAP.get(ticker, ticker)
        ohlcv_path = f"gs://{BUCKET_NAME}/{OHLCV_DIR}/{file_ticker}_1m_full_historical_data.parquet"

        try:
            ohlcv = pd.read_parquet(ohlcv_path)
        except Exception as e:
            print(f"    {ticker}: OHLCV load failed — {e}")
            continue

        if ohlcv.index.tz is None:
            ohlcv.index = ohlcv.index.tz_localize('UTC')
        ohlcv = ohlcv.sort_index()

        # Forward-fill OHLC, zero-fill volume
        for c in ['Open', 'High', 'Low', 'Close']:
            if c in ohlcv.columns:
                ohlcv[c] = ohlcv[c].ffill()
        if 'Volume' in ohlcv.columns:
            ohlcv['Volume'] = ohlcv['Volume'].fillna(0)

        ticker_signals = go_signals[go_signals['ticker'] == ticker]

        for _, sig in ticker_signals.iterrows():
            score_date = sig['score_date_dt']
            sig_close = sig['close'] if pd.notna(sig['close']) else None

            # Find bars on the signal date
            day_mask = ohlcv.index.tz_convert(PST).date == score_date.date()
            day_bars = ohlcv[day_mask]

            if len(day_bars) == 0:
                continue

            # Find the entry bar: first bar at or after 6:30 AM PST with Volume > 0
            day_pst = day_bars.index.tz_convert(PST)
            day_mins = day_pst.hour * 60 + day_pst.minute
            entry_candidates = day_bars[(day_mins >= 390) & (day_bars['Volume'] > 0)]  # 6:30 AM = 390

            if len(entry_candidates) == 0:
                # Fallback: use market open 6:30 AM regardless
                entry_candidates = day_bars[day_mins >= 390]
                if len(entry_candidates) == 0:
                    continue

            entry_bar_idx = entry_candidates.index[0]
            entry_price = ohlcv.loc[entry_bar_idx, 'Close']

            # Get +PNL_BARS bars after entry
            entry_pos = ohlcv.index.get_loc(entry_bar_idx)
            if entry_pos + PNL_BARS >= len(ohlcv):
                continue

            exit_bar_idx = ohlcv.index[entry_pos + PNL_BARS]
            exit_price = ohlcv.loc[exit_bar_idx, 'Close']

            pnl_pct = (exit_price / entry_price - 1) * 100

            # Also compute max favorable excursion (MFE) in the window
            window = ohlcv.iloc[entry_pos + 1: entry_pos + PNL_BARS + 1]
            if 'High' in window.columns and len(window) > 0:
                max_high = window['High'].max()
                mfe_pct = (max_high / entry_price - 1) * 100
            else:
                mfe_pct = pnl_pct

            pnl_results.append({
                'ticker': ticker,
                'score_date': sig['score_date'],
                'prob': sig['prob'],
                'threshold': sig['threshold'],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'mfe_pct': mfe_pct,
                'experiment': sig['experiment'],
            })

        del ohlcv
        _release_memory()
        print(f"    {ticker}: {len(ticker_signals)} GO signals processed")

    if pnl_results:
        pnl_df = pd.DataFrame(pnl_results)
        print(f"\n  Total GO signals with PnL: {len(pnl_df)}")

        # Overall stats
        print(f"\n  OVERALL GO SIGNAL PnL STATS:")
        print(f"    Mean PnL:    {pnl_df['pnl_pct'].mean():+.3f}%")
        print(f"    Median PnL:  {pnl_df['pnl_pct'].median():+.3f}%")
        print(f"    Std PnL:     {pnl_df['pnl_pct'].std():.3f}%")
        print(f"    Win Rate:    {(pnl_df['pnl_pct'] > 0).mean():.1%}")
        print(f"    Mean MFE:    {pnl_df['mfe_pct'].mean():+.3f}%")
        print(f"    Median MFE:  {pnl_df['mfe_pct'].median():+.3f}%")

        # Prob-PnL correlation
        if len(pnl_df) > 5:
            corr_pnl, p_corr_pnl = sp_stats.pearsonr(pnl_df['prob'], pnl_df['pnl_pct'])
            corr_mfe, p_corr_mfe = sp_stats.pearsonr(pnl_df['prob'], pnl_df['mfe_pct'])
            print(f"\n  PROB-TO-OUTCOME CORRELATION:")
            print(f"    Prob vs PnL:   r={corr_pnl:+.3f}, p={p_corr_pnl:.3f}")
            print(f"    Prob vs MFE:   r={corr_mfe:+.3f}, p={p_corr_mfe:.3f}")

            if abs(corr_pnl) > 0.3 and p_corr_pnl < 0.1:
                print(f"    → SIGNIFICANT correlation: higher prob {'= better' if corr_pnl > 0 else '= worse'} outcomes")
            elif abs(corr_pnl) > 0.15:
                print(f"    → WEAK correlation detected (not statistically significant)")
            else:
                print(f"    → NO meaningful correlation between prob and PnL")

        # Prob quintiles
        if len(pnl_df) >= 10:
            pnl_df['prob_bin'] = pd.qcut(pnl_df['prob'], q=min(5, len(pnl_df) // 3),
                                          duplicates='drop')
            print(f"\n  PnL BY PROB QUINTILE:")
            print(f"  {'Prob Range':<22} {'N':>4} {'AvgPnL':>8} {'WinRate':>8} {'AvgMFE':>8}")
            print(f"  {'─'*22} {'─'*4} {'─'*8} {'─'*8} {'─'*8}")

            for name, grp in pnl_df.groupby('prob_bin', observed=True):
                avg_pnl = grp['pnl_pct'].mean()
                wr = (grp['pnl_pct'] > 0).mean()
                avg_mfe = grp['mfe_pct'].mean()
                print(f"  {str(name):<22} {len(grp):>4} {avg_pnl:>+7.3f}% {wr:>7.0%} {avg_mfe:>+7.3f}%")

        # Per-ticker PnL
        print(f"\n  PER-TICKER GO SIGNAL PnL:")
        print(f"  {'Ticker':<7} {'N':>4} {'AvgProb':>8} {'AvgPnL':>8} {'WinRate':>8} {'AvgMFE':>8}")
        print(f"  {'─'*7} {'─'*4} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

        for ticker in sorted(pnl_df['ticker'].unique()):
            tdf = pnl_df[pnl_df['ticker'] == ticker]
            print(f"  {ticker:<7} {len(tdf):>4} {tdf['prob'].mean():>8.4f} "
                  f"{tdf['pnl_pct'].mean():>+7.3f}% "
                  f"{(tdf['pnl_pct'] > 0).mean():>7.0%} "
                  f"{tdf['mfe_pct'].mean():>+7.3f}%")

        # PnL over time
        pnl_df['week'] = pd.to_datetime(pnl_df['score_date']).dt.isocalendar().week.astype(int)
        print(f"\n  WEEKLY PnL TREND:")
        print(f"  {'Week':>6} {'N':>4} {'AvgPnL':>8} {'WinRate':>8} {'AvgProb':>8}")
        print(f"  {'─'*6} {'─'*4} {'─'*8} {'─'*8} {'─'*8}")

        for week, wdf in pnl_df.groupby('week'):
            print(f"  W{week:>4} {len(wdf):>4} {wdf['pnl_pct'].mean():>+7.3f}% "
                  f"{(wdf['pnl_pct'] > 0).mean():>7.0%} {wdf['prob'].mean():>8.4f}")

    else:
        print(f"\n  No PnL results computed (OHLCV data missing or no matching bars)")


# ==============================================================================
# SUMMARY REPORT
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SUMMARY — MODEL PREDICTION QUALITY ASSESSMENT")
print(f"{'='*W}")

print(f"\n  1. PROBABILITY DISTRIBUTION:")
overall_avg = df['prob'].mean()
overall_std = df['prob'].std()
go_avg = go_df['prob'].mean() if len(go_df) > 0 else 0
print(f"     Overall avg prob: {overall_avg:.4f} (std: {overall_std:.4f})")
print(f"     GO signal avg prob: {go_avg:.4f}")
if len(day_nums) > 3:
    print(f"     Trend: {trend} (slope={slope*1000:+.3f}/1000d, p={p_val:.3f})")

print(f"\n  2. SIGNAL DENSITY:")
total_go = (df['signal'] == 'GO').sum()
total_days = df['score_date'].nunique()
go_per_day = total_go / total_days if total_days > 0 else 0
print(f"     Total GO signals: {total_go} over {total_days} days ({go_per_day:.1f}/day)")
if len(day_nums) > 3:
    print(f"     GO density trend: {trend_go} (slope={slope_go:+.3f}/day)")

print(f"\n  3. PER-TICKER ASSESSMENT:")
n_declining = sum(1 for s in ticker_summaries if s['trend'] == 'DECLINING')
n_stable = sum(1 for s in ticker_summaries if s['trend'] == 'STABLE')
n_rising = sum(1 for s in ticker_summaries if s['trend'] == 'RISING')
print(f"     Declining: {n_declining}  Stable: {n_stable}  Rising: {n_rising}")

declining_tickers = [s['ticker'] for s in ticker_summaries if s['trend'] == 'DECLINING']
if declining_tickers:
    print(f"     Declining tickers: {', '.join(declining_tickers)}")

print(f"\n  4. PROB-OUTCOME CORRELATION:")
if pnl_results and len(pnl_df) > 5:
    print(f"     Prob vs PnL: r={corr_pnl:+.3f} (p={p_corr_pnl:.3f})")
    print(f"     Prob vs MFE: r={corr_mfe:+.3f} (p={p_corr_mfe:.3f})")
    if abs(corr_pnl) < 0.15:
        print(f"     → Model probability does NOT predict trade outcome magnitude")
        print(f"       (prob is a threshold signal, not a PnL predictor)")
    elif corr_pnl > 0.15:
        print(f"     → Higher prob → better outcomes (model calibration is useful)")
    else:
        print(f"     → Higher prob → WORSE outcomes (possible overfitting)")
elif pnl_results:
    print(f"     Too few GO signals with PnL data ({len(pnl_df)}) for correlation analysis")
else:
    print(f"     Could not compute (no OHLCV data matched)")

print(f"\n  5. DEGRADATION VERDICT:")
degradation_signals = 0
reasons = []

if len(day_nums) > 3 and trend == "DECLINING":
    degradation_signals += 1
    reasons.append(f"avg prob trending down (slope={slope*1000:+.3f})")
if len(day_nums) > 3 and trend_go == "DECLINING":
    degradation_signals += 1
    reasons.append(f"GO signal density declining (slope={slope_go:+.3f})")
if n_declining > n_stable + n_rising:
    degradation_signals += 1
    reasons.append(f"more tickers declining ({n_declining}) than stable/rising ({n_stable+n_rising})")
if pnl_results:
    mean_pnl = pnl_df['pnl_pct'].mean()
    if mean_pnl < 0:
        degradation_signals += 1
        reasons.append(f"avg GO signal PnL is negative ({mean_pnl:+.3f}%)")

if degradation_signals >= 3:
    print(f"     STRONG EVIDENCE of model degradation:")
elif degradation_signals >= 2:
    print(f"     MODERATE EVIDENCE of model degradation:")
elif degradation_signals >= 1:
    print(f"     WEAK EVIDENCE of possible degradation:")
else:
    print(f"     NO CLEAR EVIDENCE of model degradation")

for r in reasons:
    print(f"       - {r}")

if degradation_signals == 0:
    print(f"     Model probabilities appear stable. If live PnL is poor,")
    print(f"     investigate execution (slippage, entry timing) rather than model quality.")

print(f"\n{'='*W}")
