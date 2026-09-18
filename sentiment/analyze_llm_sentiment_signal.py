# %% ── LLM Sentiment Features vs Price Change — Signal Analysis ────────────────
#
# Tests whether LLM-extracted news features have predictive signal for
# intraday price moves at the deployed entry time (6:25 AM PT).
#
# Uses 24-hour lookback window ending at 6:00 AM PT.
# Aggregates per-article LLM scores into daily ticker features,
# then correlates with forward returns.
#
# Reads:  gs://{GCS_BUCKET}/Data/Sentiment/{ticker}_news.parquet
#         gs://{GCS_BUCKET}/Data/Sentiment/{ticker}_llm_scores.parquet
#         gs://{GCS_BUCKET}/Data/Inference_v2/{ticker}_master_inference.parquet
#
# Colab-ready, config vars at top.

import os, sys, gc, ctypes, warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import pytz
from scipy import stats as sp_stats

warnings.filterwarnings('ignore')

try:
    from google.cloud import storage as gcs_storage
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "google-cloud-storage", "pyarrow", "--quiet"])
    from google.cloud import storage as gcs_storage

try:
    from google.colab import auth
    _IN_COLAB = True
    try:
        auth.authenticate_user()
    except Exception:
        pass
except (ImportError, Exception):
    _IN_COLAB = False

import matplotlib
if not _IN_COLAB:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ==============================================================================
# CONFIG
# ==============================================================================
BUCKET_NAME   = os.environ.get("GCS_BUCKET", "")
SENTIMENT_DIR = "Data/Sentiment"
INFERENCE_DIR = "Data/Inference_v2"

PST = pytz.timezone('America/Los_Angeles')

ANALYSIS_START = '2025-01-01'
ANALYSIS_END   = '2026-08-31'

ENTRY_MINUTE   = 385
EXIT_TIME_MIN  = 780
LOOKBACK_HOURS = 24
RECENCY_HALFLIFE_H = 6.0

ANALYSIS_TICKERS = [
    'TSLA', 'AMD', 'PLTR', 'MU', 'CVX', 'RKLB',
    'SOUN', 'COIN', 'HOOD', 'OKLO', 'SMCI', 'NBIS',
]

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

LLM_SCORE_COLS = [
    'sentiment_score', 'price_impact', 'surprise_factor',
    'information_quality', 'retail_fomo_score', 'volatility_catalyst',
    'urgency', 'forward_looking', 'overreaction_risk', 'premarket_driver',
]

DAILY_FEATURE_COLS = [
    # Direct LLM aggregates (mean across articles in window)
    'llm_sentiment', 'llm_impact', 'llm_surprise', 'llm_quality',
    'llm_fomo', 'llm_volatility', 'llm_urgency', 'llm_forward',
    'llm_overreaction', 'llm_premarket',
    # Derived / composite
    'llm_catalyst_ratio', 'llm_n_articles',
    'llm_sentiment_x_impact', 'llm_surprise_x_impact',
    'llm_fomo_x_urgency', 'llm_contrarian',
    'llm_recency_sentiment', 'llm_max_impact',
    # Event type signals
    'llm_has_earnings', 'llm_has_analyst', 'llm_has_offering',
]

gcs_client = gcs_storage.Client()
bucket = gcs_client.bucket(BUCKET_NAME)

W = 100


# ==============================================================================
# DATA LOADING
# ==============================================================================
def _download_parquet(blob_path):
    local = f"/tmp/_llma_{os.path.basename(blob_path)}"
    try:
        bucket.blob(blob_path).download_to_filename(local)
        df = pd.read_parquet(local, engine='pyarrow')
        os.remove(local)
        return df
    except Exception:
        if os.path.exists(local):
            os.remove(local)
        return None


def load_news_with_scores(ticker):
    news = _download_parquet(f"{SENTIMENT_DIR}/{ticker}_news.parquet")
    scores = _download_parquet(f"{SENTIMENT_DIR}/{ticker}_llm_scores.parquet")
    if news is None or scores is None:
        return None
    merged = news.merge(scores, on='article_id', how='inner')
    if merged.empty:
        return None
    merged['published_utc'] = pd.to_datetime(merged['published_utc'], utc=True)
    return merged


def load_prices(ticker):
    file_ticker = TICKER_FILE_MAP.get(ticker, ticker)
    df = _download_parquet(f"{INFERENCE_DIR}/{file_ticker}_master_inference.parquet")
    if df is None or df.empty:
        return None
    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if dt_cols:
        df['Datetime'] = pd.to_datetime(df[dt_cols[0]], utc=True)
    else:
        num_col = next((c for c in df.columns
                       if c in ('Datetime', 'Date', 'Time', 'Timestamp', 'Level_0')),
                      df.columns[0])
        df['Datetime'] = pd.to_datetime(df[num_col], unit='ms', utc=True)
    df = df.set_index('Datetime').sort_index()
    df = df[~df.index.duplicated(keep='first')]
    col_map = {'open': 'Open', 'high': 'High', 'low': 'Low',
               'close': 'Close', 'volume': 'Volume'}
    df.columns = [col_map.get(str(c).strip().lower(), str(c).strip()) for c in df.columns]
    for c in ['Open', 'High', 'Low', 'Close']:
        if c in df.columns:
            df[c] = df[c].ffill()
    return df


# ==============================================================================
# AGGREGATE LLM SCORES INTO DAILY FEATURES
# ==============================================================================
def compute_daily_llm_features(scored_df, trading_dates):
    if scored_df is None or scored_df.empty:
        return {d: {col: 0.0 for col in DAILY_FEATURE_COLS + ['has_news']}
                for d in trading_dates}

    daily = {}
    for date_str in trading_dates:
        date = pd.Timestamp(date_str, tz='UTC')
        cutoff_end_pst = date.tz_convert(PST).replace(hour=6, minute=0, second=0)
        cutoff_start_pst = cutoff_end_pst - pd.Timedelta(hours=LOOKBACK_HOURS)
        cutoff_end_utc = cutoff_end_pst.tz_convert('UTC')
        cutoff_start_utc = cutoff_start_pst.tz_convert('UTC')

        mask = ((scored_df['published_utc'] >= cutoff_start_utc) &
                (scored_df['published_utc'] < cutoff_end_utc))
        window = scored_df[mask]

        if window.empty:
            daily[date_str] = {col: 0.0 for col in DAILY_FEATURE_COLS}
            daily[date_str]['has_news'] = False
            continue

        n = len(window)
        feats = {}

        # Direct aggregates (mean)
        col_map = {
            'sentiment_score': 'llm_sentiment',
            'price_impact': 'llm_impact',
            'surprise_factor': 'llm_surprise',
            'information_quality': 'llm_quality',
            'retail_fomo_score': 'llm_fomo',
            'volatility_catalyst': 'llm_volatility',
            'urgency': 'llm_urgency',
            'forward_looking': 'llm_forward',
            'overreaction_risk': 'llm_overreaction',
            'premarket_driver': 'llm_premarket',
        }
        for src, dst in col_map.items():
            if src in window.columns:
                feats[dst] = float(window[src].mean())
            else:
                feats[dst] = 0.0

        # Catalyst ratio
        if 'is_catalyst' in window.columns:
            feats['llm_catalyst_ratio'] = float(window['is_catalyst'].mean())
        else:
            feats['llm_catalyst_ratio'] = 0.0

        feats['llm_n_articles'] = float(n)

        # Interaction features
        sent = window.get('sentiment_score', pd.Series([0.0])).values
        impact = window.get('price_impact', pd.Series([0.0])).values

        # sentiment × impact: directional signal weighted by importance
        feats['llm_sentiment_x_impact'] = float(np.mean(sent * impact))

        # surprise × impact: unexpected + high impact = biggest movers
        surprise = window.get('surprise_factor', pd.Series([0.0])).values
        feats['llm_surprise_x_impact'] = float(np.mean(surprise * impact))

        # fomo × urgency: viral breaking news
        fomo = window.get('retail_fomo_score', pd.Series([0.0])).values
        urg = window.get('urgency', pd.Series([0.0])).values
        feats['llm_fomo_x_urgency'] = float(np.mean(fomo * urg))

        # Contrarian: negative sentiment + high overreaction risk = buy signal?
        overreact = window.get('overreaction_risk', pd.Series([0.0])).values
        feats['llm_contrarian'] = float(np.mean(-sent * overreact))

        # Recency-weighted sentiment (exp decay, 6h half-life)
        hours_ago = (cutoff_end_utc - window['published_utc']).dt.total_seconds() / 3600.0
        decay = np.exp(-np.log(2) * hours_ago.values / RECENCY_HALFLIFE_H)
        feats['llm_recency_sentiment'] = float(np.sum(sent * decay) / (np.sum(decay) + 1e-9))

        # Max impact article
        feats['llm_max_impact'] = float(impact.max()) if len(impact) > 0 else 0.0

        # Event type flags
        if 'event_type' in window.columns:
            et = window['event_type'].values
            feats['llm_has_earnings'] = float(any(e == 'earnings' for e in et))
            feats['llm_has_analyst'] = float(any(e == 'analyst_rating' for e in et))
            feats['llm_has_offering'] = float(any(e == 'offering_dilution' for e in et))
        else:
            feats['llm_has_earnings'] = 0.0
            feats['llm_has_analyst'] = 0.0
            feats['llm_has_offering'] = 0.0

        feats['has_news'] = True
        daily[date_str] = feats

    return daily


# ==============================================================================
# FORWARD RETURNS
# ==============================================================================
def compute_forward_returns(price_df, trading_dates):
    if price_df is None or price_df.empty:
        return {}

    price_pst = price_df.index.tz_convert(PST)
    mins = price_pst.hour * 60 + price_pst.minute
    date_ints = price_pst.year * 10000 + price_pst.month * 100 + price_pst.day

    daily = {}
    for date_str in trading_dates:
        dt = pd.Timestamp(date_str)
        target_date_int = dt.year * 10000 + dt.month * 100 + dt.day

        day_mask = date_ints == target_date_int
        if not day_mask.any():
            continue

        entry_mask = day_mask & (mins == ENTRY_MINUTE)
        if not entry_mask.any():
            continue

        entry_idx = np.where(entry_mask)[0][0]
        entry_price = float(price_df['Close'].iloc[entry_idx])
        if entry_price <= 0 or np.isnan(entry_price):
            continue

        day_indices = np.where(day_mask)[0]
        after_entry = day_indices[day_indices > entry_idx]
        if len(after_entry) == 0:
            continue

        result = {'entry_price': entry_price}

        for horizon_min, label in [(15, 'ret_15m'), (30, 'ret_30m'), (60, 'ret_60m')]:
            target_min = ENTRY_MINUTE + horizon_min
            horizon_mask = day_mask & (mins >= ENTRY_MINUTE) & (mins <= target_min)
            h_indices = np.where(horizon_mask)[0]

            if len(h_indices) > 1:
                window_highs = price_df['High'].iloc[h_indices[1:]].values
                window_lows = price_df['Low'].iloc[h_indices[1:]].values
                end_close = float(price_df['Close'].iloc[h_indices[-1]])
                result[label] = (end_close - entry_price) / entry_price
                result[f'max_up_{label}'] = (float(np.max(window_highs)) - entry_price) / entry_price
                result[f'max_dn_{label}'] = (float(np.min(window_lows)) - entry_price) / entry_price
            else:
                result[label] = np.nan
                result[f'max_up_{label}'] = np.nan
                result[f'max_dn_{label}'] = np.nan

        eod_mask = day_mask & (mins >= EXIT_TIME_MIN)
        eod_indices = np.where(eod_mask)[0]
        if len(eod_indices) > 0:
            eod_price = float(price_df['Close'].iloc[eod_indices[0]])
        else:
            eod_price = float(price_df['Close'].iloc[day_indices[-1]])
        result['ret_eod'] = (eod_price - entry_price) / entry_price

        all_after = day_indices[day_indices >= entry_idx]
        if len(all_after) > 1:
            result['max_up_day'] = (float(np.max(price_df['High'].iloc[all_after[1:]].values)) - entry_price) / entry_price
            result['max_dn_day'] = (float(np.min(price_df['Low'].iloc[all_after[1:]].values)) - entry_price) / entry_price

        daily[date_str] = result

    return daily


# ==============================================================================
# MAIN — COLLECT DATA
# ==============================================================================
print(f"\n{'='*W}")
print(f"  LLM SENTIMENT FEATURES vs PRICE CHANGE — SIGNAL ANALYSIS")
print(f"  {ANALYSIS_START} -> {ANALYSIS_END}")
print(f"  Entry: {ENTRY_MINUTE//60}:{ENTRY_MINUTE%60:02d} PT | Lookback: {LOOKBACK_HOURS}h")
print(f"  Features: {len(DAILY_FEATURE_COLS)} daily LLM features per ticker-day")
print(f"  Tickers: {len(ANALYSIS_TICKERS)} — {', '.join(ANALYSIS_TICKERS)}")
print(f"{'='*W}")

all_rows = []
coverage_stats = []

for ti, ticker in enumerate(ANALYSIS_TICKERS):
    print(f"  [{ti+1}/{len(ANALYSIS_TICKERS)}] {ticker}...", end=' ')

    scored_df = load_news_with_scores(ticker)
    price_df = load_prices(ticker)

    if price_df is None:
        print("no price data")
        continue

    if scored_df is None:
        print("no LLM scores")
        continue

    price_pst = price_df.index.tz_convert(PST)
    date_ints = price_pst.year * 10000 + price_pst.month * 100 + price_pst.day
    mins = price_pst.hour * 60 + price_pst.minute

    analysis_start_int = int(ANALYSIS_START.replace('-', ''))
    analysis_end_int = int(ANALYSIS_END.replace('-', ''))

    entry_bars = ((mins == ENTRY_MINUTE) &
                  (date_ints >= analysis_start_int) &
                  (date_ints <= analysis_end_int))
    if not entry_bars.any():
        print("no entry bars")
        continue

    trading_date_ints = sorted(set(date_ints[entry_bars]))
    trading_dates = [f"{d//10000}-{(d%10000)//100:02d}-{d%100:02d}" for d in trading_date_ints]

    daily_feats = compute_daily_llm_features(scored_df, trading_dates)
    daily_rets = compute_forward_returns(price_df, trading_dates)

    del price_df, scored_df
    gc.collect()

    n_with_news = 0
    for date_str in trading_dates:
        if date_str not in daily_rets:
            continue
        feats = daily_feats.get(date_str, {})
        rets = daily_rets[date_str]

        if feats.get('has_news', False):
            n_with_news += 1

        row = {
            'ticker': ticker,
            'date': date_str,
            **{k: feats.get(k, 0.0) for k in DAILY_FEATURE_COLS},
            'has_news': feats.get('has_news', False),
            **{k: v for k, v in rets.items() if k != 'entry_price'},
        }
        all_rows.append(row)

    coverage = n_with_news / len(trading_dates) * 100 if trading_dates else 0
    coverage_stats.append({'ticker': ticker, 'days': len(trading_dates),
                           'days_with_news': n_with_news, 'coverage_pct': coverage})
    print(f"{len(trading_dates)} days, {n_with_news} with LLM scores ({coverage:.0f}%)")

if not all_rows:
    print("\n  NO DATA — run score_news_llm.py first.")
    sys.exit(0)

df = pd.DataFrame(all_rows)
df_news = df[df['has_news']].copy()
df_no_news = df[~df['has_news']].copy()

total_days = sum(c['days'] for c in coverage_stats)
total_with_news = sum(c['days_with_news'] for c in coverage_stats)


# ==============================================================================
# AGGREGATE FEATURE CORRELATION
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  FEATURE CORRELATION — LLM features vs forward returns")
print(f"  {total_with_news:,}/{total_days:,} ticker-days with LLM-scored news ({total_with_news/total_days*100:.1f}%)")
print(f"{'='*W}")

ret_targets = ['ret_15m', 'ret_30m', 'ret_60m', 'ret_eod',
               'max_up_ret_60m', 'max_dn_ret_60m']

print(f"\n  {'Feature':<24}", end='')
for rt in ret_targets:
    print(f" {rt:>14}", end='')
print()
print(f"  {'─'*24}", end='')
for _ in ret_targets:
    print(f" {'─'*14}", end='')
print()

feature_results = []
for feat in DAILY_FEATURE_COLS:
    print(f"  {feat:<24}", end='')
    feat_row = {'feature': feat}
    for rt in ret_targets:
        valid = df_news[[feat, rt]].dropna()
        if len(valid) < 20 or valid[feat].std() < 1e-6:
            print(f" {'N/A':>14}", end='')
            continue
        rho, pval = sp_stats.spearmanr(valid[feat], valid[rt])
        sig = '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        print(f" {rho:>+.4f}{sig:<2}p{pval:.3f}"[:14].rjust(14), end='')
        feat_row[f'rho_{rt}'] = rho
        feat_row[f'p_{rt}'] = pval
    print()
    feature_results.append(feat_row)

fr_df = pd.DataFrame(feature_results)


# ==============================================================================
# CONDITIONAL RETURNS — by LLM feature buckets
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  CONDITIONAL RETURNS — by LLM feature buckets")
print(f"{'='*W}")

print(f"\n  {'Group':<26} {'N':>6} {'ret_15m':>9} {'ret_30m':>9} {'ret_60m':>9} "
      f"{'max_up_60':>10} {'max_dn_60':>10} {'ret_eod':>9}")
print(f"  {'─'*90}")

buckets = [
    # Sentiment direction
    ('Bearish (<-0.3)', df_news[df_news['llm_sentiment'] < -0.3]),
    ('Mild Bear (-0.3,0)', df_news[(df_news['llm_sentiment'] >= -0.3) & (df_news['llm_sentiment'] < 0)]),
    ('Mild Bull (0,+0.3)', df_news[(df_news['llm_sentiment'] >= 0) & (df_news['llm_sentiment'] < 0.3)]),
    ('Bullish (>+0.3)', df_news[df_news['llm_sentiment'] >= 0.3]),

    # Impact level
    ('Low Impact (<3)', df_news[df_news['llm_impact'] < 3]),
    ('Med Impact (3-5)', df_news[(df_news['llm_impact'] >= 3) & (df_news['llm_impact'] < 5)]),
    ('High Impact (>=5)', df_news[df_news['llm_impact'] >= 5]),

    # Catalyst vs non-catalyst
    ('Catalyst >50%', df_news[df_news['llm_catalyst_ratio'] > 0.5]),
    ('Non-catalyst', df_news[df_news['llm_catalyst_ratio'] <= 0.1]),

    # Surprise
    ('High Surprise (>=5)', df_news[df_news['llm_surprise'] >= 5]),
    ('Low Surprise (<3)', df_news[df_news['llm_surprise'] < 3]),

    # FOMO
    ('High FOMO (>=5)', df_news[df_news['llm_fomo'] >= 5]),
    ('Low FOMO (<3)', df_news[df_news['llm_fomo'] < 3]),

    # Interaction: bullish catalyst surprise
    ('Bull+Catal+Surprise', df_news[(df_news['llm_sentiment'] > 0.2) &
                                     (df_news['llm_catalyst_ratio'] > 0.5) &
                                     (df_news['llm_surprise'] >= 4)]),
    ('Bear+Catal+Surprise', df_news[(df_news['llm_sentiment'] < -0.2) &
                                     (df_news['llm_catalyst_ratio'] > 0.5) &
                                     (df_news['llm_surprise'] >= 4)]),

    # Contrarian: bearish + high overreaction
    ('Contrarian Long', df_news[(df_news['llm_sentiment'] < -0.2) &
                                 (df_news['llm_overreaction'] >= 5)]),
    ('Contrarian Short', df_news[(df_news['llm_sentiment'] > 0.2) &
                                  (df_news['llm_overreaction'] >= 5)]),

    # Premarket driver
    ('Premarket High (>=5)', df_news[df_news['llm_premarket'] >= 5]),

    # Event types
    ('Earnings Day', df_news[df_news['llm_has_earnings'] > 0.5]),
    ('Analyst Rating', df_news[df_news['llm_has_analyst'] > 0.5]),
    ('Offering/Dilution', df_news[df_news['llm_has_offering'] > 0.5]),

    # No news baseline
    ('No News', df_no_news),
]

for label, sub in buckets:
    if len(sub) < 5:
        continue
    r15 = sub['ret_15m'].mean()
    r30 = sub['ret_30m'].mean()
    r60 = sub['ret_60m'].mean()
    mu60 = sub['max_up_ret_60m'].mean() if 'max_up_ret_60m' in sub else 0
    md60 = sub['max_dn_ret_60m'].mean() if 'max_dn_ret_60m' in sub else 0
    reod = sub['ret_eod'].mean()
    print(f"  {label:<26} {len(sub):>6} {r15:>+8.3%} {r30:>+8.3%} {r60:>+8.3%} "
          f"{mu60:>+9.3%} {md60:>+9.3%} {reod:>+8.3%}")


# ==============================================================================
# PER-TICKER ANALYSIS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  PER-TICKER — best LLM feature for ret_60m")
print(f"{'='*W}")

print(f"\n  {'Ticker':<8} {'N':>5} {'Best Feature':<26} {'rho':>7} {'p':>7} "
      f"{'Bull_avg':>9} {'Bear_avg':>9}")
print(f"  {'─'*80}")

ticker_results = []
for ticker in sorted(df['ticker'].unique()):
    tk = df[(df['ticker'] == ticker) & df['has_news']].dropna(subset=['ret_60m'])
    if len(tk) < 15:
        continue

    best_feat, best_rho, best_p = None, 0, 1
    for feat in DAILY_FEATURE_COLS:
        if tk[feat].std() < 1e-6:
            continue
        rho, pval = sp_stats.spearmanr(tk[feat], tk['ret_60m'])
        if abs(rho) > abs(best_rho):
            best_feat, best_rho, best_p = feat, rho, pval

    bull_avg = tk[tk['llm_sentiment'] > 0.2]['ret_60m'].mean()
    bear_avg = tk[tk['llm_sentiment'] < -0.2]['ret_60m'].mean()

    ticker_results.append({
        'ticker': ticker, 'n': len(tk),
        'best_feat': best_feat or 'none', 'rho': best_rho, 'pval': best_p,
        'bull_avg': bull_avg, 'bear_avg': bear_avg,
    })

    sig = ' *' if best_p < 0.05 else ''
    bull_str = f"{bull_avg:+.3%}" if not np.isnan(bull_avg) else "N/A"
    bear_str = f"{bear_avg:+.3%}" if not np.isnan(bear_avg) else "N/A"
    print(f"  {ticker:<8} {len(tk):>5} {(best_feat or 'none'):<26} "
          f"{best_rho:>+6.3f} {best_p:>6.3f}{sig} {bull_str:>9} {bear_str:>9}")

tr_df = pd.DataFrame(ticker_results)
if len(tr_df) > 0:
    n_sig = (tr_df['pval'] < 0.05).sum()
    print(f"\n  Significant (p<0.05): {n_sig}/{len(tr_df)} tickers")


# ==============================================================================
# HEAD-TO-HEAD: LLM vs Polygon raw sentiment
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  HEAD-TO-HEAD — LLM features vs Polygon raw sentiment")
print(f"{'='*W}")

print(f"\n  Compare absolute rho for ret_60m:")
print(f"  {'Feature Source':<30} {'Mean |rho|':>10} {'Max |rho|':>10} {'Sig (p<0.05)':>12}")
print(f"  {'─'*65}")

if len(fr_df) > 0 and 'rho_ret_60m' in fr_df.columns:
    llm_rhos = fr_df['rho_ret_60m'].dropna().abs()
    llm_ps = fr_df['p_ret_60m'].dropna()
    print(f"  {'LLM (21 features)':<30} {llm_rhos.mean():>10.4f} {llm_rhos.max():>10.4f} "
          f"{(llm_ps < 0.05).sum():>8}/{len(llm_ps)}")
    print(f"  {'Polygon raw (12 features)':<30} {'~0.015':>10} {'~0.022':>10} {'1/11':>12}")


# ==============================================================================
# CHARTS
# ==============================================================================
fig, axes = plt.subplots(2, 3, figsize=(22, 14))
fig.suptitle(
    f'LLM Sentiment Signal Analysis\n'
    f'{ANALYSIS_START} to {ANALYSIS_END} | {len(ANALYSIS_TICKERS)} tickers | '
    f'{total_with_news:,}/{total_days:,} ticker-days with news ({total_with_news/total_days*100:.0f}%)',
    fontsize=12, fontweight='bold')

# [0,0] Feature correlation heatmap
ax = axes[0, 0]
if len(fr_df) > 0:
    rho_cols = [c for c in fr_df.columns if c.startswith('rho_')]
    if rho_cols:
        heat = fr_df.set_index('feature')[rho_cols].dropna(how='all')
        heat.columns = [c.replace('rho_', '') for c in heat.columns]
        im = ax.imshow(heat.values, cmap='RdBu_r', vmin=-0.15, vmax=0.15, aspect='auto')
        ax.set_yticks(range(len(heat)))
        ax.set_yticklabels(heat.index, fontsize=6)
        ax.set_xticks(range(len(heat.columns)))
        ax.set_xticklabels(heat.columns, fontsize=7, rotation=30)
        for i in range(len(heat)):
            for j in range(len(heat.columns)):
                val = heat.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:+.3f}', ha='center', va='center', fontsize=5,
                            color='white' if abs(val) > 0.07 else 'black')
        ax.set_title('Feature-Return Correlations (Spearman rho)')
        plt.colorbar(im, ax=ax, shrink=0.6)

# [0,1] Conditional returns by sentiment bucket
ax = axes[0, 1]
sent_buckets = ['Bearish', 'Mild Bear', 'Mild Bull', 'Bullish', 'No News']
sent_dfs = [
    df_news[df_news['llm_sentiment'] < -0.3],
    df_news[(df_news['llm_sentiment'] >= -0.3) & (df_news['llm_sentiment'] < 0)],
    df_news[(df_news['llm_sentiment'] >= 0) & (df_news['llm_sentiment'] < 0.3)],
    df_news[df_news['llm_sentiment'] >= 0.3],
    df_no_news,
]
ret_means = [b['ret_60m'].mean() * 100 if len(b) > 0 else 0 for b in sent_dfs]
ret_counts = [len(b) for b in sent_dfs]
bar_colors = ['#C62828', '#EF5350', '#66BB6A', '#2E7D32', '#9E9E9E']
bars = ax.bar(sent_buckets, ret_means, color=bar_colors, alpha=0.8)
for bar, count in zip(bars, ret_counts):
    y = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, y + (0.01 if y >= 0 else -0.04),
            f'n={count}', ha='center', fontsize=8)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel('Mean 60m Return (%)')
ax.set_title('60m Returns by LLM Sentiment')
ax.tick_params(axis='x', rotation=20, labelsize=8)
ax.grid(axis='y', alpha=0.3)

# [0,2] Impact level vs returns
ax = axes[0, 2]
impact_buckets = ['Low\n(<3)', 'Medium\n(3-5)', 'High\n(>=5)']
impact_dfs = [
    df_news[df_news['llm_impact'] < 3],
    df_news[(df_news['llm_impact'] >= 3) & (df_news['llm_impact'] < 5)],
    df_news[df_news['llm_impact'] >= 5],
]
for label, sub, color in zip(impact_buckets, impact_dfs,
                              ['#78909C', '#FFA000', '#C62828']):
    if len(sub) < 5:
        continue
    means = [sub[h].mean() * 100 for h in ['ret_15m', 'ret_30m', 'ret_60m', 'ret_eod']
             if h in sub.columns]
    ax.plot(['15m', '30m', '60m', 'EOD'], means, 'o-', color=color,
            linewidth=2, markersize=6, label=f'{label} (n={len(sub)})')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel('Mean Return (%)')
ax.set_title('Returns by Price Impact Level')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# [1,0] Interaction features scatter
ax = axes[1, 0]
for feat, color, label in [
    ('llm_sentiment_x_impact', '#1565C0', 'Sent×Impact'),
    ('llm_contrarian', '#C62828', 'Contrarian'),
    ('llm_surprise_x_impact', '#2E7D32', 'Surprise×Impact'),
]:
    valid = df_news[[feat, 'ret_60m']].dropna()
    if len(valid) > 10 and valid[feat].std() > 1e-6:
        rho, pval = sp_stats.spearmanr(valid[feat], valid['ret_60m'])
        ax.scatter(valid[feat], valid['ret_60m'] * 100, alpha=0.1, s=6, color=color,
                   label=f'{label} rho={rho:+.3f} p={pval:.3f}')
ax.axhline(0, color='black', linewidth=0.8)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_ylabel('60m Return (%)')
ax.set_title('Interaction Features vs Return')
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

# [1,1] Per-ticker best rho
ax = axes[1, 1]
if len(tr_df) > 0:
    tr_sorted = tr_df.sort_values('rho')
    bar_colors = ['#2E7D32' if r > 0 else '#C62828' for r in tr_sorted['rho']]
    edge_colors = ['black' if p < 0.05 else 'none' for p in tr_sorted['pval']]
    ax.barh(tr_sorted['ticker'], tr_sorted['rho'], color=bar_colors,
            edgecolor=edge_colors, linewidth=2, alpha=0.8)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Best feature rho vs ret_60m')
    ax.set_title('Per-Ticker Best Correlation\n(black border = p<0.05)')
    ax.grid(axis='x', alpha=0.3)

# [1,2] Summary text
ax = axes[1, 2]
ax.axis('off')
lines = [
    f"LLM SENTIMENT SIGNAL SUMMARY",
    f"{'─'*55}",
    f"Period:    {ANALYSIS_START} to {ANALYSIS_END}",
    f"Tickers:   {len(ANALYSIS_TICKERS)}",
    f"Coverage:  {total_with_news:,}/{total_days:,} ({total_with_news/total_days*100:.0f}%)",
    f"Features:  {len(DAILY_FEATURE_COLS)} LLM-derived daily features",
    f"",
]
if len(fr_df) > 0 and 'rho_ret_60m' in fr_df.columns:
    sig_feats = fr_df[fr_df['p_ret_60m'] < 0.05]
    lines.append(f"SIGNIFICANT FEATURES (p<0.05 for ret_60m):")
    if len(sig_feats) > 0:
        for _, row in sig_feats.iterrows():
            lines.append(f"  {row['feature']}: rho={row['rho_ret_60m']:+.4f}")
    else:
        lines.append(f"  None at aggregate level")

if len(tr_df) > 0:
    lines.extend([
        f"",
        f"PER-TICKER: {(tr_df['pval'] < 0.05).sum()}/{len(tr_df)} significant",
        f"Mean |best_rho|: {tr_df['rho'].abs().mean():.3f}",
    ])

contrarian_df = df_news[(df_news['llm_sentiment'] < -0.2) & (df_news['llm_overreaction'] >= 5)]
if len(contrarian_df) > 5:
    lines.extend([
        f"",
        f"CONTRARIAN SIGNAL (bear+overreact):",
        f"  N={len(contrarian_df)}, ret_60m={contrarian_df['ret_60m'].mean():+.3%}",
    ])

ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
chart_path = '/tmp/llm_sentiment_signal.png'
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"\n  Chart saved: {chart_path}")
if _IN_COLAB:
    plt.show()
plt.close()


# ==============================================================================
# SECOND CHART: deep dives
# ==============================================================================
fig2, axes2 = plt.subplots(2, 3, figsize=(22, 14))
fig2.suptitle('LLM Signal Deep Dives', fontsize=13, fontweight='bold')

# [0,0] Catalyst vs non-catalyst return curves
ax = axes2[0, 0]
cat_high = df_news[df_news['llm_catalyst_ratio'] > 0.5]
cat_low = df_news[df_news['llm_catalyst_ratio'] <= 0.1]
for sub, color, label in [
    (cat_high, '#C62828', f'Catalyst (n={len(cat_high)})'),
    (cat_low, '#78909C', f'Non-catalyst (n={len(cat_low)})'),
    (df_no_news, '#BDBDBD', f'No news (n={len(df_no_news)})'),
]:
    if len(sub) < 5:
        continue
    means_up = [sub[f'max_up_{h}'].mean() * 100 for h in ['ret_15m', 'ret_30m', 'ret_60m']
                if f'max_up_{h}' in sub.columns]
    means_dn = [sub[f'max_dn_{h}'].mean() * 100 for h in ['ret_15m', 'ret_30m', 'ret_60m']
                if f'max_dn_{h}' in sub.columns]
    if means_up:
        ax.plot(['15m', '30m', '60m'][:len(means_up)], means_up, 'o-', color=color,
                linewidth=2, label=f'{label} up')
        ax.plot(['15m', '30m', '60m'][:len(means_dn)], means_dn, 'o--', color=color,
                linewidth=1.5, alpha=0.6)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel('Max Move (%)')
ax.set_title('Catalyst vs Non-Catalyst\n(solid=upside, dashed=downside)')
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

# [0,1] Event type returns
ax = axes2[0, 1]
if 'llm_has_earnings' in df_news.columns:
    event_returns = []
    for event, col in [('Earnings', 'llm_has_earnings'),
                        ('Analyst', 'llm_has_analyst'),
                        ('Offering', 'llm_has_offering')]:
        sub = df_news[df_news[col] > 0.5]
        if len(sub) >= 3:
            event_returns.append({'event': event, 'n': len(sub),
                                   'ret_60m': sub['ret_60m'].mean() * 100,
                                   'max_up': sub['max_up_ret_60m'].mean() * 100 if 'max_up_ret_60m' in sub else 0,
                                   'max_dn': sub['max_dn_ret_60m'].mean() * 100 if 'max_dn_ret_60m' in sub else 0})
    if event_returns:
        ev_df = pd.DataFrame(event_returns)
        x = range(len(ev_df))
        ax.bar([i-0.2 for i in x], ev_df['ret_60m'], 0.2, label='ret_60m', color='#1565C0')
        ax.bar([i for i in x], ev_df['max_up'], 0.2, label='max_up', color='#2E7D32')
        ax.bar([i+0.2 for i in x], ev_df['max_dn'], 0.2, label='max_dn', color='#C62828')
        ax.set_xticks(list(x))
        labels = [f"{r['event']}\n(n={r['n']})" for _, r in ev_df.iterrows()]
        ax.set_xticklabels(labels, fontsize=8)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_ylabel('Mean Return (%)')
        ax.set_title('Returns by Event Type')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

# [0,2] FOMO score distribution vs abs return
ax = axes2[0, 2]
valid = df_news[['llm_fomo', 'ret_60m']].dropna()
if len(valid) > 10:
    valid['abs_ret'] = valid['ret_60m'].abs()
    fomo_bins = pd.cut(valid['llm_fomo'], bins=[-0.1, 2, 4, 6, 10],
                        labels=['0-2', '3-4', '5-6', '7-10'])
    grouped = valid.groupby(fomo_bins, observed=True)['abs_ret'].agg(['mean', 'count'])
    if len(grouped) > 0:
        colors = ['#78909C', '#FFA000', '#EF5350', '#C62828'][:len(grouped)]
        bars = ax.bar(range(len(grouped)), grouped['mean'] * 100, color=colors, alpha=0.8)
        ax.set_xticks(range(len(grouped)))
        labels = [f"{idx}\n(n={int(row['count'])})" for idx, row in grouped.iterrows()]
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel('Mean |60m Return| (%)')
        ax.set_title('FOMO Score vs Absolute Return\n(volatility predictor?)')
        ax.grid(axis='y', alpha=0.3)

# [1,0] Monthly signal stability
ax = axes2[1, 0]
df_monthly = df_news[['llm_sentiment_x_impact', 'ret_60m', 'date']].dropna().copy()
if len(df_monthly) > 30:
    df_monthly['month'] = pd.to_datetime(df_monthly['date']).dt.to_period('M')
    monthly_corr = []
    for month, grp in df_monthly.groupby('month'):
        if len(grp) >= 10:
            r, p = sp_stats.spearmanr(grp['llm_sentiment_x_impact'], grp['ret_60m'])
            monthly_corr.append({'month': month.start_time, 'rho': r, 'p': p})
    if monthly_corr:
        mc_df = pd.DataFrame(monthly_corr)
        bar_colors = ['#2E7D32' if r > 0 else '#C62828' for r in mc_df['rho']]
        ax.bar(mc_df['month'], mc_df['rho'], width=25, color=bar_colors, alpha=0.7)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.set_ylabel('Spearman rho')
        ax.set_title('Monthly: sent×impact ↔ ret_60m')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)

# [1,1] Contrarian signal deep dive
ax = axes2[1, 1]
contrarian_bins = [
    ('Strong Contr.', df_news[(df_news['llm_contrarian'] > 2)]),
    ('Mild Contr.', df_news[(df_news['llm_contrarian'] > 0.5) & (df_news['llm_contrarian'] <= 2)]),
    ('Neutral', df_news[(df_news['llm_contrarian'] >= -0.5) & (df_news['llm_contrarian'] <= 0.5)]),
    ('Anti-Contr.', df_news[df_news['llm_contrarian'] < -0.5]),
]
labels, means, counts = [], [], []
for label, sub in contrarian_bins:
    if len(sub) >= 3:
        labels.append(label)
        means.append(sub['ret_60m'].mean() * 100)
        counts.append(len(sub))
if labels:
    bar_colors = ['#1565C0', '#42A5F5', '#78909C', '#EF5350'][:len(labels)]
    bars = ax.bar(labels, means, color=bar_colors, alpha=0.8)
    for bar, count in zip(bars, counts):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, y + 0.01 * np.sign(y),
                f'n={count}', ha='center', fontsize=8)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('Mean 60m Return (%)')
    ax.set_title('Contrarian Feature vs Returns')
    ax.tick_params(axis='x', labelsize=8)
    ax.grid(axis='y', alpha=0.3)

# [1,2] Premarket driver vs actual early return
ax = axes2[1, 2]
valid = df_news[['llm_premarket', 'ret_15m']].dropna()
if len(valid) > 10:
    pre_bins = pd.cut(valid['llm_premarket'], bins=[-0.1, 2, 4, 6, 10],
                       labels=['0-2', '3-4', '5-6', '7-10'])
    grouped = valid.groupby(pre_bins, observed=True).agg(
        mean_ret=('ret_15m', 'mean'),
        abs_ret=('ret_15m', lambda x: x.abs().mean()),
        count=('ret_15m', 'count'),
    )
    if len(grouped) > 0:
        x = range(len(grouped))
        ax.bar([i-0.15 for i in x], grouped['mean_ret'] * 100, 0.3,
               label='Mean ret_15m', color='#1565C0', alpha=0.8)
        ax.bar([i+0.15 for i in x], grouped['abs_ret'] * 100, 0.3,
               label='Mean |ret_15m|', color='#FF6F00', alpha=0.8)
        ax.set_xticks(list(x))
        labels = [f"{idx}\n(n={int(row['count'])})" for idx, row in grouped.iterrows()]
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel('Return (%)')
        ax.set_title('Premarket Driver vs 15m Return')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
chart2_path = '/tmp/llm_sentiment_signal_deep.png'
plt.savefig(chart2_path, dpi=150, bbox_inches='tight')
print(f"  Chart 2 saved: {chart2_path}")
if _IN_COLAB:
    plt.show()
plt.close()


# ==============================================================================
# CLEANUP
# ==============================================================================
gc.collect()
try:
    ctypes.CDLL("libc.so.6").malloc_trim(0)
except Exception:
    pass

print(f"\n{'='*W}")
print(f"  LLM SENTIMENT SIGNAL ANALYSIS COMPLETE")
print(f"{'='*W}")
