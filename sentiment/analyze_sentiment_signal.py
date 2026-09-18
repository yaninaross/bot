# %% ── Sentiment vs Price Change Signal Analysis (v2) ─────────────────────────
#
# Tests whether Polygon sentiment has predictive signal for intraday price moves
# at the deployed entry time (6:25 AM PT).
#
# 24-hour lookback window ending at 6:00 AM PT on the trading day.
#
# Computes 12 sentiment features per ticker-day:
#
# Core sentiment (3):
#   1. sent_net_ratio  — (n_pos - n_neg) / n_total, bounded [-1, +1]
#   2. sent_polarity   — n_opinionated / n_total (how decisive vs neutral)
#   3. sent_consensus  — 1 if all same direction, 0 if mixed (agreement)
#
# Volume/attention (3):
#   4. news_count      — raw article count in window
#   5. news_z          — z-score of article count vs ticker's 30-day rolling mean
#   6. sent_with_ratio — fraction of articles that have Polygon sentiment labels
#
# Momentum (3):
#   7. sent_accel      — today's net_ratio minus 7-day rolling mean of net_ratio
#   8. sent_streak     — consecutive days of same-sign sentiment
#   9. sent_reversal   — 1 if sentiment flipped sign vs yesterday, else 0
#
# Time-decay (2):
#  10. sent_recency    — recency-weighted net ratio (exp decay, half-life 6h)
#  11. sent_freshness  — hours since most recent article (lower = fresher)
#
# Cross-asset (1):
#  12. sent_sector_avg — average net_ratio across all tickers that day (market mood)
#
# Forward returns measured from entry bar (minute 385 = 6:25 AM PT):
#   - 15m, 30m, 60m close-to-close returns
#   - max upside / max downside in each window
#   - EOD return
#
# Reads:  gs://{GCS_BUCKET}/Data/Sentiment/{ticker}_news.parquet
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

DEPLOY_TICKERS = [
    'ACHR', 'ACMR', 'AEHR', 'ALAB', 'AMD', 'APLD', 'ARM', 'ASTS', 'AVAV', 'AXTI',
    'BBAI', 'BE', 'BEAM', 'BKSY', 'BLZE', 'BTBT', 'CAT', 'CEG', 'CIFR', 'CLSK',
    'COIN', 'CORZ', 'CRDO', 'CVX', 'ETN', 'FCEL', 'FLNC', 'GEV', 'HOOD', 'HUT',
    'IREN', 'JOBY', 'KEEL', 'KTOS', 'LUNR',
    'MARA', 'MNTS', 'MRNA', 'MSTR', 'MU', 'MVIS', 'MXL', 'NBIS', 'NEE', 'NNE',
    'NTLA', 'OKLO', 'ONDS', 'PL', 'PLTR', 'PLUG', 'RCAT', 'RDW', 'RIOT', 'RKLB',
    'RXT', 'SLS', 'SMCI', 'SMR', 'SNDK', 'SOUN', 'TSLA', 'VOYG', 'VRT', 'VST',
    'WOLF', 'WULF', 'XOM',
]

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

SENT_FEATURE_COLS = [
    'sent_net_ratio', 'sent_polarity', 'sent_consensus',
    'news_count', 'news_z', 'sent_with_ratio',
    'sent_accel', 'sent_streak', 'sent_reversal',
    'sent_recency', 'sent_freshness',
    'sent_sector_avg',
]

gcs_client = gcs_storage.Client()
bucket = gcs_client.bucket(BUCKET_NAME)

W = 100


# ==============================================================================
# DATA LOADING
# ==============================================================================
def _download_parquet(blob_path):
    local = f"/tmp/_sent_analysis_{os.path.basename(blob_path)}"
    try:
        bucket.blob(blob_path).download_to_filename(local)
        df = pd.read_parquet(local, engine='pyarrow')
        os.remove(local)
        return df
    except Exception:
        if os.path.exists(local):
            os.remove(local)
        return None


def load_sentiment(ticker):
    df = _download_parquet(f"{SENTIMENT_DIR}/{ticker}_news.parquet")
    if df is None or df.empty:
        return None
    df['published_utc'] = pd.to_datetime(df['published_utc'], utc=True)
    return df


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
# COMPUTE SENTIMENT FEATURES — 12 features per ticker-day
# ==============================================================================
SENT_MAP = {'positive': 1, 'negative': -1, 'neutral': 0, 'none': 0}


def _window_features(window_df, cutoff_end_utc):
    """Compute core + time-decay features from articles in the 24h window."""
    n_total = len(window_df)
    sentiments = window_df['sentiment'].map(SENT_MAP).fillna(0).values
    n_pos = (sentiments > 0).sum()
    n_neg = (sentiments < 0).sum()
    n_opinionated = n_pos + n_neg
    has_sent = (window_df['sentiment'] != 'none').sum()

    # Core sentiment
    net_ratio = (n_pos - n_neg) / n_total if n_total > 0 else 0.0
    polarity = n_opinionated / n_total if n_total > 0 else 0.0
    if n_opinionated > 0:
        consensus = abs(n_pos - n_neg) / n_opinionated
    else:
        consensus = 0.0

    # Attention
    sent_with_ratio = has_sent / n_total if n_total > 0 else 0.0

    # Recency-weighted net ratio (exponential decay, half-life 6h)
    hours_ago = (cutoff_end_utc - window_df['published_utc']).dt.total_seconds() / 3600.0
    decay = np.exp(-np.log(2) * hours_ago.values / RECENCY_HALFLIFE_H)
    recency_weighted = float(np.sum(sentiments * decay) / (np.sum(decay) + 1e-9))

    # Freshness: hours since most recent article
    freshness = float(hours_ago.min()) if n_total > 0 else LOOKBACK_HOURS

    return {
        'sent_net_ratio': net_ratio,
        'sent_polarity': polarity,
        'sent_consensus': consensus,
        'news_count': n_total,
        'sent_with_ratio': sent_with_ratio,
        'sent_recency': recency_weighted,
        'sent_freshness': freshness,
        'n_positive': int(n_pos),
        'n_negative': int(n_neg),
    }


def compute_all_sentiment_features(sentiment_df, trading_dates):
    """
    Compute all 12 sentiment features for each trading date.
    Uses 24h lookback ending at 6:00 AM PT.
    Momentum features (accel, streak, reversal) require sequential processing.
    """
    if sentiment_df is None or sentiment_df.empty:
        empty = {col: 0.0 for col in SENT_FEATURE_COLS}
        empty.update({'has_news': False, 'n_positive': 0, 'n_negative': 0, 'news_count': 0})
        return {d: dict(empty) for d in trading_dates}

    daily = {}
    prev_net_ratio = 0.0
    prev_sign = 0
    streak = 0
    rolling_counts = []
    rolling_net_ratios = []

    for date_str in trading_dates:
        date = pd.Timestamp(date_str, tz='UTC')
        cutoff_end_pst = date.tz_convert(PST).replace(hour=6, minute=0, second=0)
        cutoff_start_pst = cutoff_end_pst - pd.Timedelta(hours=LOOKBACK_HOURS)
        cutoff_end_utc = cutoff_end_pst.tz_convert('UTC')
        cutoff_start_utc = cutoff_start_pst.tz_convert('UTC')

        mask = ((sentiment_df['published_utc'] >= cutoff_start_utc) &
                (sentiment_df['published_utc'] < cutoff_end_utc))
        window = sentiment_df[mask]

        if window.empty:
            feats = {col: 0.0 for col in SENT_FEATURE_COLS}
            feats.update({'has_news': False, 'n_positive': 0, 'n_negative': 0})
            rolling_counts.append(0)
            rolling_net_ratios.append(0.0)
            streak = 0
            prev_sign = 0
            prev_net_ratio = 0.0
            daily[date_str] = feats
            continue

        core = _window_features(window, cutoff_end_utc)
        net_ratio = core['sent_net_ratio']

        # News volume z-score vs 30-day rolling mean
        rolling_counts.append(core['news_count'])
        rolling_net_ratios.append(net_ratio)
        if len(rolling_counts) >= 7:
            lookback = rolling_counts[-30:]
            mu = np.mean(lookback)
            sigma = np.std(lookback)
            news_z = (core['news_count'] - mu) / sigma if sigma > 0.5 else 0.0
        else:
            news_z = 0.0

        # Sentiment acceleration vs 7-day rolling mean
        if len(rolling_net_ratios) >= 3:
            lookback_nr = rolling_net_ratios[-7:]
            sent_accel = net_ratio - np.mean(lookback_nr)
        else:
            sent_accel = 0.0

        # Streak and reversal
        cur_sign = 1 if net_ratio > 0.1 else (-1 if net_ratio < -0.1 else 0)
        if cur_sign != 0 and cur_sign == prev_sign:
            streak += 1
        elif cur_sign != 0:
            streak = 1
        else:
            streak = 0
        reversal = 1.0 if (cur_sign != 0 and prev_sign != 0 and cur_sign != prev_sign) else 0.0
        prev_sign = cur_sign if cur_sign != 0 else prev_sign
        prev_net_ratio = net_ratio

        feats = {
            'sent_net_ratio': net_ratio,
            'sent_polarity': core['sent_polarity'],
            'sent_consensus': core['sent_consensus'],
            'news_count': core['news_count'],
            'news_z': np.clip(news_z, -5, 5),
            'sent_with_ratio': core['sent_with_ratio'],
            'sent_accel': np.clip(sent_accel, -2, 2),
            'sent_streak': float(streak * cur_sign),
            'sent_reversal': reversal,
            'sent_recency': core['sent_recency'],
            'sent_freshness': core['sent_freshness'],
            'sent_sector_avg': 0.0,  # filled in cross-asset pass
            'has_news': True,
            'n_positive': core['n_positive'],
            'n_negative': core['n_negative'],
        }
        daily[date_str] = feats

    return daily


# ==============================================================================
# COMPUTE FORWARD RETURNS FROM ENTRY BAR
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
print(f"  SENTIMENT vs PRICE CHANGE — SIGNAL ANALYSIS (v2)")
print(f"  {ANALYSIS_START} -> {ANALYSIS_END}")
print(f"  Entry: {ENTRY_MINUTE//60}:{ENTRY_MINUTE%60:02d} PT | Lookback: {LOOKBACK_HOURS}h")
print(f"  Features: {len(SENT_FEATURE_COLS)} sentiment features per ticker-day")
print(f"  Tickers: {len(DEPLOY_TICKERS)}")
print(f"{'='*W}")

all_rows = []
coverage_stats = []
ticker_daily_sent = {}

for ti, ticker in enumerate(DEPLOY_TICKERS):
    print(f"  [{ti+1}/{len(DEPLOY_TICKERS)}] {ticker}...", end=' ')

    sent_df = load_sentiment(ticker)
    price_df = load_prices(ticker)

    if price_df is None:
        print("no price data")
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

    daily_sent = compute_all_sentiment_features(sent_df, trading_dates)
    daily_rets = compute_forward_returns(price_df, trading_dates)
    ticker_daily_sent[ticker] = daily_sent

    del price_df, sent_df
    gc.collect()

    n_with_news = 0
    for date_str in trading_dates:
        if date_str not in daily_rets:
            continue
        sent = daily_sent.get(date_str, {})
        rets = daily_rets[date_str]

        if sent.get('has_news', False):
            n_with_news += 1

        row = {
            'ticker': ticker,
            'date': date_str,
            **{k: sent.get(k, 0.0) for k in SENT_FEATURE_COLS},
            'has_news': sent.get('has_news', False),
            'n_positive': sent.get('n_positive', 0),
            'n_negative': sent.get('n_negative', 0),
            **{k: v for k, v in rets.items() if k != 'entry_price'},
        }
        all_rows.append(row)

    coverage = n_with_news / len(trading_dates) * 100 if trading_dates else 0
    coverage_stats.append({'ticker': ticker, 'days': len(trading_dates),
                           'days_with_news': n_with_news, 'coverage_pct': coverage})
    print(f"{len(trading_dates)} days, {n_with_news} with news ({coverage:.0f}%)")

if not all_rows:
    print("\n  NO DATA — check that sentiment download completed.")
    sys.exit(0)

df = pd.DataFrame(all_rows)

# --- Fill sent_sector_avg (cross-asset pass) ---
for date_str in df['date'].unique():
    day_mask = df['date'] == date_str
    day_news = df.loc[day_mask & df['has_news'], 'sent_net_ratio']
    sector_avg = day_news.mean() if len(day_news) > 0 else 0.0
    df.loc[day_mask, 'sent_sector_avg'] = sector_avg


# ==============================================================================
# AGGREGATE ANALYSIS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  AGGREGATE RESULTS")
print(f"{'='*W}")

cov_df = pd.DataFrame(coverage_stats)
total_days = cov_df['days'].sum()
total_with_news = cov_df['days_with_news'].sum()
print(f"\n  Coverage: {total_with_news:,}/{total_days:,} ticker-days with news "
      f"({total_with_news/total_days*100:.1f}%)")
print(f"  >50% coverage: {(cov_df['coverage_pct'] > 50).sum()}/{len(cov_df)} tickers")
print(f"  >20% coverage: {(cov_df['coverage_pct'] > 20).sum()}/{len(cov_df)} tickers")

df_news = df[df['has_news']].copy()
df_no_news = df[~df['has_news']].copy()
df_pos = df_news[df_news['sent_net_ratio'] > 0.1].copy()
df_neg = df_news[df_news['sent_net_ratio'] < -0.1].copy()
df_neutral = df_news[(df_news['sent_net_ratio'] >= -0.1) & (df_news['sent_net_ratio'] <= 0.1)].copy()

print(f"\n  Sentiment distribution (net_ratio threshold +-0.1):")
print(f"    Positive (>+0.1):  {len(df_pos):>5} days")
print(f"    Negative (<-0.1):  {len(df_neg):>5} days")
print(f"    Neutral:           {len(df_neutral):>5} days")
print(f"    No news:           {len(df_no_news):>5} days")


# ==============================================================================
# FEATURE-BY-FEATURE CORRELATION WITH RETURNS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  FEATURE CORRELATION — each sentiment feature vs forward returns")
print(f"{'='*W}")

ret_targets = ['ret_15m', 'ret_30m', 'ret_60m', 'ret_eod',
               'max_up_ret_60m', 'max_dn_ret_60m', 'max_up_day', 'max_dn_day']

print(f"\n  {'Feature':<18}", end='')
for rt in ret_targets:
    print(f" {rt:>12}", end='')
print()
print(f"  {'─'*18}", end='')
for _ in ret_targets:
    print(f" {'─'*12}", end='')
print()

feature_corrs = {}
for feat in SENT_FEATURE_COLS:
    feature_corrs[feat] = {}
    print(f"  {feat:<18}", end='')
    for rt in ret_targets:
        valid = df_news[[feat, rt]].dropna()
        if len(valid) < 20:
            print(f" {'N/A':>12}", end='')
            continue
        rho, pval = sp_stats.spearmanr(valid[feat], valid[rt])
        feature_corrs[feat][rt] = (rho, pval)
        sig = '*' if pval < 0.05 else ' '
        print(f" {rho:>+.3f}{sig:>1}p{pval:.2f}"[:12].rjust(12), end='')
    print()


# ==============================================================================
# CONDITIONAL RETURNS — by sentiment bucket
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  CONDITIONAL RETURNS — by sentiment bucket")
print(f"{'='*W}")

print(f"\n  {'Group':<14} {'N':>6} {'ret_15m':>9} {'ret_30m':>9} {'ret_60m':>9} "
      f"{'max_up_60':>10} {'max_dn_60':>10} {'ret_eod':>9}")
print(f"  {'─'*80}")

for label, sub in [
    ('Strong_Neg', df_news[df_news['sent_net_ratio'] < -0.5]),
    ('Mild_Neg', df_news[(df_news['sent_net_ratio'] >= -0.5) & (df_news['sent_net_ratio'] < -0.1)]),
    ('Neutral', df_news[(df_news['sent_net_ratio'] >= -0.1) & (df_news['sent_net_ratio'] <= 0.1)]),
    ('Mild_Pos', df_news[(df_news['sent_net_ratio'] > 0.1) & (df_news['sent_net_ratio'] <= 0.5)]),
    ('Strong_Pos', df_news[df_news['sent_net_ratio'] > 0.5]),
    ('No_News', df_no_news),
    ('High_Volume', df_news[df_news['news_z'] > 1.0]),
    ('Fresh(<2h)', df_news[df_news['sent_freshness'] < 2.0]),
    ('Stale(>12h)', df_news[df_news['sent_freshness'] > 12.0]),
    ('Reversal', df_news[df_news['sent_reversal'] > 0.5]),
    ('High_Polarity', df_news[df_news['sent_polarity'] > 0.7]),
    ('Consensus>0.8', df_news[df_news['sent_consensus'] > 0.8]),
]:
    if len(sub) < 3:
        continue
    r15 = sub['ret_15m'].mean() if 'ret_15m' in sub else 0
    r30 = sub['ret_30m'].mean() if 'ret_30m' in sub else 0
    r60 = sub['ret_60m'].mean() if 'ret_60m' in sub else 0
    mu60 = sub['max_up_ret_60m'].mean() if 'max_up_ret_60m' in sub else 0
    md60 = sub['max_dn_ret_60m'].mean() if 'max_dn_ret_60m' in sub else 0
    reod = sub['ret_eod'].mean() if 'ret_eod' in sub else 0
    print(f"  {label:<14} {len(sub):>6} {r15:>+8.3%} {r30:>+8.3%} {r60:>+8.3%} "
          f"{mu60:>+9.3%} {md60:>+9.3%} {reod:>+8.3%}")


# ==============================================================================
# PER-TICKER BEST FEATURE
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  PER-TICKER — best sentiment feature for ret_60m")
print(f"{'='*W}")

print(f"\n  {'Ticker':<8} {'N':>5} {'Cov%':>5} {'Best Feature':<18} {'rho':>7} {'p':>7} "
      f"{'Pos_avg':>8} {'Neg_avg':>8}")
print(f"  {'─'*75}")

ticker_best = []
for ticker in sorted(df['ticker'].unique()):
    tk = df[(df['ticker'] == ticker) & df['has_news']].dropna(subset=['ret_60m'])
    if len(tk) < 10:
        continue
    cov_pct = next((c['coverage_pct'] for c in coverage_stats if c['ticker'] == ticker), 0)

    best_feat, best_rho, best_p = None, 0, 1
    for feat in SENT_FEATURE_COLS:
        if tk[feat].std() < 1e-6:
            continue
        rho, pval = sp_stats.spearmanr(tk[feat], tk['ret_60m'])
        if abs(rho) > abs(best_rho):
            best_feat, best_rho, best_p = feat, rho, pval

    pos_avg = tk[tk['sent_net_ratio'] > 0.1]['ret_60m'].mean()
    neg_avg = tk[tk['sent_net_ratio'] < -0.1]['ret_60m'].mean()

    ticker_best.append({
        'ticker': ticker, 'n': len(tk), 'cov_pct': cov_pct,
        'best_feat': best_feat or 'none', 'rho': best_rho, 'pval': best_p,
        'pos_avg': pos_avg, 'neg_avg': neg_avg,
    })

    sig = ' *' if best_p < 0.05 else ''
    pos_str = f"{pos_avg:+.3%}" if not np.isnan(pos_avg) else "N/A"
    neg_str = f"{neg_avg:+.3%}" if not np.isnan(neg_avg) else "N/A"
    print(f"  {ticker:<8} {len(tk):>5} {cov_pct:>4.0f}% {(best_feat or 'none'):<18} "
          f"{best_rho:>+6.3f} {best_p:>6.3f}{sig} {pos_str:>8} {neg_str:>8}")

tb_df = pd.DataFrame(ticker_best)
if len(tb_df) > 0:
    n_sig = (tb_df['pval'] < 0.05).sum()
    feat_counts = tb_df['best_feat'].value_counts()
    print(f"\n  Significant (p<0.05): {n_sig}/{len(tb_df)} tickers")
    print(f"  Most common best feature: {feat_counts.index[0]} ({feat_counts.iloc[0]} tickers)")
    if len(feat_counts) > 1:
        print(f"  Runner up: {feat_counts.index[1]} ({feat_counts.iloc[1]} tickers)")


# ==============================================================================
# SIGNAL QUALITY — aggregate Spearman for each feature
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SIGNAL QUALITY — aggregate ranking of sentiment features")
print(f"{'='*W}")

print(f"\n  {'Feature':<18} {'rho_60m':>8} {'p_60m':>8} {'rho_eod':>8} {'p_eod':>8} "
      f"{'rho_maxup':>9} {'rho_maxdn':>9}")
print(f"  {'─'*70}")

feat_rankings = []
for feat in SENT_FEATURE_COLS:
    valid = df_news[[feat, 'ret_60m', 'ret_eod', 'max_up_ret_60m', 'max_dn_ret_60m']].dropna()
    if len(valid) < 20 or valid[feat].std() < 1e-6:
        continue
    r60, p60 = sp_stats.spearmanr(valid[feat], valid['ret_60m'])
    reod, peod = sp_stats.spearmanr(valid[feat], valid['ret_eod'])
    rup, _ = sp_stats.spearmanr(valid[feat], valid['max_up_ret_60m'])
    rdn, _ = sp_stats.spearmanr(valid[feat], valid['max_dn_ret_60m'])
    sig60 = '*' if p60 < 0.05 else ' '
    sigeod = '*' if peod < 0.05 else ' '
    print(f"  {feat:<18} {r60:>+7.4f}{sig60} {p60:>7.4f} {reod:>+7.4f}{sigeod} {peod:>7.4f} "
          f"{rup:>+8.4f} {rdn:>+8.4f}")
    feat_rankings.append({'feature': feat, 'rho_60m': r60, 'p_60m': p60,
                          'rho_eod': reod, 'rho_maxup': rup, 'rho_maxdn': rdn})

fr_df = pd.DataFrame(feat_rankings)


# ==============================================================================
# CHARTS
# ==============================================================================
fig, axes = plt.subplots(2, 3, figsize=(22, 14))

fig.suptitle(
    f'Sentiment Signal Analysis (v2) — 12 Features\n'
    f'{ANALYSIS_START} to {ANALYSIS_END} | {len(DEPLOY_TICKERS)} tickers | '
    f'{total_with_news:,}/{total_days:,} ticker-days with news ({total_with_news/total_days*100:.0f}%) | '
    f'{LOOKBACK_HOURS}h window',
    fontsize=12, fontweight='bold')

# [0,0] Feature correlation heatmap (feature x return target)
ax = axes[0, 0]
if len(fr_df) > 0:
    heat_data = fr_df.set_index('feature')[['rho_60m', 'rho_eod', 'rho_maxup', 'rho_maxdn']]
    im = ax.imshow(heat_data.values, cmap='RdBu_r', vmin=-0.08, vmax=0.08, aspect='auto')
    ax.set_yticks(range(len(heat_data)))
    ax.set_yticklabels(heat_data.index, fontsize=7)
    ax.set_xticks(range(len(heat_data.columns)))
    ax.set_xticklabels(['ret_60m', 'ret_eod', 'max_up', 'max_dn'], fontsize=8, rotation=30)
    for i in range(len(heat_data)):
        for j in range(len(heat_data.columns)):
            val = heat_data.values[i, j]
            ax.text(j, i, f'{val:+.3f}', ha='center', va='center', fontsize=7,
                    color='white' if abs(val) > 0.04 else 'black')
    ax.set_title('Feature-Return Correlations\n(Spearman rho)')
    plt.colorbar(im, ax=ax, shrink=0.6)

# [0,1] Conditional returns by sentiment bucket
ax = axes[0, 1]
buckets = ['Str_Neg', 'Mild_Neg', 'Neutral', 'Mild_Pos', 'Str_Pos', 'No_News']
bucket_dfs = [
    df_news[df_news['sent_net_ratio'] < -0.5],
    df_news[(df_news['sent_net_ratio'] >= -0.5) & (df_news['sent_net_ratio'] < -0.1)],
    df_news[(df_news['sent_net_ratio'] >= -0.1) & (df_news['sent_net_ratio'] <= 0.1)],
    df_news[(df_news['sent_net_ratio'] > 0.1) & (df_news['sent_net_ratio'] <= 0.5)],
    df_news[df_news['sent_net_ratio'] > 0.5],
    df_no_news,
]
ret_means = [b['ret_60m'].mean() * 100 if len(b) > 0 else 0 for b in bucket_dfs]
ret_counts = [len(b) for b in bucket_dfs]
bar_colors = ['#C62828', '#EF5350', '#78909C', '#66BB6A', '#2E7D32', '#9E9E9E']
bars = ax.bar(buckets, ret_means, color=bar_colors, alpha=0.8)
for bar, count in zip(bars, ret_counts):
    y = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, y + (0.01 if y >= 0 else -0.04),
            f'n={count}', ha='center', fontsize=7)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel('Mean 60m Return (%)')
ax.set_title('60m Returns by Sentiment Bucket')
ax.tick_params(axis='x', rotation=30, labelsize=8)
ax.grid(axis='y', alpha=0.3)

# [0,2] Scatter: sent_recency (time-decay weighted) vs ret_60m
ax = axes[0, 2]
valid = df_news[['sent_recency', 'ret_60m']].dropna()
if len(valid) > 10:
    ax.scatter(valid['sent_recency'], valid['ret_60m'] * 100,
               alpha=0.12, s=8, color='#1565C0')
    rho, pval = sp_stats.spearmanr(valid['sent_recency'], valid['ret_60m'])
    z = np.polyfit(valid['sent_recency'], valid['ret_60m'] * 100, 1)
    x_line = np.linspace(valid['sent_recency'].min(), valid['sent_recency'].max(), 50)
    ax.plot(x_line, np.polyval(z, x_line), 'r-', linewidth=2,
            label=f'rho={rho:+.3f}, p={pval:.3f}')
    ax.legend(fontsize=9)
ax.axhline(0, color='black', linewidth=0.8)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Recency-Weighted Sentiment')
ax.set_ylabel('60m Return (%)')
ax.set_title('Time-Decay Sentiment vs 60m Return')
ax.grid(alpha=0.3)

# [1,0] Coverage per ticker
ax = axes[1, 0]
cov_sorted = pd.DataFrame(coverage_stats).sort_values('coverage_pct', ascending=True)
colors = ['#2E7D32' if c >= 50 else '#FFA000' if c >= 20 else '#C62828'
          for c in cov_sorted['coverage_pct']]
ax.barh(cov_sorted['ticker'], cov_sorted['coverage_pct'], color=colors, alpha=0.8)
ax.set_xlabel('% of trading days with news (24h window)')
ax.set_title('News Coverage per Ticker')
ax.axvline(50, color='black', linestyle='--', alpha=0.3)
ax.set_xlim(0, 100)
ax.tick_params(axis='y', labelsize=6)
ax.grid(axis='x', alpha=0.3)

# [1,1] Per-ticker best rho
ax = axes[1, 1]
if len(tb_df) > 0:
    tb_sorted = tb_df.sort_values('rho')
    bar_colors = ['#2E7D32' if r > 0 else '#C62828' for r in tb_sorted['rho']]
    edge_colors = ['black' if p < 0.05 else 'none' for p in tb_sorted['pval']]
    ax.barh(tb_sorted['ticker'], tb_sorted['rho'], color=bar_colors,
            edgecolor=edge_colors, linewidth=1.5, alpha=0.8)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Best feature rho vs ret_60m')
    ax.set_title('Per-Ticker Best Correlation\n(black border = p<0.05)')
    ax.tick_params(axis='y', labelsize=6)
    ax.grid(axis='x', alpha=0.3)

# [1,2] News volume vs absolute return
ax = axes[1, 2]
valid = df_news[['news_count', 'ret_60m']].dropna()
if len(valid) > 10:
    valid['abs_ret'] = valid['ret_60m'].abs()
    rho, pval = sp_stats.spearmanr(valid['news_count'], valid['abs_ret'])
    ax.scatter(valid['news_count'], valid['abs_ret'] * 100,
               alpha=0.12, s=8, color='#FF6F00')
    ax.set_xlabel('Article Count (24h window)')
    ax.set_ylabel('|60m Return| (%)')
    ax.set_title(f'News Volume vs Volatility\nrho={rho:+.3f}, p={pval:.3f}')
    ax.grid(alpha=0.3)

plt.tight_layout()
chart_path = '/tmp/sentiment_signal_v2.png'
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"\n  Chart 1 saved: {chart_path}")
if _IN_COLAB:
    plt.show()
plt.close()


# ==============================================================================
# SECOND CHART: directional signal deep dive
# ==============================================================================
fig2, axes2 = plt.subplots(2, 3, figsize=(22, 14))
fig2.suptitle(
    'Directional Signal Deep Dive — LONG & SHORT',
    fontsize=13, fontweight='bold')

# [0,0] LONG: positive sentiment → upside capture
ax = axes2[0, 0]
for sent_group, color, label in [
    (df_news[df_news['sent_net_ratio'] > 0.5], '#2E7D32', 'Strong Pos'),
    (df_news[(df_news['sent_net_ratio'] > 0.1) & (df_news['sent_net_ratio'] <= 0.5)], '#66BB6A', 'Mild Pos'),
    (df_no_news, '#9E9E9E', 'No News'),
    (df_news[(df_news['sent_net_ratio'] >= -0.5) & (df_news['sent_net_ratio'] < -0.1)], '#EF5350', 'Mild Neg'),
    (df_news[df_news['sent_net_ratio'] < -0.5], '#C62828', 'Strong Neg'),
]:
    if len(sent_group) < 3:
        continue
    means = [sent_group[h].mean() * 100 if h in sent_group.columns else 0
             for h in ['max_up_ret_15m', 'max_up_ret_30m', 'max_up_ret_60m', 'max_up_day']]
    ax.plot(['15m', '30m', '60m', 'EOD'], means, 'o-', color=color, linewidth=2,
            markersize=6, label=f'{label} (n={len(sent_group)})')
ax.set_ylabel('Mean Max Upside (%)')
ax.set_title('LONG Signal — Max Upside by Sentiment')
ax.legend(fontsize=7, loc='upper left')
ax.grid(alpha=0.3)

# [0,1] SHORT: negative sentiment → downside capture
ax = axes2[0, 1]
for sent_group, color, label in [
    (df_news[df_news['sent_net_ratio'] < -0.5], '#C62828', 'Strong Neg'),
    (df_news[(df_news['sent_net_ratio'] >= -0.5) & (df_news['sent_net_ratio'] < -0.1)], '#EF5350', 'Mild Neg'),
    (df_no_news, '#9E9E9E', 'No News'),
    (df_news[(df_news['sent_net_ratio'] > 0.1) & (df_news['sent_net_ratio'] <= 0.5)], '#66BB6A', 'Mild Pos'),
    (df_news[df_news['sent_net_ratio'] > 0.5], '#2E7D32', 'Strong Pos'),
]:
    if len(sent_group) < 3:
        continue
    means = [sent_group[h].mean() * 100 if h in sent_group.columns else 0
             for h in ['max_dn_ret_15m', 'max_dn_ret_30m', 'max_dn_ret_60m', 'max_dn_day']]
    ax.plot(['15m', '30m', '60m', 'EOD'], means, 'o-', color=color, linewidth=2,
            markersize=6, label=f'{label} (n={len(sent_group)})')
ax.set_ylabel('Mean Max Drawdown (%)')
ax.set_title('SHORT Signal — Max Downside by Sentiment')
ax.legend(fontsize=7, loc='lower left')
ax.grid(alpha=0.3)

# [0,2] Freshness effect: fresh vs stale news
ax = axes2[0, 2]
fresh = df_news[df_news['sent_freshness'] < 3.0]
mid = df_news[(df_news['sent_freshness'] >= 3.0) & (df_news['sent_freshness'] < 12.0)]
stale = df_news[df_news['sent_freshness'] >= 12.0]
for sub, color, label in [
    (fresh, '#1565C0', f'Fresh <3h (n={len(fresh)})'),
    (mid, '#FFA000', f'Mid 3-12h (n={len(mid)})'),
    (stale, '#78909C', f'Stale >12h (n={len(stale)})'),
]:
    if len(sub) < 5:
        continue
    valid_sub = sub.dropna(subset=['ret_60m'])
    if len(valid_sub) < 5:
        continue
    rho, _ = sp_stats.spearmanr(valid_sub['sent_net_ratio'], valid_sub['ret_60m'])
    ax.bar(label.split('(')[0].strip(), rho, color=color, alpha=0.8)
    ax.text(label.split('(')[0].strip(), rho + 0.002 * np.sign(rho),
            f'rho={rho:+.3f}', ha='center', fontsize=8)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel('Spearman rho (net_ratio vs ret_60m)')
ax.set_title('Signal Decay — Does Freshness Matter?')
ax.tick_params(axis='x', labelsize=8)
ax.grid(axis='y', alpha=0.3)

# [1,0] Consensus effect: high consensus vs mixed
ax = axes2[1, 0]
high_con = df_news[df_news['sent_consensus'] > 0.8]
low_con = df_news[(df_news['sent_consensus'] <= 0.8) & (df_news['sent_polarity'] > 0.3)]
for sub, color, label in [
    (high_con, '#1565C0', f'High consensus >0.8 (n={len(high_con)})'),
    (low_con, '#FFA000', f'Mixed consensus (n={len(low_con)})'),
]:
    if len(sub) < 5:
        continue
    valid_sub = sub.dropna(subset=['ret_60m'])
    if len(valid_sub) < 5:
        continue
    rho, _ = sp_stats.spearmanr(valid_sub['sent_net_ratio'], valid_sub['ret_60m'])
    ax.bar(label.split('(')[0].strip(), rho, color=color, alpha=0.8)
    ax.text(label.split('(')[0].strip(), rho + 0.002 * np.sign(rho),
            f'rho={rho:+.3f}', ha='center', fontsize=8)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel('Spearman rho')
ax.set_title('Consensus Effect — Aligned vs Mixed Sentiment')
ax.grid(axis='y', alpha=0.3)

# [1,1] Monthly signal stability
ax = axes2[1, 1]
df_monthly = df_news[['sent_net_ratio', 'ret_60m', 'date']].dropna().copy()
if len(df_monthly) > 30:
    df_monthly['month'] = pd.to_datetime(df_monthly['date']).dt.to_period('M')
    monthly_corr = []
    for month, grp in df_monthly.groupby('month'):
        if len(grp) >= 15:
            r, p = sp_stats.spearmanr(grp['sent_net_ratio'], grp['ret_60m'])
            monthly_corr.append({'month': month.start_time, 'rho': r, 'p': p, 'n': len(grp)})
    if monthly_corr:
        mc_df = pd.DataFrame(monthly_corr)
        bar_colors = ['#2E7D32' if r > 0 else '#C62828' for r in mc_df['rho']]
        ax.bar(mc_df['month'], mc_df['rho'], width=25, color=bar_colors, alpha=0.7)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.set_ylabel('Spearman rho')
        ax.set_title('Monthly net_ratio ↔ ret_60m Correlation')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)

# [1,2] Summary table
ax = axes2[1, 2]
ax.axis('off')
lines = [
    f"SENTIMENT SIGNAL SUMMARY (v2)",
    f"{'─'*55}",
    f"Period:  {ANALYSIS_START} to {ANALYSIS_END}",
    f"Tickers: {len(DEPLOY_TICKERS)} | Window: {LOOKBACK_HOURS}h",
    f"Total:   {total_days:,} ticker-days | News: {total_with_news:,} ({total_with_news/total_days*100:.0f}%)",
    f"",
    f"CONDITIONAL RETURNS (60m):",
]
for label, sub in [('  Strong Pos', df_news[df_news['sent_net_ratio'] > 0.5]),
                    ('  Mild Pos', df_news[(df_news['sent_net_ratio'] > 0.1) & (df_news['sent_net_ratio'] <= 0.5)]),
                    ('  Neutral', df_news[(df_news['sent_net_ratio'] >= -0.1) & (df_news['sent_net_ratio'] <= 0.1)]),
                    ('  Mild Neg', df_news[(df_news['sent_net_ratio'] >= -0.5) & (df_news['sent_net_ratio'] < -0.1)]),
                    ('  Strong Neg', df_news[df_news['sent_net_ratio'] < -0.5]),
                    ('  No News', df_no_news)]:
    if len(sub) > 0:
        lines.append(f"{label}: {sub['ret_60m'].mean():+.3%} (n={len(sub)})")

if len(fr_df) > 0:
    best = fr_df.loc[fr_df['rho_60m'].abs().idxmax()]
    lines.extend([
        f"",
        f"BEST FEATURE: {best['feature']}",
        f"  rho_60m = {best['rho_60m']:+.4f} (p={best['p_60m']:.4f})",
    ])

if len(tb_df) > 0:
    lines.extend([
        f"",
        f"PER-TICKER: {(tb_df['pval'] < 0.05).sum()}/{len(tb_df)} significant",
    ])

ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
chart2_path = '/tmp/sentiment_signal_v2_directional.png'
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
print(f"  SENTIMENT SIGNAL ANALYSIS (v2) COMPLETE")
print(f"{'='*W}")
