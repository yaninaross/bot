"""
analyze_regime_breadth.py — Market Regime Breadth Analysis
==========================================================
Is the June 2026 spike-quality degradation market-wide or specific
to our trained small-cap/speculative asset universe?

Compares 7 intraday regime metrics across 3 groups:
  Group A: Deployed tickers (S3 live)
  Group B: Full trained pool (not deployed)
  Group C: Broad market benchmarks (SPY, QQQ, IWM, etc.)

Metrics per month (Apr / May / Jun 2026):
  1. Daily Range %          — (H-L)/C
  2. Upside Spike Rate      — days with (H/O - 1) > 5%
  3. Downside Crash Rate    — days with (L/O - 1) < -5%
  4. Mean Daily Return      — close-to-close
  5. Intraday Reversal Rate — (H-O)>5% but C<O (the Cat-B problem)
  6. Tail Ratio             — 95th up / |5th down| (asymmetry)
  7. Mean Reversion Score   — 1 - |EOD ret|/max excursion

Colab-ready (no if __name__). Requires: GCS, pandas.
"""

import os, io, gc, sys, subprocess, warnings, ctypes
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np, pandas as pd, pytz

warnings.filterwarnings('ignore')

def _release_memory():
    gc.collect()
    try: ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception: pass

try:
    from google.cloud import storage as gcs_storage
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
        "google-cloud-storage", "pyarrow", "gcsfs", "--quiet"])
    from google.cloud import storage as gcs_storage

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install",
            "yfinance", "--quiet"])
        import yfinance as yf
        HAS_YF = True
    except Exception:
        HAS_YF = False

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass


# ==============================================================================
# CONFIG
# ==============================================================================
BUCKET_NAME = "time-series-bucket-ys"
OHLCV_DIR = "Data/Inference"
INFERENCE_DIR = "Data/Inference_v2"

ANALYSIS_START = '2026-04-01'
ANALYSIS_END   = '2026-06-29'
MONTHS = ['2026-04', '2026-05', '2026-06']
MONTH_LABELS = ['Apr', 'May', 'Jun']

PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

UPSIDE_SPIKE_THRESH = 0.05
DOWNSIDE_CRASH_THRESH = -0.05

# ── ASSET GROUPS ──
DEPLOY_TICKERS = ['HUT', 'FLNC', 'CIFR', 'CLSK', 'KEEL', 'RCAT', 'RIOT', 'APLD', 'WULF']

TRAINED_POOL = [
    'ASTS', 'LUNR', 'BBAI', 'MARA', 'RKLB', 'MVIS', 'ACHR', 'SPIR',
    'ALAB', 'BTBT', 'CORZ', 'COIN', 'MSTR', 'HOOD', 'OKLO', 'SMR',
    'BKSY', 'MNTS', 'PLUG', 'BE', 'FCEL', 'BGDE', 'RDW', 'VOYG',
]

BENCHMARKS = ['SPY', 'QQQ', 'IWM', 'TLT', 'BTCUSD', 'ARKK', 'WGMI']

TICKER_GROUPS = {}
for t in DEPLOY_TICKERS: TICKER_GROUPS[t] = 'A_deployed'
for t in TRAINED_POOL:   TICKER_GROUPS[t] = 'B_trained'
for t in BENCHMARKS:     TICKER_GROUPS[t] = 'C_benchmark'

ALL_TICKERS = DEPLOY_TICKERS + TRAINED_POOL + BENCHMARKS

MKT_OPEN_MIN  = 390   # 6:30 AM PST = 390 min from midnight
MKT_CLOSE_MIN = 780   # 1:00 PM PST = 780 min from midnight

gcs_client = gcs_storage.Client()


# ==============================================================================
# HELPERS
# ==============================================================================
def heal_data(df):
    df = df.resample('1min').asfreq()
    for c in ['Open', 'High', 'Low', 'Close']:
        if c in df.columns: df[c] = df[c].ffill()
    for c in ['Volume']:
        if c in df.columns: df[c] = df[c].fillna(0)
    return df


def load_ohlcv_gcs(ticker):
    """Load OHLCV from GCS parquet. Returns PST-indexed DataFrame or None."""
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    for dir_path in [INFERENCE_DIR, OHLCV_DIR]:
        for suffix in ['_master_inference.parquet', '_1m_full_historical_data.parquet',
                        '_ohlcv_1min_pst.parquet']:
            uri = f"gs://{BUCKET_NAME}/{dir_path}/{clean}{suffix}"
            try:
                df = pd.read_parquet(uri)
                df = df.reset_index()
                dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
                if dt_cols:
                    df['Datetime'] = pd.to_datetime(df[dt_cols[0]], utc=True)
                else:
                    df['Datetime'] = pd.to_datetime(df[df.columns[0]], unit='ms', utc=True)
                df = df.set_index('Datetime').sort_index()
                df = df[~df.index.duplicated(keep='first')]
                std_map = {'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'}
                lower_map = {k.lower():v for k,v in std_map.items()}
                df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip()) for c in df.columns]
                start = pd.Timestamp(ANALYSIS_START, tz='UTC') - pd.Timedelta(days=5)
                end = pd.Timestamp(ANALYSIS_END, tz='UTC') + pd.Timedelta(days=2)
                df = df[(df.index >= start) & (df.index <= end)]
                df = heal_data(df)
                df.index = df.index.tz_convert(PST)
                return df
            except Exception:
                continue
    return None


def load_ohlcv_yfinance(ticker):
    """Fallback: load daily OHLCV from yfinance. Returns PST-localized daily DataFrame or None."""
    if not HAS_YF:
        return None
    yf_ticker = ticker
    if ticker == 'BTCUSD':
        yf_ticker = 'BTC-USD'
    try:
        obj = yf.Ticker(yf_ticker)
        df = obj.history(start=ANALYSIS_START, end=ANALYSIS_END, interval='1d')
        if df.empty:
            return None
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert(PST)
        df = df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
        for c in ['Open','High','Low','Close','Volume']:
            if c not in df.columns:
                alt = c.capitalize()
                if alt in df.columns:
                    df[c] = df[alt]
        return df
    except Exception:
        return None


def load_ticker(ticker):
    """Try GCS first, then yfinance. Returns (df, source_str)."""
    df = load_ohlcv_gcs(ticker)
    if df is not None:
        return df, 'GCS_1min'
    df = load_ohlcv_yfinance(ticker)
    if df is not None:
        return df, 'yfinance_daily'
    return None, None


def compute_daily_metrics_1min(df, month_str):
    """Compute daily regime metrics from 1-min intraday data for a single month."""
    month_mask = df.index.strftime('%Y-%m') == month_str
    sub = df[month_mask]
    if sub.empty:
        return None

    pst_mins = sub.index.hour * 60 + sub.index.minute
    mkt = sub[(pst_mins >= MKT_OPEN_MIN) & (pst_mins <= MKT_CLOSE_MIN)]
    if len(mkt) < 100:
        return None

    daily = mkt.groupby(mkt.index.date).agg(
        day_open=('Open', 'first'),
        day_high=('High', 'max'),
        day_low=('Low', 'min'),
        day_close=('Close', 'last'),
    ).dropna()

    if len(daily) < 5:
        return None

    daily['range_pct'] = (daily['day_high'] - daily['day_low']) / daily['day_close']
    daily['up_exc'] = daily['day_high'] / daily['day_open'] - 1
    daily['dn_exc'] = daily['day_low'] / daily['day_open'] - 1
    daily['daily_ret'] = daily['day_close'] / daily['day_open'] - 1

    n_days = len(daily)
    spike_days = (daily['up_exc'] > UPSIDE_SPIKE_THRESH).sum()
    crash_days = (daily['dn_exc'] < DOWNSIDE_CRASH_THRESH).sum()
    reversal_days = ((daily['up_exc'] > UPSIDE_SPIKE_THRESH) & (daily['daily_ret'] < 0)).sum()

    daily['max_exc'] = daily[['up_exc', 'dn_exc']].abs().max(axis=1)
    mean_rev_mask = daily['max_exc'] > 0.001
    if mean_rev_mask.sum() > 3:
        mean_rev_score = (1 - daily.loc[mean_rev_mask, 'daily_ret'].abs() / daily.loc[mean_rev_mask, 'max_exc']).mean()
    else:
        mean_rev_score = np.nan

    ret_1m = mkt['Close'].pct_change().dropna()
    if len(ret_1m) > 50:
        pos_ret = ret_1m[ret_1m > 0]
        neg_ret = ret_1m[ret_1m < 0]
        up_95 = pos_ret.quantile(0.95) if len(pos_ret) > 10 else 0
        dn_05 = neg_ret.quantile(0.05) if len(neg_ret) > 10 else 0
        tail_ratio = (up_95 / abs(dn_05)) if dn_05 != 0 else np.nan
    else:
        tail_ratio = np.nan

    return {
        'n_days': n_days,
        'range_pct': daily['range_pct'].mean() * 100,
        'spike_rate': spike_days / n_days * 100,
        'crash_rate': crash_days / n_days * 100,
        'mean_ret': daily['daily_ret'].mean() * 100,
        'reversal_rate': reversal_days / n_days * 100 if n_days > 0 else 0,
        'tail_ratio': tail_ratio,
        'mean_rev_score': mean_rev_score,
    }


def compute_daily_metrics_daily(df, month_str):
    """Compute daily regime metrics from daily bars (yfinance fallback)."""
    month_mask = df.index.strftime('%Y-%m') == month_str
    daily = df[month_mask].copy()
    if len(daily) < 5:
        return None

    daily['range_pct'] = (daily['High'] - daily['Low']) / daily['Close']
    daily['up_exc'] = daily['High'] / daily['Open'] - 1
    daily['dn_exc'] = daily['Low'] / daily['Open'] - 1
    daily['daily_ret'] = daily['Close'] / daily['Open'] - 1

    n_days = len(daily)
    spike_days = (daily['up_exc'] > UPSIDE_SPIKE_THRESH).sum()
    crash_days = (daily['dn_exc'] < DOWNSIDE_CRASH_THRESH).sum()
    reversal_days = ((daily['up_exc'] > UPSIDE_SPIKE_THRESH) & (daily['daily_ret'] < 0)).sum()

    daily['max_exc'] = daily[['up_exc', 'dn_exc']].abs().max(axis=1)
    mean_rev_mask = daily['max_exc'] > 0.001
    if mean_rev_mask.sum() > 3:
        mean_rev_score = (1 - daily.loc[mean_rev_mask, 'daily_ret'].abs() / daily.loc[mean_rev_mask, 'max_exc']).mean()
    else:
        mean_rev_score = np.nan

    daily_rets = daily['daily_ret']
    pos_ret = daily_rets[daily_rets > 0]
    neg_ret = daily_rets[daily_rets < 0]
    up_95 = pos_ret.quantile(0.95) if len(pos_ret) > 3 else 0
    dn_05 = neg_ret.quantile(0.05) if len(neg_ret) > 3 else 0
    tail_ratio = (up_95 / abs(dn_05)) if dn_05 != 0 else np.nan

    return {
        'n_days': n_days,
        'range_pct': daily['range_pct'].mean() * 100,
        'spike_rate': spike_days / n_days * 100,
        'crash_rate': crash_days / n_days * 100,
        'mean_ret': daily['daily_ret'].mean() * 100,
        'reversal_rate': reversal_days / n_days * 100 if n_days > 0 else 0,
        'tail_ratio': tail_ratio,
        'mean_rev_score': mean_rev_score,
    }


# ==============================================================================
# MAIN ANALYSIS
# ==============================================================================
W = 120
print("=" * W)
print(f"  MARKET REGIME BREADTH ANALYSIS — Is June Degradation Market-Wide?")
print("=" * W)
print(f"  Period: {ANALYSIS_START} → {ANALYSIS_END}")
print(f"  Group A (deployed): {len(DEPLOY_TICKERS)} tickers")
print(f"  Group B (trained):  {len(TRAINED_POOL)} tickers")
print(f"  Group C (benchmark): {len(BENCHMARKS)} tickers")
print()

# ── Load all tickers ──
print(f"  Loading OHLCV data...")
ohlcv_data = {}
daily_closes = {}
load_sources = {}

for ticker in ALL_TICKERS:
    df, src = load_ticker(ticker)
    if df is not None:
        ohlcv_data[ticker] = df
        load_sources[ticker] = src
        if src == 'GCS_1min':
            closes = df.groupby(df.index.date)['Close'].last().dropna()
        else:
            closes = df['Close'].dropna()
            closes.index = pd.to_datetime(closes.index).date if hasattr(closes.index, 'date') else closes.index
        daily_closes[ticker] = closes
        print(f"    {ticker:>8}: {len(df):>8,} bars  ({src})")
    else:
        print(f"    {ticker:>8}: MISSING")

loaded = set(ohlcv_data.keys())
print(f"\n  Loaded {len(loaded)}/{len(ALL_TICKERS)} tickers")
_release_memory()


# ==============================================================================
# TABLE 1: Per-Ticker Monthly Metrics
# ==============================================================================
print()
print("=" * W)
print(f"  TABLE 1: PER-TICKER MONTHLY REGIME METRICS")
print("=" * W)

all_rows = []

for ticker in ALL_TICKERS:
    if ticker not in ohlcv_data:
        continue
    df = ohlcv_data[ticker]
    src = load_sources[ticker]
    group = TICKER_GROUPS[ticker]

    for month_str, month_label in zip(MONTHS, MONTH_LABELS):
        if src == 'GCS_1min':
            m = compute_daily_metrics_1min(df, month_str)
        else:
            m = compute_daily_metrics_daily(df, month_str)
        if m is None:
            continue
        m['ticker'] = ticker
        m['group'] = group
        m['month'] = month_label
        all_rows.append(m)

df_metrics = pd.DataFrame(all_rows)

if df_metrics.empty:
    print("  No data loaded — cannot compute metrics.")
else:
    metric_cols = ['range_pct', 'spike_rate', 'crash_rate', 'mean_ret',
                   'reversal_rate', 'tail_ratio', 'mean_rev_score']
    col_labels = ['Range%', 'Spike%', 'Crash%', 'Ret%', 'Rev%', 'TailR', 'MeanRev']

    for grp_name, grp_label in [('A_deployed', 'GROUP A — DEPLOYED'),
                                 ('B_trained',  'GROUP B — TRAINED POOL'),
                                 ('C_benchmark','GROUP C — BENCHMARKS')]:
        grp = df_metrics[df_metrics['group'] == grp_name]
        if grp.empty:
            continue
        print(f"\n  {grp_label}")
        print(f"  {'Ticker':<8} {'Mon':>4}  {'Days':>4}  {col_labels[0]:>7} {col_labels[1]:>7} "
              f"{col_labels[2]:>7} {col_labels[3]:>7} {col_labels[4]:>7} {col_labels[5]:>7} {col_labels[6]:>7}")
        print(f"  {'─'*8} {'─'*4}  {'─'*4}  {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")

        for ticker in sorted(grp['ticker'].unique()):
            tk_rows = grp[grp['ticker'] == ticker].sort_values('month')
            for _, r in tk_rows.iterrows():
                tr = r['tail_ratio']
                mr = r['mean_rev_score']
                tr_s = f"{tr:7.2f}" if pd.notna(tr) else "    N/A"
                mr_s = f"{mr:7.2f}" if pd.notna(mr) else "    N/A"
                print(f"  {r['ticker']:<8} {r['month']:>4}  {r['n_days']:>4}  "
                      f"{r['range_pct']:>7.2f} {r['spike_rate']:>7.1f} "
                      f"{r['crash_rate']:>7.1f} {r['mean_ret']:>+7.2f} "
                      f"{r['reversal_rate']:>7.1f} {tr_s} {mr_s}")

    _release_memory()


    # ==========================================================================
    # TABLE 2: Group-Level Summary (mean ± std)
    # ==========================================================================
    print()
    print("=" * W)
    print(f"  TABLE 2: GROUP-LEVEL SUMMARY (mean ± std per month)")
    print("=" * W)

    for grp_name, grp_label in [('A_deployed', 'A_deployed'),
                                 ('B_trained',  'B_trained'),
                                 ('C_benchmark','C_benchmark')]:
        grp = df_metrics[df_metrics['group'] == grp_name]
        if grp.empty:
            continue
        print(f"\n  {grp_label}")
        print(f"  {'Month':>5}  {'N':>3}  {'Range%':>12} {'Spike%':>12} {'Crash%':>12} "
              f"{'Ret%':>12} {'Rev%':>12} {'TailR':>12} {'MeanRev':>12}")
        print(f"  {'─'*5}  {'─'*3}  {'─'*12} {'─'*12} {'─'*12} {'─'*12} {'─'*12} {'─'*12} {'─'*12}")

        for month in MONTH_LABELS:
            sub = grp[grp['month'] == month]
            if sub.empty:
                continue
            n = len(sub)
            parts = []
            for col in metric_cols:
                m_val = sub[col].mean()
                s_val = sub[col].std()
                if pd.notna(m_val):
                    parts.append(f"{m_val:>5.1f}±{s_val:>4.1f}")
                else:
                    parts.append(f"{'N/A':>12}")
            print(f"  {month:>5}  {n:>3}  {'  '.join(parts)}")


    # ==========================================================================
    # TABLE 3: April/May Avg → June Delta
    # ==========================================================================
    print()
    print("=" * W)
    print(f"  TABLE 3: REGIME SHIFT — (Apr+May avg) → June delta")
    print("=" * W)
    print(f"  {'Group':<14} {'Metric':<12} {'AprMay':>8} {'June':>8} {'Delta':>8} {'Direction'}")
    print(f"  {'─'*14} {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*20}")

    shift_summary = []

    for grp_name, grp_label in [('A_deployed', 'A_deployed'),
                                 ('B_trained',  'B_trained'),
                                 ('C_benchmark','C_benchmark')]:
        grp = df_metrics[df_metrics['group'] == grp_name]
        if grp.empty:
            continue

        pre = grp[grp['month'].isin(['Apr', 'May'])]
        jun = grp[grp['month'] == 'Jun']

        for col, label in zip(metric_cols, col_labels):
            pre_mean = pre[col].mean()
            jun_mean = jun[col].mean()
            if pd.isna(pre_mean) or pd.isna(jun_mean):
                continue
            delta = jun_mean - pre_mean

            if col == 'spike_rate':
                direction = "WORSE (fewer spikes)" if delta < -3 else ("BETTER" if delta > 3 else "STABLE")
            elif col == 'crash_rate':
                direction = "WORSE (more crashes)" if delta > 3 else ("BETTER" if delta < -3 else "STABLE")
            elif col == 'reversal_rate':
                direction = "WORSE (more reversals)" if delta > 3 else ("BETTER" if delta < -3 else "STABLE")
            elif col == 'range_pct':
                direction = "HIGHER VOL" if delta > 1 else ("LOWER VOL" if delta < -1 else "STABLE")
            elif col == 'mean_ret':
                direction = "BEARISH" if delta < -0.3 else ("BULLISH" if delta > 0.3 else "FLAT")
            elif col == 'tail_ratio':
                direction = "SKEW WORSE" if delta < -0.1 else ("SKEW BETTER" if delta > 0.1 else "STABLE")
            elif col == 'mean_rev_score':
                direction = "MORE REVERSAL" if delta > 0.05 else ("LESS REVERSAL" if delta < -0.05 else "STABLE")
            else:
                direction = ""

            print(f"  {grp_label:<14} {label:<12} {pre_mean:>8.2f} {jun_mean:>8.2f} {delta:>+8.2f} {direction}")
            shift_summary.append({
                'group': grp_label, 'metric': label,
                'pre': pre_mean, 'jun': jun_mean, 'delta': delta,
                'direction': direction,
            })

        print()


    # ==========================================================================
    # TABLE 4: Cross-Asset Correlation (intra-group)
    # ==========================================================================
    print()
    print("=" * W)
    print(f"  TABLE 4: INTRA-GROUP CORRELATION (daily returns)")
    print("=" * W)
    print(f"  Higher intra-group correlation = factor-driven moves (BTC, risk-off)")
    print(f"  If correlation spikes in June → correlated sector drawdown, not idiosyncratic")
    print()

    for grp_name, grp_label, tickers_list in [
        ('A_deployed', 'A_deployed', DEPLOY_TICKERS),
        ('B_trained',  'B_trained',  TRAINED_POOL),
    ]:
        tickers_with_data = [t for t in tickers_list if t in daily_closes]
        if len(tickers_with_data) < 3:
            print(f"  {grp_label}: too few tickers with data ({len(tickers_with_data)})")
            continue

        close_df = pd.DataFrame({t: daily_closes[t] for t in tickers_with_data})
        close_df.index = pd.to_datetime(close_df.index)
        returns_df = close_df.pct_change().dropna(how='all')

        print(f"  {grp_label}  ({len(tickers_with_data)} tickers)")
        print(f"  {'Month':>5}  {'MeanCorr':>10} {'MedianCorr':>12} {'MaxCorr':>10} {'MinCorr':>10}")
        print(f"  {'─'*5}  {'─'*10} {'─'*12} {'─'*10} {'─'*10}")

        for month_str, month_label in zip(MONTHS, MONTH_LABELS):
            mask = returns_df.index.strftime('%Y-%m') == month_str
            sub_ret = returns_df[mask].dropna(axis=1, how='all')
            if sub_ret.shape[1] < 3:
                continue
            corr = sub_ret.corr()
            upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
            vals = upper.stack().values
            if len(vals) == 0:
                continue
            print(f"  {month_label:>5}  {vals.mean():>10.3f} {np.median(vals):>12.3f} "
                  f"{vals.max():>10.3f} {vals.min():>10.3f}")

        print()

    # also show BTC correlation with deployed tickers
    if 'BTCUSD' in daily_closes:
        btc_rets = pd.Series(daily_closes['BTCUSD']).pct_change().dropna()
        btc_rets.index = pd.to_datetime(btc_rets.index)

        print(f"\n  BTC Correlation with Deployed Tickers")
        print(f"  {'Ticker':<8} {'Apr':>8} {'May':>8} {'Jun':>8} {'Δ(Jun-Pre)':>10}")
        print(f"  {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*10}")

        for ticker in DEPLOY_TICKERS:
            if ticker not in daily_closes:
                continue
            tk_rets = pd.Series(daily_closes[ticker]).pct_change().dropna()
            tk_rets.index = pd.to_datetime(tk_rets.index)
            merged = pd.concat([btc_rets, tk_rets], axis=1, keys=['BTC', ticker]).dropna()

            month_corrs = []
            for month_str in MONTHS:
                mask = merged.index.strftime('%Y-%m') == month_str
                sub = merged[mask]
                if len(sub) >= 5:
                    month_corrs.append(sub['BTC'].corr(sub[ticker]))
                else:
                    month_corrs.append(np.nan)

            if all(pd.notna(c) for c in month_corrs):
                pre_avg = (month_corrs[0] + month_corrs[1]) / 2
                delta = month_corrs[2] - pre_avg
                print(f"  {ticker:<8} {month_corrs[0]:>8.3f} {month_corrs[1]:>8.3f} "
                      f"{month_corrs[2]:>8.3f} {delta:>+10.3f}")

    _release_memory()


    # ==========================================================================
    # DIAGNOSIS
    # ==========================================================================
    print()
    print("=" * W)
    print(f"  DIAGNOSIS: MARKET-WIDE vs SECTOR-SPECIFIC")
    print("=" * W)

    df_shifts = pd.DataFrame(shift_summary)

    if not df_shifts.empty:
        a_shifts = df_shifts[df_shifts['group'] == 'A_deployed']
        b_shifts = df_shifts[df_shifts['group'] == 'B_trained']
        c_shifts = df_shifts[df_shifts['group'] == 'C_benchmark']

        def get_delta(sub, metric):
            row = sub[sub['metric'] == metric]
            return row['delta'].values[0] if len(row) > 0 else 0

        a_spike_d = get_delta(a_shifts, 'Spike%')
        c_spike_d = get_delta(c_shifts, 'Spike%')
        a_rev_d   = get_delta(a_shifts, 'Rev%')
        c_rev_d   = get_delta(c_shifts, 'Rev%')
        a_crash_d = get_delta(a_shifts, 'Crash%')
        c_crash_d = get_delta(c_shifts, 'Crash%')
        a_range_d = get_delta(a_shifts, 'Range%')
        c_range_d = get_delta(c_shifts, 'Range%')

        print()
        print(f"  Key comparisons (Deployed vs Benchmark delta):")
        print(f"    Spike Rate change:    Deployed {a_spike_d:+.1f}pp  vs  Benchmark {c_spike_d:+.1f}pp")
        print(f"    Reversal Rate change: Deployed {a_rev_d:+.1f}pp  vs  Benchmark {c_rev_d:+.1f}pp")
        print(f"    Crash Rate change:    Deployed {a_crash_d:+.1f}pp  vs  Benchmark {c_crash_d:+.1f}pp")
        print(f"    Range % change:       Deployed {a_range_d:+.1f}pp  vs  Benchmark {c_range_d:+.1f}pp")

        print()
        spike_worse_a = a_spike_d < -3
        rev_worse_a   = a_rev_d > 3
        range_up_a    = a_range_d > 1
        spike_worse_c = c_spike_d < -3
        rev_worse_c   = c_rev_d > 3
        range_up_c    = c_range_d > 1

        if spike_worse_a and not spike_worse_c:
            print(f"  → SECTOR-SPECIFIC: Spike rate dropped for trained assets but NOT benchmarks")
            print(f"    This suggests the speculative small-cap universe changed, not the broad market.")
        elif spike_worse_a and spike_worse_c:
            print(f"  → MARKET-WIDE: Both trained assets AND benchmarks show spike rate decline")
            print(f"    Broad market regime change — not fixable by retraining on same universe.")
        elif not spike_worse_a and not spike_worse_c:
            print(f"  → SPIKE RATE STABLE: The problem is NOT fewer spikes — it's spike quality.")
        else:
            print(f"  → MIXED: Benchmark spike rate dropped but trained assets didn't — unusual.")

        if rev_worse_a and not rev_worse_c:
            print(f"  → SECTOR-SPECIFIC: Reversal rate (spike then crash) rose for trained assets only")
            print(f"    Classic Cat-B/Cat-E: spikes appear but immediately reverse.")
        elif rev_worse_a and rev_worse_c:
            print(f"  → MARKET-WIDE: Reversals increased across all asset classes")

        if range_up_a and range_up_c:
            print(f"  → VOLATILITY UP MARKET-WIDE: Daily range increased for both groups")
            print(f"    Higher vol + more reversals = choppy environment hostile to directional bets.")
        elif range_up_a and not range_up_c:
            print(f"  → VOLATILITY SECTOR-SPECIFIC: Only trained assets got more volatile")

        print()
        print(f"  Summary:")
        n_sector = sum([spike_worse_a and not spike_worse_c,
                        rev_worse_a and not rev_worse_c,
                        range_up_a and not range_up_c])
        n_market = sum([spike_worse_a and spike_worse_c,
                        rev_worse_a and rev_worse_c,
                        range_up_a and range_up_c])
        n_stable = sum([not spike_worse_a and not spike_worse_c,
                        not rev_worse_a and not rev_worse_c,
                        not range_up_a and not range_up_c])

        if n_market > n_sector:
            print(f"    VERDICT: Primarily MARKET-WIDE regime shift ({n_market}/3 metrics)")
            print(f"    Action: Regime gate / pause strategy during high-reversal periods")
        elif n_sector > n_market:
            print(f"    VERDICT: Primarily SECTOR-SPECIFIC degradation ({n_sector}/3 metrics)")
            print(f"    Action: Retrain on recent data / add sector-specific regime filter")
        else:
            print(f"    VERDICT: MIXED — some market-wide, some sector-specific")
            print(f"    Action: Both sector filter AND regime gate may be needed")

print()
print("=" * W)
print(f"  Analysis complete.")
print("=" * W)
