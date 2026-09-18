"""
analyze_opening_swings.py — Opening Price Swing Analysis (All OHLCV Assets)
===========================================================================
Scans ALL tickers with OHLCV data in GCS. For each trading day, measures
the price swing in the first N minutes after market open (6:30 AM PST):

  - Max drawdown from open (deepest dip)
  - Max runup from open (highest rip)
  - Which comes first (dip-before-rip vs rip-before-dip)
  - Swing amplitude (total range) and direction bias
  - Time to trough / time to peak
  - Pattern classification: dip_then_rip, rip_then_dip, mostly_up, mostly_down, choppy
  - Dip fill rates at 0.5/1/1.5/2/3% levels
  - Dip-then-recover rates (dip >= X% but EOD positive)

Ranks assets by swing amplitude and identifies limit-entry candidates.

Run in Colab or any environment with GCS access.
"""

import gc
import io
import warnings
import numpy as np
import pandas as pd
import pytz
from datetime import datetime

warnings.filterwarnings('ignore')

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

from google.cloud import storage as gcs_storage

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BUCKET_NAME   = os.environ.get("GCS_BUCKET", "")
INFERENCE_DIR = "Data/Inference"

PST = pytz.timezone('America/Los_Angeles')

MARKET_OPEN_MINUTE = 390      # 6:30 AM PST = 390 minutes from midnight
SWING_WINDOW_MINUTES = 60     # analyze first 60 minutes after open
EOD_MINUTE = 780              # 1:00 PM PST = 780 minutes from midnight

DIP_LEVELS = [0.005, 0.01, 0.015, 0.02, 0.03]

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

EXCLUDE_TICKERS = {
    'SPY', 'VIX', 'TLT', 'IWM', 'SMH', 'ITA', 'XLE', 'BTCUSD',
    'QQQ', 'HYG', 'USO', 'SLV', 'MOD',
}

DATA_START_DATE = "2022-01-01"

SECTORS = {
    'crypto':      ['RIOT', 'CLSK', 'HUT', 'MARA', 'BTBT', 'CIFR', 'IREN', 'APLD',
                    'CORZ', 'COIN', 'MSTR', 'KEEL', 'WULF', 'HOOD'],
    'defense':     ['LUNR', 'RKLB', 'KTOS', 'AVAV', 'LMT', 'NOC', 'RTX', 'LHX',
                    'GD', 'BKSY', 'PL', 'RDW', 'MNTS', 'NNDM', 'SPIR'],
    'energy':      ['FCEL', 'PLUG', 'BE', 'FLNC', 'SMR', 'OKLO', 'CCJ', 'CEG',
                    'NEE', 'DUK', 'VST', 'GEV', 'NRG', 'TLN'],
    'space_evtol': ['ASTS', 'ACHR', 'JOBY', 'RCAT', 'BBAI', 'SMLR', 'VOYG'],
    'nuclear':     ['SMR', 'OKLO', 'CCJ', 'CEG', 'NXE', 'NBIS', 'BWXT'],
    'tech':        ['NVDA', 'AMD', 'PLTR', 'SMCI', 'TSLA', 'VRT', 'POWL',
                    'ALAB', 'MVIS', 'MU'],
}
SECTOR_LOOKUP = {}
for _sector, _tickers in SECTORS.items():
    for _t in _tickers:
        if _t not in SECTOR_LOOKUP:
            SECTOR_LOOKUP[_t] = _sector

TOP_N_DETAIL = 20
W = 140

client = gcs_storage.Client()
bucket = client.bucket(BUCKET_NAME)

# ==============================================================================
# DISCOVER ALL OHLCV ASSETS IN GCS
# ==============================================================================
def discover_tickers():
    blobs = bucket.list_blobs(prefix=f"{INFERENCE_DIR}/")
    tickers = set()
    for b in blobs:
        name = b.name.replace(f"{INFERENCE_DIR}/", "")
        if '/' in name:
            continue
        if name.endswith('_1m_full_historical_data.parquet'):
            t = name.replace('_1m_full_historical_data.parquet', '')
            tickers.add(t)
        elif name.endswith('_master_inference.parquet'):
            t = name.replace('_master_inference.parquet', '')
            tickers.add(t)
    tickers -= EXCLUDE_TICKERS
    return sorted(tickers)


# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_ohlcv(ticker):
    clean = ticker.replace("I:", "").replace("X:", "")
    file_ticker = TICKER_FILE_MAP.get(clean, clean)
    for suffix in ['_1m_full_historical_data', '_master_inference']:
        blob_path = f"{INFERENCE_DIR}/{file_ticker}{suffix}.parquet"
        blob = bucket.blob(blob_path)
        if blob.exists():
            df = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
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
            df = df.set_index('Datetime')
            df = df[~df.index.duplicated(keep='first')]
            std_map = {'open': 'Open', 'high': 'High', 'low': 'Low',
                       'close': 'Close', 'volume': 'Volume'}
            lower_map = {k.lower(): v for k, v in std_map.items()}
            df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip())
                          for c in df.columns]
            df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()
            df = df[df.index.notna()].sort_index()
            cutoff = pd.Timestamp(DATA_START_DATE, tz='UTC')
            df = df[df.index >= cutoff]
            keep = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
            return df[keep]
    return None


def heal_data(df):
    df = df.resample('1min').asfreq()
    for c in ['Open', 'High', 'Low', 'Close']:
        if c in df.columns:
            df[c] = df[c].ffill()
    for c in ['Volume']:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df


# ==============================================================================
# PER-DAY SWING METRICS
# ==============================================================================
def compute_daily_swings(df):
    df_pst = df.copy()
    df_pst.index = df_pst.index.tz_convert(PST)
    mins = df_pst.index.hour * 60 + df_pst.index.minute

    window_start = MARKET_OPEN_MINUTE
    window_end = MARKET_OPEN_MINUTE + SWING_WINDOW_MINUTES

    in_window = (mins >= window_start) & (mins <= window_end) & (df_pst['Volume'] > 0)
    in_eod = (mins >= EOD_MINUTE) & (df_pst['Volume'] > 0)

    window_bars = df_pst[in_window].copy()
    window_bars['minute'] = window_bars.index.hour * 60 + window_bars.index.minute
    window_bars['date'] = window_bars.index.date

    eod_bars = df_pst[in_eod].copy()
    eod_bars['date'] = eod_bars.index.date
    eod_closes = eod_bars.groupby('date')['Close'].first()

    results = []
    for date, day_bars in window_bars.groupby('date'):
        if len(day_bars) < 3:
            continue

        open_bar = day_bars.iloc[0]
        open_price = float(open_bar['Close'])
        if open_price <= 0:
            continue

        highs = day_bars['High'].values
        lows = day_bars['Low'].values
        minutes = day_bars['minute'].values

        max_high = highs.max()
        min_low = lows.min()

        max_runup_pct = (max_high - open_price) / open_price
        max_drawdown_pct = (min_low - open_price) / open_price

        peak_idx = np.argmax(highs)
        trough_idx = np.argmin(lows)

        time_to_peak = int(minutes[peak_idx]) - window_start
        time_to_trough = int(minutes[trough_idx]) - window_start

        dip_first = trough_idx < peak_idx

        abs_dd = abs(max_drawdown_pct)
        swing_amplitude = max_runup_pct + abs_dd
        swing_ratio = max_runup_pct / abs_dd if abs_dd > 1e-6 else 999.0

        if abs_dd < 0.003 and max_runup_pct >= 0.005:
            pattern = 'mostly_up'
        elif max_runup_pct < 0.003 and abs_dd >= 0.005:
            pattern = 'mostly_down'
        elif dip_first and abs_dd >= 0.005 and max_runup_pct >= 0.005:
            pattern = 'dip_then_rip'
        elif not dip_first and abs_dd >= 0.005 and max_runup_pct >= 0.005:
            pattern = 'rip_then_dip'
        else:
            pattern = 'choppy'

        eod_close = float(eod_closes.get(date, np.nan))
        eod_pnl = (eod_close - open_price) / open_price if not np.isnan(eod_close) else np.nan

        results.append({
            'date': date,
            'open_price': open_price,
            'max_drawdown_pct': max_drawdown_pct,
            'max_runup_pct': max_runup_pct,
            'time_to_trough': time_to_trough,
            'time_to_peak': time_to_peak,
            'dip_first': dip_first,
            'swing_amplitude': swing_amplitude,
            'swing_ratio': swing_ratio,
            'pattern': pattern,
            'eod_pnl': eod_pnl,
        })

    return pd.DataFrame(results)


# ==============================================================================
# PER-TICKER AGGREGATION
# ==============================================================================
def aggregate_ticker(ticker, daily_df):
    n_days = len(daily_df)
    if n_days < 10:
        return None

    pattern_counts = daily_df['pattern'].value_counts()
    dominant_pattern = pattern_counts.index[0]
    consistency = pattern_counts.iloc[0] / n_days

    dip_fill = {}
    dip_recover = {}
    for lvl in DIP_LEVELS:
        dipped = daily_df['max_drawdown_pct'] <= -lvl
        fill_rate = dipped.mean()
        dip_fill[lvl] = fill_rate

        if dipped.any():
            recovered = daily_df.loc[dipped, 'eod_pnl'] > 0
            dip_recover[lvl] = recovered.mean() if not recovered.empty else 0.0
        else:
            dip_recover[lvl] = 0.0

    dd = daily_df['max_drawdown_pct']
    ru = daily_df['max_runup_pct']

    result = {
        'ticker': ticker,
        'sector': SECTOR_LOOKUP.get(ticker, 'other'),
        'n_days': n_days,
        'med_drawdown': dd.median(),
        'mean_drawdown': dd.mean(),
        'std_drawdown': dd.std(),
        'med_runup': ru.median(),
        'mean_runup': ru.mean(),
        'std_runup': ru.std(),
        'med_amplitude': daily_df['swing_amplitude'].median(),
        'mean_amplitude': daily_df['swing_amplitude'].mean(),
        'med_swing_ratio': daily_df['swing_ratio'].clip(upper=10).median(),
        'med_time_to_trough': daily_df['time_to_trough'].median(),
        'med_time_to_peak': daily_df['time_to_peak'].median(),
        'pct_dip_first': daily_df['dip_first'].mean(),
        'dominant_pattern': dominant_pattern,
        'consistency': consistency,
        'med_eod_pnl': daily_df['eod_pnl'].median(),
    }

    for p in ['dip_then_rip', 'rip_then_dip', 'mostly_up', 'mostly_down', 'choppy']:
        result[f'pct_{p}'] = pattern_counts.get(p, 0) / n_days

    for lvl in DIP_LEVELS:
        key = f"{lvl*100:.1f}".replace('.', '_')
        result[f'fill_{key}pct'] = dip_fill[lvl]
        result[f'recover_{key}pct'] = dip_recover[lvl]

    best_fill_lvl = None
    for lvl in DIP_LEVELS:
        if dip_fill[lvl] >= 0.60 and dip_recover[lvl] >= 0.50:
            best_fill_lvl = lvl
            break

    result['limit_candidate'] = best_fill_lvl is not None
    result['best_dip_pct'] = best_fill_lvl

    return result


# ==============================================================================
# MAIN LOOP
# ==============================================================================
print(f"{'='*W}")
print(f"  OPENING SWING ANALYSIS")
print(f"  Window: first {SWING_WINDOW_MINUTES} minutes after market open (6:30 AM PST)")
print(f"  Data: {DATA_START_DATE} → present")
print(f"{'='*W}")
print()

print("Discovering tickers in GCS...")
all_tickers = discover_tickers()
print(f"  Found {len(all_tickers)} tickers (excl. context/indices)")
print()

all_summaries = []
all_daily = {}

for i, ticker in enumerate(all_tickers):
    try:
        df = load_ohlcv(ticker)
        if df is None:
            print(f"  [{i+1}/{len(all_tickers)}] {ticker}: no data")
            continue
        df = heal_data(df)
        daily_df = compute_daily_swings(df)
        if daily_df.empty or len(daily_df) < 10:
            print(f"  [{i+1}/{len(all_tickers)}] {ticker}: <10 trading days, skip")
            del df, daily_df
            gc.collect()
            continue

        summary = aggregate_ticker(ticker, daily_df)
        if summary:
            all_summaries.append(summary)
            all_daily[ticker] = daily_df
            print(f"  [{i+1}/{len(all_tickers)}] {ticker}: {summary['n_days']} days, "
                  f"amp={summary['med_amplitude']:.1%}, "
                  f"dd={summary['med_drawdown']:.1%}, "
                  f"ru={summary['med_runup']:.1%}, "
                  f"dip1%={summary['fill_1_0pct']:.0%}")
        del df
        gc.collect()
    except Exception as e:
        print(f"  [{i+1}/{len(all_tickers)}] {ticker}: ERROR — {e}")

results_df = pd.DataFrame(all_summaries)
if results_df.empty:
    print("\nNo tickers processed. Check GCS access.")
else:
    results_df = results_df.sort_values('med_amplitude', ascending=False).reset_index(drop=True)

    # ==============================================================================
    # DASHBOARD — RANKED TABLE
    # ==============================================================================
    print()
    print(f"{'='*W}")
    print(f"  OPENING SWING RANKINGS — Top {len(results_df)} tickers by median swing amplitude")
    print(f"{'─'*W}")
    print(f"  {'Rank':>4}  {'Ticker':<6}  {'Sector':<10}  {'Days':>5}  "
          f"{'Med DD':>8}  {'Med RU':>8}  {'Med Amp':>8}  {'Ratio':>6}  "
          f"{'DipFirst':>8}  {'Dominant':<14}  {'Consist':>7}  "
          f"{'Fill@1%':>7}  {'Fill@2%':>7}  {'Recov@1%':>8}  {'Limit?'}")
    print(f"  {'─'*W}")

    for idx, row in results_df.iterrows():
        limit_str = f"YES dip={row['best_dip_pct']*100:.1f}%" if row['limit_candidate'] else "no"
        print(f"  {idx+1:>4}  {row['ticker']:<6}  {row['sector']:<10}  {row['n_days']:>5}  "
              f"{row['med_drawdown']:>7.1%}  {row['med_runup']:>7.1%}  "
              f"{row['med_amplitude']:>7.1%}  {row['med_swing_ratio']:>5.1f}x  "
              f"{row['pct_dip_first']:>7.0%}  {row['dominant_pattern']:<14}  "
              f"{row['consistency']:>6.0%}  "
              f"{row['fill_1_0pct']:>6.0%}  {row['fill_2_0pct']:>6.0%}  "
              f"{row['recover_1_0pct']:>7.0%}  {limit_str}")

    # ==============================================================================
    # LIMIT ENTRY CANDIDATES
    # ==============================================================================
    candidates = results_df[results_df['limit_candidate']].copy()
    print()
    print(f"{'='*W}")
    print(f"  LIMIT ENTRY CANDIDATES ({len(candidates)} tickers)")
    print(f"  Criteria: fill_rate >= 60% at some dip level AND recover_rate >= 50%")
    print(f"{'─'*W}")

    if candidates.empty:
        print("  No tickers meet the criteria. Consider relaxing thresholds.")
    else:
        candidates = candidates.sort_values('med_amplitude', ascending=False)
        print(f"  {'Ticker':<6}  {'Sector':<10}  {'Best Dip':>8}  "
              f"{'Fill Rate':>9}  {'Recover':>8}  {'Med Amp':>8}  "
              f"{'Med DD':>8}  {'Med RU':>8}  {'DipFirst':>8}  {'EOD PnL':>8}")
        print(f"  {'─'*W}")
        for _, row in candidates.iterrows():
            lvl = row['best_dip_pct']
            key = f"{lvl*100:.1f}".replace('.', '_')
            fill = row[f'fill_{key}pct']
            recov = row[f'recover_{key}pct']
            print(f"  {row['ticker']:<6}  {row['sector']:<10}  "
                  f"{lvl:>7.1%}  {fill:>8.0%}  {recov:>7.0%}  "
                  f"{row['med_amplitude']:>7.1%}  "
                  f"{row['med_drawdown']:>7.1%}  {row['med_runup']:>7.1%}  "
                  f"{row['pct_dip_first']:>7.0%}  {row['med_eod_pnl']:>7.2%}")

    # ==============================================================================
    # DIP FILL CURVE — AGGREGATE
    # ==============================================================================
    print()
    print(f"{'='*W}")
    print(f"  DIP FILL CURVE (median across all tickers)")
    print(f"{'─'*W}")
    for lvl in DIP_LEVELS:
        key = f"{lvl*100:.1f}".replace('.', '_')
        med_fill = results_df[f'fill_{key}pct'].median()
        med_recov = results_df[f'recover_{key}pct'].median()
        n_above_60 = (results_df[f'fill_{key}pct'] >= 0.60).sum()
        print(f"  Dip >= {lvl:>5.1%}:  fill={med_fill:>5.0%} (median)   "
              f"recover={med_recov:>5.0%} (median)   "
              f"{n_above_60} tickers >= 60% fill")

    # ==============================================================================
    # PER-TICKER DETAIL BLOCKS (top N by amplitude)
    # ==============================================================================
    print()
    print(f"{'='*W}")
    print(f"  PER-TICKER DETAIL (Top {TOP_N_DETAIL} by swing amplitude)")
    print(f"{'='*W}")

    for idx, row in results_df.head(TOP_N_DETAIL).iterrows():
        ticker = row['ticker']
        daily_df = all_daily.get(ticker)
        if daily_df is None:
            continue

        print()
        print(f"  {'─'*W}")
        print(f"  {ticker} ({row['sector']})  —  {row['n_days']} trading days")
        print(f"  {'─'*W}")

        print(f"    Drawdown:    med={row['med_drawdown']:>7.2%}  "
              f"mean={row['mean_drawdown']:>7.2%}  std={row['std_drawdown']:>6.2%}")
        print(f"    Runup:       med={row['med_runup']:>7.2%}  "
              f"mean={row['mean_runup']:>7.2%}  std={row['std_runup']:>6.2%}")
        print(f"    Amplitude:   med={row['med_amplitude']:>7.2%}  "
              f"mean={row['mean_amplitude']:>7.2%}")
        print(f"    Swing ratio: {row['med_swing_ratio']:.1f}x  "
              f"(>1 = bullish bias)")
        print(f"    Timing:      trough @ +{row['med_time_to_trough']:.0f}min  "
              f"peak @ +{row['med_time_to_peak']:.0f}min  "
              f"dip_first={row['pct_dip_first']:.0%}")
        print(f"    EOD PnL:     med={row['med_eod_pnl']:>7.2%}")
        print()

        print(f"    Pattern distribution:")
        for p in ['dip_then_rip', 'rip_then_dip', 'mostly_up', 'mostly_down', 'choppy']:
            pct = row.get(f'pct_{p}', 0)
            bar = '█' * int(pct * 40)
            print(f"      {p:<16} {pct:>5.0%}  {bar}")

        print()
        print(f"    Dip fill curve:")
        print(f"      {'Dip Level':>10}  {'Fill Rate':>9}  {'Recover':>8}  "
              f"{'Avg PnL if filled':>18}")
        for lvl in DIP_LEVELS:
            key = f"{lvl*100:.1f}".replace('.', '_')
            fill = row[f'fill_{key}pct']
            recov = row[f'recover_{key}pct']

            dipped = daily_df['max_drawdown_pct'] <= -lvl
            if dipped.any():
                avg_pnl = daily_df.loc[dipped, 'eod_pnl'].mean()
                avg_pnl_str = f"{avg_pnl:>7.2%}" if not np.isnan(avg_pnl) else "   N/A"
            else:
                avg_pnl_str = "   N/A"

            bar = '█' * int(fill * 30)
            print(f"      {lvl:>9.1%}  {fill:>8.0%}  {recov:>7.0%}  "
                  f"{avg_pnl_str:>18}  {bar}")

        p10_dd = daily_df['max_drawdown_pct'].quantile(0.10)
        p25_dd = daily_df['max_drawdown_pct'].quantile(0.25)
        p50_dd = daily_df['max_drawdown_pct'].quantile(0.50)
        p75_dd = daily_df['max_drawdown_pct'].quantile(0.75)
        p90_dd = daily_df['max_drawdown_pct'].quantile(0.90)
        print()
        print(f"    Drawdown percentiles:  "
              f"P10={p10_dd:.2%}  P25={p25_dd:.2%}  P50={p50_dd:.2%}  "
              f"P75={p75_dd:.2%}  P90={p90_dd:.2%}")

    # ==============================================================================
    # SECTOR SUMMARY
    # ==============================================================================
    print()
    print(f"{'='*W}")
    print(f"  SECTOR SUMMARY")
    print(f"{'─'*W}")
    print(f"  {'Sector':<12}  {'Count':>5}  {'Med Amp':>8}  {'Med DD':>8}  "
          f"{'Med RU':>8}  {'Limit Cands':>11}  {'Avg Fill@1%':>11}")
    print(f"  {'─'*W}")

    for sector in sorted(results_df['sector'].unique()):
        sec_df = results_df[results_df['sector'] == sector]
        n_cands = sec_df['limit_candidate'].sum()
        print(f"  {sector:<12}  {len(sec_df):>5}  "
              f"{sec_df['med_amplitude'].median():>7.1%}  "
              f"{sec_df['med_drawdown'].median():>7.1%}  "
              f"{sec_df['med_runup'].median():>7.1%}  "
              f"{n_cands:>11}  "
              f"{sec_df['fill_1_0pct'].mean():>10.0%}")

    print()
    print(f"{'='*W}")
    print(f"  Done. {len(results_df)} tickers analyzed.")
    print(f"{'='*W}")
