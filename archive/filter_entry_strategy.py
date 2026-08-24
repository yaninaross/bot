"""
filter_entry_strategy.py — Asset Entry Strategy Filter
======================================================
For each ticker with OHLCV data, computes opening swing statistics and
classifies into one of three entry strategies:

  LIMIT_ENTRY  — consistent dip-first pattern, deep enough dip,
                 acceptable recovery. Place limit buy at drawdown.
  MARKET_BUY   — bullish open bias (rip-first or mostly-up), enter at market.
  SKIP         — bearish or choppy open with negative EOD PnL.

Also produces a recommended dip_pct for LIMIT_ENTRY assets and flags
which deployed tickers should switch entry mode.

Uses memory-efficient processing (trim_to_market_hours, malloc_trim).
Run in Colab or any environment with GCS access.
"""

import gc
import io
import ctypes
import warnings
import numpy as np
import pandas as pd
import pytz

warnings.filterwarnings('ignore')

try:
    _libc = ctypes.CDLL("libc.so.6")
    def _malloc_trim():
        _libc.malloc_trim(0)
        gc.collect()
except Exception:
    def _malloc_trim():
        gc.collect()

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

from google.cloud import storage as gcs_storage

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BUCKET_NAME   = "time-series-bucket-ys"
INFERENCE_DIR = "Data/Inference"

PST = pytz.timezone('America/Los_Angeles')

MARKET_OPEN_MINUTE = 390
SWING_WINDOW_MINUTES = 60
EOD_MINUTE = 780

DATA_START_DATE = "2022-01-01"
MIN_DAYS = 100

DIP_LEVELS = [0.005, 0.01, 0.015, 0.02, 0.03]

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

EXCLUDE_TICKERS = {
    'SPY', 'VIX', 'TLT', 'IWM', 'SMH', 'ITA', 'XLE', 'BTCUSD',
    'QQQ', 'HYG', 'USO', 'SLV', 'MOD',
}

# ── Classification thresholds ────────────────────────────────────────────────
# LIMIT_ENTRY: dip-first ≥ this fraction of days AND median dip deep enough
LIMIT_DIP_FIRST_MIN = 0.45
LIMIT_MED_DD_MIN    = -0.008       # median drawdown must be ≥ 0.8% (negative)
LIMIT_FILL_MIN      = 0.50         # fill rate at recommended dip level
LIMIT_RECOVER_MIN   = 0.40         # recovery rate at recommended dip level

# MARKET_BUY: runup-biased opens
MARKET_SWING_RATIO_MIN = 1.2       # median swing_ratio (runup/|dd|) ≥ 1.2
MARKET_RUNUP_MIN       = 0.005     # median runup ≥ 0.5%

# SKIP: negative EOD PnL or very choppy
SKIP_EOD_MAX = -0.002              # median EOD PnL worse than -0.2%

# ── Deployed tickers (for cross-reference) ───────────────────────────────────
DEPLOYED = {
    'EXP_152': [],   # populated from BQ at runtime
    'EXP_160': [],   # populated from BQ at runtime
    'EXP_170': ['ALAB', 'BE', 'OKLO', 'RKLB'],
    'EXP_185': ['PL', 'RDW', 'SMCI'],
}
ALL_DEPLOYED = set()
for _v in DEPLOYED.values():
    ALL_DEPLOYED.update(_v)

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

W = 130

client = gcs_storage.Client()
bucket = client.bucket(BUCKET_NAME)


# ==============================================================================
# DISCOVER TICKERS
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
# DATA LOADING (memory-efficient)
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


def trim_to_market_hours(df):
    pst_idx = df.index.tz_convert(PST)
    mins = pst_idx.hour * 60 + pst_idx.minute
    mask = (mins >= 360) & (mins <= 840)
    return df[mask].copy()


def heal_market_hours(df):
    for c in ['Open', 'High', 'Low', 'Close']:
        if c in df.columns:
            df[c] = df[c].ffill()
    for c in ['Volume']:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df


# ==============================================================================
# PER-DAY SWING METRICS (vectorized as much as possible)
# ==============================================================================
def compute_daily_swings(df):
    pst_idx = df.index.tz_convert(PST)
    mins_arr = pst_idx.hour * 60 + pst_idx.minute

    window_start = MARKET_OPEN_MINUTE
    window_end = MARKET_OPEN_MINUTE + SWING_WINDOW_MINUTES

    in_window = (mins_arr >= window_start) & (mins_arr <= window_end) & (df['Volume'] > 0)
    in_eod = (mins_arr >= EOD_MINUTE) & (df['Volume'] > 0)

    w_df = df[in_window]
    w_dates = pst_idx[in_window].date
    w_mins = mins_arr[in_window]

    eod_df = df[in_eod]
    eod_dates = pst_idx[in_eod].date
    eod_closes = pd.Series(eod_df['Close'].values, index=eod_dates)
    eod_closes = eod_closes.groupby(eod_closes.index).first()

    close_vals = w_df['Close'].values
    high_vals = w_df['High'].values
    low_vals = w_df['Low'].values
    vol_vals = w_df['Volume'].values
    min_vals = w_mins.values if hasattr(w_mins, 'values') else np.array(w_mins)
    date_arr = np.array(w_dates)

    results = []
    for date in pd.unique(w_dates):
        dm = date_arr == date
        d_close = close_vals[dm]
        d_high = high_vals[dm]
        d_low = low_vals[dm]
        d_vol = vol_vals[dm]
        d_mins = min_vals[dm]

        real = d_vol > 0
        if real.sum() < 3:
            continue

        first_real = np.argmax(real)
        open_price = float(d_close[first_real])
        if open_price <= 0:
            continue

        max_high = d_high.max()
        min_low = d_low.min()
        max_runup_pct = (max_high - open_price) / open_price
        max_drawdown_pct = (min_low - open_price) / open_price

        peak_idx = np.argmax(d_high)
        trough_idx = np.argmin(d_low)
        time_to_peak = int(d_mins[peak_idx]) - window_start
        time_to_trough = int(d_mins[trough_idx]) - window_start
        dip_first = trough_idx < peak_idx

        abs_dd = abs(max_drawdown_pct)
        swing_amplitude = max_runup_pct + abs_dd
        swing_ratio = max_runup_pct / abs_dd if abs_dd > 1e-6 else 99.0

        eod_close = float(eod_closes.get(date, np.nan))
        eod_pnl = (eod_close - open_price) / open_price if not np.isnan(eod_close) else np.nan

        results.append({
            'date': date,
            'max_drawdown_pct': max_drawdown_pct,
            'max_runup_pct': max_runup_pct,
            'time_to_trough': time_to_trough,
            'time_to_peak': time_to_peak,
            'dip_first': dip_first,
            'swing_amplitude': swing_amplitude,
            'swing_ratio': swing_ratio,
            'eod_pnl': eod_pnl,
        })

    return pd.DataFrame(results)


# ==============================================================================
# CLASSIFICATION LOGIC
# ==============================================================================
def classify_ticker(stats):
    """
    Returns (strategy, reason, recommended_dip_pct).

    Priority order:
      1. SKIP if median EOD PnL is deeply negative (losing money on average)
      2. LIMIT_ENTRY if dip-first is frequent enough AND deep enough AND recovers
      3. MARKET_BUY if runup-biased
      4. SKIP otherwise (choppy / unpredictable)
    """
    med_eod = stats['med_eod_pnl']
    med_dd = stats['med_drawdown']
    med_ru = stats['med_runup']
    dip_first_pct = stats['pct_dip_first']
    med_ratio = stats['med_swing_ratio']

    # ── 1. Hard SKIP: consistently losing money ──────────────────────────────
    if not np.isnan(med_eod) and med_eod < SKIP_EOD_MAX:
        return 'SKIP', f"negative EOD ({med_eod:+.2%})", None

    # ── 2. LIMIT_ENTRY candidate ─────────────────────────────────────────────
    if dip_first_pct >= LIMIT_DIP_FIRST_MIN and med_dd <= LIMIT_MED_DD_MIN:
        best_dip = None
        best_score = -1
        for lvl in DIP_LEVELS:
            fill = stats.get(f'fill_{lvl}', 0)
            recov = stats.get(f'recov_{lvl}', 0)
            if fill >= LIMIT_FILL_MIN and recov >= LIMIT_RECOVER_MIN:
                score = fill * recov
                if score > best_score:
                    best_score = score
                    best_dip = lvl

        if best_dip is not None:
            fill = stats.get(f'fill_{best_dip}', 0)
            recov = stats.get(f'recov_{best_dip}', 0)
            return ('LIMIT_ENTRY',
                    f"dip_first={dip_first_pct:.0%} fill={fill:.0%} recov={recov:.0%}",
                    best_dip)

        p25_dd = stats.get('p25_drawdown', med_dd)
        suggested = round(abs(p25_dd) * 0.6, 4)
        suggested = max(0.003, min(suggested, 0.03))
        return ('LIMIT_ENTRY',
                f"dip_first={dip_first_pct:.0%} (use dip={suggested:.1%} from P25 DD)",
                suggested)

    # ── 3. MARKET_BUY: bullish open bias ─────────────────────────────────────
    if med_ratio >= MARKET_SWING_RATIO_MIN and med_ru >= MARKET_RUNUP_MIN:
        return 'MARKET_BUY', f"ratio={med_ratio:.1f}x runup={med_ru:+.2%}", None

    # ── 4. Default: SKIP ─────────────────────────────────────────────────────
    return 'SKIP', f"choppy (ratio={med_ratio:.1f}x, dip_first={dip_first_pct:.0%})", None


def compute_ticker_stats(ticker, daily_df):
    n = len(daily_df)
    dd = daily_df['max_drawdown_pct']
    ru = daily_df['max_runup_pct']

    stats = {
        'ticker': ticker,
        'sector': SECTOR_LOOKUP.get(ticker, 'other'),
        'n_days': n,
        'med_drawdown': dd.median(),
        'mean_drawdown': dd.mean(),
        'std_drawdown': dd.std(),
        'p25_drawdown': dd.quantile(0.25),
        'med_runup': ru.median(),
        'mean_runup': ru.mean(),
        'med_amplitude': daily_df['swing_amplitude'].median(),
        'med_swing_ratio': daily_df['swing_ratio'].clip(upper=10).median(),
        'pct_dip_first': daily_df['dip_first'].mean(),
        'med_time_to_trough': daily_df['time_to_trough'].median(),
        'med_eod_pnl': daily_df['eod_pnl'].median(),
        'mean_eod_pnl': daily_df['eod_pnl'].mean(),
        'deployed': ticker in ALL_DEPLOYED,
    }

    for lvl in DIP_LEVELS:
        dipped = dd <= -lvl
        fill_rate = dipped.mean()
        if dipped.any():
            valid_eod = daily_df.loc[dipped, 'eod_pnl'].dropna()
            recov_rate = (valid_eod > 0).mean() if len(valid_eod) > 0 else 0.0
        else:
            recov_rate = 0.0
        stats[f'fill_{lvl}'] = fill_rate
        stats[f'recov_{lvl}'] = recov_rate

    return stats


# ==============================================================================
# MAIN
# ==============================================================================
print(f"{'='*W}")
print(f"  ASSET ENTRY STRATEGY FILTER")
print(f"  Window: first {SWING_WINDOW_MINUTES} min after open")
print(f"  Data: {DATA_START_DATE} → present  |  Min days: {MIN_DAYS}")
print(f"{'='*W}")
print()

print("Discovering tickers...")
all_tickers = discover_tickers()
print(f"  Found {len(all_tickers)} tickers")
print()

all_stats = []

for i, ticker in enumerate(all_tickers):
    try:
        df = load_ohlcv(ticker)
        if df is None:
            continue
        df = trim_to_market_hours(df)
        df = heal_market_hours(df)

        daily_df = compute_daily_swings(df)
        del df
        _malloc_trim()

        if daily_df.empty or len(daily_df) < MIN_DAYS:
            n = 0 if daily_df.empty else len(daily_df)
            if n > 0:
                print(f"  [{i+1}/{len(all_tickers)}] {ticker}: {n} days (< {MIN_DAYS}, skip)")
            continue

        stats = compute_ticker_stats(ticker, daily_df)
        strategy, reason, dip_pct = classify_ticker(stats)
        stats['strategy'] = strategy
        stats['reason'] = reason
        stats['rec_dip_pct'] = dip_pct
        all_stats.append(stats)

        flag = " [DEPLOYED]" if stats['deployed'] else ""
        print(f"  [{i+1}/{len(all_tickers)}] {ticker}: {stats['n_days']} days → "
              f"{strategy:<12} {reason}{flag}")

    except Exception as e:
        print(f"  [{i+1}/{len(all_tickers)}] {ticker}: ERROR — {e}")

results = pd.DataFrame(all_stats)

if results.empty:
    print("\nNo tickers processed. Check GCS access.")
else:
    # ==============================================================================
    # DASHBOARD — STRATEGY BUCKETS
    # ==============================================================================
    for strat in ['LIMIT_ENTRY', 'MARKET_BUY', 'SKIP']:
        sub = results[results['strategy'] == strat].copy()
        sub = sub.sort_values('med_amplitude', ascending=False)

        print()
        print(f"{'='*W}")
        print(f"  {strat} — {len(sub)} tickers")
        print(f"{'─'*W}")

        if strat == 'LIMIT_ENTRY':
            print(f"  {'Ticker':<8} {'Sector':<12} {'Days':>5}  {'MedDD':>7}  {'MedRU':>7}  "
                  f"{'DipFirst':>8}  {'Fill@dip':>8}  {'Recov':>6}  {'Dip%':>6}  "
                  f"{'EOD PnL':>8}  {'Deployed'}")
            print(f"  {'─'*W}")
            for _, r in sub.iterrows():
                dip = r['rec_dip_pct']
                fill_key = f"fill_{dip}" if dip in DIP_LEVELS else None
                fill_str = f"{r.get(fill_key, 0):.0%}" if fill_key else "—"
                recov_key = f"recov_{dip}" if dip in DIP_LEVELS else None
                recov_str = f"{r.get(recov_key, 0):.0%}" if recov_key else "—"
                dep = "YES" if r['deployed'] else ""
                print(f"  {r['ticker']:<8} {r['sector']:<12} {r['n_days']:>5}  "
                      f"{r['med_drawdown']:>+6.1%}  {r['med_runup']:>+6.1%}  "
                      f"{r['pct_dip_first']:>7.0%}  {fill_str:>8}  {recov_str:>6}  "
                      f"{dip:>5.1%}  {r['med_eod_pnl']:>+7.2%}  {dep}")

        elif strat == 'MARKET_BUY':
            print(f"  {'Ticker':<8} {'Sector':<12} {'Days':>5}  {'MedDD':>7}  {'MedRU':>7}  "
                  f"{'Ratio':>6}  {'DipFirst':>8}  {'EOD PnL':>8}  {'Deployed'}")
            print(f"  {'─'*W}")
            for _, r in sub.iterrows():
                dep = "YES" if r['deployed'] else ""
                print(f"  {r['ticker']:<8} {r['sector']:<12} {r['n_days']:>5}  "
                      f"{r['med_drawdown']:>+6.1%}  {r['med_runup']:>+6.1%}  "
                      f"{r['med_swing_ratio']:>5.1f}x  {r['pct_dip_first']:>7.0%}  "
                      f"{r['med_eod_pnl']:>+7.2%}  {dep}")

        else:
            print(f"  {'Ticker':<8} {'Sector':<12} {'Days':>5}  {'MedDD':>7}  {'MedRU':>7}  "
                  f"{'Reason':<50}  {'Deployed'}")
            print(f"  {'─'*W}")
            for _, r in sub.iterrows():
                dep = "YES" if r['deployed'] else ""
                print(f"  {r['ticker']:<8} {r['sector']:<12} {r['n_days']:>5}  "
                      f"{r['med_drawdown']:>+6.1%}  {r['med_runup']:>+6.1%}  "
                      f"{r['reason']:<50}  {dep}")

    # ==============================================================================
    # DEPLOYED TICKER REVIEW
    # ==============================================================================
    deployed_results = results[results['deployed']].copy()
    if not deployed_results.empty:
        print()
        print(f"{'='*W}")
        print(f"  DEPLOYED TICKER REVIEW — Entry Strategy Recommendation")
        print(f"{'─'*W}")
        print(f"  {'Ticker':<8} {'Current':>10}  {'Recommended':<14} {'Dip%':>6}  "
              f"{'DipFirst':>8}  {'MedDD':>7}  {'Ratio':>6}  {'Reason'}")
        print(f"  {'─'*W}")

        for _, r in deployed_results.sort_values('ticker').iterrows():
            current = "static"
            for exp_name, tickers in DEPLOYED.items():
                if r['ticker'] in tickers:
                    current = "static"
                    break

            dip_str = f"{r['rec_dip_pct']:.1%}" if r['rec_dip_pct'] else "—"
            match = "OK" if (r['strategy'] == 'MARKET_BUY' and current == 'static') else ""
            if r['strategy'] == 'LIMIT_ENTRY' and current == 'static':
                match = "SWITCH"
            elif r['strategy'] == 'SKIP':
                match = "REVIEW"
            print(f"  {r['ticker']:<8} {current:>10}  {r['strategy']:<14} {dip_str:>6}  "
                  f"{r['pct_dip_first']:>7.0%}  {r['med_drawdown']:>+6.1%}  "
                  f"{r['med_swing_ratio']:>5.1f}x  {r['reason']}  {match}")

    # ==============================================================================
    # SECTOR STRATEGY MIX
    # ==============================================================================
    print()
    print(f"{'='*W}")
    print(f"  SECTOR STRATEGY MIX")
    print(f"{'─'*W}")
    print(f"  {'Sector':<14} {'Total':>5}  {'LIMIT':>6}  {'MARKET':>7}  {'SKIP':>5}  "
          f"{'%LIMIT':>7}  {'AvgDD':>7}  {'AvgRU':>7}")
    print(f"  {'─'*W}")

    for sec in sorted(results['sector'].unique()):
        sec_df = results[results['sector'] == sec]
        n_limit = (sec_df['strategy'] == 'LIMIT_ENTRY').sum()
        n_market = (sec_df['strategy'] == 'MARKET_BUY').sum()
        n_skip = (sec_df['strategy'] == 'SKIP').sum()
        pct_limit = n_limit / len(sec_df) if len(sec_df) > 0 else 0
        print(f"  {sec:<14} {len(sec_df):>5}  {n_limit:>6}  {n_market:>7}  {n_skip:>5}  "
              f"{pct_limit:>6.0%}  {sec_df['med_drawdown'].mean():>+6.1%}  "
              f"{sec_df['med_runup'].mean():>+6.1%}")

    # ==============================================================================
    # SUMMARY
    # ==============================================================================
    n_limit = (results['strategy'] == 'LIMIT_ENTRY').sum()
    n_market = (results['strategy'] == 'MARKET_BUY').sum()
    n_skip = (results['strategy'] == 'SKIP').sum()

    print()
    print(f"{'='*W}")
    print(f"  SUMMARY")
    print(f"{'─'*W}")
    print(f"  Total tickers analyzed: {len(results)}")
    print(f"  LIMIT_ENTRY:  {n_limit:>3}  ({n_limit/len(results):.0%})")
    print(f"  MARKET_BUY:   {n_market:>3}  ({n_market/len(results):.0%})")
    print(f"  SKIP:         {n_skip:>3}  ({n_skip/len(results):.0%})")
    print()

    limit_deployed = results[(results['strategy'] == 'LIMIT_ENTRY') & results['deployed']]
    skip_deployed = results[(results['strategy'] == 'SKIP') & results['deployed']]
    if len(limit_deployed) > 0:
        print(f"  Deployed tickers that should SWITCH to limit_entry:")
        for _, r in limit_deployed.iterrows():
            print(f"    {r['ticker']}: dip_pct={r['rec_dip_pct']:.1%}  ({r['reason']})")
    if len(skip_deployed) > 0:
        print(f"  Deployed tickers flagged for REVIEW (SKIP recommendation):")
        for _, r in skip_deployed.iterrows():
            print(f"    {r['ticker']}: {r['reason']}")

    print()
    print(f"  NOTE: These are UNCONDITIONAL statistics. Limit entry is most effective")
    print(f"  when conditioned on a spike model signal (prob > threshold). The")
    print(f"  recommended dip_pct values here are conservative baselines — actual")
    print(f"  deployment should use walk-forward optimized values from robust_wf_v12.")
    print()
    print(f"{'='*W}")
    print(f"  Done.")
    print(f"{'='*W}")
