"""
analyze_tpsl_volatility.py — TP/SL Calibration vs Actual Volatility
====================================================================
Investigates whether dynamic TP/SL modes are generating unreasonable
stop-loss levels by comparing configured SL/TP against actual pre-market
volatility for each live GO signal.

For each trade:
  - Pre-market range (6:00-6:30 PST high-low as % of open)
  - SL% vs actual daily pre-market range%
  - MFE (max favorable excursion before exit)
  - MAE (max adverse excursion before exit)
  - Time to SL hit (if SL was hit)
  - Whether SL was tighter or wider than normal pre-market noise

Analysis by tpsl_mode: static, atr, atr_adaptive, regime_scaled, trailing_decay

Run in Colab or locally with GCS + BQ access.
"""

import gc, warnings
import numpy as np
import pandas as pd
import pytz
from google.cloud import bigquery, storage as gcs_storage

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

warnings.filterwarnings('ignore')

PST = pytz.timezone('America/Los_Angeles')
BUCKET = "time-series-bucket-ys"
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

bq_client = bigquery.Client()
gcs_client = gcs_storage.Client()

W = 130
EXIT_TIME_MIN = 780  # 1:00 PM PST = 780 minutes from midnight

# ==============================================================================
# STEP 1: LOAD GO SIGNALS FROM BQ
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TP/SL VOLATILITY CALIBRATION ANALYSIS")
print(f"  Comparing configured TP/SL vs actual pre-market volatility")
print(f"{'='*W}")

print("\n  Loading GO signals from BQ (score_date >= 2026-05-13)...")

q = """
SELECT score_date, Ticker as ticker, experiment, prob, Signal as signal,
       Filter_Status as filter_status,
       tp_pct, sl_pct, tpsl_mode, atr_pct_entry, entry_minute, close,
       vix_regime_idx, trail_start, trail_end, VIX_Value as vix_value, seed
FROM `ys-time-series.scoring_logs.live_predictions`
WHERE score_date >= '2026-05-13' AND Signal = 'GO' AND Filter_Status = 'OK'
ORDER BY score_date, Ticker
"""
signals_raw = bq_client.query(q).to_dataframe()
print(f"    Raw GO signals: {len(signals_raw)}")

# Drop duplicates on (score_date, ticker), keep='first'
signals = signals_raw.drop_duplicates(subset=['score_date', 'ticker'], keep='first').copy()
signals['tpsl_mode'] = signals['tpsl_mode'].fillna('static').replace('', 'static')
print(f"    After dedup: {len(signals)} unique (date, ticker) trades")
print(f"    Tickers: {sorted(signals['ticker'].unique())}")
print(f"    Dates: {signals['score_date'].min()} to {signals['score_date'].max()}")
print(f"    Modes: {dict(signals['tpsl_mode'].value_counts())}")

# ==============================================================================
# STEP 2: LOAD OHLCV PER TICKER FROM GCS (cached)
# ==============================================================================
print(f"\n  Loading 1-minute OHLCV from GCS...")
ohlcv_cache = {}

for ticker in sorted(signals['ticker'].unique()):
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    loaded = False
    for path_prefix in ['Data/Inference_v2/', 'Data/Inference/']:
        for suffix in ['_master_inference.parquet', '_1m_full_historical_data.parquet']:
            uri = f"gs://{BUCKET}/{path_prefix}{clean}{suffix}"
            try:
                d = pd.read_parquet(uri)
                d = d.reset_index()
                dt_cols = [c for c in d.columns if pd.api.types.is_datetime64_any_dtype(d[c])]
                if dt_cols:
                    d['Datetime'] = pd.to_datetime(d[dt_cols[0]], utc=True)
                else:
                    d['Datetime'] = pd.to_datetime(d[d.columns[0]], unit='ms', utc=True)
                d = d.set_index('Datetime').sort_index()
                d = d[~d.index.duplicated(keep='first')]
                std_map = {'open': 'Open', 'high': 'High', 'low': 'Low',
                           'close': 'Close', 'volume': 'Volume'}
                d.columns = [std_map.get(str(c).strip().lower(), str(c).strip()) for c in d.columns]
                d.index = d.index.tz_convert(PST)
                # Trim to recent data only
                trim = pd.Timestamp('2026-05-01', tz=PST)
                d = d[d.index >= trim]
                ohlcv_cache[ticker] = d[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                print(f"    {ticker}: {len(d):,} bars from {path_prefix}{clean}{suffix}")
                loaded = True
                del d
                gc.collect()
                break
            except Exception:
                continue
        if loaded:
            break
    if not loaded:
        print(f"    {ticker}: FAILED to load")

print(f"    {len(ohlcv_cache)} tickers loaded")

# ==============================================================================
# STEP 3: COMPUTE PER-TRADE METRICS
# ==============================================================================
print(f"\n  Computing trade-level metrics...")

trades = []
for _, row in signals.iterrows():
    ticker = row['ticker']
    if ticker not in ohlcv_cache:
        continue

    d = ohlcv_cache[ticker]
    target_date = pd.Timestamp(str(row['score_date'])).date()
    em = int(row['entry_minute'])
    tp = float(row['tp_pct']) if pd.notna(row['tp_pct']) and row['tp_pct'] > 0 else 0.16
    sl = float(row['sl_pct']) if pd.notna(row['sl_pct']) and row['sl_pct'] > 0 else 0.08
    mode = str(row['tpsl_mode']) if pd.notna(row['tpsl_mode']) else 'static'
    if mode == '' or mode == 'None':
        mode = 'static'

    # Get day's data
    day_data = d[d.index.date == target_date]
    if day_data.empty:
        continue

    mins = day_data.index.hour * 60 + day_data.index.minute

    # --- Pre-market range: 6:00-6:30 PST = 360-390 minutes ---
    premarket = day_data[(mins >= 360) & (mins <= 390)]
    if len(premarket) > 0:
        pm_high = float(premarket['High'].max())
        pm_low = float(premarket['Low'].min())
        pm_open = float(premarket['Open'].iloc[0])
        pm_range_pct = (pm_high - pm_low) / pm_open if pm_open > 0 else 0
    else:
        pm_range_pct = np.nan
        pm_open = np.nan

    # --- Entry bar ---
    entry_bars = day_data[mins == em]
    if entry_bars.empty:
        continue

    entry_price = float(entry_bars['Close'].iloc[0])
    entry_time_idx = entry_bars.index[0]

    # --- After entry ---
    after_entry = day_data[day_data.index > entry_time_idx]
    if after_entry.empty:
        continue

    tp_price = entry_price * (1 + tp)
    sl_price = entry_price * (1 - sl)

    # --- Compute MFE, MAE, time-to-exit ---
    mfe = 0.0   # max favorable excursion (highest % above entry)
    mae = 0.0   # max adverse excursion (deepest % below entry)
    mfe_before_exit = 0.0
    exit_type = 'EOD'
    exit_minute = None
    time_to_exit_mins = None
    pnl = 0.0

    for bar_idx, (bar_ts, bar_row) in enumerate(after_entry.iterrows()):
        bar_min = bar_ts.hour * 60 + bar_ts.minute
        bar_high = float(bar_row['High'])
        bar_low = float(bar_row['Low'])
        bar_close = float(bar_row['Close'])

        # Update running MFE/MAE
        current_mfe = (bar_high - entry_price) / entry_price
        current_mae = (entry_price - bar_low) / entry_price
        mfe = max(mfe, current_mfe)
        mae = max(mae, current_mae)

        # Check SL hit
        if bar_low <= sl_price:
            mfe_before_exit = mfe  # MFE before this bar
            exit_type = 'SL'
            pnl = -sl
            exit_minute = bar_min
            time_to_exit_mins = bar_idx + 1  # bars after entry
            break

        # Check TP hit
        if bar_high >= tp_price:
            mfe_before_exit = mfe
            exit_type = 'TP'
            pnl = tp
            exit_minute = bar_min
            time_to_exit_mins = bar_idx + 1
            break

        # Check EOD exit time
        if bar_min >= EXIT_TIME_MIN:
            mfe_before_exit = mfe
            exit_type = 'EOD'
            pnl = (bar_close - entry_price) / entry_price
            exit_minute = bar_min
            time_to_exit_mins = bar_idx + 1
            break
    else:
        # Reached end of day data
        last_close = float(after_entry['Close'].iloc[-1])
        pnl = (last_close - entry_price) / entry_price
        mfe_before_exit = mfe
        exit_type = 'EOD'
        last_min = after_entry.index[-1].hour * 60 + after_entry.index[-1].minute
        exit_minute = last_min
        time_to_exit_mins = len(after_entry)

    # EOD return (what would have happened if held to close)
    eod_price = float(after_entry['Close'].iloc[-1])
    eod_return = (eod_price - entry_price) / entry_price

    # SL vs pre-market range assessment
    sl_vs_range = None
    if not np.isnan(pm_range_pct) and pm_range_pct > 0:
        sl_vs_range = sl / pm_range_pct  # >1 means SL is wider than pre-mkt range

    # Compute 14-day ATR at entry (from available data before this date)
    pre_data = d[d.index.date < target_date]
    if len(pre_data) > 100:
        daily = pre_data.resample('D').agg({'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
        if len(daily) >= 14:
            prev_close = daily['Close'].shift(1)
            tr = pd.concat([
                daily['High'] - daily['Low'],
                (daily['High'] - prev_close).abs(),
                (daily['Low'] - prev_close).abs()
            ], axis=1).max(axis=1)
            atr_14 = tr.rolling(14).mean().iloc[-1]
            atr_pct_actual = atr_14 / daily['Close'].iloc[-1] if daily['Close'].iloc[-1] > 0 else np.nan
        else:
            atr_pct_actual = np.nan
    else:
        atr_pct_actual = np.nan

    trades.append({
        'date': str(row['score_date']),
        'ticker': ticker,
        'experiment': row['experiment'],
        'mode': mode,
        'seed': int(row['seed']) if pd.notna(row['seed']) else 0,
        'prob': float(row['prob']),
        'entry_price': entry_price,
        'entry_minute': em,
        'tp_pct': tp,
        'sl_pct': sl,
        'atr_pct_entry': float(row['atr_pct_entry']) if pd.notna(row['atr_pct_entry']) else np.nan,
        'atr_pct_actual': atr_pct_actual,
        'vix_value': float(row['vix_value']) if pd.notna(row['vix_value']) else np.nan,
        'vix_regime_idx': int(row['vix_regime_idx']) if pd.notna(row['vix_regime_idx']) else -1,
        'trail_start': float(row['trail_start']) if pd.notna(row['trail_start']) else np.nan,
        'trail_end': float(row['trail_end']) if pd.notna(row['trail_end']) else np.nan,
        'pm_range_pct': pm_range_pct,
        'mfe_pct': mfe,
        'mae_pct': mae,
        'mfe_before_exit': mfe_before_exit,
        'pnl': pnl,
        'exit_type': exit_type,
        'exit_minute': exit_minute,
        'time_to_exit_bars': time_to_exit_mins,
        'eod_return': eod_return,
        'sl_vs_range': sl_vs_range,
    })

del ohlcv_cache
gc.collect()

tdf = pd.DataFrame(trades)
tdf['date'] = pd.to_datetime(tdf['date'])
print(f"    {len(tdf)} trades computed")
print(f"    Exit types: {dict(tdf['exit_type'].value_counts())}")
print(f"    Modes: {dict(tdf['mode'].value_counts())}")

# ==============================================================================
# TABLE 1: PER-TRADE DETAIL
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 1: PER-TRADE DETAIL — TP/SL vs Actual Volatility")
print(f"{'='*W}")
print(f"  {'Date':<12} {'Ticker':<6} {'Mode':<15} {'SL%':>5} {'TP%':>5} "
      f"{'PmRng%':>7} {'MFE%':>6} {'MAE%':>6} {'T2Exit':>7} {'Outcome':>8} "
      f"{'PnL':>7} {'EOD%':>6} {'SLvRng':>7} {'Flag'}")
print(f"  {'─'*W}")

for _, r in tdf.sort_values('date').iterrows():
    # Flags
    flags = []
    if r['exit_type'] == 'SL' and r['eod_return'] > 0:
        flags.append('TIGHT_SL')
    if r['sl_vs_range'] is not None and not np.isnan(r['sl_vs_range']):
        if r['sl_vs_range'] < 0.8:
            flags.append('SL<NOISE')
        elif r['sl_vs_range'] > 3.0:
            flags.append('SL_WIDE')
    if r['sl_pct'] > r['tp_pct'] and r['tp_pct'] > 0:
        flags.append('SL>TP')
    if r['mfe_before_exit'] > 0.02 and r['exit_type'] == 'SL':
        flags.append('WHIPSAW')
    if r['exit_type'] == 'SL' and r['time_to_exit_bars'] is not None and r['time_to_exit_bars'] <= 5:
        flags.append('FAST_SL')

    flag_str = ','.join(flags) if flags else ''

    sl_vs_str = f"{r['sl_vs_range']:>6.2f}x" if r['sl_vs_range'] is not None and not np.isnan(r['sl_vs_range']) else '    N/A'
    pm_str = f"{r['pm_range_pct']*100:>6.2f}" if not np.isnan(r['pm_range_pct']) else '   N/A'
    t2e_str = f"{r['time_to_exit_bars']:>5.0f}m" if r['time_to_exit_bars'] is not None else '   N/A'

    print(f"  {r['date'].strftime('%Y-%m-%d'):<12} {r['ticker']:<6} {r['mode']:<15} "
          f"{r['sl_pct']*100:>4.1f}% {r['tp_pct']*100:>4.1f}% "
          f"{pm_str}% {r['mfe_pct']*100:>5.1f}% {r['mae_pct']*100:>5.1f}% "
          f"{t2e_str} {r['exit_type']:>8} "
          f"{r['pnl']*100:>+6.2f}% {r['eod_return']*100:>+5.1f}% "
          f"{sl_vs_str} {flag_str}")


# ==============================================================================
# TABLE 2: ANALYSIS BY TPSL_MODE
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 2: ANALYSIS BY TPSL_MODE")
print(f"{'='*W}")
print(f"  {'Mode':<16} {'N':>4} {'AvgPnL':>8} {'WR':>6} {'AvgSL%':>7} {'AvgTP%':>7} "
      f"{'AvgPmR%':>8} {'SL/Rng':>7} {'AvgMFE':>7} {'AvgMAE':>7} "
      f"{'SLhit%':>7} {'TPhit%':>7} {'SL>TP%':>7} {'TightSL':>8}")
print(f"  {'─'*W}")

for mode in ['static', 'atr', 'atr_adaptive', 'regime_scaled', 'regime', 'trailing_decay']:
    md = tdf[tdf['mode'] == mode]
    if md.empty:
        continue
    n = len(md)
    avg_pnl = md['pnl'].mean()
    wr = (md['pnl'] > 0).mean()
    avg_sl = md['sl_pct'].mean()
    avg_tp = md['tp_pct'].mean()
    pm_valid = md[md['pm_range_pct'].notna()]
    avg_pm = pm_valid['pm_range_pct'].mean() if len(pm_valid) > 0 else np.nan
    avg_sl_vs = pm_valid['sl_vs_range'].mean() if len(pm_valid) > 0 else np.nan
    avg_mfe = md['mfe_pct'].mean()
    avg_mae = md['mae_pct'].mean()
    sl_rate = (md['exit_type'] == 'SL').mean()
    tp_rate = (md['exit_type'] == 'TP').mean()
    sl_gt_tp = (md['sl_pct'] > md['tp_pct']).mean()
    tight_sl = ((md['exit_type'] == 'SL') & (md['eod_return'] > 0)).sum()

    pm_str = f"{avg_pm*100:>7.2f}" if not np.isnan(avg_pm) else '    N/A'
    svr_str = f"{avg_sl_vs:>6.2f}x" if not np.isnan(avg_sl_vs) else '    N/A'

    print(f"  {mode:<16} {n:>4} {avg_pnl*100:>+7.2f}% {wr*100:>5.0f}% "
          f"{avg_sl*100:>6.1f}% {avg_tp*100:>6.1f}% "
          f"{pm_str}% {svr_str} {avg_mfe*100:>6.2f}% {avg_mae*100:>6.2f}% "
          f"{sl_rate*100:>6.0f}% {tp_rate*100:>6.0f}% {sl_gt_tp*100:>6.0f}% "
          f"{tight_sl:>5}/{(md['exit_type']=='SL').sum()}")


# ==============================================================================
# TABLE 3: PER-TICKER SUMMARY
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 3: PER-TICKER SUMMARY")
print(f"{'='*W}")
print(f"  {'Ticker':<7} {'N':>3} {'Mode(s)':<20} {'AvgSL%':>7} {'AvgTP%':>7} "
      f"{'PmRng%':>7} {'SL/Rng':>7} {'MFE%':>6} {'MAE%':>6} "
      f"{'WR':>5} {'AvgPnL':>8} {'SLhit':>6} {'TPhit':>6} {'Tight':>6}")
print(f"  {'─'*W}")

for ticker in sorted(tdf['ticker'].unique()):
    td = tdf[tdf['ticker'] == ticker]
    n = len(td)
    modes = '/'.join(sorted(td['mode'].unique()))
    avg_sl = td['sl_pct'].mean()
    avg_tp = td['tp_pct'].mean()
    pm_valid = td[td['pm_range_pct'].notna()]
    avg_pm = pm_valid['pm_range_pct'].mean() if len(pm_valid) > 0 else np.nan
    avg_svr = pm_valid['sl_vs_range'].mean() if len(pm_valid) > 0 else np.nan
    avg_mfe = td['mfe_pct'].mean()
    avg_mae = td['mae_pct'].mean()
    wr = (td['pnl'] > 0).mean()
    avg_pnl = td['pnl'].mean()
    sl_hits = (td['exit_type'] == 'SL').sum()
    tp_hits = (td['exit_type'] == 'TP').sum()
    tight = ((td['exit_type'] == 'SL') & (td['eod_return'] > 0)).sum()

    pm_str = f"{avg_pm*100:>6.2f}" if not np.isnan(avg_pm) else '   N/A'
    svr_str = f"{avg_svr:>6.2f}x" if not np.isnan(avg_svr) else '    N/A'

    print(f"  {ticker:<7} {n:>3} {modes:<20} {avg_sl*100:>6.1f}% {avg_tp*100:>6.1f}% "
          f"{pm_str}% {svr_str} {avg_mfe*100:>5.1f}% {avg_mae*100:>5.1f}% "
          f"{wr*100:>4.0f}% {avg_pnl*100:>+7.2f}% "
          f"{sl_hits:>4}/{n} {tp_hits:>4}/{n} {tight:>4}/{sl_hits}")


# ==============================================================================
# TABLE 4: SL CALIBRATION DEEP DIVE — SL vs Pre-Market Noise
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 4: SL CALIBRATION — Is SL tighter or wider than normal pre-market noise?")
print(f"{'='*W}")

pm_valid = tdf[tdf['pm_range_pct'].notna()].copy()
if len(pm_valid) > 0:
    # Categorize
    pm_valid['sl_category'] = 'NORMAL'
    pm_valid.loc[pm_valid['sl_vs_range'] < 0.5, 'sl_category'] = 'MUCH_TIGHTER'
    pm_valid.loc[(pm_valid['sl_vs_range'] >= 0.5) & (pm_valid['sl_vs_range'] < 1.0), 'sl_category'] = 'TIGHTER'
    pm_valid.loc[(pm_valid['sl_vs_range'] >= 1.0) & (pm_valid['sl_vs_range'] < 2.0), 'sl_category'] = 'NORMAL'
    pm_valid.loc[(pm_valid['sl_vs_range'] >= 2.0) & (pm_valid['sl_vs_range'] < 4.0), 'sl_category'] = 'WIDER'
    pm_valid.loc[pm_valid['sl_vs_range'] >= 4.0, 'sl_category'] = 'MUCH_WIDER'

    print(f"\n  {'Category':<15} {'N':>4} {'WR':>6} {'AvgPnL':>8} {'AvgSL%':>7} "
          f"{'AvgPmR%':>8} {'SLhit%':>7} {'AvgMFE':>7} {'AvgMAE':>7}")
    print(f"  {'─'*80}")

    for cat in ['MUCH_TIGHTER', 'TIGHTER', 'NORMAL', 'WIDER', 'MUCH_WIDER']:
        cd = pm_valid[pm_valid['sl_category'] == cat]
        if cd.empty:
            continue
        n = len(cd)
        wr = (cd['pnl'] > 0).mean()
        avg_pnl = cd['pnl'].mean()
        avg_sl = cd['sl_pct'].mean()
        avg_pm = cd['pm_range_pct'].mean()
        sl_rate = (cd['exit_type'] == 'SL').mean()
        avg_mfe = cd['mfe_pct'].mean()
        avg_mae = cd['mae_pct'].mean()
        print(f"  {cat:<15} {n:>4} {wr*100:>5.0f}% {avg_pnl*100:>+7.2f}% "
              f"{avg_sl*100:>6.1f}% {avg_pm*100:>7.2f}% {sl_rate*100:>6.0f}% "
              f"{avg_mfe*100:>6.2f}% {avg_mae*100:>6.2f}%")

    print(f"\n  INTERPRETATION:")
    tight = pm_valid[pm_valid['sl_vs_range'] < 1.0]
    wide = pm_valid[pm_valid['sl_vs_range'] >= 2.0]
    if len(tight) > 0:
        tight_sl_rate = (tight['exit_type'] == 'SL').mean()
        tight_tight_sl = ((tight['exit_type'] == 'SL') & (tight['eod_return'] > 0)).sum()
        print(f"    SL < pre-market range: {len(tight)} trades, SL hit rate={tight_sl_rate*100:.0f}%, "
              f"{tight_tight_sl} were TIGHT_SL (recovered by EOD)")
    if len(wide) > 0:
        wide_sl_rate = (wide['exit_type'] == 'SL').mean()
        wide_avg_loss = wide[wide['exit_type'] == 'SL']['pnl'].mean() if (wide['exit_type'] == 'SL').sum() > 0 else 0
        print(f"    SL > 2x pre-market range: {len(wide)} trades, SL hit rate={wide_sl_rate*100:.0f}%, "
              f"avg loss when SL hit={wide_avg_loss*100:+.2f}%")


# ==============================================================================
# TABLE 5: FOCUS ON WORST LOSERS — FCEL, LUNR, SMCI
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 5: FOCUS ON WORST LOSERS — FCEL, LUNR, SMCI")
print(f"{'='*W}")

for focus_ticker in ['FCEL', 'LUNR', 'SMCI', 'ALAB', 'WULF', 'RCAT']:
    td = tdf[tdf['ticker'] == focus_ticker]
    if td.empty:
        continue

    print(f"\n  === {focus_ticker} ({len(td)} trades) ===")
    print(f"  Mode(s): {', '.join(td['mode'].unique())}")
    avg_price = td['entry_price'].mean()
    print(f"  Avg entry price: ${avg_price:.2f}")

    for _, r in td.sort_values('date').iterrows():
        flags = []
        if r['exit_type'] == 'SL' and r['eod_return'] > 0:
            flags.append('TIGHT_SL')
        if r['sl_pct'] > r['tp_pct'] and r['tp_pct'] > 0:
            flags.append('SL>TP')
        if r['mfe_before_exit'] > 0.02 and r['exit_type'] == 'SL':
            flags.append('WHIPSAW')
        flag_str = f"  [{','.join(flags)}]" if flags else ''

        atr_str = f"ATR={r['atr_pct_entry']*100:.1f}%" if not np.isnan(r['atr_pct_entry']) else ''
        actual_atr_str = f"ActATR={r['atr_pct_actual']*100:.1f}%" if not np.isnan(r['atr_pct_actual']) else ''
        pm_str = f"PmRng={r['pm_range_pct']*100:.2f}%" if not np.isnan(r['pm_range_pct']) else ''

        print(f"    {r['date'].strftime('%m/%d')} ${r['entry_price']:.2f}  "
              f"SL={r['sl_pct']*100:.1f}% TP={r['tp_pct']*100:.1f}%  "
              f"MFE={r['mfe_pct']*100:+.1f}% MAE={r['mae_pct']*100:+.1f}%  "
              f"-> {r['exit_type']} {r['pnl']*100:+.1f}%  "
              f"EOD={r['eod_return']*100:+.1f}%  {atr_str} {actual_atr_str} {pm_str}{flag_str}")

    # Summary for this ticker
    sl_hits = td[td['exit_type'] == 'SL']
    if len(sl_hits) > 0:
        avg_sl_loss = sl_hits['pnl'].mean()
        avg_sl_pct = sl_hits['sl_pct'].mean()
        recovered = (sl_hits['eod_return'] > 0).sum()
        print(f"    Summary: {len(sl_hits)}/{len(td)} SL hits, avg loss={avg_sl_loss*100:+.1f}%, "
              f"avg SL={avg_sl_pct*100:.1f}%, {recovered} recovered by EOD")
    tp_hits = td[td['exit_type'] == 'TP']
    if len(tp_hits) > 0:
        print(f"    TP hits: {len(tp_hits)}/{len(td)}, avg gain={tp_hits['pnl'].mean()*100:+.1f}%")


# ==============================================================================
# TABLE 6: TP/SL ASYMMETRY ANALYSIS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 6: TP/SL ASYMMETRY ANALYSIS")
print(f"{'='*W}")

tdf_with_tp = tdf[tdf['tp_pct'] > 0].copy()
tdf_with_tp['tp_sl_ratio'] = tdf_with_tp['tp_pct'] / tdf_with_tp['sl_pct']
tdf_with_tp['sl_gt_tp'] = tdf_with_tp['sl_pct'] > tdf_with_tp['tp_pct']

print(f"\n  Trades where SL > TP (inverted risk/reward):")
inverted = tdf_with_tp[tdf_with_tp['sl_gt_tp']]
if len(inverted) > 0:
    print(f"    Count: {len(inverted)}/{len(tdf_with_tp)} ({len(inverted)/len(tdf_with_tp)*100:.0f}%)")
    print(f"    Avg SL: {inverted['sl_pct'].mean()*100:.1f}%  Avg TP: {inverted['tp_pct'].mean()*100:.1f}%")
    print(f"    WR: {(inverted['pnl']>0).mean()*100:.0f}%  Avg PnL: {inverted['pnl'].mean()*100:+.2f}%")
    print(f"    Tickers: {', '.join(sorted(inverted['ticker'].unique()))}")
    print(f"    Modes: {dict(inverted['mode'].value_counts())}")
    print(f"\n    Detail:")
    for _, r in inverted.sort_values('date').iterrows():
        print(f"      {r['date'].strftime('%m/%d')} {r['ticker']:<6} {r['mode']:<15} "
              f"SL={r['sl_pct']*100:.1f}% TP={r['tp_pct']*100:.1f}% "
              f"ratio={r['tp_sl_ratio']:.2f}x -> {r['exit_type']} {r['pnl']*100:+.1f}%")

print(f"\n  TP/SL Ratio distribution:")
for lo, hi, label in [(0, 0.5, '<0.5x (SL >> TP)'), (0.5, 1.0, '0.5-1x (SL > TP)'),
                       (1.0, 2.0, '1-2x (healthy)'), (2.0, 5.0, '2-5x (wide TP)'),
                       (5.0, 999, '>5x (very wide TP)')]:
    bucket = tdf_with_tp[(tdf_with_tp['tp_sl_ratio'] >= lo) & (tdf_with_tp['tp_sl_ratio'] < hi)]
    if bucket.empty:
        continue
    wr = (bucket['pnl'] > 0).mean()
    avg = bucket['pnl'].mean()
    print(f"    {label:<22} {len(bucket):>3} trades  WR={wr*100:>4.0f}%  avg PnL={avg*100:>+6.2f}%")


# ==============================================================================
# TABLE 7: ATR COMPARISON — Model ATR vs Actual ATR
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 7: ATR COMPARISON — Model ATR vs Actual Computed ATR")
print(f"{'='*W}")

atr_trades = tdf[(tdf['atr_pct_entry'].notna()) & (tdf['atr_pct_actual'].notna())].copy()
if len(atr_trades) > 0:
    atr_trades['atr_diff'] = atr_trades['atr_pct_entry'] - atr_trades['atr_pct_actual']
    atr_trades['atr_ratio'] = atr_trades['atr_pct_entry'] / atr_trades['atr_pct_actual']

    print(f"\n  {'Date':<12} {'Ticker':<7} {'Mode':<15} {'ModelATR':>9} {'ActualATR':>10} "
          f"{'Diff':>7} {'Ratio':>6} {'SL%':>5} {'TP%':>5} {'PnL':>7}")
    print(f"  {'─'*90}")

    for _, r in atr_trades.sort_values(['ticker', 'date']).iterrows():
        print(f"  {r['date'].strftime('%Y-%m-%d'):<12} {r['ticker']:<7} {r['mode']:<15} "
              f"{r['atr_pct_entry']*100:>8.2f}% {r['atr_pct_actual']*100:>9.2f}% "
              f"{r['atr_diff']*100:>+6.2f}% {r['atr_ratio']:>5.2f}x "
              f"{r['sl_pct']*100:>4.1f}% {r['tp_pct']*100:>4.1f}% {r['pnl']*100:>+6.2f}%")

    print(f"\n  Summary:")
    print(f"    Model ATR mean: {atr_trades['atr_pct_entry'].mean()*100:.2f}%")
    print(f"    Actual ATR mean: {atr_trades['atr_pct_actual'].mean()*100:.2f}%")
    print(f"    Avg diff: {atr_trades['atr_diff'].mean()*100:+.2f}%")
    print(f"    Avg ratio: {atr_trades['atr_ratio'].mean():.2f}x")
    overshoot = atr_trades[atr_trades['atr_ratio'] > 1.2]
    if len(overshoot) > 0:
        print(f"    Model ATR > actual by 20%+: {len(overshoot)} trades — "
              f"tickers: {', '.join(sorted(overshoot['ticker'].unique()))}")
else:
    print(f"\n  No trades with both model ATR and actual ATR available.")


# ==============================================================================
# TABLE 8: TIME TO SL HIT — Are stops getting taken out too fast?
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 8: TIME TO SL HIT — How quickly are stops triggered?")
print(f"{'='*W}")

sl_trades = tdf[tdf['exit_type'] == 'SL'].copy()
if len(sl_trades) > 0:
    print(f"\n  SL hits: {len(sl_trades)}/{len(tdf)} trades ({len(sl_trades)/len(tdf)*100:.0f}%)")

    for mode in sorted(sl_trades['mode'].unique()):
        md = sl_trades[sl_trades['mode'] == mode]
        avg_t = md['time_to_exit_bars'].mean()
        med_t = md['time_to_exit_bars'].median()
        fast = (md['time_to_exit_bars'] <= 5).sum()
        recovered = (md['eod_return'] > 0).sum()
        print(f"\n    {mode}: {len(md)} SL hits")
        print(f"      Avg time to SL: {avg_t:.0f} bars  Median: {med_t:.0f} bars  "
              f"Fast (<5 bars): {fast}")
        print(f"      Recovered by EOD: {recovered}/{len(md)} ({recovered/len(md)*100:.0f}%)")
        print(f"      Avg SL%: {md['sl_pct'].mean()*100:.1f}%  "
              f"Avg MFE before SL: {md['mfe_before_exit'].mean()*100:.2f}%  "
              f"Avg loss: {md['pnl'].mean()*100:+.2f}%")

    # Overall
    print(f"\n    ALL MODES:")
    avg_t = sl_trades['time_to_exit_bars'].mean()
    fast = (sl_trades['time_to_exit_bars'] <= 5).sum()
    recovered = (sl_trades['eod_return'] > 0).sum()
    had_mfe = (sl_trades['mfe_before_exit'] > 0.02).sum()
    print(f"      Total SL hits: {len(sl_trades)}")
    print(f"      Avg time to SL: {avg_t:.0f} bars")
    print(f"      Fast SL (<5 bars): {fast} ({fast/len(sl_trades)*100:.0f}%)")
    print(f"      Recovered by EOD: {recovered} ({recovered/len(sl_trades)*100:.0f}%)")
    print(f"      Had MFE > 2% before SL: {had_mfe} ({had_mfe/len(sl_trades)*100:.0f}%) — WHIPSAW")


# ==============================================================================
# TABLE 9: JUNE VS MAY VOLATILITY COMPARISON
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  TABLE 9: VOLATILITY SHIFT — May vs June 2026")
print(f"{'='*W}")

tdf['month'] = tdf['date'].dt.month
may = tdf[tdf['month'] == 5]
jun = tdf[tdf['month'] == 6]

if len(may) > 0 and len(jun) > 0:
    print(f"\n  {'Metric':<25} {'May':>10} {'June':>10} {'Delta':>10}")
    print(f"  {'─'*55}")
    metrics = [
        ('Avg pre-mkt range %', 'pm_range_pct'),
        ('Avg MFE %', 'mfe_pct'),
        ('Avg MAE %', 'mae_pct'),
        ('Avg SL %', 'sl_pct'),
        ('Avg TP %', 'tp_pct'),
    ]
    for label, col in metrics:
        may_val = may[col].mean() * 100
        jun_val = jun[col].mean() * 100
        delta = jun_val - may_val
        print(f"  {label:<25} {may_val:>9.2f}% {jun_val:>9.2f}% {delta:>+9.2f}%")

    print(f"\n  {'':25} {'May':>10} {'June':>10}")
    print(f"  {'Trades':<25} {len(may):>10} {len(jun):>10}")
    print(f"  {'Win rate':<25} {(may['pnl']>0).mean()*100:>9.0f}% {(jun['pnl']>0).mean()*100:>9.0f}%")
    print(f"  {'Avg PnL':<25} {may['pnl'].mean()*100:>+9.2f}% {jun['pnl'].mean()*100:>+9.2f}%")
    print(f"  {'SL hit rate':<25} {(may['exit_type']=='SL').mean()*100:>9.0f}% {(jun['exit_type']=='SL').mean()*100:>9.0f}%")

    # Per-ticker volatility shift
    common_tickers = set(may['ticker'].unique()) & set(jun['ticker'].unique())
    if common_tickers:
        print(f"\n  Per-ticker volatility shift (tickers in both months):")
        print(f"  {'Ticker':<7} {'May PmR%':>9} {'Jun PmR%':>9} {'May ATR%':>9} {'Jun ATR%':>9} {'May PnL':>8} {'Jun PnL':>8}")
        print(f"  {'─'*60}")
        for t in sorted(common_tickers):
            mt = may[may['ticker'] == t]
            jt = jun[jun['ticker'] == t]
            may_pm = mt['pm_range_pct'].mean() * 100 if mt['pm_range_pct'].notna().any() else np.nan
            jun_pm = jt['pm_range_pct'].mean() * 100 if jt['pm_range_pct'].notna().any() else np.nan
            may_atr = mt['atr_pct_actual'].mean() * 100 if mt['atr_pct_actual'].notna().any() else np.nan
            jun_atr = jt['atr_pct_actual'].mean() * 100 if jt['atr_pct_actual'].notna().any() else np.nan
            may_pnl = mt['pnl'].mean() * 100
            jun_pnl = jt['pnl'].mean() * 100
            may_pm_s = f"{may_pm:>8.2f}" if not np.isnan(may_pm) else "     N/A"
            jun_pm_s = f"{jun_pm:>8.2f}" if not np.isnan(jun_pm) else "     N/A"
            may_atr_s = f"{may_atr:>8.2f}" if not np.isnan(may_atr) else "     N/A"
            jun_atr_s = f"{jun_atr:>8.2f}" if not np.isnan(jun_atr) else "     N/A"
            print(f"  {t:<7} {may_pm_s}% {jun_pm_s}% {may_atr_s}% {jun_atr_s}% "
                  f"{may_pnl:>+7.2f}% {jun_pnl:>+7.2f}%")


# ==============================================================================
# SUMMARY & DIAGNOSIS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SUMMARY: TP/SL CALIBRATION DIAGNOSIS")
print(f"{'='*W}")

total = len(tdf)
sl_hits = (tdf['exit_type'] == 'SL').sum()
tp_hits = (tdf['exit_type'] == 'TP').sum()
eod = (tdf['exit_type'] == 'EOD').sum()
wr = (tdf['pnl'] > 0).mean()
avg_pnl = tdf['pnl'].mean()
tight_sl_count = ((tdf['exit_type'] == 'SL') & (tdf['eod_return'] > 0)).sum()
inverted_count = ((tdf['sl_pct'] > tdf['tp_pct']) & (tdf['tp_pct'] > 0)).sum()
whipsaw_count = ((tdf['exit_type'] == 'SL') & (tdf['mfe_before_exit'] > 0.02)).sum()

print(f"\n  Overall: {total} trades, WR={wr*100:.0f}%, avg PnL={avg_pnl*100:+.2f}%")
print(f"  Exit distribution: SL={sl_hits} ({sl_hits/total*100:.0f}%)  "
      f"TP={tp_hits} ({tp_hits/total*100:.0f}%)  "
      f"EOD={eod} ({eod/total*100:.0f}%)")

print(f"\n  KEY FINDINGS:")
if tight_sl_count > 0:
    print(f"    1. TIGHT SL: {tight_sl_count} trades hit SL but recovered by EOD "
          f"({tight_sl_count}/{sl_hits} = {tight_sl_count/sl_hits*100:.0f}% of SL hits)")
    print(f"       -> Stops are too tight for normal intraday noise")

if inverted_count > 0:
    inv = tdf[(tdf['sl_pct'] > tdf['tp_pct']) & (tdf['tp_pct'] > 0)]
    print(f"\n    2. INVERTED RISK/REWARD: {inverted_count} trades have SL > TP")
    print(f"       Tickers: {', '.join(sorted(inv['ticker'].unique()))}")
    print(f"       Modes: {dict(inv['mode'].value_counts())}")
    print(f"       -> Negative expected value: losing MORE on wrong trades than winning on right ones")

if whipsaw_count > 0:
    print(f"\n    3. WHIPSAW: {whipsaw_count} trades had >2% MFE before getting stopped out")
    print(f"       -> Price went in our favor, then reversed to hit SL")

# Mode-level diagnosis
print(f"\n  MODE-LEVEL DIAGNOSIS:")
for mode in sorted(tdf['mode'].unique()):
    md = tdf[tdf['mode'] == mode]
    n = len(md)
    wr = (md['pnl'] > 0).mean()
    avg = md['pnl'].mean()
    sl_r = (md['exit_type'] == 'SL').mean()
    inv = ((md['sl_pct'] > md['tp_pct']) & (md['tp_pct'] > 0)).sum()

    issues = []
    if sl_r > 0.5:
        issues.append(f"HIGH SL RATE ({sl_r*100:.0f}%)")
    if inv > 0:
        issues.append(f"{inv} SL>TP trades")
    if md['sl_pct'].mean() > 0.10:
        issues.append(f"AVG SL={md['sl_pct'].mean()*100:.0f}% (too wide?)")
    if avg < 0:
        issues.append(f"NEGATIVE avg PnL")

    issue_str = '; '.join(issues) if issues else 'OK'
    print(f"    {mode:<16} N={n:>3}  WR={wr*100:>4.0f}%  avg={avg*100:>+6.2f}%  "
          f"SL rate={sl_r*100:>3.0f}%  Issues: {issue_str}")

# Recommendations
print(f"\n  RECOMMENDATIONS:")
if tight_sl_count > 2:
    print(f"    1. WIDEN SL for static mode tickers — {tight_sl_count} premature stop-outs")
    print(f"       Consider SL = max(current_SL, 1.5x pre-market range)")

if inverted_count > 0:
    print(f"    2. ENFORCE SL <= TP constraint — {inverted_count} trades violate this")
    print(f"       ATR and regime_scaled modes are generating SL > TP")

# Check specific problem cases
fcel = tdf[tdf['ticker'] == 'FCEL']
if len(fcel) > 0 and fcel['sl_pct'].max() > 0.10:
    print(f"    3. FCEL: atr_adaptive SL ranges from {fcel['sl_pct'].min()*100:.1f}% "
          f"to {fcel['sl_pct'].max()*100:.1f}% — too wide for a ${fcel['entry_price'].mean():.0f} stock")

alab = tdf[tdf['ticker'] == 'ALAB']
if len(alab) > 0:
    inv_alab = alab[alab['sl_pct'] > alab['tp_pct']]
    if len(inv_alab) > 0:
        print(f"    4. ALAB: ATR mode gives SL={inv_alab['sl_pct'].mean()*100:.1f}% > "
              f"TP={inv_alab['tp_pct'].mean()*100:.1f}% — INVERTED risk/reward")

wulf = tdf[tdf['ticker'] == 'WULF']
if len(wulf) > 0:
    inv_wulf = wulf[wulf['sl_pct'] > wulf['tp_pct']]
    if len(inv_wulf) > 0:
        print(f"    5. WULF: regime_scaled gives SL={inv_wulf['sl_pct'].mean()*100:.1f}% > "
              f"TP={inv_wulf['tp_pct'].mean()*100:.1f}% — INVERTED risk/reward")

print(f"\n{'='*W}")
print(f"  DONE — analyze_tpsl_volatility.py complete")
print(f"{'='*W}")
