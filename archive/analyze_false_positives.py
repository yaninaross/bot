"""
analyze_false_positives.py — False Positive Signal Analysis
============================================================
Queries today's BQ live_predictions for GO signals, then loads
1-minute OHLCV data to replay what happened after entry: did the
trade hit TP, SL, or expire at EOD? Also reconstructs regime
filter state (VIX/TLT/SPY) at entry time.

Set TARGET_DATE below, or leave as 'today' to use current date.
"""

import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import pytz

from google.cloud import bigquery, storage

warnings.filterwarnings('ignore')

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

# ==============================================================================
# CONFIG
# ==============================================================================
BQ_TABLE    = f"{GCP_PROJECT}.scoring_logs.live_predictions"
BUCKET_NAME = os.environ.get("GCS_BUCKET", "")
EXIT_TIME_MIN = 780
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
PST = pytz.timezone('America/Los_Angeles')

TRACE_TICKER = 'RKLB'

W = 130

bq_client  = bigquery.Client()
gcs_client = storage.Client()

# ==============================================================================
# DATE — set to specific date or 'today'
# ==============================================================================
TARGET_DATE = 'today'   # ← change to e.g. '2026-05-13' for a specific date

if TARGET_DATE == 'today':
    TARGET_DATE = pd.Timestamp.now(tz=PST).strftime('%Y-%m-%d')

print(f"{'='*W}")
print(f"  FALSE POSITIVE ANALYSIS — {TARGET_DATE}")
print(f"{'='*W}")

# ==============================================================================
# STEP 1: QUERY BQ FOR TODAY'S SIGNALS
# ==============================================================================
query = f"""
SELECT *
FROM `{BQ_TABLE}`
WHERE score_date = '{TARGET_DATE}'
ORDER BY run_timestamp, ticker
"""

df = bq_client.query(query).to_dataframe()
df.columns = df.columns.str.lower()

if df.empty:
    print(f"\n  No predictions logged for {TARGET_DATE}.")
    raise SystemExit(0)

df_display = df.sort_values('bar_timestamp').drop_duplicates(
    subset=['ticker', 'experiment', 'seed'], keep='last')

print(f"\n  TABLE 1: FINAL PREDICTIONS FOR {TARGET_DATE}  ({len(df)} raw rows → {len(df_display)} unique)")
print(f"  {'─'*(W-2)}")
print(f"  {'Ticker':<7} {'Experiment':<14} {'Seed':>6} {'Prob':>7} {'Thresh':>7} "
      f"{'Signal':>7} {'Entry':>6} {'Close':>8} {'TP':>5} {'SL':>5} "
      f"{'Bar Timestamp':<22}")
print(f"  {'─'*(W-2)}")

for _, r in df_display.iterrows():
    em = int(r['entry_minute'])
    entry_str = f"{em//60}:{em%60:02d}"
    bar_ts = pd.to_datetime(r['bar_timestamp'])
    bar_str = bar_ts.strftime('%Y-%m-%d %H:%M') if pd.notna(bar_ts) else '?'
    print(f"  {r['ticker']:<7} {r['experiment']:<14} {int(r['seed']):>6} "
          f"{r['prob']:>7.4f} {r['threshold']:>7.3f} "
          f"{r['signal']:>7} {entry_str:>6} {r['close']:>8.2f} "
          f"{r['tp_pct']:>4.0%} {r['sl_pct']:>4.0%} "
          f"{bar_str:<22}")

go_signals = df[df['signal'] == 'GO'].copy()
no_signals = df[df['signal'] == 'NO'].copy()

n_go_raw = len(go_signals)
go_signals = go_signals.sort_values('bar_timestamp').drop_duplicates(
    subset=['ticker', 'experiment', 'seed'], keep='last')

print(f"\n  Total: {len(df)} predictions, {n_go_raw} GO (→ {len(go_signals)} unique), {len(no_signals)} NO")

if go_signals.empty:
    print("\n  No GO signals to analyze.")
    raise SystemExit(0)

# ==============================================================================
# HELPER: LOAD OHLCV IN PST
# ==============================================================================
def load_ohlcv_pst(ticker):
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    for suffix in ['_master_inference', '_1m_full_historical_data']:
        uri = f"gs://{BUCKET_NAME}/Data/Inference_v2/{clean}{suffix}.parquet"
        try:
            df = pd.read_parquet(uri)
            df = df.reset_index()
            dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
            if dt_cols:
                df['Datetime'] = pd.to_datetime(df[dt_cols[0]], utc=True)
            else:
                num_col = next((c for c in df.columns
                               if c in ('Timestamp', 'Date', 'Time')), df.columns[0])
                df['Datetime'] = pd.to_datetime(df[num_col], unit='ms', utc=True)
            df = df.set_index('Datetime').sort_index()
            df = df[~df.index.duplicated(keep='first')]
            df.index = df.index.tz_convert(PST)
            return df
        except Exception:
            continue
    return None


# ==============================================================================
# STEP 1b: REGIME FILTER TRACE FOR TRACE_TICKER
# ==============================================================================
if TRACE_TICKER:
    trace_rows = df[df['ticker'] == TRACE_TICKER].sort_values('run_timestamp').copy()
    if trace_rows.empty:
        print(f"\n  No rows for {TRACE_TICKER} — skipping trace")
    else:
        print(f"\n\n{'='*W}")
        print(f"  REGIME FILTER TRACE — {TRACE_TICKER}  ({len(trace_rows)} logged iterations)")
        print(f"{'='*W}")

        target_date = pd.Timestamp(TARGET_DATE, tz=PST)
        score_date_py = target_date.date()

        df_vix_raw = load_ohlcv_pst('VIX')
        df_tlt_raw = load_ohlcv_pst('TLT')
        df_spy_raw = load_ohlcv_pst('SPY')

        tlt_prev_close = spy_prev_close = None
        if df_tlt_raw is not None:
            prev_tlt = df_tlt_raw[df_tlt_raw.index.date < score_date_py]
            tlt_prev_close = float(prev_tlt['Close'].iloc[-1]) if not prev_tlt.empty else None
        if df_spy_raw is not None:
            prev_spy = df_spy_raw[df_spy_raw.index.date < score_date_py]
            spy_prev_close = float(prev_spy['Close'].iloc[-1]) if not prev_spy.empty else None

        cfg = DEPLOY_CONFIGS.get(TRACE_TICKER, {})
        vix_thresh = cfg.get('vix_threshold', 99)
        tlt_crash  = cfg.get('tlt_crash', -99)
        spy_ignore = cfg.get('spy_ignore', 99)

        print(f"\n  Thresholds: VIX>{vix_thresh:.0f}  TLT<{tlt_crash:+.2f}%  SPY>{spy_ignore:+.2f}%  (ALL three → BLOCKED)")
        if tlt_prev_close is not None:
            print(f"  TLT prev close: ${tlt_prev_close:.2f}")
        if spy_prev_close is not None:
            print(f"  SPY prev close: ${spy_prev_close:.2f}")

        print(f"\n  {'run_ts (PST)':<18} {'bar_ts':>8} {'prob':>7} {'sig':>4} "
              f"{'VIX':>7} {'VIX?':>5} "
              f"{'TLT%':>8} {'TLT?':>5} "
              f"{'SPY%':>8} {'SPY?':>5} "
              f"{'FILTER':>8}")
        print(f"  {'─'*(W-2)}")

        for _, row in trace_rows.iterrows():
            run_ts_utc = pd.to_datetime(row['run_timestamp'], utc=True)
            run_ts_pst = run_ts_utc.tz_convert(PST)
            bar_ts_utc = pd.to_datetime(row['bar_timestamp'], utc=True)
            bar_ts_pst = bar_ts_utc.tz_convert(PST)

            window_end = run_ts_pst
            window_start = window_end - timedelta(minutes=30)

            vix_val = tlt_gap = spy_gap = None

            if df_vix_raw is not None:
                vix_bars = df_vix_raw[(df_vix_raw.index >= window_start) & (df_vix_raw.index <= window_end)]
                if not vix_bars.empty:
                    vix_val = float(vix_bars['Close'].median())

            if df_tlt_raw is not None and tlt_prev_close is not None:
                tlt_bars = df_tlt_raw[(df_tlt_raw.index >= window_start) & (df_tlt_raw.index <= window_end)]
                if not tlt_bars.empty:
                    tlt_median = float(tlt_bars['Close'].median())
                    tlt_gap = ((tlt_median - tlt_prev_close) / tlt_prev_close) * 100

            if df_spy_raw is not None and spy_prev_close is not None:
                spy_bars = df_spy_raw[(df_spy_raw.index >= window_start) & (df_spy_raw.index <= window_end)]
                if not spy_bars.empty:
                    spy_median = float(spy_bars['Close'].median())
                    spy_gap = ((spy_median - spy_prev_close) / spy_prev_close) * 100

            vix_bad = vix_val is not None and vix_val > vix_thresh
            tlt_bad = tlt_gap is not None and tlt_gap < tlt_crash
            spy_bad = spy_gap is not None and spy_gap > spy_ignore

            blocked = vix_bad and tlt_bad and spy_bad

            vix_s = f"{vix_val:>7.2f}" if vix_val is not None else "    N/A"
            tlt_s = f"{tlt_gap:>+7.3f}%" if tlt_gap is not None else "     N/A"
            spy_s = f"{spy_gap:>+7.3f}%" if spy_gap is not None else "     N/A"

            print(f"  {run_ts_pst.strftime('%H:%M:%S'):<18} "
                  f"{bar_ts_pst.strftime('%H:%M'):>8} "
                  f"{row['prob']:>7.4f} "
                  f"{'GO' if row['signal'] == 'GO' else 'NO':>4} "
                  f"{vix_s} {'FAIL' if vix_bad else '  OK':>5} "
                  f"{tlt_s} {'FAIL' if tlt_bad else '  OK':>5} "
                  f"{spy_s} {'FAIL' if spy_bad else '  OK':>5} "
                  f"{'BLOCKED' if blocked else '   OK':>8}")

        if df_tlt_raw is not None:
            tlt_today = df_tlt_raw[df_tlt_raw.index.date == score_date_py]
            if not tlt_today.empty and tlt_prev_close:
                print(f"\n  TLT BAR-BY-BAR (first 40 bars today):")
                print(f"    {'time':>8} {'Close':>8} {'gap%':>8}")
                for ts, bar in tlt_today.head(40).iterrows():
                    g = ((float(bar['Close']) - tlt_prev_close) / tlt_prev_close) * 100
                    marker = " <<<" if abs(g - tlt_crash) < 0.02 else ""
                    print(f"    {ts.strftime('%H:%M'):>8} {bar['Close']:>8.2f} {g:>+7.3f}%{marker}")


# ==============================================================================
# STEP 2: REPLAY TRADES
# ==============================================================================
def replay_trade(df_ohlcv, entry_time, tp_pct, sl_pct):
    entry_bar = df_ohlcv[df_ohlcv.index >= entry_time]
    if entry_bar.empty:
        return None

    entry_price = float(entry_bar['Close'].iloc[0])
    entry_ts    = entry_bar.index[0]
    entry_date  = entry_ts.date()
    tp_price    = entry_price * (1 + tp_pct)
    sl_price    = entry_price * (1 - sl_pct)

    after_entry = df_ohlcv[df_ohlcv.index > entry_ts]
    same_day    = after_entry[after_entry.index.date == entry_date]

    exit_reason = 'EOD'
    exit_price  = entry_price
    exit_ts     = entry_ts
    exit_pnl    = 0.0

    tp_ts = sl_ts = eod_ts = None
    tp_price_actual = sl_price_actual = eod_price_actual = None

    for ts, row in same_day.iterrows():
        minute = ts.hour * 60 + ts.minute

        if row['Low'] <= sl_price:
            exit_reason = 'SL'
            exit_price  = sl_price
            exit_pnl    = -sl_pct
            exit_ts     = ts
            sl_ts = ts
            sl_price_actual = row['Low']
            break

        if row['High'] >= tp_price:
            exit_reason = 'TP'
            exit_price  = tp_price
            exit_pnl    = tp_pct
            exit_ts     = ts
            tp_ts = ts
            tp_price_actual = row['High']
            break

        if minute >= EXIT_TIME_MIN:
            exit_reason = 'TIME'
            exit_price  = row['Close']
            exit_pnl    = (exit_price - entry_price) / entry_price
            exit_ts     = ts
            eod_ts = ts
            eod_price_actual = row['Close']
            break

    if exit_reason == 'EOD' and not same_day.empty:
        last = same_day.iloc[-1]
        exit_price = float(last['Close'])
        exit_pnl   = (exit_price - entry_price) / entry_price
        exit_ts    = same_day.index[-1]
        last_minute = exit_ts.hour * 60 + exit_ts.minute
        if last_minute < EXIT_TIME_MIN:
            exit_reason = 'PARTIAL'

    high_of_day = float(same_day['High'].max()) if not same_day.empty else entry_price
    low_of_day  = float(same_day['Low'].min()) if not same_day.empty else entry_price
    max_favorable = (high_of_day - entry_price) / entry_price
    max_adverse   = (entry_price - low_of_day) / entry_price

    price_path = []
    checkpoints = [1, 5, 10, 15, 30, 60, 120]
    for mins in checkpoints:
        target_time = entry_ts + timedelta(minutes=mins)
        nearby = same_day[(same_day.index >= target_time - timedelta(seconds=30)) &
                          (same_day.index <= target_time + timedelta(seconds=30))]
        if not nearby.empty:
            p = float(nearby['Close'].iloc[0])
            chg = (p - entry_price) / entry_price
            price_path.append((mins, p, chg))

    return {
        'entry_price': entry_price,
        'entry_ts': entry_ts,
        'exit_reason': exit_reason,
        'exit_price': exit_price,
        'exit_pnl': exit_pnl,
        'exit_ts': exit_ts,
        'tp_price': tp_price,
        'sl_price': sl_price,
        'high_of_day': high_of_day,
        'low_of_day': low_of_day,
        'max_favorable': max_favorable,
        'max_adverse': max_adverse,
        'price_path': price_path,
        'bars_after_entry': len(same_day),
    }


# ==============================================================================
# STEP 3: REGIME STATE AT ENTRY TIME
# ==============================================================================
def get_regime_at_entry(entry_minute):
    target_date = pd.Timestamp(TARGET_DATE, tz=PST)

    regime = {}

    for asset, key in [('VIX', 'vix'), ('SPY', 'spy'), ('TLT', 'tlt')]:
        df = load_ohlcv_pst(asset)
        if df is None:
            regime[key] = None
            regime[f'{key}_detail'] = 'data unavailable'
            continue

        entry_time = target_date + timedelta(minutes=entry_minute)
        window_start = entry_time - timedelta(minutes=30)

        bars = df[(df.index >= window_start) & (df.index <= entry_time)]
        median_close = float(bars['Close'].median()) if not bars.empty else None

        if key == 'vix':
            regime['vix'] = median_close
            regime['vix_detail'] = f"VIX median(30min)={median_close:.2f}" if median_close else 'N/A'
        else:
            prev_day = df[df.index.date < target_date.date()]
            prev_close = float(prev_day['Close'].iloc[-1]) if not prev_day.empty else None
            if median_close is not None and prev_close is not None and prev_close != 0:
                gap = ((median_close - prev_close) / prev_close) * 100
            else:
                gap = None
            regime[f'{key}_gap'] = gap
            regime[f'{key}_detail'] = (
                f"{asset} prev_close={prev_close:.2f} → entry_median={median_close:.2f} = {gap:+.3f}%"
                if gap is not None else 'N/A'
            )

    return regime


def check_regime_filter(cfg_row, regime):
    vix_thresh = cfg_row['vix_threshold'] if 'vix_threshold' in cfg_row else 99
    tlt_crash  = cfg_row['tlt_crash']     if 'tlt_crash' in cfg_row else -99
    spy_ignore = cfg_row['spy_ignore']    if 'spy_ignore' in cfg_row else 99

    vix_val = regime.get('vix')
    tlt_val = regime.get('tlt_gap')
    spy_val = regime.get('spy_gap')

    vix_bad = vix_val is not None and vix_val > vix_thresh
    tlt_bad = tlt_val is not None and tlt_val < tlt_crash
    spy_bad = spy_val is not None and spy_val > spy_ignore

    blocked = vix_bad and tlt_bad and spy_bad
    return {
        'blocked': blocked,
        'vix_bad': vix_bad,
        'tlt_bad': tlt_bad,
        'spy_bad': spy_bad,
        'vix_val': vix_val,
        'tlt_val': tlt_val,
        'spy_val': spy_val,
        'vix_thresh': vix_thresh,
        'tlt_crash': tlt_crash,
        'spy_ignore': spy_ignore,
    }


# ==============================================================================
# STEP 4: DEPLOY CONFIGS (for regime thresholds not in BQ)
# ==============================================================================
DEPLOY_CONFIGS = {
    'BGDE': {'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.10},
    'WULF': {'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.24},
    'HOOD': {'vix_threshold': 10.0, 'tlt_crash': -0.30, 'spy_ignore': 0.00},
    'KTOS': {'vix_threshold': 15.0, 'tlt_crash': -0.20, 'spy_ignore': 0.00},
    'RKLB': {'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.00},
    'PLTR': {'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.20},
    'JOBY': {'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.24},
    'LUNR': {'vix_threshold': 16.0, 'tlt_crash': -0.10, 'spy_ignore': 0.00},
    'APLD': {'vix_threshold': 22.0, 'tlt_crash': -0.10, 'spy_ignore': 0.00},
    'RIOT': {'vix_threshold': 18.0, 'tlt_crash': -0.10, 'spy_ignore': 0.20},
    'KEEL': {'vix_threshold': 10.0, 'tlt_crash': -0.16, 'spy_ignore': 0.00},
    'RCAT': {'vix_threshold': 10.0, 'tlt_crash': -0.20, 'spy_ignore': 0.02},
    'BBAI': {'vix_threshold': 10.0, 'tlt_crash': -0.50, 'spy_ignore': 0.48},
    'ACHR': {'vix_threshold': 10.0, 'tlt_crash': -0.50, 'spy_ignore': 0.28},
    'ASTS': {'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.20},
    'VRT':  {'vix_threshold': 15.0, 'tlt_crash': -0.50, 'spy_ignore': 0.00},
    'HUT':  {'vix_threshold': 15.0, 'tlt_crash':  0.00, 'spy_ignore': 0.10},
    'BTBT': {'vix_threshold': 15.0, 'tlt_crash': -0.10, 'spy_ignore': 0.30},
    'CIFR': {'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.48},
    'COIN': {'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.44},
    'IREN': {'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.02},
    'CLSK': {'vix_threshold': 10.0, 'tlt_crash': -0.34, 'spy_ignore': 0.00},
    'MARA': {'vix_threshold': 10.0, 'tlt_crash': -0.20, 'spy_ignore': 0.00},
    'BE':   {'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.30},
}


# ==============================================================================
# STEP 5: ANALYZE EACH GO SIGNAL
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  DETAILED GO SIGNAL ANALYSIS")
print(f"{'='*W}")

trade_results = []

for _, sig in go_signals.iterrows():
    ticker = sig['ticker']
    entry_minute = int(sig['entry_minute'])

    print(f"\n{'─'*W}")
    print(f"  {ticker}  |  {sig['experiment']}  |  seed {int(sig['seed'])}  |  "
          f"prob={sig['prob']:.4f} (thresh={sig['threshold']:.3f})  |  "
          f"entry={entry_minute//60}:{entry_minute%60:02d}")
    print(f"{'─'*W}")

    # --- Trade replay ---
    df_ohlcv = load_ohlcv_pst(ticker)
    if df_ohlcv is None:
        print(f"  OHLCV data not available for {ticker}")
        continue

    entry_time = pd.Timestamp(TARGET_DATE, tz=PST) + timedelta(minutes=entry_minute)
    trade = replay_trade(df_ohlcv, entry_time, sig['tp_pct'], sig['sl_pct'])

    if trade is None:
        print(f"  No OHLCV bars found at/after entry time {entry_time}")
        continue

    outcome_emoji = '+' if trade['exit_pnl'] > 0 else '-'
    print(f"\n  TRADE REPLAY:")
    print(f"    Entry:     ${trade['entry_price']:.2f} at {trade['entry_ts'].strftime('%H:%M')}")
    print(f"    TP target: ${trade['tp_price']:.2f} (+{sig['tp_pct']:.0%})")
    print(f"    SL target: ${trade['sl_price']:.2f} (-{sig['sl_pct']:.0%})")
    print(f"    Exit:      ${trade['exit_price']:.2f} at {trade['exit_ts'].strftime('%H:%M')} "
          f"→ {trade['exit_reason']}")
    print(f"    PnL:       {trade['exit_pnl']:+.2%} {outcome_emoji}")
    print(f"    Max favorable excursion: {trade['max_favorable']:+.2%} "
          f"(high=${trade['high_of_day']:.2f})")
    print(f"    Max adverse excursion:   {trade['max_adverse']:+.2%} "
          f"(low=${trade['low_of_day']:.2f})")
    print(f"    Bars after entry: {trade['bars_after_entry']}")

    if trade['price_path']:
        print(f"\n  PRICE PATH AFTER ENTRY:")
        print(f"    {'T+min':>6} {'Price':>8} {'Change':>8}")
        print(f"    {'─'*6} {'─'*8} {'─'*8}")
        for mins, price, chg in trade['price_path']:
            print(f"    T+{mins:>3}m ${price:>7.2f} {chg:>+7.2%}")

    # --- Regime filter ---
    deploy_cfg = DEPLOY_CONFIGS.get(ticker, {})
    merged_cfg = {**sig.to_dict(), **deploy_cfg}
    regime = get_regime_at_entry(entry_minute)
    regime_check = check_regime_filter(merged_cfg, regime)

    print(f"\n  REGIME STATE AT ENTRY:")
    print(f"    {regime.get('vix_detail', 'N/A')}")
    print(f"      → threshold: {regime_check['vix_thresh']:.0f}  |  "
          f"{'FAIL' if regime_check['vix_bad'] else 'OK'}")
    print(f"    {regime.get('tlt_detail', 'N/A')}")
    print(f"      → crash threshold: {regime_check['tlt_crash']:+.2f}%  |  "
          f"{'FAIL' if regime_check['tlt_bad'] else 'OK'}")
    print(f"    {regime.get('spy_detail', 'N/A')}")
    print(f"      → ignore threshold: {regime_check['spy_ignore']:+.2f}%  |  "
          f"{'FAIL' if regime_check['spy_bad'] else 'OK'}")
    print(f"    Regime filter: {'BLOCKED' if regime_check['blocked'] else 'PASSED'}")

    # --- Pre-entry price action ---
    pre_entry_start = entry_time - timedelta(minutes=30)
    pre_bars = df_ohlcv[(df_ohlcv.index >= pre_entry_start) & (df_ohlcv.index < entry_time)]
    if not pre_bars.empty:
        pre_open  = float(pre_bars['Open'].iloc[0])
        pre_close = float(pre_bars['Close'].iloc[-1])
        pre_high  = float(pre_bars['High'].max())
        pre_low   = float(pre_bars['Low'].min())
        pre_vol   = pre_bars['Volume'].sum() if 'Volume' in pre_bars.columns else 0
        pre_chg   = (pre_close - pre_open) / pre_open if pre_open != 0 else 0

        print(f"\n  PRE-ENTRY (30 min before):")
        print(f"    Open=${pre_open:.2f} → Close=${pre_close:.2f}  "
              f"({pre_chg:+.2%})  High=${pre_high:.2f}  Low=${pre_low:.2f}  "
              f"Vol={pre_vol:,.0f}")

    # --- Volume analysis ---
    if 'Volume' in df_ohlcv.columns:
        entry_bar_data = df_ohlcv[df_ohlcv.index == trade['entry_ts']]
        if not entry_bar_data.empty:
            entry_vol = float(entry_bar_data['Volume'].iloc[0])
            same_time_hist = df_ohlcv[
                (df_ohlcv.index.hour == trade['entry_ts'].hour) &
                (df_ohlcv.index.minute == trade['entry_ts'].minute) &
                (df_ohlcv.index.date != trade['entry_ts'].date())
            ]
            avg_vol = float(same_time_hist['Volume'].mean()) if not same_time_hist.empty else 0
            rvol = entry_vol / avg_vol if avg_vol > 0 else 0

            print(f"\n  ENTRY BAR VOLUME:")
            print(f"    Entry bar volume: {entry_vol:,.0f}")
            print(f"    Avg same-time volume (hist): {avg_vol:,.0f}")
            print(f"    RVOL: {rvol:.1f}x")

    trade_results.append({
        'ticker': ticker,
        'experiment': sig['experiment'],
        'prob': sig['prob'],
        'threshold': sig['threshold'],
        'tp_pct': sig['tp_pct'],
        'sl_pct': sig['sl_pct'],
        'entry_price': trade['entry_price'],
        'exit_reason': trade['exit_reason'],
        'exit_pnl': trade['exit_pnl'],
        'max_favorable': trade['max_favorable'],
        'max_adverse': trade['max_adverse'],
        'regime_blocked': regime_check['blocked'],
        'vix': regime_check['vix_val'],
        'tlt_gap': regime_check['tlt_val'],
        'spy_gap': regime_check['spy_val'],
    })


# ==============================================================================
# STEP 6: SUMMARY
# ==============================================================================
if trade_results:
    print(f"\n\n{'='*W}")
    print(f"  SUMMARY — {TARGET_DATE}")
    print(f"{'='*W}")

    results_df = pd.DataFrame(trade_results)

    print(f"\n  {'Ticker':<7} {'Prob':>7} {'Entry$':>8} {'Exit':>5} {'PnL':>8} "
          f"{'MaxFav':>8} {'MaxAdv':>8} {'Regime':>8} "
          f"{'VIX':>6} {'TLT%':>7} {'SPY%':>7}")
    print(f"  {'─'*(W-2)}")

    for _, r in results_df.iterrows():
        regime_str = 'BLOCK' if r['regime_blocked'] else 'OK'
        vix_str = f"{r['vix']:.1f}" if r['vix'] is not None else 'N/A'
        tlt_str = f"{r['tlt_gap']:+.2f}" if r['tlt_gap'] is not None else 'N/A'
        spy_str = f"{r['spy_gap']:+.2f}" if r['spy_gap'] is not None else 'N/A'
        print(f"  {r['ticker']:<7} {r['prob']:>7.4f} {r['entry_price']:>8.2f} "
              f"{r['exit_reason']:>5} {r['exit_pnl']:>+7.2%} "
              f"{r['max_favorable']:>+7.2%} {r['max_adverse']:>+7.2%} "
              f"{regime_str:>8} {vix_str:>6} {tlt_str:>7} {spy_str:>7}")

    total_pnl = results_df['exit_pnl'].sum()
    wins = (results_df['exit_pnl'] > 0).sum()
    losses = (results_df['exit_pnl'] <= 0).sum()
    blocked = results_df['regime_blocked'].sum()

    print(f"\n  GO signals:      {len(results_df)}")
    print(f"  Wins / Losses:   {wins} / {losses}")
    print(f"  Total PnL:       {total_pnl:+.2%}")
    print(f"  Regime blocked:  {blocked}")

    if losses > 0:
        losers = results_df[results_df['exit_pnl'] <= 0]
        print(f"\n  LOSING TRADES DIAGNOSIS:")
        for _, r in losers.iterrows():
            reached_tp = r['max_favorable'] >= r['tp_pct']
            if reached_tp:
                diag = f"price reached TP level but SL hit first (intra-bar)"
            elif r['max_favorable'] < r['tp_pct'] * 0.25:
                diag = f"never got close to TP ({r['max_favorable']:+.2%} vs TP {r['tp_pct']:+.0%})"
            else:
                diag = f"approached TP ({r['max_favorable']:+.2%} of {r['tp_pct']:+.0%}) but reversed"
            print(f"    {r['ticker']}: {r['exit_pnl']:+.2%} via {r['exit_reason']}  |  {diag}")

print(f"\n{'='*W}")
print(f"  DONE")
print(f"{'='*W}")
