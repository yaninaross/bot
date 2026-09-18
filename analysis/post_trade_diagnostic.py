"""
post_trade_diagnostic.py — Post-Trade Signal Diagnostic
========================================================
Pulls GO signals from BigQuery, fetches actual price data from Polygon,
reconstructs regime conditions, simulates trades, and diagnoses failures.

Usage:
  python post_trade_diagnostic.py                    # last 2 trading days
  python post_trade_diagnostic.py --date 2026-04-29  # specific date
  python post_trade_diagnostic.py --days 5           # last 5 days
"""

import time
import numpy as np
import pandas as pd
import pytz
from datetime import datetime, timedelta
from polygon import RESTClient
from google.cloud import bigquery
from google.cloud import storage as gcs_storage

# ==============================================================================
# CONFIG
# ==============================================================================
POLYGON_API_KEY = os.environ["POLYGON_API_KEY"]
BQ_TABLE        = f"{GCP_PROJECT}.scoring_logs.live_predictions"
BUCKET_NAME     = os.environ.get("GCS_BUCKET", "")
INFERENCE_DIR   = "Data/Inference"
EXIT_TIME_MIN   = 780  # 13:00 ET

ET  = pytz.timezone('US/Eastern')
PST = pytz.timezone('US/Pacific')
UTC = pytz.utc

poly_client = RESTClient(POLYGON_API_KEY)
bq_client   = bigquery.Client()
gcs_client  = gcs_storage.Client()

ASSET_CONFIGS = {
    'RIOT': {'vix_threshold': 10.0, 'tlt_crash': -0.40, 'spy_ignore': 0.10},
    'IREN': {'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.04},
    'WULF': {'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.18},
    'CIFR': {'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.50},
    'BTBT': {'vix_threshold': 15.0, 'tlt_crash':  0.00, 'spy_ignore': 0.00},
    'APLD': {'vix_threshold': 10.0, 'tlt_crash': -0.50, 'spy_ignore': 0.00},
    'CLSK': {'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.10},
    'HUT':  {'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.00},
    'RCAT': {'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.00},
    'LUNR': {'vix_threshold': 16.0, 'tlt_crash': -0.10, 'spy_ignore': 0.00},
    'RKLB': {'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.00},
    'SPIR': {'vix_threshold': 16.0, 'tlt_crash': -0.10, 'spy_ignore': 0.10},
    'PLTR': {'vix_threshold': 10.0, 'tlt_crash': -0.20, 'spy_ignore': 0.10},
    'JOBY': {'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.00},
    'ASTS': {'vix_threshold': 15.0, 'tlt_crash':  0.00, 'spy_ignore': 0.00},
    'KTOS': {'vix_threshold': 10.0, 'tlt_crash': -0.10, 'spy_ignore': 0.00},
}

DEPLOYED_SEEDS = {
    ('97_MICRO',    'RIOT'): 9703,
    ('97_MICRO',    'IREN'): 9715,
    ('97_MICRO',    'WULF'): 9715,
    ('97_MICRO',    'CIFR'): 9714,
    ('110_MICRO',   'BTBT'): 11006,
    ('112_MICRO',   'APLD'): 11204,
    ('112_MICRO',   'CLSK'): 11205,
    ('112_MICRO',   'HUT'):  11206,
    ('103_DEFENSE', 'RCAT'): 10304,
    ('103_DEFENSE', 'LUNR'): 10311,
    ('103_DEFENSE', 'RKLB'): 10308,
    ('103_DEFENSE', 'SPIR'): 10310,
    ('103_DEFENSE', 'PLTR'): 10311,
    ('103_DEFENSE', 'JOBY'): 10307,
    ('103_DEFENSE', 'ASTS'): 10308,
    ('103_DEFENSE', 'KTOS'): 10312,
}


# ==============================================================================
# LAYER 1: FETCH SIGNALS FROM BQ
# ==============================================================================
def fetch_signals(start_date, end_date):
    seed_conditions = " OR ".join(
        f"(experiment = '{exp}' AND ticker = '{tkr}' AND seed = {seed})"
        for (exp, tkr), seed in DEPLOYED_SEEDS.items()
    )
    query = f"""
    SELECT ticker, experiment, seed, prob, threshold, signal,
           tp_pct, sl_pct, entry_minute, close, score_date,
           run_timestamp, bar_timestamp
    FROM `{BQ_TABLE}`
    WHERE score_date BETWEEN '{start_date}' AND '{end_date}'
      AND signal = 'GO'
      AND ({seed_conditions})
    ORDER BY score_date, ticker, run_timestamp
    """
    df = bq_client.query(query).to_dataframe()
    if df.empty:
        print("   No GO signals found for deployed seeds in date range.")
        return df
    df['prob_margin'] = df['prob'] - df['threshold']
    return df


def aggregate_signals(signals_df):
    """Group by (ticker, date) — each group = multiple scoring runs of the same signal."""
    records = []
    for (ticker, date_val), grp in signals_df.groupby(['ticker', 'score_date']):
        last = grp.iloc[-1]
        records.append({
            'ticker': ticker,
            'score_date': date_val,
            'experiment': last['experiment'],
            'seed': last['seed'],
            'prob_avg': grp['prob'].mean(),
            'prob_min': grp['prob'].min(),
            'prob_max': grp['prob'].max(),
            'threshold': last['threshold'],
            'prob_margin_avg': grp['prob_margin'].mean(),
            'tp_pct': last['tp_pct'],
            'sl_pct': last['sl_pct'],
            'entry_minute': last['entry_minute'],
            'close': last['close'],
            'n_runs': len(grp),
            'first_run': grp['run_timestamp'].min(),
            'last_run': grp['run_timestamp'].max(),
        })
    return pd.DataFrame(records)


# ==============================================================================
# LAYER 2: REGIME RECONSTRUCTION (from Polygon)
# ==============================================================================
_regime_bars_cache = {}


def _fetch_day_bars(ticker, date_str):
    cache_key = (ticker, date_str)
    if cache_key in _regime_bars_cache:
        return _regime_bars_cache[cache_key]

    d = pd.Timestamp(date_str)
    prev_d = d - timedelta(days=5)
    all_bars = []
    try:
        for a in poly_client.list_aggs(ticker, 1, 'minute', prev_d, d + timedelta(days=1), limit=50000):
            all_bars.append({
                'timestamp': pd.to_datetime(a.timestamp, unit='ms', utc=True),
                'Open': a.open, 'High': a.high, 'Low': a.low,
                'Close': a.close, 'Volume': a.volume,
            })
    except Exception as e:
        print(f"      Polygon fetch failed for {ticker}: {e}")
        _regime_bars_cache[cache_key] = pd.DataFrame()
        return pd.DataFrame()

    if not all_bars:
        _regime_bars_cache[cache_key] = pd.DataFrame()
        return pd.DataFrame()

    df = pd.DataFrame(all_bars).set_index('timestamp').sort_index()
    df = df[~df.index.duplicated(keep='last')]
    _regime_bars_cache[cache_key] = df
    return df


def reconstruct_regime(date_str, entry_minute):
    d_et = pd.Timestamp(date_str, tz=ET)

    vix_df = _fetch_day_bars('I:VIX', date_str)
    tlt_df = _fetch_day_bars('TLT', date_str)
    spy_df = _fetch_day_bars('SPY', date_str)

    def _to_et(df):
        if df.empty:
            return df
        out = df.copy()
        if out.index.tz is None:
            out.index = out.index.tz_localize('UTC')
        out.index = out.index.tz_convert(ET)
        return out

    vix_et = _to_et(vix_df)
    tlt_et = _to_et(tlt_df)
    spy_et = _to_et(spy_df)

    entry_h, entry_m = divmod(entry_minute, 60)
    entry_ts = d_et.replace(hour=entry_h, minute=entry_m)
    window_start = entry_ts - timedelta(minutes=30)

    vix_level = None
    if not vix_et.empty:
        bars = vix_et[(vix_et.index >= window_start) & (vix_et.index <= entry_ts)]
        if len(bars):
            vix_level = float(bars['Close'].median())

    tlt_gap = None
    if not tlt_et.empty:
        prev = tlt_et[tlt_et.index.date < d_et.date()]
        bars = tlt_et[(tlt_et.index >= window_start) & (tlt_et.index <= entry_ts)]
        if len(prev) > 0 and len(bars) > 0:
            prev_close = float(prev['Close'].iloc[-1])
            curr_close = float(bars['Close'].median())
            if prev_close != 0:
                tlt_gap = ((curr_close - prev_close) / prev_close) * 100

    spy_gap = None
    if not spy_et.empty:
        prev = spy_et[spy_et.index.date < d_et.date()]
        bars = spy_et[(spy_et.index >= window_start) & (spy_et.index <= entry_ts)]
        if len(prev) > 0 and len(bars) > 0:
            prev_close = float(prev['Close'].iloc[-1])
            curr_close = float(bars['Close'].median())
            if prev_close != 0:
                spy_gap = ((curr_close - prev_close) / prev_close) * 100

    return {'vix': vix_level, 'tlt_gap': tlt_gap, 'spy_gap': spy_gap}


def check_filter(ticker, regime):
    cfg = ASSET_CONFIGS.get(ticker)
    if cfg is None or regime is None:
        return True, {}

    vix_bad = regime['vix']     is not None and regime['vix']     > cfg['vix_threshold']
    tlt_bad = regime['tlt_gap'] is not None and regime['tlt_gap'] < cfg['tlt_crash']
    spy_bad = regime['spy_gap'] is not None and regime['spy_gap'] > cfg['spy_ignore']
    blocked = vix_bad and tlt_bad and spy_bad

    details = {
        'vix_value': regime['vix'], 'vix_threshold': cfg['vix_threshold'], 'vix_bad': vix_bad,
        'tlt_value': regime['tlt_gap'], 'tlt_threshold': cfg['tlt_crash'], 'tlt_bad': tlt_bad,
        'spy_value': regime['spy_gap'], 'spy_threshold': cfg['spy_ignore'], 'spy_bad': spy_bad,
        'would_block': blocked,
        'bad_count': sum([vix_bad, tlt_bad, spy_bad]),
    }
    return not blocked, details


# ==============================================================================
# LAYER 3: TRADE OUTCOME SIMULATION
# ==============================================================================
def fetch_ticker_bars(ticker, date_str):
    d = pd.Timestamp(date_str)
    all_bars = []
    try:
        for a in poly_client.list_aggs(ticker, 1, 'minute', d, d + timedelta(days=1), limit=50000):
            all_bars.append({
                'timestamp': pd.to_datetime(a.timestamp, unit='ms', utc=True),
                'Open': a.open, 'High': a.high, 'Low': a.low,
                'Close': a.close, 'Volume': a.volume,
            })
    except Exception as e:
        print(f"      Polygon fetch failed for {ticker}: {e}")
        return pd.DataFrame()

    if not all_bars:
        return pd.DataFrame()

    df = pd.DataFrame(all_bars).set_index('timestamp').sort_index()
    df = df[~df.index.duplicated(keep='last')]
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert(ET)
    df['minute_of_day'] = df.index.hour * 60 + df.index.minute
    return df


def simulate_trade(bars_df, entry_minute, tp_pct, sl_pct, exit_minute=EXIT_TIME_MIN):
    if bars_df.empty:
        return None

    entry_bars = bars_df[bars_df['minute_of_day'] == entry_minute]
    if entry_bars.empty:
        nearby = bars_df[(bars_df['minute_of_day'] >= entry_minute - 2) &
                         (bars_df['minute_of_day'] <= entry_minute + 2)]
        if nearby.empty:
            return None
        entry_bars = nearby.iloc[[0]]

    entry_price = float(entry_bars['Close'].iloc[0])
    entry_ts    = entry_bars.index[0]
    tp_target   = entry_price * (1 + tp_pct)
    sl_stop     = entry_price * (1 - sl_pct)

    after_entry = bars_df[bars_df.index > entry_ts]
    if after_entry.empty:
        return {
            'entry_price': entry_price, 'exit_price': entry_price,
            'pnl_pct': 0.0, 'outcome': 'NO_DATA', 'exit_time': None,
            'mae': 0.0, 'mfe': 0.0, 'duration_min': 0,
        }

    max_high = entry_price
    min_low  = entry_price
    exit_price = None
    outcome    = None
    exit_ts    = None

    for ts, bar in after_entry.iterrows():
        if bar['minute_of_day'] > exit_minute and ts.date() == entry_ts.date():
            exit_price = float(bar['Close'])
            outcome = 'EOD_EXIT'
            exit_ts = ts
            break

        if ts.date() != entry_ts.date():
            prev_bar = bars_df[bars_df.index < ts].iloc[-1]
            exit_price = float(prev_bar['Close'])
            outcome = 'EOD_EXIT'
            exit_ts = prev_bar.name
            break

        if float(bar['Low']) <= sl_stop:
            exit_price = sl_stop
            outcome = 'SL_HIT'
            exit_ts = ts
            break
        if float(bar['High']) >= tp_target:
            exit_price = tp_target
            outcome = 'TP_HIT'
            exit_ts = ts
            break

        max_high = max(max_high, float(bar['High']))
        min_low  = min(min_low, float(bar['Low']))

    if exit_price is None:
        last = after_entry.iloc[-1]
        exit_price = float(last['Close'])
        outcome = 'EOD_EXIT'
        exit_ts = last.name

    max_high = max(max_high, exit_price)
    min_low  = min(min_low, exit_price)

    pnl_pct = (exit_price - entry_price) / entry_price
    mfe = (max_high - entry_price) / entry_price
    mae = (entry_price - min_low) / entry_price
    duration = int((exit_ts - entry_ts).total_seconds() / 60) if exit_ts else 0

    return {
        'entry_price': round(entry_price, 4),
        'exit_price': round(exit_price, 4),
        'pnl_pct': round(pnl_pct, 4),
        'outcome': outcome,
        'exit_time': exit_ts,
        'mae': round(mae, 4),
        'mfe': round(mfe, 4),
        'duration_min': duration,
    }


# ==============================================================================
# LAYER 4: CORRELATION ANALYSIS
# ==============================================================================
def analyze_correlations(diagnostics_by_date):
    results = {}
    for date_str, diags in diagnostics_by_date.items():
        pnls = {d['ticker']: d['trade']['pnl_pct']
                for d in diags if d.get('trade') and d['trade'].get('pnl_pct') is not None}
        if not pnls:
            continue

        n_signals = len(pnls)
        n_losses  = sum(1 for p in pnls.values() if p < 0)
        loss_rate = n_losses / n_signals if n_signals > 0 else 0
        avg_pnl   = np.mean(list(pnls.values()))

        spy_df = _fetch_day_bars('SPY', date_str)
        spy_return = None
        if not spy_df.empty:
            spy_et = spy_df.copy()
            if spy_et.index.tz is None:
                spy_et.index = spy_et.index.tz_localize('UTC')
            spy_et.index = spy_et.index.tz_convert(ET)
            day_bars = spy_et[spy_et.index.date == pd.Timestamp(date_str).date()]
            if len(day_bars) > 1:
                spy_return = (float(day_bars['Close'].iloc[-1]) - float(day_bars['Open'].iloc[0])) / float(day_bars['Open'].iloc[0])

        all_negative = all(p < 0 for p in pnls.values())
        is_macro_crash = all_negative and n_signals >= 3 and loss_rate == 1.0

        results[date_str] = {
            'n_signals': n_signals, 'n_losses': n_losses,
            'loss_rate': loss_rate, 'avg_pnl': avg_pnl,
            'spy_return': spy_return, 'is_macro_crash': is_macro_crash,
            'pnls': pnls,
        }

    return results


# ==============================================================================
# LAYER 5: DATA FRESHNESS CHECK
# ==============================================================================
def check_data_freshness(ticker, score_date):
    clean = ticker.replace("I:", "").replace("X:", "")
    uri = f"gs://{BUCKET_NAME}/{INFERENCE_DIR}/{clean}_master_inference.parquet"
    try:
        df = pd.read_parquet(uri)
        df = df.reset_index()
        dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        if dt_cols:
            last_ts = pd.to_datetime(df[dt_cols[0]].max(), utc=True)
        else:
            return None
        score_ts = pd.Timestamp(score_date, tz='UTC').replace(hour=14)
        staleness_hours = (score_ts - last_ts).total_seconds() / 3600
        return round(staleness_hours, 1)
    except Exception:
        return None


# ==============================================================================
# DIAGNOSIS ENGINE
# ==============================================================================
def diagnose_signal(signal, regime, filter_details, trade, data_age_hours, is_macro_day):
    tags = []

    if data_age_hours is not None and data_age_hours > 24:
        tags.append('STALE_DATA')

    if filter_details.get('bad_count', 0) >= 2:
        tags.append('REGIME_MISS')

    if is_macro_day:
        tags.append('MACRO_CRASH')

    if trade and trade.get('pnl_pct') is not None and trade['pnl_pct'] < 0:
        prob_margin = signal.get('prob_margin_avg', signal.get('prob_margin', 0))
        if prob_margin < 0.02:
            tags.append('LOW_CONFIDENCE')
        elif prob_margin > 0.10:
            tags.append('MODEL_OVERCONFIDENT')

    if not tags:
        if trade and trade.get('pnl_pct', 0) >= 0:
            tags.append('WIN')
        else:
            tags.append('NORMAL_LOSS')

    return tags


# ==============================================================================
# REPORT PRINTER
# ==============================================================================
def print_report(all_diagnostics, correlation_results):
    from collections import Counter
    W = 120
    print(f"\n{'='*W}")
    print(f"  POST-TRADE DIAGNOSTIC REPORT  (deployed seeds only)")
    print(f"{'='*W}")

    for date_str in sorted(set(d['date'] for d in all_diagnostics)):
        day_diags = [d for d in all_diagnostics if d['date'] == date_str]
        corr = correlation_results.get(date_str, {})

        spy_str = f"{corr['spy_return']:+.2%}" if corr.get('spy_return') is not None else "N/A"
        n_with_trade = sum(1 for d in day_diags if d.get('trade', {}).get('pnl_pct') is not None)
        n_losses = sum(1 for d in day_diags
                       if d.get('trade', {}).get('pnl_pct') is not None
                       and d['trade']['pnl_pct'] < 0)
        print(f"\n{'─'*W}")
        print(f"  DATE: {date_str}   |   Tickers signaled: {len(day_diags)}   "
              f"|   With trade data: {n_with_trade}   "
              f"|   Losses: {n_losses}/{n_with_trade}   "
              f"|   SPY: {spy_str}")
        if corr.get('is_macro_crash'):
            print(f"  *** CORRELATED MACRO CRASH — ALL SIGNALS LOST ***")
        print(f"{'─'*W}")

        regime_printed = False
        for d in day_diags:
            if not regime_printed and d.get('regime'):
                r = d['regime']
                vix_s = f"{r['vix']:.1f}" if r['vix'] is not None else "N/A"
                tlt_s = f"{r['tlt_gap']:+.2f}%" if r['tlt_gap'] is not None else "N/A"
                spy_s = f"{r['spy_gap']:+.2f}%" if r['spy_gap'] is not None else "N/A"
                print(f"  Regime at entry: VIX={vix_s}  TLT={tlt_s}  SPY={spy_s}")
                regime_printed = True

        print()
        print(f"  {'Ticker':<6} {'Exp':<12} {'Runs':>4} {'Prob':>12} {'Thr':>5} {'Margin':>7} "
              f"{'Entry':>6} {'Close':>7} {'TP/SL':>10} "
              f"{'Result':>8} {'PnL':>7} {'MAE':>6} {'MFE':>6} {'Dur':>5} {'Diagnosis'}")
        print(f"  {'─'*6} {'─'*12} {'─'*4} {'─'*12} {'─'*5} {'─'*7} "
              f"{'─'*6} {'─'*7} {'─'*10} "
              f"{'─'*8} {'─'*7} {'─'*6} {'─'*6} {'─'*5} {'─'*20}")

        for d in day_diags:
            s = d['signal']
            t = d.get('trade', {})
            tags = d.get('tags', [])

            entry_m = s.get('entry_minute', 0)
            entry_str = f"{entry_m // 60}:{entry_m % 60:02d}"
            close = s.get('close', 0)
            tp_pct = s.get('tp_pct', 0)
            sl_pct = s.get('sl_pct', 0)
            tpsl_str = f"+{tp_pct:.0%}/-{sl_pct:.0%}"

            n_runs = s.get('n_runs', 1)
            prob_avg = s.get('prob_avg', s.get('prob', 0))
            prob_min = s.get('prob_min', prob_avg)
            prob_max = s.get('prob_max', prob_avg)
            if prob_min == prob_max:
                prob_str = f"{prob_avg:.3f}"
            else:
                prob_str = f"{prob_min:.3f}-{prob_max:.3f}"

            margin = s.get('prob_margin_avg', s.get('prob_margin', 0))

            pnl_str = f"{t['pnl_pct']:+.2%}" if t.get('pnl_pct') is not None else "--"
            mae_str = f"{t['mae']:.2%}" if t.get('mae') is not None else "--"
            mfe_str = f"{t['mfe']:.2%}" if t.get('mfe') is not None else "--"
            dur_str = f"{t['duration_min']}m" if t.get('duration_min') is not None else "--"
            outcome = t.get('outcome', '--')
            tag_str = ', '.join(tags)

            pnl_color = ''
            pnl_reset = ''
            if t.get('pnl_pct') is not None:
                if t['pnl_pct'] < 0:
                    pnl_color = '\033[91m'
                    pnl_reset = '\033[0m'
                else:
                    pnl_color = '\033[92m'
                    pnl_reset = '\033[0m'

            print(f"  {s['ticker']:<6} {s['experiment']:<12} {n_runs:>4} "
                  f"{prob_str:>12} {s['threshold']:>5.3f} {margin:>+7.3f} "
                  f"{entry_str:>6} {close:>7.2f} {tpsl_str:>10} "
                  f"{outcome:>8} {pnl_color}{pnl_str:>7}{pnl_reset} "
                  f"{mae_str:>6} {mfe_str:>6} {dur_str:>5} {tag_str}")

            fd = d.get('filter_details', {})
            if fd and fd.get('bad_count', 0) > 0:
                parts = []
                if fd.get('vix_bad'):
                    parts.append(f"VIX {fd['vix_value']:.1f}>{fd['vix_threshold']:.0f}")
                if fd.get('tlt_bad'):
                    parts.append(f"TLT {fd['tlt_value']:+.2f}<{fd['tlt_threshold']:+.2f}")
                if fd.get('spy_bad'):
                    parts.append(f"SPY {fd['spy_value']:+.2f}>{fd['spy_threshold']:+.2f}")
                status = "BLOCKED" if fd.get('would_block') else f"{fd['bad_count']}/3 bad"
                print(f"         filter: {' | '.join(parts)}  [{status}]")

    # Summary
    has_trade = [d for d in all_diagnostics
                 if d.get('trade', {}).get('pnl_pct') is not None]
    no_data = [d for d in all_diagnostics
               if d.get('trade', {}).get('pnl_pct') is None]

    print(f"\n{'='*W}")
    print(f"  SUMMARY")
    print(f"{'='*W}")

    total = len(has_trade)
    if total == 0:
        print("  No trades with data to analyze.")
        print(f"  Tickers with no trade data: {', '.join(d['ticker'] for d in no_data)}")
        print(f"\n{'='*W}")
        return

    wins = sum(1 for d in has_trade if d['trade']['pnl_pct'] >= 0)
    losses = total - wins
    avg_pnl = np.mean([d['trade']['pnl_pct'] for d in has_trade])
    total_pnl = sum(d['trade']['pnl_pct'] for d in has_trade)

    all_tags = []
    for d in all_diagnostics:
        all_tags.extend(d.get('tags', []))
    tag_counts = Counter(all_tags)

    print(f"  Unique signals: {len(all_diagnostics)} tickers across "
          f"{len(set(d['date'] for d in all_diagnostics))} day(s)")
    print(f"  With trade data: {total}   |   No data: {len(no_data)}")
    print(f"  Wins / Losses:   {wins} / {losses}  ({wins/total:.0%} win rate)")
    print(f"  Avg PnL:         {avg_pnl:+.2%}")
    print(f"  Total PnL:       {total_pnl:+.2%}")

    if no_data:
        print(f"\n  No trade data (pre-market illiquid?):")
        for d in no_data:
            print(f"    {d['ticker']} on {d['date']}")

    print(f"\n  Failure mode breakdown:")
    for tag, count in tag_counts.most_common():
        pct = count / len(all_diagnostics)
        print(f"    {tag:<25} {count:>3}  ({pct:.0%})")

    print(f"\n{'='*W}")


# ==============================================================================
# MAIN
# ==============================================================================
def main(date=None, days=2, skip_freshness=True):
    """
    Run post-trade diagnostic.

    Args:
        date: Specific date string 'YYYY-MM-DD', or None for last N days
        days: Number of trading days back (default 2)
        skip_freshness: Skip GCS data freshness check (default True — broken for past dates)

    Colab usage:
        main()                          # last 2 trading days
        main(date='2026-04-29')         # specific date
        main(days=5)                    # last 5 days
    """
    if date:
        start_date = date
        end_date   = date
    else:
        end_date   = datetime.now(ET).strftime('%Y-%m-%d')
        start_date = (datetime.now(ET) - timedelta(days=days + 2)).strftime('%Y-%m-%d')

    print(f"\n{'='*80}")
    print(f"  POST-TRADE DIAGNOSTIC  (deployed seeds only)")
    print(f"  Date range: {start_date} -> {end_date}")
    print(f"  Deployed tickers: {len(DEPLOYED_SEEDS)}")
    print(f"{'='*80}")

    # Layer 1: Fetch deployed GO signals
    print(f"\n  [1/5] Fetching GO signals from BigQuery (deployed seeds only)...")
    signals_df = fetch_signals(start_date, end_date)
    if signals_df.empty:
        print("  No deployed GO signals found. Exiting.")
        return

    # Aggregate: one row per (ticker, date), across multiple scoring runs
    agg_df = aggregate_signals(signals_df)
    print(f"  Found {len(signals_df)} raw GO rows -> {len(agg_df)} unique (ticker, date) signals")
    print()
    for _, row in agg_df.iterrows():
        prob_str = f"{row['prob_min']:.3f}-{row['prob_max']:.3f}" if row['prob_min'] != row['prob_max'] else f"{row['prob_avg']:.3f}"
        print(f"    {str(row['score_date'])[:10]}  {row['ticker']:<6}  {row['experiment']:<15}  "
              f"runs={row['n_runs']:>2}  prob={prob_str}  thr={row['threshold']:.3f}  "
              f"margin={row['prob_margin_avg']:+.3f}")

    # Process each unique (ticker, date) signal
    all_diagnostics = []
    diagnostics_by_date = {}

    unique_dates = sorted(agg_df['score_date'].unique())

    for date_val in unique_dates:
        date_str = str(date_val)[:10]
        day_signals = agg_df[agg_df['score_date'] == date_val]
        diagnostics_by_date[date_str] = []

        # Layer 2: Regime reconstruction (once per date)
        print(f"\n  [2/5] Reconstructing regime for {date_str}...")
        median_entry = int(day_signals['entry_minute'].median())
        regime = reconstruct_regime(date_str, median_entry)
        vix_s = f"{regime['vix']:.1f}" if regime['vix'] is not None else "N/A"
        tlt_s = f"{regime['tlt_gap']:+.2f}%" if regime['tlt_gap'] is not None else "N/A"
        spy_s = f"{regime['spy_gap']:+.2f}%" if regime['spy_gap'] is not None else "N/A"
        print(f"    VIX={vix_s}  TLT={tlt_s}  SPY={spy_s}")

        for _, sig in day_signals.iterrows():
            ticker = sig['ticker']
            entry_min = int(sig['entry_minute'])

            # Per-ticker regime at its specific entry_minute
            if entry_min != median_entry:
                ticker_regime = reconstruct_regime(date_str, entry_min)
            else:
                ticker_regime = regime

            filter_ok, filter_details = check_filter(ticker, ticker_regime)

            # Layer 3: Trade simulation (ONCE per ticker per day)
            print(f"  [3/5] Simulating {ticker} trade on {date_str}...")
            bars = fetch_ticker_bars(ticker, date_str)
            trade = simulate_trade(bars, entry_min, sig['tp_pct'], sig['sl_pct'])
            if trade:
                pnl_str = f"{trade['pnl_pct']:+.2%}"
                print(f"    {ticker}: {trade['outcome']}  PnL={pnl_str}  "
                      f"MAE={trade['mae']:.2%}  MFE={trade['mfe']:.2%}")
            else:
                print(f"    {ticker}: No trade data (pre-market illiquid?)")
                trade = {}
            time.sleep(0.3)

            # Layer 5: Data freshness (disabled by default — reads current file, not score-time snapshot)
            data_age = None
            if not skip_freshness:
                data_age = check_data_freshness(ticker, date_str)

            diag = {
                'date': date_str,
                'ticker': ticker,
                'signal': dict(sig),
                'regime': ticker_regime,
                'filter_ok': filter_ok,
                'filter_details': filter_details,
                'trade': trade,
                'data_age': data_age,
                'tags': [],
            }
            all_diagnostics.append(diag)
            diagnostics_by_date[date_str].append(diag)

    # Layer 4: Correlation analysis
    print(f"\n  [4/5] Analyzing cross-signal correlations...")
    correlation_results = analyze_correlations(diagnostics_by_date)
    for date_str, corr in correlation_results.items():
        macro_tag = " *** MACRO CRASH ***" if corr['is_macro_crash'] else ""
        spy_str = f"{corr['spy_return']:+.2%}" if corr['spy_return'] is not None else "N/A"
        print(f"    {date_str}: {corr['n_losses']}/{corr['n_signals']} losses, "
              f"avg PnL={corr['avg_pnl']:+.2%}, SPY={spy_str}{macro_tag}")

    # Apply diagnosis tags
    for diag in all_diagnostics:
        date_str = diag['date']
        is_macro = correlation_results.get(date_str, {}).get('is_macro_crash', False)
        diag['tags'] = diagnose_signal(
            diag['signal'], diag['regime'], diag['filter_details'],
            diag['trade'], diag['data_age'], is_macro,
        )

    # Print report
    print_report(all_diagnostics, correlation_results)


if __name__ == '__main__':
    main()
