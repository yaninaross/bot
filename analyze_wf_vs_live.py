"""
WF Backtest vs Live Performance Comparison
============================================
Compares walkforward optimization predictions against actual live outcomes.
Identifies config mismatches, performance deltas, and overfitting signals.

BQ tables:
  - WF results: ys-time-series.scoring_logs.wf_optimization_results
    (model_version = 'robust_wf_v9')
  - Live predictions: ys-time-series.scoring_logs.live_predictions

Run in Colab or on the VM after daily_score_combined.py has been logging to BQ.
"""

import pandas as pd
import numpy as np
import math
from google.cloud import bigquery

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)
pd.set_option('display.max_rows', 200)

client = bigquery.Client(project='ys-time-series')

# =====================================================
# CONFIG: Deployed tickers and their configs
# (extracted from daily_score_combined.py as of 2026-06-08)
# =====================================================
DEPLOYED = {
    'APLD': '152', 'IREN': '152', 'LUNR': '152',
    'CIFR': '160', 'FCEL': '160', 'KEEL': '160', 'PLUG': '160', 'RCAT': '160',
    'ALAB': '170', 'ASTS': '170', 'BE': '170', 'BTBT': '170',
    'CLSK': '170', 'OKLO': '170', 'RKLB': '170', 'WULF': '170',
    'PL': '185', 'RDW': '185', 'RIOT': '185', 'SMCI': '185',
}

DEPLOYED_CONFIGS = {
    'APLD': {'seed': 15203, 'threshold': 0.550, 'entry_minute': 370, 'tpsl_mode': 'atr_adaptive',
             'tp_pct': 0.0, 'sl_pct': 0.0, 'tp_mult': 1.5, 'sl_mult': 1.0,
             'wf_wr': '100%', 'wf_pnl': '4.4%', 'wf_n': 4, 'confidence': 'LOW'},
    'IREN': {'seed': -1, 'threshold': 0.600, 'entry_minute': 380, 'tpsl_mode': 'static',
             'tp_pct': 0.08, 'sl_pct': 0.03,
             'wf_wr': '50%', 'wf_pnl': '2.2%', 'wf_n': 2, 'confidence': 'LOW'},
    'LUNR': {'seed': -1, 'threshold': 0.500, 'entry_minute': 380, 'tpsl_mode': 'regime_scaled',
             'tp_pct': 0.30, 'sl_pct': 0.08,
             'wf_wr': '60%', 'wf_pnl': '8.9%', 'wf_n': 5, 'confidence': 'MED'},
    'CIFR': {'seed': -1, 'threshold': 0.500, 'entry_minute': 375, 'tpsl_mode': 'static',
             'tp_pct': 0.12, 'sl_pct': 0.08,
             'wf_wr': '67%', 'wf_pnl': '4.4%', 'wf_n': 3, 'confidence': 'LOW'},
    'FCEL': {'seed': 16005, 'threshold': 0.450, 'entry_minute': 370, 'tpsl_mode': 'atr_adaptive',
             'tp_pct': 0.0, 'sl_pct': 0.0, 'tp_mult': 2.5, 'sl_mult': 1.25,
             'wf_wr': '50%', 'wf_pnl': '4.8%', 'wf_n': 10, 'confidence': 'HIGH'},
    'KEEL': {'seed': -1, 'threshold': 0.600, 'entry_minute': 375, 'tpsl_mode': 'regime_scaled',
             'tp_pct': 0.16, 'sl_pct': 0.14,
             'wf_wr': '100%', 'wf_pnl': '5.9%', 'wf_n': 3, 'confidence': 'LOW'},
    'PLUG': {'seed': 16015, 'threshold': 0.500, 'entry_minute': 370, 'tpsl_mode': 'trailing_decay',
             'tp_pct': 0.0, 'sl_pct': 0.0, 'sl_mult': 1.5, 'trail_start_mult': 2.0, 'trail_end_mult': 0.5,
             'wf_wr': '100%', 'wf_pnl': '3.2%', 'wf_n': 2, 'confidence': 'LOW'},
    'RCAT': {'seed': 16013, 'threshold': 0.500, 'entry_minute': 370, 'tpsl_mode': 'atr',
             'tp_pct': 0.0, 'sl_pct': 0.0, 'tp_mult': 2.0, 'sl_mult': 2.0,
             'wf_wr': '100%', 'wf_pnl': '4.9%', 'wf_n': 2, 'confidence': 'LOW'},
    'ALAB': {'seed': 17507, 'threshold': 0.450, 'entry_minute': 360, 'tpsl_mode': 'static',
             'tp_pct': 0.12, 'sl_pct': 0.03,
             'wf_wr': '62%', 'wf_pnl': '3.1%', 'wf_n': 8, 'confidence': 'MED'},
    'ASTS': {'seed': -1, 'threshold': 0.600, 'entry_minute': 370, 'tpsl_mode': 'atr_adaptive',
             'tp_pct': 0.0, 'sl_pct': 0.0, 'tp_mult': 2.0, 'sl_mult': 2.0,
             'wf_wr': '71%', 'wf_pnl': '4.6%', 'wf_n': 7, 'confidence': 'MED'},
    'BE':   {'seed': 17500, 'threshold': 0.550, 'entry_minute': 360, 'tpsl_mode': 'static',
             'tp_pct': 0.25, 'sl_pct': 0.14,
             'wf_wr': '64%', 'wf_pnl': '2.0%', 'wf_n': 11, 'confidence': 'HIGH'},
    'BTBT': {'seed': 17510, 'threshold': 0.500, 'entry_minute': 360, 'tpsl_mode': 'atr_adaptive',
             'tp_pct': 0.0, 'sl_pct': 0.0, 'tp_mult': 1.5, 'sl_mult': 1.0,
             'wf_wr': '100%', 'wf_pnl': '3.4%', 'wf_n': 4, 'confidence': 'LOW'},
    'CLSK': {'seed': 17504, 'threshold': 0.450, 'entry_minute': 380, 'tpsl_mode': 'regime_scaled',
             'tp_pct': 0.06, 'sl_pct': 0.05,
             'wf_wr': '78%', 'wf_pnl': '3.0%', 'wf_n': 9, 'confidence': 'MED'},
    'OKLO': {'seed': 17501, 'threshold': 0.500, 'entry_minute': 370, 'tpsl_mode': 'static',
             'tp_pct': 0.12, 'sl_pct': 0.05,
             'wf_wr': '100%', 'wf_pnl': '5.4%', 'wf_n': 3, 'confidence': 'LOW'},
    'RKLB': {'seed': 17514, 'threshold': 0.750, 'entry_minute': 375, 'tpsl_mode': 'static',
             'tp_pct': 0.08, 'sl_pct': 0.08,
             'wf_wr': '75%', 'wf_pnl': '2.5%', 'wf_n': 8, 'confidence': 'MED'},
    'WULF': {'seed': 17511, 'threshold': 0.500, 'entry_minute': 365, 'tpsl_mode': 'atr_adaptive',
             'tp_pct': 0.0, 'sl_pct': 0.0, 'tp_mult': 1.5, 'sl_mult': 1.0,
             'wf_wr': '100%', 'wf_pnl': '3.9%', 'wf_n': 3, 'confidence': 'LOW'},
    'PL':   {'seed': 18507, 'threshold': 0.500, 'entry_minute': 360, 'tpsl_mode': 'static',
             'tp_pct': 0.06, 'sl_pct': 0.03,
             'wf_wr': '67%', 'wf_pnl': '2.7%', 'wf_n': 3, 'confidence': 'LOW'},
    'RDW':  {'seed': 18503, 'threshold': 0.600, 'entry_minute': 370, 'tpsl_mode': 'atr',
             'tp_pct': 0.0, 'sl_pct': 0.0, 'tp_mult': 2.0, 'sl_mult': 1.0,
             'wf_wr': '50%', 'wf_pnl': '2.5%', 'wf_n': 8, 'confidence': 'MED'},
    'RIOT': {'seed': 18503, 'threshold': 0.500, 'entry_minute': 375, 'tpsl_mode': 'trailing_decay',
             'tp_pct': 0.0, 'sl_pct': 0.0, 'sl_mult': 2.0, 'trail_start_mult': 2.0, 'trail_end_mult': 1.0,
             'wf_wr': '100%', 'wf_pnl': '3.2%', 'wf_n': 3, 'confidence': 'LOW'},
    'SMCI': {'seed': -1, 'threshold': 0.500, 'entry_minute': 360, 'tpsl_mode': 'static',
             'tp_pct': 0.06, 'sl_pct': 0.06,
             'wf_wr': '80%', 'wf_pnl': '2.7%', 'wf_n': 5, 'confidence': 'MED'},
}


W = 130

# =====================================================
# STEP 1: Load WF optimization results from BQ
# =====================================================
print("=" * W)
print("  STEP 1: WF OPTIMIZATION RESULTS FROM BIGQUERY")
print("=" * W)

q_wf = """
SELECT ticker, seed, entry_time, entry_minute, prob_thresh, tpsl_mode,
       tp_pct, sl_pct, tp_mult, sl_mult, trail_start_mult, trail_end_mult,
       val_a_avg_pnl, val_a_win_rate, val_a_count, survived_a,
       val_b_avg_pnl, val_b_win_rate, val_b_count, survived_b,
       val_a_start, val_a_end, val_b_start, val_b_end,
       opt_start, opt_end, experiment, model_version,
       is_ensemble, n_seeds_avg,
       vix_threshold, tlt_crash, spy_ignore, vol_filter_pct
FROM `ys-time-series.scoring_logs.wf_optimization_results`
WHERE model_version = 'robust_wf_v9'
  AND survived_a = TRUE
  AND val_b_start = '2026-04-01'
ORDER BY ticker, seed
"""
wf = client.query(q_wf).to_dataframe()
print(f"  Total WF rows (survived_a, val_b from 2026-04-01): {len(wf)}")
print(f"  Tickers in WF: {sorted(wf.ticker.unique())}")
print(f"  val_b_start values: {sorted(wf.val_b_start.astype(str).unique())}")
print(f"  val_b_end values: {sorted(wf.val_b_end.astype(str).unique())}")
print(f"  Experiments: {sorted(wf.experiment.unique())}")
print()

# Composite score: avg_pnl * win_rate * log2(1+N) using val_b (test period)
wf['composite'] = wf['val_b_avg_pnl'] * wf['val_b_win_rate'] * np.log2(1 + wf['val_b_count'])

# Filter to deployed tickers
wf_dep = wf[wf.ticker.isin(DEPLOYED.keys())].copy()
print(f"  WF rows for deployed tickers: {len(wf_dep)}")
print(f"  Deployed tickers found in WF: {sorted(wf_dep.ticker.unique())}")
missing = set(DEPLOYED.keys()) - set(wf_dep.ticker.unique())
if missing:
    print(f"  *** WARNING: Deployed tickers NOT in WF results: {missing}")
print()


# =====================================================
# STEP 2: Config comparison — WF best vs deployed
# =====================================================
print("=" * W)
print("  STEP 2: CONFIG COMPARISON -- WF BEST vs DEPLOYED")
print("=" * W)
print()

best_wf = {}
for t in sorted(DEPLOYED.keys()):
    dc = DEPLOYED_CONFIGS[t]
    subset = wf_dep[wf_dep.ticker == t]
    if len(subset) == 0:
        print(f"  {t} ({DEPLOYED[t]}): NO WF ROWS FOUND")
        continue

    best = subset.loc[subset['composite'].idxmax()]
    best_wf[t] = best
    mismatches = []

    # Seed (skip for ensemble, seed=-1)
    if dc['seed'] != -1:
        wf_seed = int(best['seed'])
        if wf_seed != dc['seed']:
            mismatches.append(f"SEED: wf={wf_seed} deploy={dc['seed']}")

    # Threshold
    if abs(best['prob_thresh'] - dc['threshold']) > 0.001:
        mismatches.append(f"THR: wf={best['prob_thresh']:.3f} deploy={dc['threshold']:.3f}")

    # Entry minute
    if int(best['entry_minute']) != dc['entry_minute']:
        mismatches.append(f"ENTRY: wf={int(best['entry_minute'])} deploy={dc['entry_minute']}")

    # TPSL mode
    wf_mode = str(best['tpsl_mode']).strip() if pd.notna(best['tpsl_mode']) else ''
    if wf_mode != dc['tpsl_mode']:
        mismatches.append(f"MODE: wf={wf_mode} deploy={dc['tpsl_mode']}")

    # Static TP/SL
    if dc['tpsl_mode'] in ('static', 'regime_scaled'):
        if abs(best['tp_pct'] - dc.get('tp_pct', 0)) > 0.001:
            mismatches.append(f"TP: wf={best['tp_pct']:.3f} deploy={dc['tp_pct']:.3f}")
        if abs(best['sl_pct'] - dc.get('sl_pct', 0)) > 0.001:
            mismatches.append(f"SL: wf={best['sl_pct']:.3f} deploy={dc['sl_pct']:.3f}")

    # ATR mults
    if dc['tpsl_mode'] in ('atr', 'atr_adaptive'):
        if dc.get('tp_mult') and pd.notna(best.get('tp_mult')):
            if abs(best['tp_mult'] - dc['tp_mult']) > 0.01:
                mismatches.append(f"TP_MULT: wf={best['tp_mult']:.2f} deploy={dc['tp_mult']:.2f}")
        if dc.get('sl_mult') and pd.notna(best.get('sl_mult')):
            if abs(best['sl_mult'] - dc['sl_mult']) > 0.01:
                mismatches.append(f"SL_MULT: wf={best['sl_mult']:.2f} deploy={dc['sl_mult']:.2f}")

    # Trailing decay mults
    if dc['tpsl_mode'] == 'trailing_decay':
        for k in ('sl_mult', 'trail_start_mult', 'trail_end_mult'):
            if dc.get(k) is not None and pd.notna(best.get(k)):
                if abs(best[k] - dc[k]) > 0.01:
                    mismatches.append(f"{k.upper()}: wf={best[k]:.2f} deploy={dc[k]:.2f}")

    # VIX/TLT/SPY regime
    if pd.notna(best.get('vix_threshold')):
        wf_vix = best['vix_threshold']
        dc_vix = dc.get('vix_threshold', 999.0) if 'vix_threshold' not in dc else 999.0
        # Not a critical mismatch — regime filter params not stored in DEPLOYED_CONFIGS

    status = "MATCH" if not mismatches else "MISMATCH"
    mismatch_str = "; ".join(mismatches) if mismatches else "all params match"

    print(f"  {t:<6} ({DEPLOYED[t]:>3}) | {status:<8} | {mismatch_str}")
    print(f"         WF best: seed={int(best['seed'])}, thr={best['prob_thresh']:.3f}, "
          f"entry={int(best['entry_minute'])}, mode={best['tpsl_mode']}, "
          f"tp={best['tp_pct']:.3f}, sl={best['sl_pct']:.3f}, "
          f"tp_m={best.get('tp_mult', 'N/A')}, sl_m={best.get('sl_mult', 'N/A')}")
    print(f"         WF stats: val_b pnl={best['val_b_avg_pnl']:.4f}, wr={best['val_b_win_rate']:.2f}, "
          f"N={int(best['val_b_count'])}, composite={best['composite']:.6f}")
    print(f"         WF period: val_a={best['val_a_start']} to {best['val_a_end']}, "
          f"val_b={best['val_b_start']} to {best['val_b_end']}")
    print(f"         WF alternatives: {len(subset)} configs (composite range "
          f"[{subset['composite'].min():.6f}, {subset['composite'].max():.6f}])")
    if mismatches:
        for m in mismatches:
            print(f"         >>> {m}")
    print()


# =====================================================
# STEP 3: Live predictions from BQ
# =====================================================
print("=" * W)
print("  STEP 3: LIVE PREDICTION SIGNALS")
print("=" * W)

tickers_str = ", ".join(f"'{t}'" for t in DEPLOYED.keys())
q_live = f"""
SELECT score_date, ticker, experiment, seed, prob, threshold,
       signal, tp_pct, sl_pct, entry_minute, close,
       filter_status, tpsl_mode, atr_pct_entry,
       vix_value, tlt_gap_pct, spy_gap_pct,
       run_timestamp
FROM `ys-time-series.scoring_logs.live_predictions`
WHERE ticker IN ({tickers_str})
  AND score_date >= '2026-04-01'
ORDER BY score_date, ticker
"""
live = client.query(q_live).to_dataframe()
print(f"  Total live prediction rows since 2026-04-01: {len(live)}")
if len(live) > 0:
    print(f"  Date range: {live.score_date.min()} to {live.score_date.max()}")
    print(f"  Tickers in live: {sorted(live.ticker.unique())}")
print()

# Filter to GO signals
go_signals = live[(live.signal == 'GO') & (live.filter_status == 'OK')].copy()
all_go_signals = live[live.signal == 'GO'].copy()
blocked_signals = live[(live.signal == 'GO') & (live.filter_status != 'OK')]
no_signals = live[live.signal != 'GO']

print(f"  Total scored:  {len(live)}")
print(f"  GO signals:    {len(all_go_signals)} (of which {len(blocked_signals)} blocked by filter)")
print(f"  Traded (GO+OK): {len(go_signals)}")
print(f"  No signal:     {len(no_signals)}")
print()

# Per-ticker summary
print(f"  {'Ticker':<6} | {'Scored':>6} | {'Signal':>6} | {'GO+OK':>5} | {'Blocked':>7} | {'GO dates'}")
print(f"  {'-'*100}")
for t in sorted(DEPLOYED.keys()):
    t_all = live[live.ticker == t]
    t_signal = all_go_signals[all_go_signals.ticker == t]
    t_go = go_signals[go_signals.ticker == t]
    t_blocked = blocked_signals[blocked_signals.ticker == t]
    dates = sorted(t_go.score_date.astype(str).unique()) if len(t_go) > 0 else []
    dates_str = ', '.join(dates[:5])
    if len(dates) > 5:
        dates_str += f'... (+{len(dates)-5} more)'
    print(f"  {t:<6} | {len(t_all):>6} | {len(t_signal):>6} | {len(t_go):>5} | "
          f"{len(t_blocked):>7} | {dates_str}")
print()


# =====================================================
# STEP 4: Compute live outcomes using GCS OHLCV data
# =====================================================
print("=" * W)
print("  STEP 4: LIVE OUTCOME COMPUTATION (from GCS OHLCV)")
print("=" * W)
print()

BUCKET = "time-series-bucket-ys"
INFERENCE_DIR = "Data/Inference_v2"
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

import pytz
PST = pytz.timezone('America/Los_Angeles')

live_stats = {}

for t in sorted(DEPLOYED.keys()):
    dc = DEPLOYED_CONFIGS[t]
    t_go = go_signals[go_signals.ticker == t]

    live_stats[t] = {
        'n_scored': len(live[live.ticker == t]),
        'n_signal': len(all_go_signals[all_go_signals.ticker == t]),
        'n_go': len(t_go),
        'n_blocked': len(blocked_signals[blocked_signals.ticker == t]),
    }

    if len(t_go) == 0:
        live_stats[t].update({'wins': 0, 'losses': 0, 'live_n': 0,
                              'live_wr': None, 'live_avg_pnl': None, 'pnls': []})
        continue

    file_ticker = TICKER_FILE_MAP.get(t, t)
    uri = f"gs://{BUCKET}/{INFERENCE_DIR}/{file_ticker}_master_inference.parquet"
    try:
        ohlcv = pd.read_parquet(uri)
        ohlcv = ohlcv.reset_index()
        # Find datetime column
        dt_cols = [c for c in ohlcv.columns if pd.api.types.is_datetime64_any_dtype(ohlcv[c])]
        if dt_cols:
            ohlcv['Datetime'] = pd.to_datetime(ohlcv[dt_cols[0]], utc=True)
        else:
            num_col = next((c for c in ohlcv.columns
                           if c in ('Datetime', 'Date', 'Time', 'Timestamp', 'Level_0')),
                          ohlcv.columns[0])
            ohlcv['Datetime'] = pd.to_datetime(ohlcv[num_col], unit='ms', utc=True)
        ohlcv = ohlcv.set_index('Datetime').sort_index()

        # Standardize column names
        col_lower = {c: c.capitalize() for c in ohlcv.columns
                     if c.lower() in ('open', 'high', 'low', 'close', 'volume')}
        ohlcv = ohlcv.rename(columns=col_lower)

        # Ensure standard cols exist
        for std_c in ('Open', 'High', 'Low', 'Close', 'Volume'):
            if std_c not in ohlcv.columns:
                lc = std_c.lower()
                if lc in ohlcv.columns:
                    ohlcv[std_c] = ohlcv[lc]

        wins = 0
        losses = 0
        pnls = []
        trade_details = []

        for _, sig in t_go.iterrows():
            score_date = pd.Timestamp(sig['score_date'])
            entry_price = float(sig['close'])
            if entry_price <= 0:
                continue

            # Get TP/SL from the signal log
            tp_pct = float(sig['tp_pct']) if pd.notna(sig['tp_pct']) and float(sig['tp_pct']) > 0 else None
            sl_pct = float(sig['sl_pct']) if pd.notna(sig['sl_pct']) and float(sig['sl_pct']) > 0 else None

            # Fallback to deployed config if signal log has 0
            if tp_pct is None or tp_pct <= 0:
                tp_pct = dc.get('tp_pct', 0.06) if dc.get('tp_pct', 0) > 0 else 0.06
            if sl_pct is None or sl_pct <= 0:
                sl_pct = dc.get('sl_pct', 0.03) if dc.get('sl_pct', 0) > 0 else 0.03

            tp_price = entry_price * (1 + tp_pct)
            sl_price = entry_price * (1 - sl_pct)

            # Get all bars from entry to EOD
            # Entry is at entry_minute PST; market closes at 4pm ET = 1pm PT = 13:00 PST = minute 780
            entry_pst = PST.localize(pd.Timestamp(score_date.year, score_date.month, score_date.day,
                                                   int(sig['entry_minute']) // 60,
                                                   int(sig['entry_minute']) % 60))
            entry_utc = entry_pst.astimezone(pytz.utc)
            eod_pst = PST.localize(pd.Timestamp(score_date.year, score_date.month, score_date.day, 13, 0))
            eod_utc = eod_pst.astimezone(pytz.utc)

            # Bars after entry until EOD
            day_bars = ohlcv[(ohlcv.index > entry_utc) & (ohlcv.index <= eod_utc)]

            if len(day_bars) == 0:
                continue

            # Check TP/SL hits bar by bar
            hit = False
            exit_type = 'eod'
            exit_pnl = 0

            for _, bar in day_bars.iterrows():
                h = float(bar.get('High', entry_price))
                l = float(bar.get('Low', entry_price))

                if l <= sl_price:
                    # SL hit first (conservative: check SL before TP on same bar)
                    exit_pnl = -sl_pct
                    losses += 1
                    exit_type = 'SL'
                    hit = True
                    break
                elif h >= tp_price:
                    exit_pnl = tp_pct
                    wins += 1
                    exit_type = 'TP'
                    hit = True
                    break

            if not hit:
                # EOD exit
                eod_close = float(day_bars['Close'].iloc[-1])
                exit_pnl = (eod_close - entry_price) / entry_price
                if exit_pnl > 0:
                    wins += 1
                else:
                    losses += 1
                exit_type = 'EOD'

            pnls.append(exit_pnl)
            trade_details.append({
                'date': str(score_date.date()),
                'entry': entry_price,
                'tp_target': tp_price,
                'sl_target': sl_price,
                'exit_type': exit_type,
                'pnl': exit_pnl,
            })

        live_stats[t].update({
            'wins': wins,
            'losses': losses,
            'live_n': wins + losses,
            'live_wr': wins / (wins + losses) if (wins + losses) > 0 else None,
            'live_avg_pnl': float(np.mean(pnls)) if pnls else None,
            'pnls': pnls,
            'trade_details': trade_details,
        })

        print(f"  {t}: {wins + losses} trades ({wins}W/{losses}L), "
              f"WR={wins/(wins+losses)*100:.0f}%, "
              f"avg_pnl={np.mean(pnls)*100:+.2f}%" if pnls else f"  {t}: 0 trades")

    except Exception as e:
        print(f"  {t}: OHLCV load failed -- {e}")
        live_stats[t].update({'wins': 0, 'losses': 0, 'live_n': 0,
                              'live_wr': None, 'live_avg_pnl': None, 'pnls': []})

print()


# =====================================================
# STEP 5: WF test period overlap check
# =====================================================
print("=" * W)
print("  STEP 5: WF TEST PERIOD vs LIVE TRADING DATES")
print("=" * W)
print()

for t in sorted(DEPLOYED.keys()):
    if t not in best_wf:
        print(f"  {t}: no WF data")
        continue
    bw = best_wf[t]
    vb_start = str(bw['val_b_start'])
    vb_end = str(bw['val_b_end'])

    t_go = go_signals[go_signals.ticker == t]
    if len(t_go) > 0:
        live_dates = sorted(t_go.score_date.astype(str).unique())
        first_live = live_dates[0]
        last_live = live_dates[-1]
        overlap = first_live <= vb_end
        status = "OVERLAP (IN-SAMPLE DATA LEAK!)" if overlap else "OK (out-of-sample)"
        print(f"  {t}: WF test={vb_start} to {vb_end}, live={first_live} to {last_live} -- {status}")
    else:
        print(f"  {t}: WF test={vb_start} to {vb_end}, live=NO TRADES")

print()


# =====================================================
# STEP 6: FINAL COMPARISON TABLE
# =====================================================
print("=" * W)
print("  STEP 6: COMPARISON TABLE -- WF BACKTEST vs LIVE PERFORMANCE")
print("=" * W)
print()

# Parse WF stats from deployed configs
def parse_pct(s):
    return float(s.replace('%', '')) / 100

header = (f"{'Ticker':<6} | {'Exp':>4} | {'Conf':<4} | "
          f"{'WF PnL':>7} | {'WF WR':>5} | {'WF N':>4} | "
          f"{'Live PnL':>8} | {'Live WR':>7} | {'Live N':>6} | "
          f"{'d(PnL)':>7} | {'d(WR)':>6} | {'Notes'}")
print(header)
print("-" * W)

flags = []
n_worse = 0
n_better = 0
n_no_data = 0

for t in sorted(DEPLOYED.keys()):
    dc = DEPLOYED_CONFIGS[t]
    ls = live_stats.get(t, {})

    wf_pnl_pct = parse_pct(dc['wf_pnl'])
    wf_wr_pct = parse_pct(dc['wf_wr'])
    wf_n = dc['wf_n']

    live_pnl = ls.get('live_avg_pnl')
    live_wr = ls.get('live_wr')
    live_n = ls.get('live_n', 0) or 0

    notes = []

    if dc['confidence'] == 'LOW':
        notes.append(f"LOW(N={wf_n})")

    if live_n == 0:
        n_no_data += 1
        notes.append("NO_LIVE_TRADES")
        print(f"{t:<6} | {DEPLOYED[t]:>4} | {dc['confidence']:<4} | "
              f"{wf_pnl_pct:>+7.3f} | {wf_wr_pct:>4.0%} | {wf_n:>4} | "
              f"{'--':>8} | {'--':>7} | {'--':>6} | "
              f"{'--':>7} | {'--':>6} | {' '.join(notes)}")
        continue

    # Compute deltas
    delta_pnl = live_pnl - wf_pnl_pct if live_pnl is not None else None
    delta_wr = live_wr - wf_wr_pct if live_wr is not None else None

    if delta_pnl is not None and delta_pnl < -0.01:
        n_worse += 1
        notes.append("WORSE")
        flags.append(f"{t}: live {live_pnl:+.3f} vs WF {wf_pnl_pct:+.3f} (delta {delta_pnl:+.3f})")
    elif delta_pnl is not None and delta_pnl > 0.01:
        n_better += 1
        notes.append("BETTER")

    # Check if live PnL is negative
    if live_pnl is not None and live_pnl < 0:
        notes.append("NEGATIVE_PnL!")

    live_pnl_str = f"{live_pnl:>+8.3f}" if live_pnl is not None else f"{'--':>8}"
    live_wr_str = f"{live_wr:>6.0%}" if live_wr is not None else f"{'--':>6}"
    d_pnl_str = f"{delta_pnl:>+7.3f}" if delta_pnl is not None else f"{'--':>7}"
    d_wr_str = f"{delta_wr:>+5.0%}" if delta_wr is not None else f"{'--':>5}"

    print(f"{t:<6} | {DEPLOYED[t]:>4} | {dc['confidence']:<4} | "
          f"{wf_pnl_pct:>+7.3f} | {wf_wr_pct:>4.0%} | {wf_n:>4} | "
          f"{live_pnl_str} | {live_wr_str:>7} | {live_n:>6} | "
          f"{d_pnl_str} | {d_wr_str:>6} | {' '.join(notes)}")

print()


# =====================================================
# STEP 7: TRADE-BY-TRADE DETAIL FOR LOSING TICKERS
# =====================================================
print("=" * W)
print("  STEP 7: TRADE DETAILS (all GO+OK trades)")
print("=" * W)
print()

for t in sorted(DEPLOYED.keys()):
    ls = live_stats.get(t, {})
    details = ls.get('trade_details', [])
    if not details:
        continue
    print(f"  {t} ({DEPLOYED[t]}):")
    print(f"    {'Date':<12} {'Entry':>8} {'TP@':>8} {'SL@':>8} {'Exit':>5} {'PnL':>8}")
    for td in details:
        pnl_color = '+' if td['pnl'] > 0 else ''
        print(f"    {td['date']:<12} ${td['entry']:>7.2f} ${td['tp_target']:>7.2f} "
              f"${td['sl_target']:>7.2f} {td['exit_type']:>5} {td['pnl']*100:>+7.2f}%")
    avg = np.mean([td['pnl'] for td in details])
    wr = sum(1 for td in details if td['pnl'] > 0) / len(details) * 100
    print(f"    --- avg={avg*100:+.2f}%  WR={wr:.0f}%  N={len(details)}")
    print()


# =====================================================
# STEP 8: SUMMARY & RED FLAGS
# =====================================================
print("=" * W)
print("  STEP 8: SUMMARY & RED FLAGS")
print("=" * W)

total_live_trades = sum((ls.get('live_n', 0) or 0) for ls in live_stats.values())
total_wins = sum((ls.get('wins', 0) or 0) for ls in live_stats.values())
total_losses = sum((ls.get('losses', 0) or 0) for ls in live_stats.values())

all_pnls = []
for ls in live_stats.values():
    if ls.get('pnls'):
        all_pnls.extend(ls['pnls'])

print(f"\n  PORTFOLIO LIVE PERFORMANCE:")
print(f"    Total trades: {total_live_trades}")
print(f"    Wins: {total_wins}")
print(f"    Losses: {total_losses}")
if total_live_trades > 0:
    print(f"    Win rate: {total_wins/total_live_trades*100:.1f}%")
if all_pnls:
    print(f"    Avg PnL per trade: {np.mean(all_pnls)*100:+.2f}%")
    print(f"    Median PnL: {np.median(all_pnls)*100:+.2f}%")
    print(f"    Total cumulative PnL: {sum(all_pnls)*100:+.2f}% (sum of all trade PnLs)")
    print(f"    Best trade: {max(all_pnls)*100:+.2f}%")
    print(f"    Worst trade: {min(all_pnls)*100:+.2f}%")

print(f"\n  DELTA ANALYSIS:")
print(f"    Tickers performing WORSE than WF: {n_worse}")
print(f"    Tickers performing BETTER than WF: {n_better}")
print(f"    Tickers with NO live trades: {n_no_data}")

if flags:
    print(f"\n  --- UNDERPERFORMERS (live PnL > 1% worse than WF) ---")
    for f in flags:
        print(f"    {f}")

# Confidence breakdown
low_conf = [t for t, dc in DEPLOYED_CONFIGS.items() if dc['confidence'] == 'LOW']
med_conf = [t for t, dc in DEPLOYED_CONFIGS.items() if dc['confidence'] == 'MED']
high_conf = [t for t, dc in DEPLOYED_CONFIGS.items() if dc['confidence'] == 'HIGH']
print(f"\n  CONFIDENCE BREAKDOWN:")
print(f"    HIGH (N>=10): {len(high_conf)} tickers -- {', '.join(sorted(high_conf))}")
print(f"    MED  (N=5-9): {len(med_conf)} tickers -- {', '.join(sorted(med_conf))}")
print(f"    LOW  (N<=4):  {len(low_conf)} tickers -- {', '.join(sorted(low_conf))}")
print(f"    LOW confidence = {len(low_conf)/len(DEPLOYED_CONFIGS)*100:.0f}% of portfolio")

# Check if low-confidence tickers are the worst performers
if all_pnls:
    low_pnls = []
    high_pnls = []
    for t, ls in live_stats.items():
        if ls.get('pnls') and len(ls['pnls']) > 0:
            avg = np.mean(ls['pnls'])
            if DEPLOYED_CONFIGS[t]['confidence'] == 'LOW':
                low_pnls.append(avg)
            else:
                high_pnls.append(avg)
    if low_pnls:
        print(f"    LOW conf avg PnL: {np.mean(low_pnls)*100:+.2f}%")
    if high_pnls:
        print(f"    MED/HIGH conf avg PnL: {np.mean(high_pnls)*100:+.2f}%")

print(f"\n  KEY RED FLAGS:")
print(f"    1. {len(low_conf)}/20 deployed tickers have LOW confidence (N<=4 WF trades)")
print(f"       100% WR on 2-3 trades = noise, not signal")
print(f"       Affected: {', '.join(sorted(low_conf))}")
print(f"    2. Deploy config was reshuffled between v8 and v9")
print(f"       Memory says v8 deployed ASTS/BE/CLSK/HUT/IREN (152), CIFR/MVIS/PLUG/RCAT (160)")
print(f"       Code now has v9 with completely different ticker assignments")
print(f"    3. WF test period val_b_start=2026-04-01 may overlap with live trading dates")
print(f"    4. KEEL: SL(14%) nearly equals TP(16%) -- regime scaling at VIX>25 makes SL>TP")
print(f"    5. BE: lowest WF PnL at 2.0% -- barely above 1.8% deploy threshold")
print(f"    6. MVIS was removed 2026-06-03 after -30% crash -- same could happen to others")
print()
print("=" * W)
print("  Script complete. Run this in Colab to get live outcomes.")
print("=" * W)
