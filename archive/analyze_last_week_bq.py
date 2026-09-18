"""
analyze_last_week_bq.py — Last week S3 live predictions vs price actuals
========================================================================
Pulls BQ live_predictions for Jun 22-29 and cross-references against
intraday price data. Day-by-day breakdown with fill sim and trade outcome.

Colab-ready. No GPU needed.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from google.cloud import bigquery, storage as gcs_storage

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

# ── CONFIG ──
START_DATE = '2026-06-22'
END_DATE   = '2026-06-29'
BQ_TABLE   = f"{GCP_PROJECT}.scoring_logs.live_predictions"
BUCKET     = os.environ.get("GCS_BUCKET", "")
INF_DIR    = 'Data/Inference_v2'
SLIPPAGE   = 0.0015
EXIT_TIME  = 780  # 1pm PT
CAPITAL    = 200_000
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
W = 110

bq_client  = bigquery.Client()
gcs_client = gcs_storage.Client()

# ══════════════════════════════════════════════════════════════════════════════
# 1. PULL BQ LIVE PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*W}")
print(f"  S3 LIVE PREDICTIONS vs ACTUALS — {START_DATE} → {END_DATE}")
print(f"{'='*W}")

query = f"""
SELECT score_date, ticker, experiment,
       AVG(prob) as avg_prob, ANY_VALUE(threshold) as threshold,
       ANY_VALUE(signal) as signal,
       ANY_VALUE(tp_pct) as tp_pct, ANY_VALUE(sl_pct) as sl_pct,
       ANY_VALUE(entry_minute) as entry_minute,
       ANY_VALUE(close) as signal_close,
       ANY_VALUE(filter_status) as filter_status,
       ANY_VALUE(vix_value) as vix_value,
       ANY_VALUE(tlt_gap_pct) as tlt_gap_pct,
       ANY_VALUE(spy_gap_pct) as spy_gap_pct,
       ANY_VALUE(tpsl_mode) as tpsl_mode,
       ANY_VALUE(dip_pct) as dip_pct,
       ANY_VALUE(max_wait_minutes) as max_wait_minutes,
       COUNT(*) as n_seeds
FROM `{BQ_TABLE}`
WHERE score_date BETWEEN '{START_DATE}' AND '{END_DATE}'
GROUP BY score_date, ticker, experiment
ORDER BY score_date, ticker
"""
signals = bq_client.query(query).to_dataframe()
print(f"  {len(signals)} signal-level rows from BQ")

# ══════════════════════════════════════════════════════════════════════════════
# 2. LOAD PRICE DATA + SIMULATE FILLS
# ══════════════════════════════════════════════════════════════════════════════
_price_cache = {}

def load_prices(ticker):
    if ticker in _price_cache:
        return _price_cache[ticker]
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    uri = f"gs://{BUCKET}/{INF_DIR}/{clean}_master_inference.parquet"
    try:
        df = pd.read_parquet(uri).reset_index()
        dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        if dt_cols:
            df['Datetime'] = pd.to_datetime(df[dt_cols[0]], utc=True)
        else:
            nc = next((c for c in df.columns if c in ('Datetime','Date','Time','Timestamp')), df.columns[0])
            df['Datetime'] = pd.to_datetime(df[nc], unit='ms', utc=True)
        df = df.set_index('Datetime').sort_index()
        df = df[~df.index.duplicated(keep='first')]
        df.index = df.index.tz_convert('America/Los_Angeles')
        col_map = {c.lower(): c for c in df.columns}
        for old, new in [('open','Open'),('high','High'),('low','Low'),('close','Close'),('volume','Volume')]:
            if old in col_map and col_map[old] != new:
                df = df.rename(columns={col_map[old]: new})
        _price_cache[ticker] = df
        return df
    except Exception as e:
        print(f"    load_prices({ticker}): FAILED — {e}")
        return None


def simulate_limit_entry(pdf, d_int, em, dip_pct, max_wait, tp, sl):
    d_arr = (pdf.index.year*10000 + pdf.index.month*100 + pdf.index.day).values.astype(np.int64)
    t_arr = (pdf.index.hour*60 + pdf.index.minute).values.astype(np.int64)

    sig_bars = np.where((d_arr == d_int) & (t_arr == em))[0]
    if len(sig_bars) == 0:
        return {'status': 'NO_BAR'}

    sig_idx = sig_bars[0]
    signal_price = float(pdf['Close'].values[sig_idx])
    limit_price = signal_price * (1 - dip_pct)

    fill_idx = None
    for k in range(sig_idx + 1, len(pdf)):
        if d_arr[k] != d_int:
            break
        if t_arr[k] - em > max_wait:
            break
        if pdf['Low'].values[k] <= limit_price:
            fill_idx = k
            break

    if fill_idx is None:
        return {'status': 'UNFILLED', 'signal_price': signal_price, 'limit_price': limit_price}

    entry_price = limit_price
    tp_price = entry_price * (1 + tp)
    sl_price = entry_price * (1 - sl)

    exit_type = 'EOD'
    pnl = 0.0
    mfe = 0.0
    mae = 0.0
    exit_min = 0

    for k in range(fill_idx + 1, len(pdf)):
        if d_arr[k] != d_int:
            pnl = (float(pdf['Close'].values[k-1]) - entry_price) / entry_price
            exit_min = int(t_arr[k-1])
            break
        h = float(pdf['High'].values[k])
        lo = float(pdf['Low'].values[k])
        mfe = max(mfe, (h - entry_price) / entry_price)
        mae = max(mae, (entry_price - lo) / entry_price)

        if lo <= sl_price and h >= tp_price:
            pnl = -sl if np.random.random() < 0.5 else tp
            exit_type = 'SL+TP'
            exit_min = int(t_arr[k])
            break
        if lo <= sl_price:
            pnl = -sl
            exit_type = 'SL'
            exit_min = int(t_arr[k])
            break
        if h >= tp_price:
            pnl = tp
            exit_type = 'TP'
            exit_min = int(t_arr[k])
            break
        if t_arr[k] >= EXIT_TIME:
            pnl = (float(pdf['Close'].values[k]) - entry_price) / entry_price
            exit_type = 'EOD'
            exit_min = int(t_arr[k])
            break

    pnl -= 2 * SLIPPAGE

    eod_bars = np.where((d_arr == d_int) & (t_arr == EXIT_TIME))[0]
    eod_price = float(pdf['Close'].values[eod_bars[0]]) if len(eod_bars) > 0 else entry_price
    eod_ret = (eod_price - entry_price) / entry_price

    return {
        'status': 'FILLED', 'signal_price': signal_price, 'entry_price': entry_price,
        'limit_price': limit_price, 'pnl': pnl, 'exit_type': exit_type,
        'mfe': mfe, 'mae': mae, 'eod_ret': eod_ret, 'exit_min': exit_min,
    }


results = []
for _, sig in signals.iterrows():
    ticker = sig['ticker']
    d = sig['score_date']
    d_int = int(d.strftime('%Y%m%d'))
    em = int(sig['entry_minute'])
    tp = float(sig['tp_pct'])
    sl = float(sig['sl_pct'])
    dip_pct = float(sig['dip_pct']) if pd.notna(sig['dip_pct']) else 0.01
    max_wait = int(sig['max_wait_minutes']) if pd.notna(sig['max_wait_minutes']) else 30
    is_go = sig['signal'] == 'GO'
    is_ok = sig['filter_status'] == 'OK'

    row = {
        'date': d, 'ticker': ticker, 'experiment': sig['experiment'],
        'prob': sig['avg_prob'], 'threshold': sig['threshold'],
        'signal': sig['signal'], 'filter': sig['filter_status'],
        'tp': tp, 'sl': sl, 'dip_pct': dip_pct, 'max_wait': max_wait,
        'signal_close': sig['signal_close'],
        'vix': sig['vix_value'], 'tlt_gap': sig['tlt_gap_pct'], 'spy_gap': sig['spy_gap_pct'],
        'n_seeds': sig['n_seeds'],
    }

    pdf = load_prices(ticker)
    if pdf is None:
        row.update({'outcome': 'NO_DATA', 'pnl': 0})
        results.append(row)
        continue

    sim = simulate_limit_entry(pdf, d_int, em, dip_pct, max_wait, tp, sl)

    if sim['status'] == 'NO_BAR':
        row.update({'outcome': 'NO_BAR', 'pnl': 0})
    elif sim['status'] == 'UNFILLED':
        row.update({'outcome': 'UNFILLED', 'pnl': 0,
                    'signal_price': sim['signal_price'], 'limit_price': sim['limit_price']})
    else:
        if is_go and is_ok:
            tag = 'TRADED'
        elif is_go and not is_ok:
            tag = 'BLOCKED_GO'
        else:
            tag = 'BELOW_THRESH'
        row.update({
            'outcome': tag, 'pnl': sim['pnl'],
            'signal_price': sim['signal_price'], 'entry_price': sim['entry_price'],
            'exit_type': sim['exit_type'], 'mfe': sim['mfe'], 'mae': sim['mae'],
            'eod_ret': sim['eod_ret'],
        })
    results.append(row)

rdf = pd.DataFrame(results)

# ══════════════════════════════════════════════════════════════════════════════
# 3. SIGNAL OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n  SIGNAL OVERVIEW:")
print(f"  {'Category':<22} {'Count':>6}  Description")
print(f"  {'─'*65}")
cats = {
    'TRADED':       rdf[rdf['outcome'] == 'TRADED'],
    'BLOCKED_GO':   rdf[rdf['outcome'] == 'BLOCKED_GO'],
    'BELOW_THRESH': rdf[rdf['outcome'] == 'BELOW_THRESH'],
    'UNFILLED':     rdf[rdf['outcome'] == 'UNFILLED'],
}
descs = {
    'TRADED':       'GO + OK — actually traded',
    'BLOCKED_GO':   'GO + BLOCKED — regime filter killed',
    'BELOW_THRESH': 'Prob < threshold — no signal',
    'UNFILLED':     'Limit price never reached',
}
for cat, sub in cats.items():
    print(f"  {cat:<22} {len(sub):>6}  {descs[cat]}")
print(f"  {'TOTAL':<22} {len(rdf):>6}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. DAY-BY-DAY BREAKDOWN — ALL SIGNALS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*W}")
print(f"  DAY-BY-DAY: every signal, what happened")
print(f"{'='*W}")

for d in sorted(rdf['date'].unique()):
    day = rdf[rdf['date'] == d].sort_values('ticker')
    traded = day[day['outcome'] == 'TRADED']
    blocked = day[day['outcome'] == 'BLOCKED_GO']
    below = day[day['outcome'] == 'BELOW_THRESH']
    unfilled_d = day[day['outcome'] == 'UNFILLED']

    n_traded = len(traded)
    traded_filled = traded[traded['pnl'] != 0] if 'pnl' in traded.columns else traded
    day_pnl = traded_filled['pnl'].sum() if not traded_filled.empty else 0
    day_wins = len(traded_filled[traded_filled['pnl'] > 0]) if not traded_filled.empty else 0
    alloc = CAPITAL / max(n_traded, 1) if n_traded > 0 else 0
    dollar = alloc * day_pnl if n_traded > 0 else 0

    print(f"\n  ┌─ {d} ─────────────────────────────────────────────────────────────────────────────┐")
    print(f"  │  Traded: {n_traded}  Blocked: {len(blocked)}  Below thresh: {len(below)}  "
          f"Unfilled: {len(unfilled_d)}  │  Day PnL: {day_pnl:+.2%}  ${dollar:+,.0f}")
    print(f"  └{'─'*(W-3)}┘")

    # Traded signals
    if not traded.empty:
        print(f"\n    TRADED (GO + OK):")
        print(f"    {'Ticker':<7} {'Prob':>6} {'Entry$':>9} {'TP':>5} {'SL':>5} "
              f"{'Exit':>4} {'PnL':>8} {'MFE':>7} {'MAE':>7} {'EOD_ret':>8} {'W/L':>4}")
        print(f"    {'─'*82}")
        for _, t in traded.iterrows():
            if t['outcome'] == 'TRADED' and 'entry_price' in t and pd.notna(t.get('entry_price')):
                wl = 'W' if t['pnl'] > 0 else 'L'
                print(f"    {t['ticker']:<7} {t['prob']:>5.3f} ${t['entry_price']:>8.2f} "
                      f"{t['tp']:>4.0%} {t['sl']:>4.0%} {t.get('exit_type','?'):>4} "
                      f"{t['pnl']:>+7.2%} {t.get('mfe',0):>+6.2%} {t.get('mae',0):>+6.2%} "
                      f"{t.get('eod_ret',0):>+7.2%} {wl:>4}")
            elif t['outcome'] in ('UNFILLED',):
                print(f"    {t['ticker']:<7} {t['prob']:>5.3f}  UNFILLED (limit ${t.get('limit_price',0):.2f} not reached)")

    # Blocked GO signals — show hypothetical
    if not blocked.empty:
        print(f"\n    BLOCKED by regime filter (would have traded):")
        print(f"    {'Ticker':<7} {'Prob':>6} {'VIX':>5} {'TLT%':>7} {'SPY%':>7} "
              f"{'Hypo_PnL':>9} {'Hypo_Exit':>10} {'Verdict':<14}")
        print(f"    {'─'*78}")
        for _, b in blocked.iterrows():
            hypo_pnl = b.get('pnl', 0)
            hypo_exit = b.get('exit_type', 'UNFILL')
            if b['outcome'] == 'BLOCKED_GO' and 'entry_price' in b and pd.notna(b.get('entry_price')):
                verdict = 'FILTER SAVED' if hypo_pnl < 0 else 'FILTER COST'
                print(f"    {b['ticker']:<7} {b['prob']:>5.3f} {b['vix']:>5.1f} "
                      f"{b['tlt_gap']:>+6.2f}% {b['spy_gap']:>+6.2f}% "
                      f"{hypo_pnl:>+8.2%} {hypo_exit:>10} {verdict:<14}")
            else:
                print(f"    {b['ticker']:<7} {b['prob']:>5.3f} {b['vix']:>5.1f} "
                      f"{b['tlt_gap']:>+6.2f}% {b['spy_gap']:>+6.2f}% "
                      f"{'UNFILLED':>9} {'':>10} {'N/A':<14}")

    # Below threshold
    if not below.empty:
        tickers_below = ', '.join(f"{r['ticker']}({r['prob']:.2f}<{r['threshold']:.2f})"
                                   for _, r in below.iterrows())
        print(f"\n    Below threshold: {tickers_below}")

    # Unfilled
    if not unfilled_d.empty:
        tickers_uf = ', '.join(f"{r['ticker']}(limit ${r.get('limit_price',0):.2f})"
                               for _, r in unfilled_d.iterrows())
        print(f"    Unfilled: {tickers_uf}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. SUMMARY TABLES
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*W}")
print(f"  SUMMARY")
print(f"{'='*W}")

# Actual traded
traded_all = rdf[rdf['outcome'] == 'TRADED']
traded_filled = traded_all[traded_all.get('entry_price', pd.Series()).notna()] if 'entry_price' in traded_all.columns else traded_all
if not traded_filled.empty:
    wins = traded_filled[traded_filled['pnl'] > 0]
    losses = traded_filled[traded_filled['pnl'] <= 0]
    n = len(traded_filled)
    alloc = CAPITAL / n if n > 0 else 0
    dollar = sum(alloc * p for p in traded_filled['pnl'])
    print(f"\n  ACTUAL TRADES (GO + OK + filled):")
    print(f"    Trades: {n}, Wins: {len(wins)}, Losses: {len(losses)}, WR: {len(wins)/n*100:.0f}%")
    print(f"    Avg PnL: {traded_filled['pnl'].mean():+.2%}, Total: {traded_filled['pnl'].sum():+.2%}")
    print(f"    $PnL on ${CAPITAL/1000:.0f}K: ${dollar:+,.0f}")

# Filter impact
blocked_all = rdf[rdf['outcome'] == 'BLOCKED_GO']
blocked_filled = blocked_all[blocked_all.get('entry_price', pd.Series()).notna()] if 'entry_price' in blocked_all.columns else pd.DataFrame()
if not blocked_filled.empty:
    bw = blocked_filled[blocked_filled['pnl'] > 0]
    bl = blocked_filled[blocked_filled['pnl'] <= 0]
    print(f"\n  REGIME FILTER IMPACT (blocked GO signals that would have filled):")
    print(f"    Would-be trades: {len(blocked_filled)}, Wins: {len(bw)}, Losses: {len(bl)}")
    print(f"    Avg PnL: {blocked_filled['pnl'].mean():+.2%}, Total: {blocked_filled['pnl'].sum():+.2%}")
    saved = bl['pnl'].sum() if not bl.empty else 0
    missed = bw['pnl'].sum() if not bw.empty else 0
    print(f"    Losses avoided: {saved:+.2%} ({len(bl)} trades)")
    print(f"    Profits missed: {missed:+.2%} ({len(bw)} trades)")
    net = saved + missed
    print(f"    Net: filter {'HELPED' if abs(saved) > abs(missed) else 'HURT'} by {abs(net):.2%}")

# Hypothetical no-filter
if not traded_filled.empty or not blocked_filled.empty:
    all_hypo = pd.concat([traded_filled[['pnl']], blocked_filled[['pnl']]], ignore_index=True)
    n_h = len(all_hypo)
    alloc_h = CAPITAL / n_h if n_h > 0 else 0
    dollar_h = sum(alloc_h * p for p in all_hypo['pnl'])
    print(f"\n  COMPARISON:")
    print(f"  {'Scenario':<28} {'Trades':>7} {'WR':>6} {'AvgPnL':>9} {'TotalPnL':>10} {'$PnL':>12}")
    print(f"  {'─'*75}")
    if not traded_filled.empty:
        tw = len(traded_filled[traded_filled['pnl'] > 0])
        ta = CAPITAL / len(traded_filled) if len(traded_filled) > 0 else 0
        td = sum(ta * p for p in traded_filled['pnl'])
        print(f"  {'WITH filter (actual)':<28} {len(traded_filled):>7} {tw/len(traded_filled):>5.0%} "
              f"{traded_filled['pnl'].mean():>+8.2%} {traded_filled['pnl'].sum():>+9.2%} ${td:>+11,.0f}")
    hw = len(all_hypo[all_hypo['pnl'] > 0])
    print(f"  {'WITHOUT filter (hypothetical)':<28} {n_h:>7} {hw/n_h:>5.0%} "
          f"{all_hypo['pnl'].mean():>+8.2%} {all_hypo['pnl'].sum():>+9.2%} ${dollar_h:>+11,.0f}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. CHARTS — Avg PnL and Total PnL by day
# ══════════════════════════════════════════════════════════════════════════════
traded_filled_chart = rdf[(rdf['outcome'] == 'TRADED') & (rdf.get('entry_price', pd.Series()).notna())]
blocked_filled_chart = rdf[(rdf['outcome'] == 'BLOCKED_GO') & (rdf.get('entry_price', pd.Series()).notna())]

if not traded_filled_chart.empty or not blocked_filled_chart.empty:
    dates = sorted(rdf['date'].unique())

    actual_avg, actual_tot = [], []
    nofilter_avg, nofilter_tot = [], []

    for d in dates:
        act = traded_filled_chart[traded_filled_chart['date'] == d]
        blk = blocked_filled_chart[blocked_filled_chart['date'] == d]
        combined = pd.concat([act[['pnl']], blk[['pnl']]], ignore_index=True)

        actual_avg.append(act['pnl'].mean() * 100 if not act.empty else 0)
        actual_tot.append(act['pnl'].sum() * 100 if not act.empty else 0)
        nofilter_avg.append(combined['pnl'].mean() * 100 if not combined.empty else 0)
        nofilter_tot.append(combined['pnl'].sum() * 100 if not combined.empty else 0)

    date_labels = [d.strftime('%b %d') if hasattr(d, 'strftime') else str(d) for d in dates]
    x = np.arange(len(dates))
    bar_w = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f'S3 Live Predictions — {START_DATE} → {END_DATE}', fontsize=14, fontweight='bold')

    ax1.bar(x - bar_w/2, actual_avg, bar_w, label='Actual (with filter)', color='#2196F3', alpha=0.85)
    ax1.bar(x + bar_w/2, nofilter_avg, bar_w, label='Hypothetical (no filter)', color='#4CAF50', alpha=0.85)
    ax1.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax1.set_title('Average PnL per Trade (%)', fontsize=12)
    ax1.set_ylabel('Avg PnL (%)')
    ax1.set_xticks(x); ax1.set_xticklabels(date_labels)
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(actual_avg):
        if v != 0:
            ax1.text(i - bar_w/2, v + (0.2 if v >= 0 else -0.5), f'{v:+.1f}%', ha='center', fontsize=8)
    for i, v in enumerate(nofilter_avg):
        if v != 0:
            ax1.text(i + bar_w/2, v + (0.2 if v >= 0 else -0.5), f'{v:+.1f}%', ha='center', fontsize=8)

    ax2.bar(x - bar_w/2, actual_tot, bar_w, label='Actual (with filter)', color='#2196F3', alpha=0.85)
    ax2.bar(x + bar_w/2, nofilter_tot, bar_w, label='Hypothetical (no filter)', color='#4CAF50', alpha=0.85)
    ax2.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax2.set_title('Total PnL (%)', fontsize=12)
    ax2.set_ylabel('Total PnL (%)')
    ax2.set_xticks(x); ax2.set_xticklabels(date_labels)
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(actual_tot):
        if v != 0:
            ax2.text(i - bar_w/2, v + (0.3 if v >= 0 else -0.7), f'{v:+.1f}%', ha='center', fontsize=8)
    for i, v in enumerate(nofilter_tot):
        if v != 0:
            ax2.text(i + bar_w/2, v + (0.3 if v >= 0 else -0.7), f'{v:+.1f}%', ha='center', fontsize=8)

    plt.tight_layout()
    chart_path = '/tmp/last_week_pnl.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"\n  Charts saved: {chart_path}")
    try:
        from IPython.display import display, Image
        display(Image(filename=chart_path))
    except ImportError:
        plt.show()

print(f"\n{'='*W}")
print(f"  DONE")
print(f"{'='*W}")
