"""
analyze_june_nofilter.py — June 2026: what if we didn't have vol/dollar-vol filters?
=====================================================================================
Pulls BQ live_predictions for June, identifies which signals were blocked by
the volume filter (MIN_VOL_FILTER_PCT=25) and dollar-volume filter (MIN_DOLLAR_VOLUME=50K),
simulates fills for those blocked signals, and shows the counterfactual June.

Colab-ready. No GPU needed.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.cloud import bigquery, storage as gcs_storage

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

# ── CONFIG ──
START_DATE = '2026-06-01'
END_DATE   = '2026-06-29'
BQ_TABLE   = f"{GCP_PROJECT}.scoring_logs.live_predictions"
BUCKET     = os.environ.get("GCS_BUCKET", "")
INF_DIR    = 'Data/Inference_v2'
SLIPPAGE   = 0.0015
EXIT_TIME  = 780  # 1pm PT
CAPITAL    = 200_000
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
DOLLAR_VOL_CUTOFF = 50_000
W = 110

bq_client  = bigquery.Client()
gcs_client = gcs_storage.Client()

# ══════════════════════════════════════════════════════════════════════════════
# 1. PULL BQ LIVE PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*W}")
print(f"  JUNE FILTER ANALYSIS — {START_DATE} → {END_DATE}")
print(f"  What if we removed vol-filter floor (25%) and dollar-volume gate ($50K)?")
print(f"{'='*W}")

query = f"""
SELECT score_date, ticker, experiment,
       AVG(prob) as avg_prob, ANY_VALUE(threshold) as threshold,
       ANY_VALUE(signal) as signal,
       ANY_VALUE(tp_pct) as tp_pct, ANY_VALUE(sl_pct) as sl_pct,
       ANY_VALUE(entry_minute) as entry_minute,
       ANY_VALUE(close) as signal_close,
       ANY_VALUE(filter_status) as filter_status,
       ANY_VALUE(vol_filter_ok) as vol_filter_ok,
       ANY_VALUE(dollar_volume) as dollar_volume,
       ANY_VALUE(entry_volume) as entry_volume,
       ANY_VALUE(vol_median) as vol_median,
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
if signals.empty:
    print("  No data — check BQ table or date range")
    raise SystemExit

date_range = f"{signals['score_date'].min()} → {signals['score_date'].max()}"
print(f"  Actual date coverage: {date_range}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. CATEGORIZE WHY EACH SIGNAL WAS BLOCKED
# ══════════════════════════════════════════════════════════════════════════════

def block_reason(row):
    if row['filter_status'] == 'OK':
        return 'OK'
    reasons = []
    if str(row.get('vol_filter_ok', 'OK')).upper() in ('BLOCKED', 'FALSE', '0'):
        reasons.append('VOL')
    dv = row.get('dollar_volume')
    if pd.notna(dv) and float(dv) < DOLLAR_VOL_CUTOFF:
        reasons.append('DVOL')
    if not reasons:
        reasons.append('REGIME')
    return '+'.join(reasons)

signals['block_reason'] = signals.apply(block_reason, axis=1)

# Would this signal pass WITHOUT vol+dollar filters?
def would_pass_nofilter(row):
    if row['filter_status'] == 'OK':
        return True
    br = row['block_reason']
    vol_dollar_only = all(r in ('VOL', 'DVOL') for r in br.split('+'))
    return vol_dollar_only

signals['pass_nofilter'] = signals.apply(would_pass_nofilter, axis=1)

# ══════════════════════════════════════════════════════════════════════════════
# 3. BLOCK REASON SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
go_signals = signals[signals['signal'] == 'GO'].copy()
no_signals = signals[signals['signal'] != 'GO'].copy()

print(f"\n  SIGNAL BREAKDOWN:")
print(f"    Total signals:     {len(signals)}")
print(f"    GO (above thresh): {len(go_signals)}")
print(f"    NO (below thresh): {len(no_signals)}")

print(f"\n  GO SIGNALS — Filter status:")
print(f"    {'Status':<25} {'Count':>6}  {'%':>5}")
print(f"    {'─'*42}")
for reason in sorted(go_signals['block_reason'].unique()):
    n = len(go_signals[go_signals['block_reason'] == reason])
    pct = n / len(go_signals) * 100
    label_map = {
        'OK': 'Passed all filters',
        'VOL': 'Blocked: volume filter',
        'DVOL': 'Blocked: dollar-volume',
        'VOL+DVOL': 'Blocked: vol + dollar-vol',
        'REGIME': 'Blocked: regime (VIX/TLT/SPY)',
    }
    label = label_map.get(reason, f'Blocked: {reason}')
    print(f"    {label:<25} {n:>6}  {pct:>4.0f}%")

recoverable = go_signals[go_signals['pass_nofilter'] & (go_signals['filter_status'] != 'OK')]
print(f"\n    Recoverable by removing vol+dollar filters: {len(recoverable)} signals")

# ══════════════════════════════════════════════════════════════════════════════
# 4. SIMULATE FILLS FOR ALL GO SIGNALS
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

    for k in range(fill_idx + 1, len(pdf)):
        if d_arr[k] != d_int:
            pnl = (float(pdf['Close'].values[k-1]) - entry_price) / entry_price
            break
        h = float(pdf['High'].values[k])
        lo = float(pdf['Low'].values[k])
        mfe = max(mfe, (h - entry_price) / entry_price)
        mae = max(mae, (entry_price - lo) / entry_price)

        if lo <= sl_price and h >= tp_price:
            pnl = -sl
            exit_type = 'SL'
            break
        if lo <= sl_price:
            pnl = -sl
            exit_type = 'SL'
            break
        if h >= tp_price:
            pnl = tp
            exit_type = 'TP'
            break
        if t_arr[k] >= EXIT_TIME:
            pnl = (float(pdf['Close'].values[k]) - entry_price) / entry_price
            exit_type = 'EOD'
            break

    pnl -= 2 * SLIPPAGE

    return {
        'status': 'FILLED', 'signal_price': signal_price, 'entry_price': entry_price,
        'limit_price': limit_price, 'pnl': pnl, 'exit_type': exit_type,
        'mfe': mfe, 'mae': mae,
    }


print(f"\n  Simulating fills for all GO signals...")
results = []
for _, sig in go_signals.iterrows():
    ticker = sig['ticker']
    d = sig['score_date']
    d_int = int(d.strftime('%Y%m%d'))
    em = int(sig['entry_minute'])
    tp = float(sig['tp_pct'])
    sl = float(sig['sl_pct'])
    dip_pct = float(sig['dip_pct']) if pd.notna(sig.get('dip_pct')) else 0.01
    max_wait = int(sig['max_wait_minutes']) if pd.notna(sig.get('max_wait_minutes')) else 30

    row = {
        'date': d, 'ticker': ticker, 'experiment': sig['experiment'],
        'prob': sig['avg_prob'], 'threshold': sig['threshold'],
        'filter': sig['filter_status'], 'block_reason': sig['block_reason'],
        'pass_nofilter': sig['pass_nofilter'],
        'tp': tp, 'sl': sl, 'dip_pct': dip_pct, 'max_wait': max_wait,
        'vix': sig['vix_value'], 'tlt_gap': sig['tlt_gap_pct'], 'spy_gap': sig['spy_gap_pct'],
        'dollar_volume': sig.get('dollar_volume'), 'vol_filter_ok': sig.get('vol_filter_ok'),
    }

    pdf = load_prices(ticker)
    if pdf is None:
        row.update({'filled': False, 'pnl': 0})
        results.append(row)
        continue

    sim = simulate_limit_entry(pdf, d_int, em, dip_pct, max_wait, tp, sl)

    if sim['status'] == 'FILLED':
        row.update({
            'filled': True, 'pnl': sim['pnl'],
            'entry_price': sim['entry_price'], 'exit_type': sim['exit_type'],
            'mfe': sim['mfe'], 'mae': sim['mae'],
        })
    else:
        row.update({'filled': False, 'pnl': 0, 'fill_status': sim['status']})
    results.append(row)

rdf = pd.DataFrame(results)
filled = rdf[rdf['filled'] == True].copy()

print(f"  {len(rdf)} GO signals total, {len(filled)} would fill")

# ══════════════════════════════════════════════════════════════════════════════
# 5. ACTUAL vs NO-FILTER COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
actual = filled[(filled['filter'] == 'OK')].copy()
recovered = filled[(filled['filter'] != 'OK') & (filled['pass_nofilter'] == True)].copy()
still_blocked = filled[(filled['filter'] != 'OK') & (filled['pass_nofilter'] == False)].copy()
nofilter = pd.concat([actual, recovered], ignore_index=True)

def _add_daily_dollar(df):
    """Add dollar_pnl: each day allocates full CAPITAL equally across that day's trades."""
    if df.empty:
        return df
    df = df.copy()
    df['dollar_pnl'] = 0.0
    for d in df['date'].unique():
        mask = df['date'] == d
        n = mask.sum()
        df.loc[mask, 'dollar_pnl'] = df.loc[mask, 'pnl'] * (CAPITAL / n)
    return df

actual = _add_daily_dollar(actual)
recovered = _add_daily_dollar(recovered)
still_blocked = _add_daily_dollar(still_blocked)
nofilter = _add_daily_dollar(nofilter)

print(f"\n{'='*W}")
print(f"  ACTUAL vs NO-FILTER COMPARISON")
print(f"{'='*W}")

def print_stats(label, df, indent='  '):
    if df.empty:
        print(f"{indent}{label}: 0 trades")
        return
    n = len(df)
    wins = len(df[df['pnl'] > 0])
    wr = wins / n * 100
    avg = df['pnl'].mean()
    tot = df['pnl'].sum()
    dollar = df['dollar_pnl'].sum()
    print(f"{indent}{label}:")
    print(f"{indent}  Trades: {n}, Wins: {wins}, WR: {wr:.0f}%, "
          f"Avg PnL: {avg:+.2%}, Total: {tot:+.2%}, $PnL: ${dollar:+,.0f}")

print()
print_stats('ACTUAL (with vol+dollar filters)', actual)
print_stats('RECOVERED (would add without filters)', recovered)
print_stats('NO-FILTER scenario (actual + recovered)', nofilter)
print_stats('STILL BLOCKED (regime filter)', still_blocked)

# ── Comparison table ──
print(f"\n  {'Scenario':<35} {'Trades':>7} {'WR':>5} {'AvgPnL':>9} {'TotalPnL':>10} {'$PnL':>12}")
print(f"  {'─'*80}")

for label, df in [('ACTUAL (with filters)', actual),
                  ('NO VOL/DVOL FILTERS', nofilter)]:
    if df.empty:
        print(f"  {label:<35} {'0':>7}")
        continue
    n = len(df)
    w = len(df[df['pnl'] > 0])
    dollar = df['dollar_pnl'].sum()
    print(f"  {label:<35} {n:>7} {w/n:>4.0%} {df['pnl'].mean():>+8.2%} "
          f"{df['pnl'].sum():>+9.2%} ${dollar:>+11,.0f}")

delta_trades = len(nofilter) - len(actual)
d_nf = nofilter['dollar_pnl'].sum() if not nofilter.empty else 0
d_ac = actual['dollar_pnl'].sum() if not actual.empty else 0
delta_dollar = d_nf - d_ac
print(f"\n  Net filter removal impact: {delta_trades:+d} trades, ${delta_dollar:+,.0f}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. RECOVERED TRADES DETAIL
# ══════════════════════════════════════════════════════════════════════════════
if not recovered.empty:
    print(f"\n{'='*W}")
    print(f"  RECOVERED TRADES (currently blocked by vol/dollar filters)")
    print(f"{'='*W}")
    print(f"  {'Date':<12} {'Ticker':<7} {'Block':<10} {'Prob':>6} {'Entry$':>9} "
          f"{'TP':>5} {'SL':>5} {'Exit':>4} {'PnL':>8} {'MFE':>7} {'$Vol':>10} {'W/L':>4}")
    print(f"  {'─'*100}")
    for _, r in recovered.sort_values('date').iterrows():
        wl = 'W' if r['pnl'] > 0 else 'L'
        dv = f"${r['dollar_volume']:,.0f}" if pd.notna(r.get('dollar_volume')) else '?'
        print(f"  {str(r['date']):<12} {r['ticker']:<7} {r['block_reason']:<10} "
              f"{r['prob']:>5.3f} ${r.get('entry_price',0):>8.2f} "
              f"{r['tp']:>4.0%} {r['sl']:>4.0%} {r.get('exit_type','?'):>4} "
              f"{r['pnl']:>+7.2%} {r.get('mfe',0):>+6.2%} {dv:>10} {wl:>4}")

# ══════════════════════════════════════════════════════════════════════════════
# 7. WEEKLY BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*W}")
print(f"  WEEKLY BREAKDOWN — ACTUAL vs NO-FILTER")
print(f"{'='*W}")

if not actual.empty:
    actual = actual.copy()
    actual['week'] = pd.to_datetime(actual['date']).dt.isocalendar().week.values
if not nofilter.empty:
    nofilter = nofilter.copy()
    nofilter['week'] = pd.to_datetime(nofilter['date']).dt.isocalendar().week.values
if not recovered.empty:
    recovered = recovered.copy()
    recovered['week'] = pd.to_datetime(recovered['date']).dt.isocalendar().week.values

weeks = sorted(set(
    list(actual['week'].unique() if not actual.empty else []) +
    list(nofilter['week'].unique() if not nofilter.empty else [])
))

print(f"\n  {'Week':<8} {'|':>2} {'Act Tr':>7} {'Act WR':>7} {'Act Avg':>9} {'Act $PnL':>12} "
      f"{'|':>2} {'NF Tr':>7} {'NF WR':>7} {'NF Avg':>9} {'NF $PnL':>12} {'|':>2} {'Delta$':>10}")
print(f"  {'─'*105}")

for wk in weeks:
    aw = actual[actual['week'] == wk] if not actual.empty else pd.DataFrame()
    nw = nofilter[nofilter['week'] == wk] if not nofilter.empty else pd.DataFrame()

    a_n = len(aw)
    a_wr = len(aw[aw['pnl'] > 0]) / a_n * 100 if a_n > 0 else 0
    a_avg = aw['pnl'].mean() if a_n > 0 else 0
    a_dollar = aw['dollar_pnl'].sum() if a_n > 0 else 0

    n_n = len(nw)
    n_wr = len(nw[nw['pnl'] > 0]) / n_n * 100 if n_n > 0 else 0
    n_avg = nw['pnl'].mean() if n_n > 0 else 0
    n_dollar = nw['dollar_pnl'].sum() if n_n > 0 else 0

    delta = n_dollar - a_dollar
    print(f"  W{wk:<6} {'|':>2} {a_n:>7} {a_wr:>6.0f}% {a_avg:>+8.2%} ${a_dollar:>+11,.0f} "
          f"{'|':>2} {n_n:>7} {n_wr:>6.0f}% {n_avg:>+8.2%} ${n_dollar:>+11,.0f} {'|':>2} ${delta:>+9,.0f}")

# ══════════════════════════════════════════════════════════════════════════════
# 8. DAY-BY-DAY
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*W}")
print(f"  DAY-BY-DAY — actual trades vs what we missed")
print(f"{'='*W}")

for d in sorted(rdf['date'].unique()):
    day = rdf[rdf['date'] == d]
    day_actual = day[(day['filter'] == 'OK') & (day['filled'] == True)]
    day_recov = day[(day['filter'] != 'OK') & (day['pass_nofilter'] == True) & (day['filled'] == True)]
    day_regime = day[(day['filter'] != 'OK') & (day['pass_nofilter'] == False) & (day['filled'] == True)]
    day_unfilled = day[day['filled'] == False]

    n_act = len(day_actual)
    act_pnl = day_actual['pnl'].sum() if n_act > 0 else 0
    act_alloc = CAPITAL / n_act if n_act > 0 else 0
    act_dollar = sum(act_alloc * p for p in day_actual['pnl']) if n_act > 0 else 0

    n_nf = n_act + len(day_recov)
    nf_combined = pd.concat([day_actual[['pnl']], day_recov[['pnl']]], ignore_index=True)
    nf_alloc = CAPITAL / n_nf if n_nf > 0 else 0
    nf_dollar = sum(nf_alloc * p for p in nf_combined['pnl']) if n_nf > 0 else 0

    delta = nf_dollar - act_dollar

    print(f"\n  ┌─ {d} ───────────────────────────────────────────────────────────────────────┐")
    print(f"  │  Actual: {n_act} trades ${act_dollar:+,.0f}  │  "
          f"No-filter: {n_nf} trades ${nf_dollar:+,.0f}  │  Delta: ${delta:+,.0f}")
    print(f"  └{'─'*(W-3)}┘")

    if not day_actual.empty:
        for _, t in day_actual.sort_values('ticker').iterrows():
            wl = 'W' if t['pnl'] > 0 else 'L'
            print(f"    {wl} {t['ticker']:<7} {t['pnl']:>+7.2%} {t.get('exit_type','?'):>4}  "
                  f"entry ${t.get('entry_price',0):.2f}  prob {t['prob']:.3f}")

    if not day_recov.empty:
        print(f"    ── would recover without vol/dollar filters:")
        for _, t in day_recov.sort_values('ticker').iterrows():
            wl = 'W' if t['pnl'] > 0 else 'L'
            print(f"    {wl} {t['ticker']:<7} {t['pnl']:>+7.2%} {t.get('exit_type','?'):>4}  "
                  f"blocked:{t['block_reason']:<10} dv=${t.get('dollar_volume',0):,.0f}")

    if not day_regime.empty:
        for _, t in day_regime.sort_values('ticker').iterrows():
            print(f"    X {t['ticker']:<7} regime-blocked (VIX={t.get('vix',0):.1f})")

# ══════════════════════════════════════════════════════════════════════════════
# 9. PER-TICKER IMPACT
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*W}")
print(f"  PER-TICKER FILTER IMPACT")
print(f"{'='*W}")

tickers = sorted(rdf['ticker'].unique())
print(f"\n  {'Ticker':<7} {'Act':>4} {'Rec':>4} {'NF':>4} {'Act Avg':>9} {'NF Avg':>9} "
      f"{'Act $PnL':>12} {'NF $PnL':>12} {'Delta$':>10}")
print(f"  {'─'*85}")

for tk in tickers:
    tk_act = actual[actual['ticker'] == tk] if not actual.empty else pd.DataFrame()
    tk_rec = recovered[recovered['ticker'] == tk] if not recovered.empty else pd.DataFrame()
    tk_nf = nofilter[nofilter['ticker'] == tk] if not nofilter.empty else pd.DataFrame()

    n_a, n_r, n_nf = len(tk_act), len(tk_rec), len(tk_nf)
    a_avg = tk_act['pnl'].mean() if n_a > 0 else 0
    nf_avg = tk_nf['pnl'].mean() if n_nf > 0 else 0

    a_dollar = tk_act['dollar_pnl'].sum() if n_a > 0 else 0
    nf_dollar = tk_nf['dollar_pnl'].sum() if n_nf > 0 else 0

    delta = nf_dollar - a_dollar
    marker = ' <<<' if abs(delta) > 2000 else ''
    print(f"  {tk:<7} {n_a:>4} {n_r:>4} {n_nf:>4} {a_avg:>+8.2%} {nf_avg:>+8.2%} "
          f"${a_dollar:>+11,.0f} ${nf_dollar:>+11,.0f} ${delta:>+9,.0f}{marker}")

# ══════════════════════════════════════════════════════════════════════════════
# 10. CHARTS
# ══════════════════════════════════════════════════════════════════════════════
if not filled.empty:
    dates = sorted(rdf['date'].unique())

    act_daily_dollar = []
    nf_daily_dollar = []
    act_daily_trades = []
    nf_daily_trades = []

    for d in dates:
        da = actual[actual['date'] == d] if not actual.empty else pd.DataFrame()
        dn = nofilter[nofilter['date'] == d] if not nofilter.empty else pd.DataFrame()

        act_daily_dollar.append(da['dollar_pnl'].sum() if not da.empty else 0)
        nf_daily_dollar.append(dn['dollar_pnl'].sum() if not dn.empty else 0)
        act_daily_trades.append(len(da))
        nf_daily_trades.append(len(dn))

    date_labels = [d.strftime('%b %d') if hasattr(d, 'strftime') else str(d)[:10] for d in dates]
    x = np.arange(len(dates))
    bar_w = 0.35

    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    fig.suptitle(f'S3 June Filter Impact — {START_DATE} → {END_DATE}', fontsize=14, fontweight='bold')

    # Chart 1: Daily $PnL
    axes[0].bar(x - bar_w/2, act_daily_dollar, bar_w, label='Actual (with filters)', color='#2196F3', alpha=0.85)
    axes[0].bar(x + bar_w/2, nf_daily_dollar, bar_w, label='No vol/dollar filters', color='#4CAF50', alpha=0.85)
    axes[0].axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    axes[0].set_title('Daily $PnL ($200K capital)', fontsize=12)
    axes[0].set_ylabel('$PnL')
    axes[0].set_xticks(x); axes[0].set_xticklabels(date_labels, rotation=45, ha='right', fontsize=7)
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3, axis='y')

    # Chart 2: Cumulative $PnL
    act_cum = np.cumsum(act_daily_dollar)
    nf_cum = np.cumsum(nf_daily_dollar)
    axes[1].plot(x, act_cum, 'o-', label='Actual (with filters)', color='#2196F3', linewidth=2)
    axes[1].plot(x, nf_cum, 's-', label='No vol/dollar filters', color='#4CAF50', linewidth=2)
    axes[1].axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    axes[1].fill_between(x, act_cum, nf_cum, alpha=0.15, color='green',
                          where=[n >= a for n, a in zip(nf_cum, act_cum)])
    axes[1].fill_between(x, act_cum, nf_cum, alpha=0.15, color='red',
                          where=[n < a for n, a in zip(nf_cum, act_cum)])
    axes[1].set_title('Cumulative $PnL', fontsize=12)
    axes[1].set_ylabel('Cumulative $PnL')
    axes[1].set_xticks(x); axes[1].set_xticklabels(date_labels, rotation=45, ha='right', fontsize=7)
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    # Chart 3: Trade count
    axes[2].bar(x - bar_w/2, act_daily_trades, bar_w, label='Actual trades', color='#2196F3', alpha=0.85)
    axes[2].bar(x + bar_w/2, nf_daily_trades, bar_w, label='No-filter trades', color='#4CAF50', alpha=0.85)
    axes[2].set_title('Daily Trade Count', fontsize=12)
    axes[2].set_ylabel('Trades')
    axes[2].set_xticks(x); axes[2].set_xticklabels(date_labels, rotation=45, ha='right', fontsize=7)
    axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    chart_path = '/tmp/june_filter_impact.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"\n  Charts saved: {chart_path}")
    try:
        from IPython.display import display, Image
        display(Image(filename=chart_path))
    except ImportError:
        plt.show()

print(f"\n{'='*W}")
print(f"  DONE — filter removal impact analysis complete")
print(f"{'='*W}")
