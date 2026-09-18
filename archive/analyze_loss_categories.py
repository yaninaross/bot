"""
analyze_loss_categories.py — Categorize trading losses from live predictions
============================================================================
Queries BQ live_predictions for GO signals, loads OHLCV from GCS (cached),
computes actual trade outcomes, and categorizes each loss.

Categories:
  TIGHT_SL  — SL hit but price recovered by EOD (would have been profitable)
  MKT_SELL  — SPY gap < -0.3% (broad market selloff)
  WRONG_DIR — Max intraday upside < 1%, stock went straight down
  WHIPSAW   — SL hit then bounced past 50% of TP target
  PENNY     — Entry price < $2 (unreliable microstructure)
  LOW_CONV  — Prob within 5% of threshold (weak signal)
  LATE      — Trade in last 2 weeks (potential model decay)

Run in Colab after daily_ohlcv scripts have updated GCS.
"""

import gc, warnings
import pandas as pd, numpy as np, pytz
from google.cloud import bigquery, storage as gcs_storage

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

warnings.filterwarnings('ignore')

PST = pytz.timezone('America/Los_Angeles')
BUCKET = os.environ.get("GCS_BUCKET", "")
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

bq_client = bigquery.Client()
gcs_client = gcs_storage.Client()

# ==============================================================================
# LOAD GO SIGNALS FROM BQ
# ==============================================================================
print("Loading GO signals from live_predictions...")
q = """
SELECT score_date, ticker, experiment, seed, prob, threshold, signal,
       tp_pct, sl_pct, entry_minute, close, filter_status,
       tpsl_mode, vix_value, tlt_gap_pct, spy_gap_pct
FROM `{GCP_PROJECT}.scoring_logs.live_predictions`
WHERE signal = 'GO'
  AND score_date >= '2026-04-01'
ORDER BY score_date, ticker
"""
df = bq_client.query(q).to_dataframe()
df = df.drop_duplicates(subset=['score_date', 'ticker'], keep='last')
print(f"  {len(df)} GO signals, {df['ticker'].nunique()} tickers, "
      f"{df['score_date'].nunique()} trading days")

# ==============================================================================
# LOAD OHLCV — CACHED PER TICKER
# ==============================================================================
print("\nLoading OHLCV data (one parquet per ticker)...")
ohlcv_cache = {}
for ticker in sorted(df['ticker'].unique()):
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    uri = f"gs://{BUCKET}/Data/Inference_v2/{clean}_master_inference.parquet"
    try:
        d = pd.read_parquet(uri)
        d = d.reset_index()
        dt_cols = [c for c in d.columns if pd.api.types.is_datetime64_any_dtype(d[c])]
        if dt_cols:
            d['Datetime'] = pd.to_datetime(d[dt_cols[0]], utc=True)
        else:
            d['Datetime'] = pd.to_datetime(d[d.columns[0]], unit='ms', utc=True)
        d = d.set_index('Datetime').sort_index()
        std_map = {'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'}
        lower_map = {k.lower():v for k,v in std_map.items()}
        d.columns = [lower_map.get(str(c).strip().lower(), str(c).strip()) for c in d.columns]
        d.index = d.index.tz_convert(PST)
        trim = pd.Timestamp('2026-03-25', tz=PST)
        d = d[d.index >= trim]
        ohlcv_cache[ticker] = d[['Open','High','Low','Close','Volume']].copy()
        print(f"  {ticker}: {len(d):,} bars (trimmed)")
        del d; gc.collect()
    except Exception as e:
        print(f"  {ticker}: FAILED — {e}")
print(f"  {len(ohlcv_cache)} tickers loaded")

# ==============================================================================
# COMPUTE TRADE OUTCOMES
# ==============================================================================
print(f"\nComputing outcomes for {len(df)} trades...")
outcomes = []

for _, row in df.iterrows():
    ticker = row['ticker']
    if ticker not in ohlcv_cache:
        continue

    d = ohlcv_cache[ticker]
    target_date = pd.Timestamp(str(row['score_date'])).date()
    em = int(row['entry_minute'])
    tp = float(row['tp_pct']) if pd.notna(row['tp_pct']) and row['tp_pct'] > 0 else 0.16
    sl = float(row['sl_pct']) if pd.notna(row['sl_pct']) and row['sl_pct'] > 0 else 0.08

    day_data = d[d.index.date == target_date]
    if day_data.empty:
        continue

    mins = day_data.index.hour * 60 + day_data.index.minute
    entry_bars = day_data[mins == em]
    if entry_bars.empty:
        continue

    entry_price = float(entry_bars['Close'].iloc[0])
    after_entry = day_data[day_data.index > entry_bars.index[0]]
    if after_entry.empty:
        continue

    tp_price = entry_price * (1 + tp)
    sl_price = entry_price * (1 - sl)

    max_high = float(after_entry['High'].max())
    min_low = float(after_entry['Low'].min())
    eod_price = float(after_entry['Close'].iloc[-1])

    max_up = (max_high - entry_price) / entry_price
    max_down = (entry_price - min_low) / entry_price
    eod_return = (eod_price - entry_price) / entry_price

    hit_sl = min_low <= sl_price
    hit_tp = max_high >= tp_price

    if hit_sl and hit_tp:
        exit_type = 'BOTH'; pnl = -sl
    elif hit_sl:
        exit_type = 'SL_HIT'; pnl = -sl
    elif hit_tp:
        exit_type = 'TP_HIT'; pnl = tp
    else:
        exit_type = 'EOD'; pnl = eod_return

    pnl -= 0.003

    outcomes.append({
        'date': str(row['score_date']), 'ticker': ticker,
        'prob': float(row['prob']), 'entry_price': entry_price,
        'tp_pct': tp, 'sl_pct': sl, 'mode': row['tpsl_mode'],
        'vix': float(row['vix_value']) if pd.notna(row['vix_value']) else None,
        'spy_gap': float(row['spy_gap_pct']) if pd.notna(row['spy_gap_pct']) else None,
        'pnl': pnl, 'exit_type': exit_type,
        'max_up': max_up, 'max_down': max_down,
        'eod_return': eod_return, 'eod_price': eod_price,
    })

del ohlcv_cache; gc.collect()

odf = pd.DataFrame(outcomes)
odf['date'] = pd.to_datetime(odf['date'])

print(f"  {len(odf)} trades with outcomes")
print(f"  Exit types: {dict(odf['exit_type'].value_counts())}")

wins = odf[odf['pnl'] > 0]
losses = odf[odf['pnl'] < 0].copy()
print(f"  {len(wins)} wins ({len(wins)/len(odf)*100:.0f}%), {len(losses)} losses")

# ==============================================================================
# CATEGORIZE LOSSES
# ==============================================================================
losses['sl_too_tight'] = (losses['exit_type'] == 'SL_HIT') & (losses['eod_return'] > 0)
losses['market_selloff'] = (losses['spy_gap'].notna()) & (losses['spy_gap'] < -0.3)
losses['wrong_direction'] = (losses['max_up'] < 0.01) & (losses['eod_return'] < -0.02)
losses['whipsaw'] = (losses['exit_type'] == 'SL_HIT') & (losses['max_up'] > losses['tp_pct'] * 0.5)
losses['penny_stock'] = losses['entry_price'] < 2.0
losses['low_conviction'] = (losses['prob'] - 0.50) < 0.05
late_start = odf['date'].max() - pd.Timedelta(days=14)
losses['late_period'] = losses['date'] >= late_start

W = 120
print(f"\n{'='*W}")
print(f"  LOSS CATEGORIZATION — {len(losses)} losing trades")
print(f"{'='*W}")

categories = [
    ('SL Too Tight (recovered by EOD)', 'sl_too_tight'),
    ('Market Selloff (SPY < -0.3%)', 'market_selloff'),
    ('Wrong Direction (no upside <1%)', 'wrong_direction'),
    ('Whipsaw (SL hit, bounced 50%+ of TP)', 'whipsaw'),
    ('Penny Stock (<$2)', 'penny_stock'),
    ('Low Conviction (prob near threshold)', 'low_conviction'),
    ('Late Period (last 2 weeks)', 'late_period'),
]

print(f"\n  {'Category':<45} {'Count':>6} {'% Losses':>9} {'Avg Loss':>9} {'Total$':>8}")
print(f"  " + "-" * 80)
for label, col in categories:
    mask = losses[col]
    n = mask.sum()
    pct = n / len(losses) * 100 if len(losses) > 0 else 0
    avg = losses[mask]['pnl'].mean() * 100 if n > 0 else 0
    total = losses[mask]['pnl'].sum() * 100 if n > 0 else 0
    print(f"  {label:<45} {n:>6} {pct:>8.0f}% {avg:>+8.2f}% {total:>+7.1f}%")

# ==============================================================================
# DETAILED LOSS TABLE
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  ALL LOSING TRADES")
print(f"{'='*W}")
print(f"\n  {'Date':<12} {'Ticker':<7} {'Prob':>5} {'Entry$':>7} {'PnL':>7} {'Exit':>8} "
      f"{'MaxUp':>6} {'MaxDn':>6} {'EOD':>7} {'SL%':>5} {'TP%':>5} {'Category'}")
print(f"  " + "-" * 105)

for _, r in losses.sort_values('date').iterrows():
    cats = []
    if r['sl_too_tight']: cats.append('TIGHT_SL')
    if r['market_selloff']: cats.append('MKT_SELL')
    if r['wrong_direction']: cats.append('WRONG_DIR')
    if r['whipsaw']: cats.append('WHIPSAW')
    if r['penny_stock']: cats.append('PENNY')
    if r['low_conviction']: cats.append('LOW_CONV')
    if r['late_period']: cats.append('LATE')
    if not cats: cats.append('NORMAL')

    print(f"  {r['date'].strftime('%Y-%m-%d'):<12} {r['ticker']:<7} {r['prob']:>5.2f} "
          f"${r['entry_price']:>6.2f} {r['pnl']*100:>+6.2f}% {r['exit_type']:>8} "
          f"{r['max_up']*100:>+5.1f}% {r['max_down']*100:>+5.1f}% {r['eod_return']*100:>+6.2f}% "
          f"{r['sl_pct']*100:>4.0f}% {r['tp_pct']*100:>4.0f}% {','.join(cats)}")

# ==============================================================================
# LOSS PATTERNS SUMMARY
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  LOSS PATTERNS SUMMARY")
print(f"{'='*W}")

tight_sl = losses[losses['sl_too_tight']]
if len(tight_sl) > 0:
    print(f"\n  TIGHT SL ({len(tight_sl)} trades):")
    print(f"    Hit SL then recovered — would have been profitable at EOD")
    print(f"    Avg SL: {tight_sl['sl_pct'].mean()*100:.1f}%  Avg EOD: {tight_sl['eod_return'].mean()*100:+.1f}%")
    print(f"    Lost {tight_sl['pnl'].sum()*100:+.1f}% → would have been +{tight_sl['eod_return'].sum()*100:.1f}%")
    print(f"    Tickers: {', '.join(tight_sl['ticker'].unique())}")

wrong = losses[losses['wrong_direction']]
if len(wrong) > 0:
    print(f"\n  WRONG DIRECTION ({len(wrong)} trades):")
    print(f"    Model said spike but stock went down all day (<1% upside)")
    print(f"    Avg prob: {wrong['prob'].mean():.3f}  Avg loss: {wrong['pnl'].mean()*100:+.2f}%")
    print(f"    Tickers: {', '.join(wrong['ticker'].unique())}")

mkt = losses[losses['market_selloff']]
if len(mkt) > 0:
    print(f"\n  MARKET SELLOFF ({len(mkt)} trades):")
    print(f"    SPY gap < -0.3% — broad market dragged stocks down")
    print(f"    Avg SPY: {mkt['spy_gap'].mean():+.2f}%  Avg VIX: {mkt['vix'].mean():.1f}")
    print(f"    Dates: {', '.join(sorted(mkt['date'].dt.strftime('%m/%d').unique()))}")

whip = losses[losses['whipsaw']]
if len(whip) > 0:
    print(f"\n  WHIPSAW ({len(whip)} trades):")
    print(f"    SL hit then bounced significantly — SL was too tight for volatility")
    print(f"    Avg SL: {whip['sl_pct'].mean()*100:.1f}%  Avg max upside after: {whip['max_up'].mean()*100:+.1f}%")
    print(f"    Tickers: {', '.join(whip['ticker'].unique())}")

late = losses[losses['late_period']]
early_l = losses[~losses['late_period']]
if len(late) > 0 and len(early_l) > 0:
    print(f"\n  TEMPORAL DECAY:")
    print(f"    Early losses ({len(early_l)}): avg {early_l['pnl'].mean()*100:+.2f}%")
    print(f"    Late losses ({len(late)}): avg {late['pnl'].mean()*100:+.2f}%")
    delta = early_l['pnl'].mean() - late['pnl'].mean()
    if delta > 0:
        print(f"    Late losses are WORSE by {delta*100:.2f}% — supports decay hypothesis")

# ==============================================================================
# WIN RATE BY REGIME
# ==============================================================================
print(f"\n  WIN RATE BY VIX REGIME:")
odf['vix_bucket'] = pd.cut(odf['vix'].fillna(20), bins=[0, 15, 20, 25, 35, 100],
                            labels=['<15', '15-20', '20-25', '25-35', '35+'])
for bucket, group in odf.groupby('vix_bucket', observed=True):
    wr = (group['pnl'] > 0).mean()
    avg = group['pnl'].mean()
    print(f"    VIX {bucket}: {len(group)} trades, WR={wr*100:.0f}%, avg PnL={avg*100:+.2f}%")

# ==============================================================================
# ACTIONABLE RECOMMENDATIONS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  ACTIONABLE RECOMMENDATIONS")
print(f"{'='*W}")

n_tight = len(tight_sl)
n_wrong = len(wrong)
n_penny = losses['penny_stock'].sum()
total_losses = len(losses)

if n_tight > total_losses * 0.2:
    print(f"\n  1. WIDEN SL — {n_tight}/{total_losses} losses ({n_tight/total_losses*100:.0f}%) were tight SL recoveries")
    print(f"     Consider ATR-based SL or increasing SL by 50%")

if n_wrong > total_losses * 0.3:
    print(f"\n  2. RETRAIN MODELS — {n_wrong}/{total_losses} losses ({n_wrong/total_losses*100:.0f}%) were wrong direction")
    print(f"     Model is predicting spikes that don't happen — patterns have shifted")

if n_penny > 0:
    print(f"\n  3. ADD MIN_PRICE FILTER — {n_penny} trades on penny stocks (<$2)")
    print(f"     Already added MIN_PRICE = $2.00 to scoring script")

late_wr = (odf[odf['date'] >= late_start]['pnl'] > 0).mean() if len(odf[odf['date'] >= late_start]) > 0 else 0
early_wr = (odf[odf['date'] < late_start]['pnl'] > 0).mean() if len(odf[odf['date'] < late_start]) > 0 else 0
if late_wr < early_wr - 0.10:
    print(f"\n  4. MODEL DECAY CONFIRMED — WR dropped from {early_wr*100:.0f}% to {late_wr*100:.0f}%")
    print(f"     Retrain with March 2026 data cutoff (current: Jan 2026)")

# ==============================================================================
# WHAT-IF SCENARIO ANALYSIS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  WHAT-IF SCENARIO ANALYSIS — Simulating parameter changes")
print(f"  Baseline: {len(odf)} trades, WR={len(wins)/len(odf)*100:.0f}%, "
      f"avg PnL={odf['pnl'].mean()*100:+.2f}%, total PnL={odf['pnl'].sum()*100:+.1f}%")
print(f"{'='*W}")

# Whipsaw-prone tickers that need wider SL
WHIPSAW_TICKERS = ['ASTS', 'FCEL', 'RCAT', 'CLSK', 'LUNR', 'SPIR']

def simulate_scenario(trades_df, name, prob_bump=0.0, sl_mult=1.0,
                       min_price=0.0, whipsaw_sl_mult=1.0, min_prob=0.0,
                       max_sl_pct=None):
    t = trades_df.copy()

    # Filter: min price
    if min_price > 0:
        t = t[t['entry_price'] >= min_price]

    # Filter: prob bump (raise threshold)
    if prob_bump > 0:
        t = t[t['prob'] >= (0.50 + prob_bump)]

    # Filter: absolute min prob
    if min_prob > 0:
        t = t[t['prob'] >= min_prob]

    # Recompute PnL with wider SL for all tickers
    if sl_mult != 1.0:
        new_pnl = []
        for _, r in t.iterrows():
            new_sl = r['sl_pct'] * sl_mult
            new_sl_price = r['entry_price'] * (1 - new_sl)
            hit_new_sl = (r['entry_price'] - r['entry_price'] * (1 - r['max_down'])) >= (r['entry_price'] - new_sl_price)
            # If original was SL_HIT but wider SL wouldn't hit
            if r['exit_type'] == 'SL_HIT' and not hit_new_sl:
                new_pnl.append(r['eod_return'] - 0.003)
            elif r['exit_type'] == 'SL_HIT' and hit_new_sl:
                new_pnl.append(-new_sl - 0.003)
            else:
                new_pnl.append(r['pnl'])
        t['pnl'] = new_pnl

    # Recompute PnL with wider SL for whipsaw tickers only
    if whipsaw_sl_mult != 1.0:
        new_pnl = []
        for _, r in t.iterrows():
            if r['ticker'] in WHIPSAW_TICKERS:
                new_sl = r['sl_pct'] * whipsaw_sl_mult
                entry = r['entry_price']
                new_sl_price = entry * (1 - new_sl)
                actual_low = entry * (1 - r['max_down'])
                hit_new_sl = actual_low <= new_sl_price
                if r['exit_type'] == 'SL_HIT' and not hit_new_sl:
                    new_pnl.append(r['eod_return'] - 0.003)
                elif r['exit_type'] == 'SL_HIT' and hit_new_sl:
                    new_pnl.append(-new_sl - 0.003)
                else:
                    new_pnl.append(r['pnl'])
            else:
                new_pnl.append(r['pnl'])
        t['pnl'] = new_pnl

    # Cap max SL
    if max_sl_pct is not None:
        new_pnl = []
        for _, r in t.iterrows():
            if r['sl_pct'] > max_sl_pct and r['exit_type'] == 'SL_HIT':
                new_pnl.append(-max_sl_pct - 0.003)
            else:
                new_pnl.append(r['pnl'])
        t['pnl'] = new_pnl

    if t.empty:
        return {'name': name, 'trades': 0, 'wins': 0, 'wr': 0, 'avg_pnl': 0,
                'total_pnl': 0, 'removed': len(trades_df)}

    n = len(t)
    w = (t['pnl'] > 0).sum()
    return {
        'name': name,
        'trades': n,
        'wins': w,
        'wr': w / n,
        'avg_pnl': t['pnl'].mean(),
        'total_pnl': t['pnl'].sum(),
        'removed': len(trades_df) - n,
    }

scenarios = [
    simulate_scenario(odf, 'A: Baseline (current)'),
    simulate_scenario(odf, 'B: Prob +5%', prob_bump=0.05),
    simulate_scenario(odf, 'C: Prob +10%', prob_bump=0.10),
    simulate_scenario(odf, 'D: Prob +15%', prob_bump=0.15),
    simulate_scenario(odf, 'E: Min price $2', min_price=2.0),
    simulate_scenario(odf, 'F: Widen SL 50% (all)', sl_mult=1.5),
    simulate_scenario(odf, 'G: Widen SL 50% (whipsaw)', whipsaw_sl_mult=1.5),
    simulate_scenario(odf, 'H: Widen SL 100% (whipsaw)', whipsaw_sl_mult=2.0),
    simulate_scenario(odf, 'I: Cap SL at 8%', max_sl_pct=0.08),
    simulate_scenario(odf, 'J: Prob+5% + $2 + whip SL', prob_bump=0.05, min_price=2.0, whipsaw_sl_mult=1.5),
    simulate_scenario(odf, 'K: Prob+10% + $2 + whip SL', prob_bump=0.10, min_price=2.0, whipsaw_sl_mult=1.5),
    simulate_scenario(odf, 'L: Prob+5% + $2 + all SL 50%', prob_bump=0.05, min_price=2.0, sl_mult=1.5),
    simulate_scenario(odf, 'M: Prob+10% + $2 + all SL 50%', prob_bump=0.10, min_price=2.0, sl_mult=1.5),
    simulate_scenario(odf, 'N: Min prob 0.55 + $2', min_prob=0.55, min_price=2.0),
    simulate_scenario(odf, 'O: Min prob 0.60 + $2', min_prob=0.60, min_price=2.0),
    simulate_scenario(odf, 'P: Min prob 0.60 + $2 + whip', min_prob=0.60, min_price=2.0, whipsaw_sl_mult=1.5),
]

print(f"\n  {'Scenario':<35} {'Trades':>7} {'Removed':>8} {'Wins':>5} {'WR':>6} "
      f"{'AvgPnL':>8} {'TotalPnL':>10} {'vs Base':>8}")
print(f"  " + "-" * 95)

base_total = scenarios[0]['total_pnl']
for s in scenarios:
    delta = (s['total_pnl'] - base_total) * 100
    marker = ' <-- BEST' if s == max(scenarios[1:], key=lambda x: x['total_pnl']) else ''
    print(f"  {s['name']:<35} {s['trades']:>7} {s['removed']:>8} {s['wins']:>5} "
          f"{s['wr']*100:>5.0f}% {s['avg_pnl']*100:>+7.2f}% {s['total_pnl']*100:>+9.1f}% "
          f"{delta:>+7.1f}%{marker}")

# Find top 3 scenarios
ranked = sorted(scenarios[1:], key=lambda x: x['total_pnl'], reverse=True)[:3]
print(f"\n  TOP 3 SCENARIOS:")
for i, s in enumerate(ranked, 1):
    improvement = (s['total_pnl'] - base_total) * 100
    print(f"    {i}. {s['name']}")
    print(f"       {s['trades']} trades, WR={s['wr']*100:.0f}%, avg={s['avg_pnl']*100:+.2f}%, "
          f"total={s['total_pnl']*100:+.1f}% ({improvement:+.1f}% vs baseline)")

# Per-ticker impact of best scenario
best = ranked[0]
print(f"\n  BEST SCENARIO DETAILS: {best['name']}")

# Re-run best scenario to get per-ticker breakdown
# Determine which filters to apply based on name
if 'Prob+10%' in best['name'] and '$2' in best['name']:
    filtered = odf[(odf['prob'] >= 0.60) & (odf['entry_price'] >= 2.0)]
elif 'Prob+5%' in best['name'] and '$2' in best['name']:
    filtered = odf[(odf['prob'] >= 0.55) & (odf['entry_price'] >= 2.0)]
else:
    filtered = odf.copy()

if not filtered.empty:
    ticker_perf = filtered.groupby('ticker').agg(
        trades=('pnl', 'count'),
        wins=('pnl', lambda x: (x > 0).sum()),
        avg_pnl=('pnl', 'mean'),
        total_pnl=('pnl', 'sum'),
    ).sort_values('total_pnl', ascending=False)

    print(f"\n  {'Ticker':<7} {'Trades':>7} {'Wins':>5} {'WR':>6} {'AvgPnL':>8} {'TotalPnL':>10}")
    print(f"  " + "-" * 45)
    for t, r in ticker_perf.iterrows():
        print(f"  {t:<7} {int(r['trades']):>7} {int(r['wins']):>5} "
              f"{r['wins']/r['trades']*100:>5.0f}% {r['avg_pnl']*100:>+7.2f}% "
              f"{r['total_pnl']*100:>+9.1f}%")

print(f"\n{'='*W}")
