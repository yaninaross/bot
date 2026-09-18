"""
July 2026 Model Predictions vs Actual PnL Analysis
=====================================================
Pulls S3 GO signals from BQ live_predictions for July,
downloads 1-min OHLCV from GCS, simulates limit_entry trades,
and reports daily PnL breakdown.
"""

import numpy as np
import pandas as pd
import io
from datetime import datetime
from google.cloud import bigquery, storage as gcs_storage

BUCKET_NAME = os.environ.get("GCS_BUCKET", "")
EXIT_TIME_MIN = 780
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE', 'KEEL': 'KEEL'}
W = 90

bq_client = bigquery.Client()
gcs_client = gcs_storage.Client()
bucket = gcs_client.bucket(BUCKET_NAME)

# ==============================================================================
# STEP 1: Get deployed S3 configs (with dip_pct, max_wait_minutes)
# ==============================================================================
print("=" * W)
print("  JULY 2026 — MODEL PREDICTIONS vs ACTUAL PnL")
print("=" * W)

print("\n  STEP 1: Loading deployed S3 configs from WF optimizer...")

DEPLOY_THRESHOLD = 0.018
MIN_SEEDS = 3
MAX_VA_OPT_RATIO = 3.0
GRP_KEYS = ['ticker', 'experiment', 'tpsl_mode', 'entry_minute',
            'prob_thresh', 'tp_pct', 'sl_pct', 'tp_mult', 'sl_mult',
            'trail_start_mult', 'trail_end_mult', 'grace_minutes',
            'dip_pct', 'max_wait_minutes', 'vol_filter_pct']

q_wf = """
SELECT ticker, experiment, seed, tpsl_mode, entry_minute,
       prob_thresh, tp_pct, sl_pct, tp_mult, sl_mult,
       trail_start_mult, trail_end_mult, grace_minutes,
       dip_pct, max_wait_minutes, vol_filter_pct,
       val_a_avg_pnl, val_a_win_rate, val_a_count, val_a_freq,
       val_b_avg_pnl, opt_avg_pnl, is_ensemble
FROM `{GCP_PROJECT}.scoring_logs.wf_optimization_results`
WHERE model_version IN ('robust_wf_v12', 'robust_wf_v12_stitched')
  AND survived_a = TRUE
  AND val_a_count >= 3
  AND val_a_avg_pnl > 0
"""
raw_df = bq_client.query(q_wf).to_dataframe()
print(f"    {len(raw_df)} raw BQ rows loaded")

raw_df = raw_df[raw_df['opt_avg_pnl'] > 0].copy()
raw_df = raw_df[raw_df['val_a_avg_pnl'] / raw_df['opt_avg_pnl'] <= MAX_VA_OPT_RATIO]

indiv = raw_df[raw_df['is_ensemble'] == False].copy()
for k in GRP_KEYS:
    if k not in indiv.columns:
        indiv[k] = 0
    else:
        indiv[k] = indiv[k].fillna(0)

sa = indiv.groupby(GRP_KEYS).agg(
    n_seeds=('seed', 'nunique'),
    n_pos_seeds_18=('val_a_avg_pnl', lambda x: (x >= DEPLOY_THRESHOLD).sum()),
    mean_va_pnl=('val_a_avg_pnl', 'mean'),
    mean_va_wr=('val_a_win_rate', 'mean'),
    vb_pnl=('val_b_avg_pnl', 'mean'),
).reset_index()
sa['agree_18'] = (sa['n_pos_seeds_18'] / sa['n_seeds']).clip(upper=1.0)
sa = sa[sa['n_seeds'] >= MIN_SEEDS].copy()
sa_limit = sa[sa['tpsl_mode'] == 'limit_entry'].copy()
sa_limit['score'] = (sa_limit['mean_va_pnl'] ** 2) * sa_limit['agree_18']


def assign_family(exp):
    base = exp.split('_')[0]
    if not base.isdigit():
        return 'other'
    num = int(base)
    if num == 152: return '152'
    if num == 160: return '160'
    if 170 <= num <= 175: return '170'
    if 180 <= num <= 185: return '180'
    if 200 <= num <= 205: return '200'
    return 'other'


sa_limit['exp_fam'] = sa_limit['experiment'].apply(assign_family)
sa_fam = sa_limit[sa_limit['exp_fam'] != 'other'].copy()
best_per_fam = sa_fam.loc[sa_fam.groupby(['ticker', 'exp_fam'])['score'].idxmax()]
best_per_fam = best_per_fam[best_per_fam['mean_va_pnl'] >= DEPLOY_THRESHOLD]

agreement = {}
for tk in best_per_fam['ticker'].unique():
    tk_rows = best_per_fam[best_per_fam['ticker'] == tk]
    n_fams = len(tk_rows)
    n_pos = int((tk_rows['vb_pnl'] > 0).sum())
    agreement[tk] = {
        'n_fams': n_fams, 'n_pos': n_pos,
        'majority_pos': n_pos > n_fams / 2 and n_fams >= 2,
    }

s3_tickers = sorted([tk for tk, ag in agreement.items() if ag['majority_pos']])
print(f"    S3 family agreement: {len(s3_tickers)} tickers pass gate")
print(f"    S3 tickers: {', '.join(s3_tickers)}")

sa_s3 = sa_limit[sa_limit['ticker'].isin(s3_tickers)].copy()
sa_s3['score'] = (sa_s3['mean_va_pnl'] ** 2) * sa_s3['agree_18']
best_s3 = sa_s3.loc[sa_s3.groupby('ticker')['score'].idxmax()]
best_s3 = best_s3[best_s3['mean_va_pnl'] >= DEPLOY_THRESHOLD]

deploy_configs = {}
for _, row in best_s3.iterrows():
    tk = row['ticker']
    deploy_configs[tk] = {
        'prob_thresh': float(row['prob_thresh']),
        'tp': float(row['tp_pct']),
        'sl': float(row['sl_pct']),
        'dip_pct': float(row['dip_pct']),
        'max_wait': float(row['max_wait_minutes']),
        'entry_minute': int(row['entry_minute']) if pd.notna(row['entry_minute']) else 375,
        'experiment': row['experiment'],
        'avg_pnl': float(row['mean_va_pnl']),
    }

print(f"\n    Deployed configs ({len(deploy_configs)} tickers):")
for tk in sorted(deploy_configs):
    c = deploy_configs[tk]
    print(f"      {tk:6s}: TP={c['tp']:.0%} SL={c['sl']:.0%} "
          f"dip={c['dip_pct']:.1%} wait={c['max_wait']:.0f}m "
          f"prob>{c['prob_thresh']:.2f} ({c['experiment']})")

# ==============================================================================
# STEP 2: Get July GO signals from BQ
# ==============================================================================
print(f"\n  STEP 2: Loading July GO signals from BQ...")

q_signals = """
SELECT score_date, Ticker, experiment,
       AVG(prob) as mean_prob,
       ANY_VALUE(tp_pct) as tp_pct,
       ANY_VALUE(sl_pct) as sl_pct,
       ANY_VALUE(close) as close_price,
       ANY_VALUE(entry_minute) as entry_minute,
       COUNT(*) as n_seeds
FROM `{GCP_PROJECT}.scoring_logs.live_predictions`
WHERE score_date >= '2026-07-01' AND score_date <= '2026-07-16'
  AND strategy = 'S3'
  AND Signal = 'GO'
GROUP BY score_date, Ticker, experiment
ORDER BY score_date, Ticker
"""
signals_df = bq_client.query(q_signals).to_dataframe()
print(f"    {len(signals_df)} unique (date, ticker, experiment) GO signals")
print(f"    Dates: {signals_df['score_date'].min()} to {signals_df['score_date'].max()}")
print(f"    Tickers with signals: {sorted(signals_df['Ticker'].unique())}")

signal_tickers = sorted(signals_df['Ticker'].unique())

# ==============================================================================
# STEP 3: Download OHLCV data from GCS
# ==============================================================================
print(f"\n  STEP 3: Downloading OHLCV data from GCS...")

ohlcv_cache = {}
for ticker in signal_tickers:
    gcs_tk = TICKER_FILE_MAP.get(ticker, ticker)
    blob_path = f"Data/Inference/{gcs_tk}_1m_full_historical_data.parquet"
    blob = bucket.blob(blob_path)
    if not blob.exists():
        print(f"    WARNING: {blob_path} not found in GCS")
        continue
    data = blob.download_as_bytes()
    df = pd.read_parquet(io.BytesIO(data))

    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df = df.set_index('Datetime')
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    if df.index.tz is None:
        df.index = df.index.tz_localize('America/Los_Angeles')
    else:
        df.index = df.index.tz_convert('America/Los_Angeles')

    df = df[df.index >= '2026-07-01']
    df['date_int'] = df.index.strftime('%Y%m%d').astype(int)
    df['time_min'] = df.index.hour * 60 + df.index.minute
    ohlcv_cache[ticker] = df
    print(f"    {ticker}: {len(df)} bars loaded (Jul 2026)")

print(f"    {len(ohlcv_cache)} tickers loaded")

# ==============================================================================
# STEP 4: Simulate limit_entry trades
# ==============================================================================
print(f"\n  STEP 4: Simulating limit_entry trades...")


def simulate_limit_entry(ohlcv_day, entry_minute, tp_pct, sl_pct, dip_pct, max_wait, exit_time=EXIT_TIME_MIN):
    bars = ohlcv_day[ohlcv_day['time_min'] >= entry_minute].copy()
    if len(bars) < 2:
        return None

    signal_bar = bars.iloc[0]
    signal_price = signal_bar['Close']
    limit_price = signal_price * (1.0 - dip_pct)

    et = entry_minute
    fill_idx = None
    for i in range(1, len(bars)):
        if bars.iloc[i]['time_min'] - et > max_wait:
            break
        if bars.iloc[i]['Low'] <= limit_price:
            fill_idx = i
            break

    if fill_idx is None:
        return {
            'entry_price': None,
            'signal_price': signal_price,
            'pnl_pct': 0.0,
            'outcome': 'NO_FILL',
            'exit_price': None,
            'exit_time_min': None,
        }

    entry_price = limit_price
    tp_target = entry_price * (1.0 + tp_pct)
    sl_stop = entry_price * (1.0 - sl_pct)

    for i in range(fill_idx + 1, len(bars)):
        bar = bars.iloc[i]
        sl_hit = bar['Low'] <= sl_stop
        tp_hit = bar['High'] >= tp_target

        if sl_hit and tp_hit:
            pnl = -sl_pct if np.random.random() < 0.5 else tp_pct
            return {
                'entry_price': entry_price,
                'signal_price': signal_price,
                'pnl_pct': pnl,
                'outcome': 'TP' if pnl > 0 else 'SL',
                'exit_price': entry_price * (1 + pnl),
                'exit_time_min': bar['time_min'],
            }
        if sl_hit:
            return {
                'entry_price': entry_price,
                'signal_price': signal_price,
                'pnl_pct': -sl_pct,
                'outcome': 'SL',
                'exit_price': sl_stop,
                'exit_time_min': bar['time_min'],
            }
        if tp_hit:
            return {
                'entry_price': entry_price,
                'signal_price': signal_price,
                'pnl_pct': tp_pct,
                'outcome': 'TP',
                'exit_price': tp_target,
                'exit_time_min': bar['time_min'],
            }
        if bar['time_min'] >= exit_time:
            pnl = (bar['Close'] - entry_price) / entry_price
            return {
                'entry_price': entry_price,
                'signal_price': signal_price,
                'pnl_pct': pnl,
                'outcome': 'TIME_EXIT',
                'exit_price': bar['Close'],
                'exit_time_min': bar['time_min'],
            }

    last = bars.iloc[-1]
    pnl = (last['Close'] - entry_price) / entry_price
    return {
        'entry_price': entry_price,
        'signal_price': signal_price,
        'pnl_pct': pnl,
        'outcome': 'EOD',
        'exit_price': last['Close'],
        'exit_time_min': last['time_min'],
    }


np.random.seed(42)

all_trades = []
for _, row in signals_df.iterrows():
    ticker = row['Ticker']
    date = row['score_date']
    date_str = str(date)

    cfg = deploy_configs.get(ticker)
    if cfg is None:
        continue

    if ticker not in ohlcv_cache:
        continue

    ohlcv = ohlcv_cache[ticker]
    date_int = int(date_str.replace('-', ''))
    day_bars = ohlcv[ohlcv['date_int'] == date_int]
    if len(day_bars) < 5:
        continue

    result = simulate_limit_entry(
        day_bars, cfg['entry_minute'],
        cfg['tp'], cfg['sl'],
        cfg['dip_pct'], cfg['max_wait']
    )

    if result is None:
        continue

    all_trades.append({
        'date': date_str,
        'ticker': ticker,
        'experiment': row['experiment'],
        'mean_prob': row['mean_prob'],
        'n_seeds': row['n_seeds'],
        'signal_price': result['signal_price'],
        'entry_price': result['entry_price'],
        'exit_price': result['exit_price'],
        'tp_pct': cfg['tp'],
        'sl_pct': cfg['sl'],
        'dip_pct': cfg['dip_pct'],
        'pnl_pct': result['pnl_pct'],
        'outcome': result['outcome'],
        'exit_time_min': result['exit_time_min'],
    })

trades_df = pd.DataFrame(all_trades)
if trades_df.empty:
    print("    NO TRADES FOUND — check data availability")
else:
    # De-duplicate: if multiple experiments signal same ticker on same day,
    # keep the one with highest mean_prob (matches production behavior)
    trades_df = trades_df.sort_values('mean_prob', ascending=False)
    trades_dedup = trades_df.drop_duplicates(subset=['date', 'ticker'], keep='first').copy()
    filled = trades_dedup[trades_dedup['outcome'] != 'NO_FILL']
    unfilled = trades_dedup[trades_dedup['outcome'] == 'NO_FILL']

    print(f"    {len(trades_dedup)} unique (date, ticker) signals")
    print(f"    {len(filled)} filled trades, {len(unfilled)} unfilled (no dip reached)")

# ==============================================================================
# STEP 5: Results
# ==============================================================================
print(f"\n{'=' * W}")
print(f"  RESULTS")
print(f"{'=' * W}")

if trades_df.empty or filled.empty:
    print("  No filled trades to analyze.")
else:
    # ── Overall Stats ──
    total_pnl = filled['pnl_pct'].sum()
    avg_pnl = filled['pnl_pct'].mean()
    n_wins = (filled['pnl_pct'] > 0).sum()
    n_losses = (filled['pnl_pct'] < 0).sum()
    n_flat = (filled['pnl_pct'] == 0).sum()
    win_rate = n_wins / len(filled) * 100 if len(filled) > 0 else 0

    print(f"\n  OVERALL STATS (July 1-16, 2026)")
    print(f"  {'─' * 50}")
    print(f"  Total trades:     {len(filled)}")
    print(f"  Wins / Losses:    {n_wins}W / {n_losses}L / {n_flat}F")
    print(f"  Win rate:         {win_rate:.1f}%")
    print(f"  Total PnL:        {total_pnl:+.2%}")
    print(f"  Avg PnL/trade:    {avg_pnl:+.2%}")
    if n_wins > 0:
        avg_win = filled[filled['pnl_pct'] > 0]['pnl_pct'].mean()
        print(f"  Avg win:          {avg_win:+.2%}")
    if n_losses > 0:
        avg_loss = filled[filled['pnl_pct'] < 0]['pnl_pct'].mean()
        print(f"  Avg loss:         {avg_loss:+.2%}")
    print(f"  Unfilled signals: {len(unfilled)}")

    # ── Daily Breakdown ──
    print(f"\n  DAILY PnL BREAKDOWN")
    print(f"  {'─' * 70}")
    print(f"  {'Date':<12} {'Trades':>6} {'W':>3} {'L':>3} {'WR%':>6} "
          f"{'Avg PnL':>9} {'Total PnL':>10} {'Cum PnL':>10}")
    print(f"  {'─' * 70}")

    cum_pnl = 0
    daily_stats = []
    for date in sorted(filled['date'].unique()):
        day = filled[filled['date'] == date]
        d_trades = len(day)
        d_wins = (day['pnl_pct'] > 0).sum()
        d_losses = (day['pnl_pct'] < 0).sum()
        d_wr = d_wins / d_trades * 100 if d_trades > 0 else 0
        d_avg = day['pnl_pct'].mean()
        d_total = day['pnl_pct'].sum()
        cum_pnl += d_total
        daily_stats.append({
            'date': date, 'trades': d_trades, 'wins': d_wins,
            'losses': d_losses, 'wr': d_wr, 'avg_pnl': d_avg,
            'total_pnl': d_total, 'cum_pnl': cum_pnl,
        })
        marker = "+++" if d_total > 0.02 else ("---" if d_total < -0.02 else "   ")
        print(f"  {date:<12} {d_trades:>6} {d_wins:>3} {d_losses:>3} {d_wr:>5.1f}% "
              f"{d_avg:>+8.2%} {d_total:>+9.2%} {cum_pnl:>+9.2%} {marker}")

    print(f"  {'─' * 70}")

    # ── Per-Ticker Breakdown ──
    print(f"\n  PER-TICKER SUMMARY")
    print(f"  {'─' * 70}")
    print(f"  {'Ticker':<8} {'Signals':>7} {'Filled':>6} {'W':>3} {'L':>3} "
          f"{'WR%':>6} {'Avg PnL':>9} {'Total PnL':>10}")
    print(f"  {'─' * 70}")

    for ticker in sorted(filled['ticker'].unique()):
        tk_all = trades_dedup[trades_dedup['ticker'] == ticker]
        tk_filled = filled[filled['ticker'] == ticker]
        t_n = len(tk_all)
        t_f = len(tk_filled)
        t_w = (tk_filled['pnl_pct'] > 0).sum()
        t_l = (tk_filled['pnl_pct'] < 0).sum()
        t_wr = t_w / t_f * 100 if t_f > 0 else 0
        t_avg = tk_filled['pnl_pct'].mean()
        t_total = tk_filled['pnl_pct'].sum()
        print(f"  {ticker:<8} {t_n:>7} {t_f:>6} {t_w:>3} {t_l:>3} "
              f"{t_wr:>5.1f}% {t_avg:>+8.2%} {t_total:>+9.2%}")

    print(f"  {'─' * 70}")

    # ── Individual Trade Log ──
    print(f"\n  TRADE LOG (all filled trades)")
    print(f"  {'─' * 90}")
    print(f"  {'Date':<12} {'Ticker':<8} {'Prob':>5} {'Signal$':>9} {'Entry$':>9} "
          f"{'Exit$':>9} {'TP/SL':>8} {'PnL%':>8} {'Outcome':<10}")
    print(f"  {'─' * 90}")

    for _, t in filled.sort_values(['date', 'ticker']).iterrows():
        tp_sl = f"{t['tp_pct']:.0%}/{t['sl_pct']:.0%}"
        print(f"  {t['date']:<12} {t['ticker']:<8} {t['mean_prob']:>5.3f} "
              f"{t['signal_price']:>9.2f} {t['entry_price']:>9.2f} "
              f"{t['exit_price']:>9.2f} {tp_sl:>8} {t['pnl_pct']:>+7.2%} "
              f"{t['outcome']:<10}")

    print(f"  {'─' * 90}")

    # ── Unfilled signals ──
    if len(unfilled) > 0:
        print(f"\n  UNFILLED SIGNALS ({len(unfilled)})")
        print(f"  {'─' * 50}")
        for _, t in unfilled.sort_values(['date', 'ticker']).iterrows():
            print(f"  {t['date']:<12} {t['ticker']:<8} prob={t['mean_prob']:.3f} "
                  f"signal$={t['signal_price']:.2f} dip={t['dip_pct']:.1%}")

    # ── Model probability vs outcome ──
    print(f"\n  MODEL PROBABILITY vs OUTCOME")
    print(f"  {'─' * 50}")
    bins = [(0, 0.48), (0.48, 0.52), (0.52, 0.56), (0.56, 1.0)]
    for lo, hi in bins:
        subset = filled[(filled['mean_prob'] >= lo) & (filled['mean_prob'] < hi)]
        if len(subset) == 0:
            continue
        wr = (subset['pnl_pct'] > 0).sum() / len(subset) * 100
        avg = subset['pnl_pct'].mean()
        print(f"  prob [{lo:.2f}, {hi:.2f}): {len(subset):>3} trades, "
              f"WR={wr:.0f}%, avg PnL={avg:+.2%}")

# ==============================================================================
# STEP 6: What-if Market Order analysis (skip dip requirement)
# ==============================================================================
print(f"\n{'=' * W}")
print(f"  WHAT-IF: MARKET ORDER ENTRY (no dip filter)")
print(f"{'=' * W}")


def simulate_market_order(ohlcv_day, entry_minute, tp_pct, sl_pct, exit_time=EXIT_TIME_MIN):
    bars = ohlcv_day[ohlcv_day['time_min'] >= entry_minute].copy()
    if len(bars) < 2:
        return None

    signal_bar = bars.iloc[0]
    entry_price = signal_bar['Close']
    tp_target = entry_price * (1.0 + tp_pct)
    sl_stop = entry_price * (1.0 - sl_pct)

    for i in range(1, len(bars)):
        bar = bars.iloc[i]
        sl_hit = bar['Low'] <= sl_stop
        tp_hit = bar['High'] >= tp_target

        if sl_hit and tp_hit:
            pnl = -sl_pct if np.random.random() < 0.5 else tp_pct
            return {
                'entry_price': entry_price,
                'pnl_pct': pnl,
                'outcome': 'TP' if pnl > 0 else 'SL',
                'exit_price': entry_price * (1 + pnl),
                'exit_time_min': bar['time_min'],
            }
        if sl_hit:
            return {
                'entry_price': entry_price,
                'pnl_pct': -sl_pct,
                'outcome': 'SL',
                'exit_price': sl_stop,
                'exit_time_min': bar['time_min'],
            }
        if tp_hit:
            return {
                'entry_price': entry_price,
                'pnl_pct': tp_pct,
                'outcome': 'TP',
                'exit_price': tp_target,
                'exit_time_min': bar['time_min'],
            }
        if bar['time_min'] >= exit_time:
            pnl = (bar['Close'] - entry_price) / entry_price
            return {
                'entry_price': entry_price,
                'pnl_pct': pnl,
                'outcome': 'TIME_EXIT',
                'exit_price': bar['Close'],
                'exit_time_min': bar['time_min'],
            }

    last = bars.iloc[-1]
    pnl = (last['Close'] - entry_price) / entry_price
    return {
        'entry_price': entry_price,
        'pnl_pct': pnl,
        'outcome': 'EOD',
        'exit_price': last['Close'],
        'exit_time_min': last['time_min'],
    }


np.random.seed(42)
mkt_trades = []

# De-dup signals first: one per (date, ticker), highest prob
sig_dedup = signals_df.sort_values('mean_prob', ascending=False).drop_duplicates(
    subset=['score_date', 'Ticker'], keep='first')

for _, row in sig_dedup.iterrows():
    ticker = row['Ticker']
    date = row['score_date']
    date_str = str(date)

    cfg = deploy_configs.get(ticker)
    if cfg is None:
        continue
    if ticker not in ohlcv_cache:
        continue

    ohlcv = ohlcv_cache[ticker]
    date_int = int(date_str.replace('-', ''))
    day_bars = ohlcv[ohlcv['date_int'] == date_int]
    if len(day_bars) < 5:
        continue

    result = simulate_market_order(
        day_bars, cfg['entry_minute'],
        cfg['tp'], cfg['sl']
    )
    if result is None:
        continue

    mkt_trades.append({
        'date': date_str,
        'ticker': ticker,
        'mean_prob': row['mean_prob'],
        'entry_price': result['entry_price'],
        'exit_price': result['exit_price'],
        'tp_pct': cfg['tp'],
        'sl_pct': cfg['sl'],
        'pnl_pct': result['pnl_pct'],
        'outcome': result['outcome'],
    })

mkt_df = pd.DataFrame(mkt_trades)

if mkt_df.empty:
    print("  No market order trades to analyze.")
else:
    total = mkt_df['pnl_pct'].sum()
    avg = mkt_df['pnl_pct'].mean()
    n_w = (mkt_df['pnl_pct'] > 0).sum()
    n_l = (mkt_df['pnl_pct'] < 0).sum()
    wr = n_w / len(mkt_df) * 100

    print(f"\n  OVERALL (market order entry at signal bar close)")
    print(f"  {'─' * 50}")
    print(f"  Total trades:     {len(mkt_df)}")
    print(f"  Wins / Losses:    {n_w}W / {n_l}L")
    print(f"  Win rate:         {wr:.1f}%")
    print(f"  Total PnL:        {total:+.2%}")
    print(f"  Avg PnL/trade:    {avg:+.2%}")
    if n_w > 0:
        print(f"  Avg win:          {mkt_df[mkt_df['pnl_pct'] > 0]['pnl_pct'].mean():+.2%}")
    if n_l > 0:
        print(f"  Avg loss:         {mkt_df[mkt_df['pnl_pct'] < 0]['pnl_pct'].mean():+.2%}")

    print(f"\n  DAILY BREAKDOWN (market order)")
    print(f"  {'─' * 70}")
    print(f"  {'Date':<12} {'Trades':>6} {'W':>3} {'L':>3} {'WR%':>6} "
          f"{'Avg PnL':>9} {'Total PnL':>10} {'Cum PnL':>10}")
    print(f"  {'─' * 70}")
    cum = 0
    for date in sorted(mkt_df['date'].unique()):
        day = mkt_df[mkt_df['date'] == date]
        d_n = len(day)
        d_w = (day['pnl_pct'] > 0).sum()
        d_l = (day['pnl_pct'] < 0).sum()
        d_wr = d_w / d_n * 100 if d_n > 0 else 0
        d_avg = day['pnl_pct'].mean()
        d_tot = day['pnl_pct'].sum()
        cum += d_tot
        marker = "+++" if d_tot > 0.02 else ("---" if d_tot < -0.02 else "   ")
        print(f"  {date:<12} {d_n:>6} {d_w:>3} {d_l:>3} {d_wr:>5.1f}% "
              f"{d_avg:>+8.2%} {d_tot:>+9.2%} {cum:>+9.2%} {marker}")
    print(f"  {'─' * 70}")

    print(f"\n  TRADE LOG (market order)")
    print(f"  {'─' * 80}")
    print(f"  {'Date':<12} {'Ticker':<8} {'Prob':>5} {'Entry$':>9} {'Exit$':>9} "
          f"{'TP/SL':>8} {'PnL%':>8} {'Outcome':<10}")
    print(f"  {'─' * 80}")
    for _, t in mkt_df.sort_values(['date', 'ticker']).iterrows():
        tp_sl = f"{t['tp_pct']:.0%}/{t['sl_pct']:.0%}"
        print(f"  {t['date']:<12} {t['ticker']:<8} {t['mean_prob']:>5.3f} "
              f"{t['entry_price']:>9.2f} {t['exit_price']:>9.2f} "
              f"{tp_sl:>8} {t['pnl_pct']:>+7.2%} {t['outcome']:<10}")
    print(f"  {'─' * 80}")

    print(f"\n  PER-TICKER (market order)")
    print(f"  {'─' * 60}")
    for ticker in sorted(mkt_df['ticker'].unique()):
        tk = mkt_df[mkt_df['ticker'] == ticker]
        t_w = (tk['pnl_pct'] > 0).sum()
        t_l = (tk['pnl_pct'] < 0).sum()
        t_wr = t_w / len(tk) * 100
        print(f"  {ticker:<8} {len(tk):>3} trades  {t_w}W/{t_l}L  "
              f"WR={t_wr:.0f}%  avg={tk['pnl_pct'].mean():+.2%}  "
              f"total={tk['pnl_pct'].sum():+.2%}")
    print(f"  {'─' * 60}")

print(f"\n{'=' * W}")
print(f"  ANALYSIS COMPLETE")
print(f"{'=' * W}")
