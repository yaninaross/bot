"""
analyze_160_decay.py — Config Decay Analysis using Exp-160 (15 months)
======================================================================
Measures how WF-optimized configs decay over time.

Approach:
  1. Load ALL v9 configs for 160 tickers from BQ (not just deployed)
  2. Load ensemble predictions + OHLCV for each ticker
  3. Apply each ticker's optimized config (prob, entry_minute, TP/SL)
     to every 2-month window from Apr 2025 through Jun 2026
  4. Show PnL, WR, trade count per window — the decay curve

Model: 160_XSECTOR (training cutoff March 2025)
WF optimization window: R2 VAL = Dec 2025 - Jan 2026

Run in Colab.
"""

import gc, io, warnings
import pandas as pd, numpy as np, pytz
from google.cloud import storage as gcs_storage, bigquery
from numba import jit

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

warnings.filterwarnings('ignore')

PST = pytz.timezone('America/Los_Angeles')
BUCKET = os.environ.get("GCS_BUCKET", "")
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
EXIT_TIME_MIN = 780
SLIPPAGE_PCT = 0.0015

TRAINING_CUTOFF = '2025-03-01'

WINDOWS = [
    ('2025-04-01', '2025-05-31', 'Apr-May 25', 1),
    ('2025-06-01', '2025-07-31', 'Jun-Jul 25', 3),
    ('2025-08-01', '2025-09-30', 'Aug-Sep 25', 5),
    ('2025-10-01', '2025-11-30', 'Oct-Nov 25', 7),
    ('2025-12-01', '2026-01-31', 'Dec-Jan 26', 9),
    ('2026-02-01', '2026-03-31', 'Feb-Mar 26', 11),
    ('2026-04-01', '2026-06-03', 'Apr-Jun 26', 13),
]

REGIME_TP_SCALE = np.array([0.70, 0.85, 1.00, 1.30, 1.60])
REGIME_SL_SCALE = np.array([0.60, 0.80, 1.00, 1.30, 1.80])

gcs = gcs_storage.Client()
bucket_obj = gcs.bucket(BUCKET)
bq_client = bigquery.Client()

# ==============================================================================
# LOAD ALL v9 CONFIGS FOR 160 FROM BQ (not just S8_mix survivors)
# ==============================================================================
print("Loading ALL v9 configs for 160 from BQ...")
q = """
SELECT ticker, experiment, seed, tpsl_mode, prob_thresh, entry_minute,
       tp_pct, sl_pct, tp_mult, sl_mult, trail_start_mult, trail_end_mult,
       vix_threshold, tlt_crash, spy_ignore,
       val_a_avg_pnl, val_a_win_rate, val_a_count,
       val_b_avg_pnl, val_b_win_rate, val_b_count,
       survived_a, survived_b, is_ensemble
FROM `{GCP_PROJECT}.scoring_logs.wf_optimization_results`
WHERE model_version = 'robust_wf_v9'
  AND experiment LIKE '160%'
  AND survived_a = TRUE
  AND val_b_start = '2026-04-01'
"""
df_configs = bq_client.query(q).to_dataframe()

best_per_ticker = df_configs.loc[df_configs.groupby('ticker')['val_a_avg_pnl'].idxmax()]
print(f"  {len(df_configs)} total configs, {len(best_per_ticker)} tickers (best config each)")

# Also grab tickers that survived but weren't the best
all_survived = df_configs.drop_duplicates(subset=['ticker'], keep='first')
EXCLUDE = {'TLT', 'VIX', 'SMLR', 'SPIR', 'IWM', 'SPY'}
best_per_ticker = best_per_ticker[~best_per_ticker['ticker'].isin(EXCLUDE)]
print(f"  {len(best_per_ticker)} tickers after exclusions")

for _, r in best_per_ticker.iterrows():
    tp_str = f"TP={r['tp_pct']*100:.0f}%" if r['tp_pct'] > 0 else f"tp_m={r.get('tp_mult',0):.1f}x"
    sl_str = f"SL={r['sl_pct']*100:.0f}%" if r['sl_pct'] > 0 else f"sl_m={r.get('sl_mult',0):.1f}x"
    seed_str = 'ENS' if r['is_ensemble'] else f"S{int(r['seed'])}"
    print(f"    {r['ticker']:<7} {r['tpsl_mode']:<14} prob>{r['prob_thresh']:.2f}  em={int(r['entry_minute'])}  "
          f"{tp_str}  {sl_str}  val={r['val_a_avg_pnl']*100:+.1f}%  test={r['val_b_avg_pnl']*100:+.1f}%  {seed_str}")

# ==============================================================================
# LOAD ENSEMBLE PREDICTIONS
# ==============================================================================
print(f"\nLoading ensemble predictions for 160_XSECTOR...")
ensemble_probs = {}

for ticker in sorted(best_per_ticker['ticker'].unique()):
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    prefix = f"Algos/Results_all_v2/Universal_Exp160_XSECTOR/{clean}/Seed_"
    blobs = list(bucket_obj.list_blobs(prefix=prefix))

    prob_series = []
    for b in blobs:
        if not b.name.endswith('_Results.csv'):
            continue
        try:
            df_csv = pd.read_csv(io.BytesIO(b.download_as_bytes()),
                                 index_col=0, parse_dates=True)
            if df_csv.index.tz is None:
                df_csv.index = df_csv.index.tz_localize('UTC')
            df_csv.index = df_csv.index.tz_convert(PST)
            if 'Prob_Spike' in df_csv.columns:
                prob_series.append(df_csv['Prob_Spike'])
        except Exception:
            continue

    if len(prob_series) >= 3:
        ens = pd.concat(prob_series, axis=1).mean(axis=1).sort_index()
        ensemble_probs[ticker] = ens
        print(f"  {ticker}: {len(prob_series)} seeds, {len(ens):,} bars")
    else:
        print(f"  {ticker}: only {len(prob_series)} seeds — skipping")

print(f"  {len(ensemble_probs)} tickers with ensemble probs")

# ==============================================================================
# LOAD OHLCV (trimmed to start before first window)
# ==============================================================================
print(f"\nLoading OHLCV...")
ohlcv_cache = {}
trim = pd.Timestamp('2025-03-15', tz=PST)

for ticker in sorted(ensemble_probs.keys()):
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
        d = d[d.index >= trim]
        ohlcv_cache[ticker] = d[['Open','High','Low','Close','Volume']].copy()
        del d; gc.collect()
    except Exception:
        pass
print(f"  {len(ohlcv_cache)} tickers loaded")

# ==============================================================================
# TRADE SIMULATION
# ==============================================================================
@jit(nopython=True)
def sim_trade(prices, highs, lows, date_ints, time_mins, entry_idx, tp, sl, exit_time):
    n = len(prices)
    if entry_idx >= n - 1:
        return 0.0
    ep = prices[entry_idx]; ed = date_ints[entry_idx]
    tp_p = ep * (1 + tp); sl_p = ep * (1 - sl)
    for k in range(entry_idx + 1, n):
        if date_ints[k] != ed:
            return (prices[k-1] - ep) / ep
        if lows[k] <= sl_p and highs[k] >= tp_p:
            return -sl if np.random.random() < 0.5 else tp
        if lows[k] <= sl_p:
            return -sl
        if highs[k] >= tp_p:
            return tp
        if time_mins[k] >= exit_time:
            return (prices[k] - ep) / ep
    return (prices[-1] - ep) / ep

def compute_daily_atr_pct(d, period=14):
    daily = d.resample('D').agg({'High':'max','Low':'min','Close':'last'}).dropna()
    pc = daily['Close'].shift(1)
    tr = pd.concat([daily['High']-daily['Low'],(daily['High']-pc).abs(),(daily['Low']-pc).abs()],axis=1).max(axis=1)
    atr = tr.rolling(period).mean() / daily['Close']
    return atr.reindex(d.index, method='ffill').fillna(0.05)

def compute_adaptive_atr_pct(d, short=5, long=22, base=14):
    daily = d.resample('D').agg({'High':'max','Low':'min','Close':'last'}).dropna()
    pc = daily['Close'].shift(1)
    tr = pd.concat([daily['High']-daily['Low'],(daily['High']-pc).abs(),(daily['Low']-pc).abs()],axis=1).max(axis=1)
    atr_b = tr.rolling(base).mean()
    accel = (tr.rolling(short).mean() / tr.rolling(long).mean()).clip(0.5, 3.0)
    atr = (atr_b * accel) / daily['Close']
    return atr.reindex(d.index, method='ffill').fillna(0.05)

# ==============================================================================
# RUN TRADES PER WINDOW
# ==============================================================================
print(f"\nSimulating trades per 2-month window...")

all_results = []

for _, cfg in best_per_ticker.iterrows():
    ticker = cfg['ticker']
    if ticker not in ensemble_probs or ticker not in ohlcv_cache:
        continue

    ens = ensemble_probs[ticker]
    d = ohlcv_cache[ticker]
    mode = cfg['tpsl_mode']
    prob_thresh = float(cfg['prob_thresh'])
    em = int(cfg['entry_minute'])
    is_ens = bool(cfg['is_ensemble'])
    seed = int(cfg['seed'])

    prices = d['Close'].values
    highs = d['High'].values
    lows = d['Low'].values
    volumes = d['Volume'].values
    dates = (d.index.year*10000 + d.index.month*100 + d.index.day).values.astype(np.int64)
    times = (d.index.hour*60 + d.index.minute).values.astype(np.int64)

    common = d.index.intersection(ens.index)
    probs = np.zeros(len(d), dtype=np.float64)
    locs = d.index.get_indexer(common)
    probs[locs] = ens.loc[common].values

    atr_std = compute_daily_atr_pct(d)
    atr_adp = compute_adaptive_atr_pct(d)

    for w_start, w_end, w_label, w_months in WINDOWS:
        ws = pd.Timestamp(w_start, tz=PST)
        we = pd.Timestamp(w_end, tz=PST) + pd.Timedelta(days=1)

        w_mask = (d.index >= ws) & (d.index < we)
        w_idx = np.where(w_mask)[0]
        if len(w_idx) == 0:
            continue

        signal_mask = (probs >= prob_thresh) & (times == em) & np.array(w_mask)
        signal_mask = signal_mask & np.array([v > 0 for v in volumes])
        indices = np.where(signal_mask)[0]

        last_date = -1
        for idx in indices:
            d_int = int(dates[idx])
            if d_int == last_date:
                continue

            if mode in ('atr', 'trailing_decay'):
                atr_val = float(atr_std.iloc[idx]) if idx < len(atr_std) else 0.05
            elif mode == 'atr_adaptive':
                atr_val = float(atr_adp.iloc[idx]) if idx < len(atr_adp) else 0.05
            else:
                atr_val = 0.05
            if np.isnan(atr_val) or atr_val <= 0:
                atr_val = 0.05

            if mode in ('atr', 'atr_adaptive'):
                tp_m = float(cfg.get('tp_mult') or 2.0)
                sl_m = float(cfg.get('sl_mult') or 1.0)
                tp = min(max(atr_val * tp_m, 0.02), 0.60)
                sl = min(max(atr_val * sl_m, 0.01), 0.25)
            elif mode == 'regime_scaled':
                tp = float(cfg['tp_pct']) if cfg['tp_pct'] > 0 else 0.16
                sl = float(cfg['sl_pct']) if cfg['sl_pct'] > 0 else 0.08
            elif mode == 'trailing_decay':
                sl_m = float(cfg.get('sl_mult') or 2.0)
                sl = min(max(atr_val * sl_m, 0.01), 0.25)
                tp = 0.60
            else:
                tp = float(cfg['tp_pct']) if cfg['tp_pct'] > 0 else 0.16
                sl = float(cfg['sl_pct']) if cfg['sl_pct'] > 0 else 0.08

            pnl = sim_trade(prices, highs, lows, dates, times, idx, tp, sl, EXIT_TIME_MIN)
            pnl = float(pnl) - 2 * SLIPPAGE_PCT

            all_results.append({
                'ticker': ticker,
                'window': w_label,
                'months_from_cutoff': w_months,
                'date_int': d_int,
                'prob': float(probs[idx]),
                'pnl': pnl,
                'tp': tp,
                'sl': sl,
                'entry_price': float(prices[idx]),
                'mode': mode,
            })
            last_date = d_int

print(f"  {len(all_results)} total trades across all windows")

del ohlcv_cache, ensemble_probs; gc.collect()

rdf = pd.DataFrame(all_results)

# ==============================================================================
# DECAY TABLE
# ==============================================================================
W = 120
print(f"\n{'='*W}")
print(f"  CONFIG DECAY ANALYSIS — 160_XSECTOR")
print(f"  {len(best_per_ticker)} tickers, each using its best v9 WF-optimized config")
print(f"  Config was optimized on: Dec 2025 - Jan 2026 (R2 OPT)")
print(f"  Config was validated on: Feb - Mar 2026 (R3 VAL)")
print(f"{'='*W}")

window_stats = rdf.groupby(['window', 'months_from_cutoff']).agg(
    trades=('pnl', 'count'),
    avg_pnl=('pnl', 'mean'),
    win_rate=('pnl', lambda x: (x > 0).mean()),
    total_pnl=('pnl', 'sum'),
    tickers=('ticker', 'nunique'),
    avg_prob=('prob', 'mean'),
).reset_index().sort_values('months_from_cutoff')

print(f"\n  {'Window':<14} {'Mo':>3} {'Trades':>7} {'Tickers':>8} {'AvgPnL':>8} {'WR':>5} "
      f"{'TotPnL':>9} {'AvgProb':>8} {'Phase'}")
print(f"  " + "-" * 85)

for _, r in window_stats.iterrows():
    mo = r['months_from_cutoff']
    if mo <= 4:
        phase = 'R1 OPT (in-sample)'
    elif mo <= 6:
        phase = 'R1 VAL'
    elif mo <= 8:
        phase = 'R2 OPT'
    elif mo <= 10:
        phase = 'R2 VAL (config found here)'
    elif mo <= 12:
        phase = 'R3 VAL'
    else:
        phase = 'TEST (pure OOS)'
    print(f"  {r['window']:<14} {int(r['months_from_cutoff']):>3} {int(r['trades']):>7} "
          f"{int(r['tickers']):>8} {r['avg_pnl']*100:>+7.2f}% {r['win_rate']*100:>4.0f}% "
          f"{r['total_pnl']*100:>+8.1f}% {r['avg_prob']:>7.3f} {phase}")

# Before/after optimization
opt_window = rdf[rdf['months_from_cutoff'] <= 10]
post_opt = rdf[rdf['months_from_cutoff'] > 10]
if len(opt_window) > 10 and len(post_opt) > 10:
    print(f"\n  BEFORE CONFIG SELECTION (months 1-10, includes OPT+VAL data):")
    print(f"    {len(opt_window)} trades, avg={opt_window['pnl'].mean()*100:+.2f}%, "
          f"WR={(opt_window['pnl']>0).mean()*100:.0f}%")
    print(f"  AFTER CONFIG SELECTION (months 11+, true out-of-sample):")
    print(f"    {len(post_opt)} trades, avg={post_opt['pnl'].mean()*100:+.2f}%, "
          f"WR={(post_opt['pnl']>0).mean()*100:.0f}%")
    decay = opt_window['pnl'].mean() - post_opt['pnl'].mean()
    print(f"  DECAY: {decay*100:+.2f}% per trade")

# Linear regression on monthly avg PnL
x = window_stats['months_from_cutoff'].values.astype(float)
y = window_stats['avg_pnl'].values
if len(x) >= 4:
    coeffs = np.polyfit(x, y, 1)
    slope = coeffs[0]
    r_sq = 1 - np.sum((y - np.polyval(coeffs, x))**2) / np.sum((y - y.mean())**2)
    print(f"\n  LINEAR TREND (on window averages):")
    print(f"    Slope: {slope*100:+.4f}% per month")
    print(f"    R²: {r_sq:.3f}")
    if slope < 0 and y[0] > 0:
        half_life = (y[0] / 2) / abs(slope)
        zero_cross = y[0] / abs(slope)
        print(f"    Half-life: {half_life:.1f} months ({half_life*30:.0f} days)")
        print(f"    Zero-crossing: {zero_cross:.1f} months")

# ==============================================================================
# PER-TICKER DECAY
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  PER-TICKER DECAY — Early (months 1-6) vs Late (months 11+)")
print(f"{'='*W}")

ticker_decay = []
for ticker, group in rdf.groupby('ticker'):
    early = group[group['months_from_cutoff'] <= 6]
    late = group[group['months_from_cutoff'] >= 11]
    if len(early) < 2 or len(late) < 2:
        continue
    ticker_decay.append({
        'ticker': ticker,
        'early_pnl': early['pnl'].mean(),
        'early_wr': (early['pnl'] > 0).mean(),
        'early_n': len(early),
        'late_pnl': late['pnl'].mean(),
        'late_wr': (late['pnl'] > 0).mean(),
        'late_n': len(late),
        'delta': late['pnl'].mean() - early['pnl'].mean(),
    })

if ticker_decay:
    ticker_decay.sort(key=lambda x: x['delta'])
    n_decaying = sum(1 for td in ticker_decay if td['delta'] < -0.01)
    n_improving = sum(1 for td in ticker_decay if td['delta'] > 0.01)
    n_stable = len(ticker_decay) - n_decaying - n_improving

    print(f"\n  {'Ticker':<7} {'Early PnL':>10} {'EarlyWR':>8} {'(N)':>4} "
          f"{'Late PnL':>10} {'LateWR':>8} {'(N)':>4} {'Delta':>8} {'Status':>10}")
    print(f"  " + "-" * 80)
    for td in ticker_decay:
        status = 'DECAYING' if td['delta'] < -0.01 else ('IMPROVING' if td['delta'] > 0.01 else 'STABLE')
        print(f"  {td['ticker']:<7} {td['early_pnl']*100:>+9.2f}% {td['early_wr']*100:>7.0f}% {td['early_n']:>4} "
              f"{td['late_pnl']*100:>+9.2f}% {td['late_wr']*100:>7.0f}% {td['late_n']:>4} "
              f"{td['delta']*100:>+7.2f}% {status:>10}")

    print(f"\n  Summary: {n_decaying} decaying, {n_stable} stable, {n_improving} improving "
          f"out of {len(ticker_decay)} tickers")

# ==============================================================================
# CHART
# ==============================================================================
import matplotlib.pyplot as plt

print(f"\n  Generating decay chart...")

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle(f'Config Decay — 160_XSECTOR (March 2025 cutoff)\n'
             f'{len(best_per_ticker)} tickers, v9 optimized configs applied to each 2-month window',
             fontsize=13, fontweight='bold')

# Top: Avg PnL per window
ax = axes[0]
ax.bar(range(len(window_stats)), window_stats['avg_pnl'].values * 100,
       color=['#4CAF50' if v > 0 else '#F44336' for v in window_stats['avg_pnl'].values],
       alpha=0.7, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(window_stats)))
ax.set_xticklabels(window_stats['window'].values, rotation=30, ha='right')
ax.axhline(y=0, color='red', linewidth=0.5, linestyle='--')
ax.axhline(y=1.8, color='green', linewidth=0.5, linestyle='--', alpha=0.5, label='1.8% deploy threshold')
ax.axvspan(3.5, 4.5, alpha=0.15, color='blue', label='R2 VAL (config selected here)')
ax.set_ylabel('Avg PnL per trade (%)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
ax.set_title('Avg PnL by 2-Month Window')

# Bottom: Win rate + trade count
ax2 = axes[1]
ax2_twin = ax2.twinx()
ax2.bar(range(len(window_stats)), window_stats['win_rate'].values * 100,
        alpha=0.5, color='#2196F3', label='Win Rate')
ax2_twin.plot(range(len(window_stats)), window_stats['trades'].values,
              color='#FF9800', linewidth=2, marker='o', label='Trade Count')
ax2.set_xticks(range(len(window_stats)))
ax2.set_xticklabels(window_stats['window'].values, rotation=30, ha='right')
ax2.axhline(y=50, color='red', linewidth=0.5, linestyle='--')
ax2.set_ylabel('Win Rate (%)')
ax2_twin.set_ylabel('Trade Count')
ax2.legend(loc='upper left', fontsize=9)
ax2_twin.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_title('Win Rate & Trade Count by Window')

plt.tight_layout()
chart_path = '/tmp/config_decay_160.png'
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"  Charts saved: {chart_path}")
try:
    from IPython.display import display, Image
    display(Image(filename=chart_path))
except ImportError:
    plt.show()

print(f"\n{'='*W}")
