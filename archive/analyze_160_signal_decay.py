"""
analyze_160_signal_decay.py — Signal Quality Decay (TP/FP/FN/TN)
================================================================
Measures model prediction quality over time, NOT portfolio PnL.

For each 2-month window, computes:
  - True Positive:  model said spike (prob >= thresh) AND spike happened (max rise >= X%)
  - False Positive: model said spike BUT no spike happened
  - False Negative: model didn't signal BUT spike happened
  - True Negative:  model didn't signal AND no spike
  - Precision, Recall, F1

Spike definition: Future_Max >= entry_price * (1 + SPIKE_THRESHOLD)
Uses raw model probabilities — no TP/SL/config, just signal accuracy.

Run in Colab.
"""

import gc, io, warnings
import pandas as pd, numpy as np, pytz
from google.cloud import storage as gcs_storage

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

warnings.filterwarnings('ignore')

PST = pytz.timezone('America/Los_Angeles')
BUCKET = os.environ.get("GCS_BUCKET", "")
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

SPIKE_THRESHOLDS = [0.05, 0.08, 0.10]
PROB_THRESHOLDS = [0.40, 0.50, 0.60]
ENTRY_MINUTES = [360, 365, 370, 375, 380]

WINDOWS = [
    ('2025-04-01', '2025-05-31', 'Apr-May 25', 1),
    ('2025-06-01', '2025-07-31', 'Jun-Jul 25', 3),
    ('2025-08-01', '2025-09-30', 'Aug-Sep 25', 5),
    ('2025-10-01', '2025-11-30', 'Oct-Nov 25', 7),
    ('2025-12-01', '2026-01-31', 'Dec-Jan 26', 9),
    ('2026-02-01', '2026-03-31', 'Feb-Mar 26', 11),
    ('2026-04-01', '2026-06-03', 'Apr-Jun 26', 13),
]

EXCLUDE = {'TLT', 'VIX', 'SMLR', 'SPIR', 'IWM', 'SPY'}

gcs = gcs_storage.Client()
bucket_obj = gcs.bucket(BUCKET)

# ==============================================================================
# LOAD ALL TICKERS WITH SCORING CSVs
# ==============================================================================
print("Finding all tickers with 160_XSECTOR scoring CSVs...")
prefix = "Algos/Results_all_v2/Universal_Exp160_XSECTOR/"
all_blobs = list(bucket_obj.list_blobs(prefix=prefix))

ticker_blobs = {}
for b in all_blobs:
    parts = b.name.replace(prefix, '').split('/')
    if len(parts) >= 2 and parts[1].endswith('_Results.csv'):
        t = parts[0]
        if t not in EXCLUDE:
            ticker_blobs.setdefault(t, []).append(b.name)

print(f"  {len(ticker_blobs)} tickers found")

# ==============================================================================
# LOAD ENSEMBLE PROBS + OHLCV, COMPUTE SIGNALS PER BAR
# ==============================================================================
print("\nLoading data and computing per-bar signals...")
trim = pd.Timestamp('2025-03-15', tz=PST)

all_bars = []

for ticker in sorted(ticker_blobs.keys()):
    # Load ensemble probs
    prob_series = []
    for blob_path in ticker_blobs[ticker]:
        try:
            df_csv = pd.read_csv(io.BytesIO(bucket_obj.blob(blob_path).download_as_bytes()),
                                 index_col=0, parse_dates=True)
            if df_csv.index.tz is None:
                df_csv.index = df_csv.index.tz_localize('UTC')
            df_csv.index = df_csv.index.tz_convert(PST)
            if 'Prob_Spike' in df_csv.columns:
                prob_series.append(df_csv['Prob_Spike'])
        except Exception:
            continue

    if len(prob_series) < 3:
        continue

    ens = pd.concat(prob_series, axis=1).mean(axis=1).sort_index()

    # Load OHLCV
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
    except Exception:
        continue

    # For each entry bar, compute: did a spike actually happen?
    dates = (d.index.year*10000 + d.index.month*100 + d.index.day).values.astype(np.int64)
    times = (d.index.hour*60 + d.index.minute).values.astype(np.int64)
    prices = d['Close'].values
    highs = d['High'].values
    volumes = d['Volume'].values

    common = d.index.intersection(ens.index)
    probs = np.zeros(len(d), dtype=np.float64)
    locs = d.index.get_indexer(common)
    probs[locs] = ens.loc[common].values

    for em in ENTRY_MINUTES:
        em_mask = (times == em) & np.array([v > 0 for v in volumes])
        em_indices = np.where(em_mask)[0]

        last_date = -1
        for idx in em_indices:
            d_int = int(dates[idx])
            if d_int == last_date:
                continue
            last_date = d_int

            entry_price = float(prices[idx])
            if entry_price <= 0:
                continue

            # Find max high after entry on the same day
            after_mask = (dates == d_int) & (np.arange(len(prices)) > idx)
            after_idx = np.where(after_mask)[0]
            if len(after_idx) == 0:
                continue

            max_high = float(highs[after_idx].max())
            max_rise = (max_high - entry_price) / entry_price

            bar_date = d.index[idx]
            months_from_cutoff = (bar_date.year - 2025) * 12 + (bar_date.month - 3)

            # Find which window
            w_label = None
            for w_start, w_end, wl, wm in WINDOWS:
                ws = pd.Timestamp(w_start, tz=PST)
                we = pd.Timestamp(w_end, tz=PST) + pd.Timedelta(days=1)
                if ws <= bar_date < we:
                    w_label = wl
                    break

            if w_label is None:
                continue

            all_bars.append({
                'ticker': ticker,
                'date_int': d_int,
                'entry_minute': em,
                'entry_price': entry_price,
                'prob': float(probs[idx]),
                'max_rise': max_rise,
                'window': w_label,
                'months_from_cutoff': months_from_cutoff,
            })

    del d, ens; gc.collect()
    print(f"  {ticker}: {len([b for b in all_bars if b['ticker']==ticker])} entry bars")

bdf = pd.DataFrame(all_bars)
print(f"\n  Total: {len(bdf)} entry bars across {bdf['ticker'].nunique()} tickers")

# ==============================================================================
# COMPUTE TP/FP/FN/TN PER WINDOW
# ==============================================================================
W = 130

for spike_thresh in SPIKE_THRESHOLDS:
    bdf['actual_spike'] = bdf['max_rise'] >= spike_thresh

    for prob_thresh in PROB_THRESHOLDS:
        bdf['predicted_spike'] = bdf['prob'] >= prob_thresh

        bdf['TP'] = bdf['predicted_spike'] & bdf['actual_spike']
        bdf['FP'] = bdf['predicted_spike'] & ~bdf['actual_spike']
        bdf['FN'] = ~bdf['predicted_spike'] & bdf['actual_spike']
        bdf['TN'] = ~bdf['predicted_spike'] & ~bdf['actual_spike']

        print(f"\n{'='*W}")
        print(f"  SIGNAL QUALITY — Spike >= {spike_thresh*100:.0f}%  |  Prob >= {prob_thresh:.0%}")
        print(f"  160_XSECTOR  |  {bdf['ticker'].nunique()} tickers  |  {len(bdf)} entry bars")
        print(f"{'='*W}")

        window_order = [w[2] for w in WINDOWS]
        window_groups = bdf.groupby('window')

        print(f"\n  {'Window':<14} {'Bars':>6} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5} "
              f"{'Prec':>6} {'Recall':>7} {'F1':>5} {'Spike%':>7} {'Signal%':>8}")
        print(f"  " + "-" * 90)

        window_metrics = []
        for wl in window_order:
            if wl not in window_groups.groups:
                continue
            g = window_groups.get_group(wl)
            tp = g['TP'].sum()
            fp = g['FP'].sum()
            fn = g['FN'].sum()
            tn = g['TN'].sum()
            prec = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * prec * recall / max(prec + recall, 1e-6)
            spike_rate = (tp + fn) / len(g)
            signal_rate = (tp + fp) / len(g)

            mo = next(w[3] for w in WINDOWS if w[2] == wl)
            window_metrics.append({
                'window': wl, 'months': mo,
                'precision': prec, 'recall': recall, 'f1': f1,
                'spike_rate': spike_rate, 'signal_rate': signal_rate,
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'bars': len(g),
            })

            print(f"  {wl:<14} {len(g):>6} {tp:>5} {fp:>5} {fn:>5} {tn:>5} "
                  f"{prec*100:>5.1f}% {recall*100:>6.1f}% {f1*100:>4.1f}% "
                  f"{spike_rate*100:>6.1f}% {signal_rate*100:>7.1f}%")

        if window_metrics:
            wm = pd.DataFrame(window_metrics)

            # Precision trend
            x = wm['months'].values.astype(float)
            y_prec = wm['precision'].values
            y_recall = wm['recall'].values

            if len(x) >= 4:
                c_prec = np.polyfit(x, y_prec, 1)
                c_recall = np.polyfit(x, y_recall, 1)
                r2_prec = 1 - np.sum((y_prec - np.polyval(c_prec, x))**2) / max(np.sum((y_prec - y_prec.mean())**2), 1e-10)
                r2_recall = 1 - np.sum((y_recall - np.polyval(c_recall, x))**2) / max(np.sum((y_recall - y_recall.mean())**2), 1e-10)

                print(f"\n  PRECISION TREND: {c_prec[0]*100:+.3f}% per month (R²={r2_prec:.3f})")
                print(f"  RECALL TREND:    {c_recall[0]*100:+.3f}% per month (R²={r2_recall:.3f})")

                if c_prec[0] < -0.005:
                    init = y_prec[0]
                    hl = (init / 2) / abs(c_prec[0]) if init > 0 else float('inf')
                    print(f"  PRECISION HALF-LIFE: {hl:.1f} months ({hl*30:.0f} days)")
                else:
                    print(f"  PRECISION: STABLE or improving")

            # Spike rate trend (market regime change)
            y_spike = wm['spike_rate'].values
            c_spike = np.polyfit(x, y_spike, 1)
            print(f"  SPIKE RATE TREND: {c_spike[0]*100:+.3f}% per month "
                  f"({'fewer spikes over time' if c_spike[0] < 0 else 'more spikes over time'})")

# ==============================================================================
# AGGREGATE: Best summary across all thresholds
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  SUMMARY — PRECISION DECAY ACROSS THRESHOLDS")
print(f"{'='*W}")
print(f"\n  {'Spike':>6} {'Prob':>5} {'Early Prec':>11} {'Late Prec':>10} {'Delta':>7} "
      f"{'Early Recall':>13} {'Late Recall':>12} {'Verdict':>12}")
print(f"  " + "-" * 80)

for spike_thresh in SPIKE_THRESHOLDS:
    bdf['actual_spike'] = bdf['max_rise'] >= spike_thresh
    for prob_thresh in PROB_THRESHOLDS:
        bdf['predicted_spike'] = bdf['prob'] >= prob_thresh
        bdf['TP'] = bdf['predicted_spike'] & bdf['actual_spike']
        bdf['FP'] = bdf['predicted_spike'] & ~bdf['actual_spike']
        bdf['FN'] = ~bdf['predicted_spike'] & bdf['actual_spike']

        early = bdf[bdf['months_from_cutoff'] <= 6]
        late = bdf[bdf['months_from_cutoff'] >= 11]

        e_tp, e_fp = early['TP'].sum(), early['FP'].sum()
        l_tp, l_fp = late['TP'].sum(), late['FP'].sum()
        e_fn, l_fn = early['FN'].sum(), late['FN'].sum()

        e_prec = e_tp / max(e_tp + e_fp, 1)
        l_prec = l_tp / max(l_tp + l_fp, 1)
        e_recall = e_tp / max(e_tp + e_fn, 1)
        l_recall = l_tp / max(l_tp + l_fn, 1)

        delta_prec = l_prec - e_prec
        verdict = 'DECAYING' if delta_prec < -0.05 else ('IMPROVING' if delta_prec > 0.05 else 'STABLE')

        print(f"  {spike_thresh*100:>5.0f}% {prob_thresh:>4.0%} {e_prec*100:>10.1f}% {l_prec*100:>9.1f}% "
              f"{delta_prec*100:>+6.1f}% {e_recall*100:>12.1f}% {l_recall*100:>11.1f}% {verdict:>12}")

# ==============================================================================
# CHART
# ==============================================================================
import matplotlib.pyplot as plt

print(f"\n  Generating charts...")

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle(f'Signal Quality Decay — 160_XSECTOR (March 2025 cutoff)\n'
             f'{bdf["ticker"].nunique()} tickers, ensemble predictions',
             fontsize=13, fontweight='bold')

colors = ['#2196F3', '#4CAF50', '#F44336']
window_labels = [w[2] for w in WINDOWS]

# Top: Precision per window
ax = axes[0]
spike_t = 0.08
bdf['actual_spike'] = bdf['max_rise'] >= spike_t
for i, pt in enumerate(PROB_THRESHOLDS):
    bdf['predicted_spike'] = bdf['prob'] >= pt
    precs = []
    for wl in window_labels:
        g = bdf[bdf['window'] == wl]
        tp = (g['predicted_spike'] & g['actual_spike']).sum()
        fp = (g['predicted_spike'] & ~g['actual_spike']).sum()
        precs.append(tp / max(tp + fp, 1))
    ax.plot(range(len(window_labels)), [p*100 for p in precs], color=colors[i],
            linewidth=2, marker='o', label=f'Prob >= {pt:.0%}')
ax.set_xticks(range(len(window_labels)))
ax.set_xticklabels(window_labels, rotation=30, ha='right')
ax.set_ylabel('Precision (%)')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_title(f'Precision Decay (spike >= {spike_t*100:.0f}%)')

# Bottom: Recall per window
ax2 = axes[1]
for i, pt in enumerate(PROB_THRESHOLDS):
    bdf['predicted_spike'] = bdf['prob'] >= pt
    recalls = []
    for wl in window_labels:
        g = bdf[bdf['window'] == wl]
        tp = (g['predicted_spike'] & g['actual_spike']).sum()
        fn = (~g['predicted_spike'] & g['actual_spike']).sum()
        recalls.append(tp / max(tp + fn, 1))
    ax2.plot(range(len(window_labels)), [r*100 for r in recalls], color=colors[i],
             linewidth=2, marker='s', label=f'Prob >= {pt:.0%}')
ax2.set_xticks(range(len(window_labels)))
ax2.set_xticklabels(window_labels, rotation=30, ha='right')
ax2.set_ylabel('Recall (%)')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_title(f'Recall Decay (spike >= {spike_t*100:.0f}%)')

plt.tight_layout()
chart_path = '/tmp/signal_decay_160.png'
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"  Charts saved: {chart_path}")
try:
    from IPython.display import display, Image
    display(Image(filename=chart_path))
except ImportError:
    plt.show()

print(f"\n{'='*W}")
