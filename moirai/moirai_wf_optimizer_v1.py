# ==============================================================================
# MOIRAI-Large Direct Signal WF Optimizer — S3-style family agreement
# ==============================================================================
# Single Colab cell. GPU not required.
#
# Loads v1 path validation summary, rejoins forward OHLCV bars from
# master_inference, runs per-(ticker, signal_family) grid sweep, then applies
# S3-style selection: family agreement gate, overfit filter, val_a^2 * agree
# scoring. Outputs deploy-ready configs, charts, GCS upload.
#
# Signal families (analogous to S3 experiment families):
#   spike_prob_2pct, spike_prob_3pct, spike_prob_4pct, spike_prob_5pct, direction_pct
#
# Selection logic (mirrors S3):
#   1. survived_a: val_a_avg_pnl > 0
#   2. Overfit filter: opt/val_a ratio <= 3.0
#   3. Best config per (ticker, signal_col family)
#   4. Family agreement: majority of families must have val_b > 0
#   5. Gate: agreement > 50% AND n_families >= 2
#   6. Score: val_a^2 * agreement_rate, pick best family per ticker
#
# Input:  gs://time-series-bucket-ys/Algos/MOIRAI/paths/path_validation_summary.parquet
# Output: gs://time-series-bucket-ys/Algos/MOIRAI/optimizer_v1/
# ==============================================================================

import os, time, gc
import numpy as np
import pandas as pd
import pytz
import matplotlib
import matplotlib.pyplot as plt

# ── CONFIG ────────────────────────────────────────────────────────────────────

BUCKET_NAME = 'time-series-bucket-ys'
GCS_SUMMARY = 'Algos/MOIRAI/paths/path_validation_summary.parquet'
GCS_INFER   = 'Data/Inference_v2/{ticker}_master_inference.parquet'
GCS_OUTPUT  = 'Algos/MOIRAI/optimizer_v1'

LOCAL_DIR   = '/tmp/moirai_optimizer_v1'
os.makedirs(LOCAL_DIR, exist_ok=True)

ENTRY_MINUTE   = 390
TRADE_WINDOW   = 60
PST = pytz.timezone('America/Los_Angeles')

WF_WINDOWS = {
    'OPT':   ('2026-03-01', '2026-04-30'),
    'VAL_A': ('2026-05-01', '2026-06-15'),
    'VAL_B': ('2026-06-16', '2026-06-30'),
    'TEST':  ('2026-07-01', '2026-07-31'),
}

SIGNAL_COLS = [
    'spike_prob_2pct', 'spike_prob_3pct', 'spike_prob_4pct',
    'spike_prob_5pct', 'direction_pct',
]

THRESHOLDS = {
    'spike_prob_2pct': [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
    'spike_prob_3pct': [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
    'spike_prob_4pct': [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
    'spike_prob_5pct': [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50],
    'direction_pct':   [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
}

TP_PCT   = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]
SL_PCT   = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
MAX_HOLD = [15, 30, 45, 60]

MIN_TRADES_PER_MONTH = 2.0
DEPLOY_THRESHOLD     = 0.018
OVERFIT_RATIO_MAX    = 3.0
MIN_FAMILIES         = 2

EXCLUDE_TICKERS = {'SPY', 'VIX', 'TLT', 'IWM', 'XLE', 'ITA'}

S3_TICKERS = {'APLD', 'CIFR', 'CLSK', 'FLNC', 'HUT', 'KEEL', 'PL', 'RCAT', 'RIOT'}


# ── DATA LOADING ──────────────────────────────────────────────────────────────

def load_summary_from_gcs():
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    local_path = f'{LOCAL_DIR}/path_validation_summary.parquet'
    if not os.path.exists(local_path):
        blob = bucket.blob(GCS_SUMMARY)
        blob.download_to_filename(local_path)
        print(f"Downloaded summary from GCS ({blob.size/1e6:.1f} MB)")
    df = pd.read_parquet(local_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    print(f"Loaded {len(df)} predictions for {df['ticker'].nunique()} tickers")
    return df


def load_master_inference(ticker):
    from google.cloud import storage
    local_path = f'/tmp/{ticker}_master_inference.parquet'
    if not os.path.exists(local_path):
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(GCS_INFER.format(ticker=ticker))
        if not blob.exists():
            return None
        blob.download_to_filename(local_path)
    df = pd.read_parquet(local_path)
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        df = df.set_index('datetime')
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    return df


def attach_forward_bars(summary_df):
    fwd_highs, fwd_lows, fwd_closes = [], [], []
    for ticker in sorted(summary_df['ticker'].unique()):
        if ticker in EXCLUDE_TICKERS:
            continue
        tdf = summary_df[summary_df['ticker'] == ticker]
        master = load_master_inference(ticker)
        if master is None:
            fwd_highs.extend([None] * len(tdf))
            fwd_lows.extend([None] * len(tdf))
            fwd_closes.extend([None] * len(tdf))
            continue
        master_pt = master.copy()
        if master_pt.index.tz is not None:
            master_pt.index = master_pt.index.tz_convert('America/Los_Angeles')
        else:
            master_pt.index = master_pt.index.tz_localize('America/Los_Angeles')
        highs = master_pt['High'].values.astype(np.float64)
        lows = master_pt['Low'].values.astype(np.float64)
        closes = master_pt['Close'].values.astype(np.float64)
        ts_to_idx = {ts: i for i, ts in enumerate(master_pt.index)}
        for _, row in tdf.iterrows():
            dt = row['datetime']
            if dt.tzinfo is None:
                dt = dt.tz_localize('America/Los_Angeles')
            elif str(dt.tzinfo) != 'America/Los_Angeles':
                dt = dt.tz_convert('America/Los_Angeles')
            idx = ts_to_idx.get(dt)
            if idx is None or idx + 1 + TRADE_WINDOW > len(highs):
                fwd_highs.append(None)
                fwd_lows.append(None)
                fwd_closes.append(None)
                continue
            fwd_highs.append(highs[idx+1:idx+1+TRADE_WINDOW].copy())
            fwd_lows.append(lows[idx+1:idx+1+TRADE_WINDOW].copy())
            fwd_closes.append(closes[idx+1:idx+1+TRADE_WINDOW].copy())
        del master, master_pt
        gc.collect()
    return fwd_highs, fwd_lows, fwd_closes


# ── TRADE SIMULATION ─────────────────────────────────────────────────────────

def simulate_trade(entry_price, fwd_highs, fwd_lows, fwd_closes,
                   tp_pct, sl_pct, max_hold):
    tp_price = entry_price * (1 + tp_pct / 100)
    sl_price = entry_price * (1 - sl_pct / 100)
    for bar in range(min(max_hold, len(fwd_highs))):
        if fwd_lows[bar] <= sl_price:
            return -sl_pct / 100
        if fwd_highs[bar] >= tp_price:
            return tp_pct / 100
    exit_price = fwd_closes[min(max_hold - 1, len(fwd_closes) - 1)]
    return (exit_price - entry_price) / entry_price


def simulate_trades_for_df(trade_df, tp, sl, mh):
    pnls = []
    for _, row in trade_df.iterrows():
        pnls.append(simulate_trade(
            row['entry_price'], row['fwd_highs'], row['fwd_lows'],
            row['fwd_closes'], tp, sl, mh))
    return pnls


def composite_score(pnls, months):
    if len(pnls) == 0:
        return -999.0
    trades_per_month = len(pnls) / max(months, 0.1)
    if trades_per_month < MIN_TRADES_PER_MONTH:
        return -999.0
    avg_pnl = np.mean(pnls)
    win_rate = np.sum(np.array(pnls) > 0) / len(pnls) * 100.0
    return avg_pnl * 0.6 + (win_rate - 50) * 0.02 + min(trades_per_month, 8) * 0.05


def period_months(start_str, end_str):
    return (pd.Timestamp(end_str) - pd.Timestamp(start_str)).days / 30.44


def get_dates(ticker_df):
    dt_col = ticker_df['datetime']
    if dt_col.dt.tz is not None:
        return dt_col.dt.tz_localize(None).dt.date
    return dt_col.dt.date


def get_period_mask(dates, start_str, end_str):
    return (dates >= pd.Timestamp(start_str).date()) & \
           (dates <= pd.Timestamp(end_str).date())


def eval_config_on_period(ticker_df, dates, sig_col, thresh, tp, sl, mh,
                           start_str, end_str):
    p_months = period_months(start_str, end_str)
    p_mask = get_period_mask(dates, start_str, end_str)
    p_df = ticker_df[p_mask]
    p_signal = p_df[p_df[sig_col] >= thresh]
    if len(p_signal) == 0:
        return {'avg_pnl': np.nan, 'win_rate': np.nan, 'trades': 0, 'tpm': 0.0}
    pnls = simulate_trades_for_df(p_signal, tp, sl, mh)
    return {
        'avg_pnl': np.mean(pnls),
        'win_rate': np.sum(np.array(pnls) > 0) / len(pnls) * 100,
        'trades': len(pnls),
        'tpm': len(pnls) / max(p_months, 0.1),
    }


# ── PER-FAMILY OPTIMIZER ─────────────────────────────────────────────────────

def run_family_optimizer(ticker_df, ticker):
    opt_start, opt_end = WF_WINDOWS['OPT']
    opt_months = period_months(opt_start, opt_end)
    dates = get_dates(ticker_df)
    opt_mask = get_period_mask(dates, opt_start, opt_end)
    opt_df = ticker_df[opt_mask]

    if len(opt_df) < 5:
        return []

    family_results = []

    for sig_col in SIGNAL_COLS:
        if sig_col not in opt_df.columns:
            continue

        best_score = -999.0
        best_config = None

        for thresh in THRESHOLDS.get(sig_col, []):
            signal_df = opt_df[opt_df[sig_col] >= thresh]
            if len(signal_df) < 3:
                continue
            for tp in TP_PCT:
                for sl in SL_PCT:
                    if sl > tp:
                        continue
                    for mh in MAX_HOLD:
                        pnls = simulate_trades_for_df(signal_df, tp, sl, mh)
                        score = composite_score(pnls, opt_months)
                        if score > best_score:
                            best_score = score
                            best_config = {
                                'ticker': ticker,
                                'signal_col': sig_col,
                                'threshold': thresh,
                                'tp_pct': tp, 'sl_pct': sl, 'max_hold': mh,
                                'opt_score': score,
                                'opt_avg_pnl': np.mean(pnls),
                                'opt_win_rate': np.sum(np.array(pnls) > 0) / len(pnls) * 100,
                                'opt_trades': len(pnls),
                            }

        if best_config is None or best_config['opt_score'] <= -999:
            continue

        sig = best_config['signal_col']
        th = best_config['threshold']
        tp = best_config['tp_pct']
        sl = best_config['sl_pct']
        mh = best_config['max_hold']

        for period_name in ('VAL_A', 'VAL_B', 'TEST'):
            p_start, p_end = WF_WINDOWS[period_name]
            stats = eval_config_on_period(ticker_df, dates, sig, th, tp, sl, mh,
                                           p_start, p_end)
            pn = period_name.lower()
            best_config[f'{pn}_avg_pnl'] = stats['avg_pnl']
            best_config[f'{pn}_win_rate'] = stats['win_rate']
            best_config[f'{pn}_trades'] = stats['trades']
            best_config[f'{pn}_tpm'] = stats['tpm']

        # Threshold stability: count adjacent thresholds with positive val_a
        thresh_list = THRESHOLDS.get(sig, [])
        thresh_idx = thresh_list.index(th) if th in thresh_list else -1
        stable_count, stable_total = 0, 0
        for offset in [-2, -1, 1, 2]:
            ni = thresh_idx + offset
            if 0 <= ni < len(thresh_list):
                stable_total += 1
                nb_stats = eval_config_on_period(
                    ticker_df, dates, sig, thresh_list[ni], tp, sl, mh,
                    *WF_WINDOWS['VAL_A'])
                if not np.isnan(nb_stats['avg_pnl']) and nb_stats['avg_pnl'] > 0:
                    stable_count += 1
        best_config['thresh_stability'] = stable_count / max(stable_total, 1)

        family_results.append(best_config)

    return family_results


# ── S3-STYLE FAMILY AGREEMENT SELECTION ───────────────────────────────────────

def s3_family_selection(family_df):
    # Step 1: survived_a
    survived = family_df[family_df['val_a_avg_pnl'] > 0].copy()
    print(f"\n  Step 1 — Survived VAL_A (val_a > 0): {len(survived)} configs "
          f"({survived['ticker'].nunique()} tickers)")

    # Step 2: overfit filter
    survived['opt_val_ratio'] = survived['opt_avg_pnl'] / survived['val_a_avg_pnl'].clip(lower=1e-6)
    survived = survived[survived['opt_val_ratio'] <= OVERFIT_RATIO_MAX]
    print(f"  Step 2 — Overfit filter (ratio <= {OVERFIT_RATIO_MAX}): "
          f"{len(survived)} configs ({survived['ticker'].nunique()} tickers)")

    # Step 3: best per (ticker, family)
    best_per_family = survived.loc[
        survived.groupby(['ticker', 'signal_col'])['val_a_avg_pnl'].idxmax()
    ].copy()
    print(f"  Step 3 — Best per (ticker, family): {len(best_per_family)} entries")

    # Step 4-6: family agreement + selection
    ticker_selections = []
    for ticker in sorted(best_per_family['ticker'].unique()):
        t_fams = best_per_family[best_per_family['ticker'] == ticker]
        n_fams = len(t_fams)

        val_b_pos = t_fams['val_b_avg_pnl'].apply(
            lambda x: x > 0 if not np.isnan(x) else False)
        n_val_b_pos = val_b_pos.sum()
        agreement = n_val_b_pos / n_fams if n_fams > 0 else 0
        passes_gate = (agreement > 0.5) and (n_fams >= MIN_FAMILIES)

        t_fams = t_fams.copy()
        t_fams['selection_score'] = t_fams['val_a_avg_pnl'] ** 2 * agreement

        best_row = t_fams.loc[t_fams['selection_score'].idxmax()].copy()
        best_row['n_families'] = n_fams
        best_row['n_val_b_pos'] = n_val_b_pos
        best_row['agreement'] = agreement
        best_row['passes_gate'] = passes_gate

        fam_details = []
        for _, fr in t_fams.iterrows():
            vb = fr['val_b_avg_pnl']
            vb_str = f"{vb*100:+.1f}%" if not np.isnan(vb) else "N/A"
            fam_details.append(f"{fr['signal_col']}({vb_str})")
        best_row['family_details'] = ' | '.join(fam_details)
        ticker_selections.append(best_row)

    return pd.DataFrame(ticker_selections)


# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════

print("Loading v1 summary...")
t_total = time.time()
summary = load_summary_from_gcs()
summary = summary[~summary['ticker'].isin(EXCLUDE_TICKERS)].reset_index(drop=True)
print(f"After excluding indices/ETFs: {len(summary)} predictions, "
      f"{summary['ticker'].nunique()} tickers")

print("\nAttaching forward OHLCV bars...")
fwd_h, fwd_l, fwd_c = attach_forward_bars(summary)
summary['fwd_highs'] = fwd_h
summary['fwd_lows'] = fwd_l
summary['fwd_closes'] = fwd_c

valid_mask = summary['fwd_highs'].notna()
print(f"Valid predictions with forward bars: {valid_mask.sum()}/{len(summary)}")
df = summary[valid_mask].reset_index(drop=True).copy()
print(f"Ready: {len(df)} predictions, {df['ticker'].nunique()} tickers "
      f"({time.time()-t_total:.0f}s)")

# ── Per-family optimization ──
print(f"\n{'='*90}")
print("Running per-family WF optimization (S3-style)...")
print(f"Grid: {len(SIGNAL_COLS)} families × thresholds × "
      f"{len(TP_PCT)} TP × {len(SL_PCT)} SL × {len(MAX_HOLD)} hold")

t0 = time.time()
all_family_results = []
tickers = sorted(df['ticker'].unique())

for i, ticker in enumerate(tickers):
    ticker_df = df[df['ticker'] == ticker].copy()
    all_family_results.extend(run_family_optimizer(ticker_df, ticker))
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(tickers)} tickers")

family_df = pd.DataFrame(all_family_results)
print(f"\nFamily optimization: {len(family_df)} configs across "
      f"{family_df['ticker'].nunique()} tickers ({time.time()-t0:.0f}s)")

# ── S3-style selection ──
print(f"\n{'='*90}")
print("S3-STYLE FAMILY AGREEMENT SELECTION")
print(f"{'='*90}")

results_df = s3_family_selection(family_df)
results_df = results_df.sort_values('val_a_avg_pnl', ascending=False).reset_index(drop=True)

# ── All tickers ──
print(f"\n  ALL TICKERS (sorted by VAL_A avg PnL)")
print(f"  {'─'*100}")
display_cols = [
    'ticker', 'signal_col', 'threshold', 'tp_pct', 'sl_pct', 'max_hold',
    'opt_avg_pnl', 'val_a_avg_pnl', 'val_a_trades',
    'val_b_avg_pnl', 'test_avg_pnl', 'test_trades',
    'n_families', 'agreement', 'passes_gate',
]
avail = [c for c in display_cols if c in results_df.columns]
fmt = {
    'opt_avg_pnl': '{:.4f}'.format, 'val_a_avg_pnl': '{:.4f}'.format,
    'val_b_avg_pnl': '{:.4f}'.format, 'test_avg_pnl': '{:.4f}'.format,
    'threshold': '{:.2f}'.format, 'agreement': '{:.0%}'.format,
}
print(results_df[avail].to_string(index=False, formatters=fmt))

# ── Family details ──
print(f"\n\n  FAMILY AGREEMENT DETAILS")
print(f"  {'─'*100}")
for _, row in results_df.iterrows():
    gate = "PASS" if row.get('passes_gate', False) else "FAIL"
    print(f"  {row['ticker']:6s} [{gate}] n_fam={int(row['n_families'])} "
          f"agree={row['agreement']:.0%}  → {row.get('family_details', '')}")

# ── Deploy-ready ──
deployed = results_df[results_df['passes_gate'] == True]
print(f"\n\n  DEPLOY-READY (gate: majority val_b > 0, n_fam >= {MIN_FAMILIES})")
print(f"  {'─'*100}")

if len(deployed) > 0:
    deploy_cols = [
        'ticker', 'signal_col', 'threshold', 'tp_pct', 'sl_pct', 'max_hold',
        'opt_avg_pnl', 'opt_win_rate', 'val_a_avg_pnl', 'val_a_win_rate',
        'val_a_trades', 'val_b_avg_pnl', 'test_avg_pnl', 'test_win_rate',
        'test_trades', 'n_families', 'agreement', 'thresh_stability',
    ]
    avail_d = [c for c in deploy_cols if c in deployed.columns]
    print(deployed[avail_d].to_string(index=False, formatters=fmt))
    print(f"\n  Deployed:       {len(deployed)} tickers")
    print(f"  Avg VAL_A PnL:  {deployed['val_a_avg_pnl'].mean()*100:.2f}%")
    test_valid = deployed[deployed['test_avg_pnl'].notna()]
    if len(test_valid) > 0:
        print(f"  Avg TEST PnL:   {test_valid['test_avg_pnl'].mean()*100:.2f}%")
    print(f"  Avg agreement:  {deployed['agreement'].mean():.0%}")
    if 'thresh_stability' in deployed.columns:
        print(f"  Avg thresh stab:{deployed['thresh_stability'].mean():.0%}")
else:
    print("  No tickers pass the family agreement gate.")

# ── Signal column ranking ──
print(f"\n\n  SIGNAL COLUMN RANKING")
print(f"  {'─'*50}")
src = deployed if len(deployed) > 0 else results_df
sig_counts = src['signal_col'].value_counts()
for sig, cnt in sig_counts.items():
    print(f"    {sig:25s}  count: {cnt}")

# ── S3 overlap ──
moirai_overlap = results_df[results_df['ticker'].isin(S3_TICKERS)]
print(f"\n\n  S3 OVERLAP ({len(moirai_overlap)}/{len(S3_TICKERS)} found)")
print(f"  {'─'*50}")
if len(moirai_overlap) > 0:
    ov_cols = ['ticker', 'val_a_avg_pnl', 'val_a_trades', 'test_avg_pnl',
               'n_families', 'agreement', 'passes_gate']
    avail_ov = [c for c in ov_cols if c in moirai_overlap.columns]
    print(moirai_overlap[avail_ov].to_string(index=False, formatters=fmt))


# ── CHARTS ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(22, 14))
fig.suptitle(
    f'MOIRAI-Large S3-Style Optimizer\n'
    f'{len(results_df)} tickers | {len(deployed)} deployed | '
    f'entry 6:30 AM PT | {len(SIGNAL_COLS)} signal families',
    fontsize=14, fontweight='bold',
)

# 1: Per-ticker VAL_A PnL
ax = axes[0, 0]
colors = ['#2e7d32' if row.get('passes_gate', False) else '#ff8f00' if row['val_a_avg_pnl'] > 0 else '#c62828'
          for _, row in results_df.iterrows()]
ax.barh(results_df['ticker'], results_df['val_a_avg_pnl'] * 100, color=colors)
ax.axvline(DEPLOY_THRESHOLD * 100, color='green', linestyle='--', alpha=0.7,
           label=f'{DEPLOY_THRESHOLD*100:.1f}% deploy')
ax.axvline(0, color='k', linewidth=0.5)
ax.set_xlabel('VAL_A Avg PnL (%)')
ax.set_title('Per-Ticker VAL_A (green=gate pass)')
ax.legend(fontsize=9)
ax.tick_params(axis='y', labelsize=5)

# 2: OPT vs VAL_A scatter
ax = axes[0, 1]
gate_colors = ['#2e7d32' if row.get('passes_gate', False) else '#c62828'
               for _, row in results_df.iterrows()]
ax.scatter(results_df['opt_avg_pnl'] * 100, results_df['val_a_avg_pnl'] * 100,
           c=gate_colors, s=30, alpha=0.7)
lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
ax.plot([lo, hi], [lo, hi], 'r--', alpha=0.5, label='no degradation')
ax.set_xlabel('OPT Avg PnL (%)')
ax.set_ylabel('VAL_A Avg PnL (%)')
ax.set_title('OPT vs VAL_A (degradation)')
ax.legend(fontsize=9)
for _, row in deployed.iterrows():
    ax.annotate(row['ticker'], (row['opt_avg_pnl']*100, row['val_a_avg_pnl']*100),
                fontsize=7, alpha=0.8)

# 3: Family agreement distribution
ax = axes[0, 2]
if 'n_families' in results_df.columns and 'agreement' in results_df.columns:
    ax.scatter(results_df['n_families'], results_df['agreement'] * 100,
               c=['#2e7d32' if g else '#c62828' for g in results_df.get('passes_gate', [False]*len(results_df))],
               s=40, alpha=0.7)
    ax.axhline(50, color='orange', linestyle='--', alpha=0.5, label='50% gate')
    ax.axvline(MIN_FAMILIES, color='blue', linestyle='--', alpha=0.5, label=f'n_fam >= {MIN_FAMILIES}')
    ax.set_xlabel('Number of signal families')
    ax.set_ylabel('Family agreement (%)')
    ax.set_title('Family Agreement Gate')
    ax.legend(fontsize=9)
    for _, row in deployed.iterrows():
        ax.annotate(row['ticker'], (row['n_families'], row['agreement']*100),
                     fontsize=7, alpha=0.8)

# 4: Monthly PnL (deployed)
ax = axes[1, 0]
if len(deployed) > 0:
    monthly_data = {}
    for _, row in deployed.iterrows():
        ticker = row['ticker']
        sig_col = row['signal_col']
        thresh = row['threshold']
        tp, sl, mh = row['tp_pct'], row['sl_pct'], int(row['max_hold'])
        t_df = df[df['ticker'] == ticker]
        for _, tr in t_df[t_df[sig_col] >= thresh].iterrows():
            dt = tr['datetime']
            mk = dt.tz_localize(None).to_period('M') if hasattr(dt, 'tz') and dt.tz is not None else dt.to_period('M')
            pnl = simulate_trade(tr['entry_price'], tr['fwd_highs'], tr['fwd_lows'],
                                  tr['fwd_closes'], tp, sl, mh)
            monthly_data.setdefault(str(mk), []).append(pnl)
    if monthly_data:
        ms = sorted(monthly_data.keys())
        avg_p = [np.mean(monthly_data[m]) * 100 for m in ms]
        n_t = [len(monthly_data[m]) for m in ms]
        bars = ax.bar(ms, avg_p, color=['#2e7d32' if p > 0 else '#c62828' for p in avg_p], alpha=0.8)
        for bar, n in zip(bars, n_t):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'n={n}', ha='center', fontsize=8)
        ax.axhline(0, color='k', linewidth=0.5)
ax.set_ylabel('Avg PnL (%)')
ax.set_title('Monthly PnL (deployed)')
ax.tick_params(axis='x', rotation=45)

# 5: Equity curve (deployed)
ax = axes[1, 1]
if len(deployed) > 0:
    all_trades = []
    for _, row in deployed.iterrows():
        ticker = row['ticker']
        sig_col = row['signal_col']
        thresh = row['threshold']
        tp, sl, mh = row['tp_pct'], row['sl_pct'], int(row['max_hold'])
        t_df = df[df['ticker'] == ticker]
        for _, tr in t_df[t_df[sig_col] >= thresh].iterrows():
            pnl = simulate_trade(tr['entry_price'], tr['fwd_highs'], tr['fwd_lows'],
                                  tr['fwd_closes'], tp, sl, mh)
            all_trades.append({'datetime': tr['datetime'], 'pnl': pnl})
    if all_trades:
        eq = pd.DataFrame(all_trades).sort_values('datetime')
        eq['cumulative'] = (1 + eq['pnl']).cumprod()
        ax.plot(eq['datetime'], eq['cumulative'], color='steelblue', linewidth=1.5)
        ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
        for pn, (ps, pe) in WF_WINDOWS.items():
            if pn in ('VAL_A', 'TEST'):
                ax.axvline(pd.Timestamp(ps), color='orange' if pn == 'VAL_A' else 'red',
                           linestyle=':', alpha=0.6, label=pn)
        ax.legend(fontsize=8)
ax.set_ylabel('Equity (start = 1.0)')
ax.set_title('Cumulative Equity (deployed)')

# 6: Threshold stability
ax = axes[1, 2]
if 'thresh_stability' in results_df.columns:
    stab = results_df['thresh_stability'].dropna()
    if len(stab) > 0:
        ax.hist(stab * 100, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
        ax.axvline(50, color='orange', linestyle='--', label='50% stable')
        ax.set_xlabel('Threshold Stability (%)')
        ax.set_ylabel('Count')
        ax.set_title('Threshold Neighborhood Stability')
        ax.legend(fontsize=9)

plt.tight_layout()
chart_path = f'{LOCAL_DIR}/optimizer_results.png'
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nChart saved to {chart_path}")


# ── SAVE + GCS UPLOAD ─────────────────────────────────────────────────────────

results_path = f'{LOCAL_DIR}/optimizer_results.csv'
results_df.to_csv(results_path, index=False)
family_path = f'{LOCAL_DIR}/all_family_configs.csv'
family_df.to_csv(family_path, index=False)

from google.cloud import storage
client = storage.Client()
bucket = client.bucket(BUCKET_NAME)
for fname in ['optimizer_results.csv', 'all_family_configs.csv', 'optimizer_results.png']:
    local_path = f'{LOCAL_DIR}/{fname}'
    if os.path.exists(local_path):
        bucket.blob(f'{GCS_OUTPUT}/{fname}').upload_from_filename(local_path)
        print(f"  → gs://{BUCKET_NAME}/{GCS_OUTPUT}/{fname}")

total_elapsed = time.time() - t_total
print(f"\nTotal elapsed: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
