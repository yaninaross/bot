# ==============================================================================
# MOIRAI-1.1-Large Path Quality Validation — Colab/Vertex Notebook with GPU
# ==============================================================================
# Cell 0: pip install (~30s, no conflicts)
# Cell 1: Config + helpers
# Cell 2: Load model from GCS (~10-15s, weights pre-cached)
# Cell 3: Generate paths + compute actuals for all tickers
# Cell 4: Validate path quality — Spearman r, per-ticker, quintiles, charts
# Cell 5: Upload paths to GCS for persistence
# ==============================================================================

# ==============================================================================
# CELL 0: PIP INSTALL (~30 seconds, no torch download, no conflicts)
# ==============================================================================
# Weights are pre-cached on GCS (uploaded from dev VM 2026-08-08).
# uni2ts installed with --no-deps so it uses the notebook's existing torch.
# No torchvision conflict, no 800MB torch download, no disk issues.

import subprocess, sys

def setup_moirai():
    _pip = lambda args: subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-q'] + args,
        capture_output=True, text=True,
    )
    _pip(['einops', 'gluonts', 'safetensors', 'lightning', 'jaxtyping',
          'hydra-core==1.3', 'multiprocess', 'orjson', 'python-dotenv',
          'google-cloud-storage', 'scipy'])
    _pip(['uni2ts', '--no-deps'])

    import torch
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    print(f"OK: torch={torch.__version__}, CUDA={torch.cuda.is_available()}")

setup_moirai()

# ==============================================================================
# CELL 1: CONFIG + HELPERS
# ==============================================================================
import os
import gc
import numpy as np
import pandas as pd

BUCKET_NAME        = 'time-series-bucket-ys'
GCS_MOIRAI_WEIGHTS = 'Algos/MOIRAI/weights'
GCS_MOIRAI_PATHS   = 'Algos/MOIRAI/paths'
LOCAL_WEIGHTS_DIR   = '/tmp/moirai_weights'
LOCAL_PATHS_DIR     = '/tmp/moirai_paths'
LOCAL_RESULTS_DIR   = '/tmp/moirai_results'

MODEL_SIZE         = 'large'   # 'small' (14M) | 'base' (91M) | 'large' (311M)
HF_REPO            = f'Salesforce/moirai-1.1-R-{MODEL_SIZE}'
CONTEXT_LENGTH     = 240       # 4 hours of 1-min bars
PREDICTION_LENGTH  = 60        # 60 bars of forward prediction
NUM_SAMPLES        = 100       # 100 sample paths per prediction

MASTER_INFERENCE_GCS = 'Data/Inference_v2/{ticker}_master_inference.parquet'

ENTRY_MINUTE       = 390       # 6:30 AM PT (hour*60 + minute)

DATE_START = '2026-03-01'
DATE_END   = '2026-07-31'

# All 98 tickers with master_inference on GCS
TICKERS = [
    'ACHR', 'ACMR', 'AEHR', 'ALAB', 'AMD', 'APLD', 'ARM', 'ASTS',
    'AVAV', 'BABA', 'BBAI', 'BE', 'BEAM', 'BGDE', 'BHP', 'BITF',
    'BKSY', 'BTBT', 'CAT', 'CCJ', 'CEG', 'CIFR', 'CLF', 'CLSK',
    'COIN', 'COP', 'CORZ', 'CRDO', 'CVX', 'DUK', 'EQIX', 'ETN',
    'FCEL', 'FCX', 'FLNC', 'GD', 'GE', 'GEV', 'HOOD', 'HUT',
    'IREN', 'ITA', 'IWM', 'JOBY', 'KEEL', 'KTOS', 'LHX', 'LIN',
    'LMT', 'LUNR', 'MARA', 'MIGI', 'MNTS', 'MOD', 'MP', 'MRNA',
    'MSTR', 'MU', 'MVIS', 'MXL', 'NBIS', 'NEE', 'NEOV', 'NNDM',
    'NNE', 'NOC', 'NTLA', 'NVDA', 'OKLO', 'ONDS', 'PL', 'PLTR',
    'PLUG', 'POWL', 'RCAT', 'RDW', 'RIOT', 'RKLB', 'RTX', 'SLS',
    'SMCI', 'SMLR', 'SMR', 'SOUN', 'SPIR', 'SPY', 'TLT', 'TSLA',
    'TWST', 'UCTT', 'VIX', 'VOYG', 'VRT', 'VST', 'WOLF', 'WULF',
    'XLE', 'XOM',
]


def load_master(ticker):
    """Load master_inference from GCS."""
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob_path = MASTER_INFERENCE_GCS.format(ticker=ticker)
    blob = bucket.blob(blob_path)

    if not blob.exists():
        print(f"  {ticker}: master_inference not found on GCS")
        return None

    local_path = f'/tmp/{ticker}_master_inference.parquet'
    if not os.path.exists(local_path) or os.path.getsize(local_path) != blob.size:
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


# ==============================================================================
# CELL 2: LOAD MODEL FROM GCS (~10-15s, weights pre-cached on GCS)
# ==============================================================================
# Run this cell → loads model into GPU memory.
# Skips if already loaded. ~15s first time, instant after.

_moirai_pipeline = None

def load_moirai_from_gcs():
    """Download weights from GCS to /tmp, load into GPU memory."""
    global _moirai_pipeline
    if _moirai_pipeline is not None:
        print("Model already loaded, skipping.")
        return _moirai_pipeline

    import time
    from google.cloud import storage

    t0 = time.time()

    local_dir = f"{LOCAL_WEIGHTS_DIR}/{MODEL_SIZE}"
    os.makedirs(local_dir, exist_ok=True)

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    prefix = f"{GCS_MOIRAI_WEIGHTS}/{MODEL_SIZE}/"
    blobs = list(bucket.list_blobs(prefix=prefix))

    downloaded = 0
    for blob in blobs:
        rel_path = blob.name[len(prefix):]
        local_path = os.path.join(local_dir, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if os.path.exists(local_path) and os.path.getsize(local_path) == blob.size:
            continue
        blob.download_to_filename(local_path)
        downloaded += 1

    t1 = time.time()
    print(f"GCS download: {t1-t0:.1f}s ({downloaded} new files, {len(blobs)} total)")

    import torch
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading MOIRAI {MODEL_SIZE} ({HF_REPO}) on {device}...")

    module = MoiraiModule.from_pretrained(local_dir)
    _moirai_pipeline = MoiraiForecast(
        module=module,
        prediction_length=PREDICTION_LENGTH,
        context_length=CONTEXT_LENGTH,
        patch_size='auto',
        num_samples=NUM_SAMPLES,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )

    t2 = time.time()
    print(f"Model load:   {t2-t1:.1f}s  |  Total cold start: {t2-t0:.1f}s")
    return _moirai_pipeline


pipeline = load_moirai_from_gcs()


# ==============================================================================
# CELL 3: GENERATE PATHS + COMPUTE ACTUALS
# ==============================================================================

def prepare_windows(df, entry_minute, date_start, date_end):
    """Extract context windows AND forward 60-bar actuals for each entry bar."""
    df_pt = df.copy()
    if df_pt.index.tz is not None:
        df_pt.index = df_pt.index.tz_convert('America/Los_Angeles')
    else:
        df_pt.index = df_pt.index.tz_localize('America/Los_Angeles')

    mins   = df_pt.index.hour * 60 + df_pt.index.minute
    closes = df_pt['Close'].values.astype(np.float64)
    highs  = df_pt['High'].values.astype(np.float64)
    lows   = df_pt['Low'].values.astype(np.float64)
    dates  = df_pt.index.date

    ds = pd.Timestamp(date_start).date()
    de = pd.Timestamp(date_end).date()

    windows = []
    max_i = len(df_pt) - PREDICTION_LENGTH
    for i in range(CONTEXT_LENGTH, max_i):
        if mins[i] != entry_minute:
            continue
        if dates[i] < ds or dates[i] > de:
            continue

        ctx = closes[i - CONTEXT_LENGTH + 1 : i + 1].astype(np.float32)
        if len(ctx) != CONTEXT_LENGTH or np.isnan(ctx).any():
            continue

        fwd_h = highs[i+1 : i+1+PREDICTION_LENGTH]
        fwd_l = lows[i+1 : i+1+PREDICTION_LENGTH]
        fwd_c = closes[i+1 : i+1+PREDICTION_LENGTH]
        if len(fwd_h) != PREDICTION_LENGTH:
            continue

        entry_price = float(ctx[-1])
        if entry_price <= 0:
            continue

        windows.append({
            'datetime': df_pt.index[i],
            'entry_price': entry_price,
            'context': ctx,
            'actual_max_return': float((fwd_h.max() - entry_price) / entry_price),
            'actual_min_return': float((fwd_l.min() - entry_price) / entry_price),
            'actual_terminal_return': float((fwd_c[-1] - entry_price) / entry_price),
        })

    return windows


def generate_paths_for_ticker(pipeline, ticker, entry_minute, date_start, date_end):
    """Generate 100 sample paths + compute path metrics + actuals for one ticker."""
    from gluonts.dataset.pandas import PandasDataset

    df = load_master(ticker)
    if df is None:
        return pd.DataFrame()

    windows = prepare_windows(df, entry_minute, date_start, date_end)
    if not windows:
        print(f"  {ticker}: 0 valid entry bars in [{date_start}, {date_end}]")
        return pd.DataFrame()

    predictor = pipeline.create_predictor(batch_size=1)
    results = []

    for w in windows:
        ctx_series = pd.Series(
            w['context'],
            index=pd.period_range(start='2000-01-01', periods=CONTEXT_LENGTH, freq='min'),
        )
        ds = PandasDataset({'target': ctx_series}, freq='min')

        for forecast in predictor.predict(ds):
            paths = forecast.samples   # (NUM_SAMPLES, PREDICTION_LENGTH)
            ep = w['entry_price']
            returns = (paths - ep) / ep
            max_per_path = returns.max(axis=1)
            min_per_path = returns.min(axis=1)

            results.append({
                'datetime': w['datetime'],
                'ticker': ticker,
                'entry_price': ep,
                'paths': paths,
                # path-based metrics
                'spike_prob_2pct': float(np.mean(max_per_path >= 0.02)),
                'spike_prob_3pct': float(np.mean(max_per_path >= 0.03)),
                'spike_prob_4pct': float(np.mean(max_per_path >= 0.04)),
                'spike_prob_5pct': float(np.mean(max_per_path >= 0.05)),
                'expected_max_return': float(max_per_path.mean()),
                'expected_min_return': float(min_per_path.mean()),
                'direction_pct': float(np.mean(returns[:, -1] > 0)),
                'risk_ratio': float(max_per_path.mean() /
                                    max(abs(min_per_path.mean()), 1e-8)),
                'path_spread': float(np.std(max_per_path)),
                # ground truth
                'actual_max_return': w['actual_max_return'],
                'actual_min_return': w['actual_min_return'],
                'actual_terminal_return': w['actual_terminal_return'],
            })

    print(f"  {ticker}: {len(results)} predictions")
    return pd.DataFrame(results)


def batch_generate_all_tickers(tickers=None, entry_minute=None,
                                date_start=None, date_end=None,
                                save_npy=True):
    """Generate paths + actuals for all tickers. Saves .npy + summary parquet."""
    tickers = tickers or TICKERS
    entry_minute = entry_minute or ENTRY_MINUTE
    date_start = date_start or DATE_START
    date_end = date_end or DATE_END

    pipeline = load_moirai_from_gcs()

    os.makedirs(LOCAL_RESULTS_DIR, exist_ok=True)
    os.makedirs(LOCAL_PATHS_DIR, exist_ok=True)
    summary_path = f'{LOCAL_RESULTS_DIR}/path_validation_summary.parquet'

    # Resume: load previously saved results
    all_results = []
    done_tickers = set()
    if os.path.exists(summary_path):
        prev = pd.read_parquet(summary_path)
        done_tickers = set(prev['ticker'].unique())
        all_results.append(prev)
        print(f"  Resuming: {len(done_tickers)} tickers already done, skipping")

    for i, ticker in enumerate(tickers):
        if ticker in done_tickers:
            continue
        print(f"\n[{i+1}/{len(tickers)}] {ticker}")
        ticker_df = generate_paths_for_ticker(
            pipeline, ticker, entry_minute, date_start, date_end,
        )
        if ticker_df.empty:
            continue

        if save_npy:
            for _, row in ticker_df.iterrows():
                dt_str = row['datetime'].strftime('%Y%m%d_%H%M')
                fname = f"{ticker}_{dt_str}.npy"
                np.save(f"{LOCAL_PATHS_DIR}/{fname}", row['paths'])

        all_results.append(ticker_df)

        # Save after every ticker so Cell 4 can run on partial results
        master = pd.concat(all_results, ignore_index=True)
        summary_cols = [c for c in master.columns if c != 'paths']
        master[summary_cols].to_parquet(summary_path)
        print(f"  Summary saved: {len(master)} predictions, "
              f"{master['ticker'].nunique()} tickers → {summary_path}")

        gc.collect()

    if not all_results:
        print("No predictions generated for any ticker.")
        return pd.DataFrame()

    master = pd.concat(all_results, ignore_index=True)
    print(f"\n{'='*60}")
    print(f"Total: {len(master)} predictions across {master['ticker'].nunique()} tickers")
    print(f"Summary → {summary_path}")
    print(f".npy paths → {LOCAL_PATHS_DIR}/")
    return master


master = batch_generate_all_tickers()


# ==============================================================================
# CELL 4: VALIDATE PATH QUALITY
# ==============================================================================

def validate_path_quality(summary_path=None):
    """Step 1 validation: Do MOIRAI-Large paths contain tradeable signal?

    Loads the summary parquet saved by batch_generate_all_tickers() and
    computes Spearman r between path-based spike probabilities and actual
    forward max returns. Produces per-ticker breakdown, quintile analysis,
    and 6-panel diagnostic chart.

    Decision thresholds:
        r >= 0.15  →  PASS  (proceed to path-to-trade simulation)
        0.10 <= r  →  MARGINAL (try consensus filter / feature augmentation)
        r <  0.10  →  FAIL  (pivot to Kronos)
    """
    from scipy.stats import spearmanr
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['figure.dpi'] = 120

    if summary_path is None:
        summary_path = f'{LOCAL_RESULTS_DIR}/path_validation_summary.parquet'
    df = pd.read_parquet(summary_path)
    df['datetime'] = pd.to_datetime(df['datetime'])

    print(f"Loaded {len(df)} predictions for {df['ticker'].nunique()} tickers")
    print(f"Date range: {df['datetime'].min().date()} → {df['datetime'].max().date()}")

    # ── 1. Overall Spearman correlations ──────────────────────────────────────
    metrics = [
        'spike_prob_2pct', 'spike_prob_3pct', 'spike_prob_4pct',
        'spike_prob_5pct', 'expected_max_return', 'direction_pct',
        'risk_ratio', 'path_spread',
    ]

    print(f"\n{'='*70}")
    print("OVERALL SPEARMAN CORRELATIONS  vs  actual_max_return")
    print(f"{'='*70}")

    correlations = {}
    for m in metrics:
        r, p = spearmanr(df[m], df['actual_max_return'])
        correlations[m] = (r, p)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  {m:30s}  r = {r:+.4f}   p = {p:.2e}  {sig}")

    # ── 2. Per-ticker breakdown ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("PER-TICKER  Spearman r  (spike_prob_3pct  vs  actual_max_return)")
    print(f"{'='*70}")

    ticker_stats = []
    for t in sorted(df['ticker'].unique()):
        tdf = df[df['ticker'] == t]
        if len(tdf) < 10:
            continue
        r, p = spearmanr(tdf['spike_prob_3pct'], tdf['actual_max_return'])
        q5_mask = tdf['spike_prob_3pct'] >= tdf['spike_prob_3pct'].quantile(0.8)
        q1_mask = tdf['spike_prob_3pct'] <= tdf['spike_prob_3pct'].quantile(0.2)
        ticker_stats.append({
            'Ticker': t,
            'N': len(tdf),
            'r': r,
            'p': p,
            'Avg_spike3': tdf['spike_prob_3pct'].mean(),
            'Avg_actual': tdf['actual_max_return'].mean(),
            'Q5_actual': tdf.loc[q5_mask, 'actual_max_return'].mean() if q5_mask.sum() > 0 else np.nan,
            'Q1_actual': tdf.loc[q1_mask, 'actual_max_return'].mean() if q1_mask.sum() > 0 else np.nan,
        })

    ticker_df = pd.DataFrame(ticker_stats).sort_values('r', ascending=False)
    fmt = {'r': '{:+.3f}'.format, 'p': '{:.3f}'.format, 'Avg_spike3': '{:.3f}'.format,
           'Avg_actual': '{:.4f}'.format, 'Q5_actual': '{:.4f}'.format, 'Q1_actual': '{:.4f}'.format}
    print(ticker_df.to_string(index=False, formatters=fmt))
    print(f"\n  Positive r: {(ticker_df['r'] > 0).sum()}/{len(ticker_df)} tickers")
    print(f"  r > 0.15:   {(ticker_df['r'] > 0.15).sum()}/{len(ticker_df)} tickers")

    # ── 3. Quintile analysis ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("QUINTILE ANALYSIS  (grouping predictions by spike_prob_3pct)")
    print(f"{'='*70}")

    try:
        df['quintile'] = pd.qcut(df['spike_prob_3pct'], 5,
                                  labels=['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)'],
                                  duplicates='drop')
    except ValueError:
        df['quintile'] = pd.cut(df['spike_prob_3pct'], 5,
                                 labels=['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)'])

    quint = df.groupby('quintile', observed=True).agg(
        N=('actual_max_return', 'count'),
        Avg_spike3=('spike_prob_3pct', 'mean'),
        Avg_actual_max=('actual_max_return', 'mean'),
        Med_actual_max=('actual_max_return', 'median'),
        Avg_terminal=('actual_terminal_return', 'mean'),
    ).reset_index()
    print(quint.to_string(index=False, float_format='%.4f'))

    q5_avg = quint.loc[quint['quintile'] == 'Q5(high)', 'Avg_actual_max']
    q1_avg = quint.loc[quint['quintile'] == 'Q1(low)', 'Avg_actual_max']
    if len(q5_avg) > 0 and len(q1_avg) > 0:
        lift = q5_avg.values[0] - q1_avg.values[0]
        print(f"\n  Q5 - Q1 lift: {lift:+.4f}  ({'signal separates' if lift > 0 else 'NO separation'})")

    # ── 4. Monthly stability ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("MONTHLY STABILITY  (Spearman r by month)")
    print(f"{'='*70}")

    df['month'] = df['datetime'].dt.to_period('M')
    monthly_stats = []
    for m, mdf in df.groupby('month'):
        if len(mdf) < 20:
            continue
        r, p = spearmanr(mdf['spike_prob_3pct'], mdf['actual_max_return'])
        monthly_stats.append({'Month': str(m), 'N': len(mdf), 'r': r, 'p': p})
    if monthly_stats:
        monthly_df = pd.DataFrame(monthly_stats)
        print(monthly_df.to_string(index=False, float_format='%.4f'))

    # ── 5. VERDICT ────────────────────────────────────────────────────────────
    r_main = correlations['spike_prob_3pct'][0]
    r_exp  = correlations['expected_max_return'][0]
    best_metric = max(correlations, key=lambda k: correlations[k][0])
    best_r = correlations[best_metric][0]

    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    print(f"  Primary metric (spike_prob_3pct):   r = {r_main:+.4f}")
    print(f"  Expected max return:                r = {r_exp:+.4f}")
    print(f"  Best metric ({best_metric}):  r = {best_r:+.4f}")

    if best_r >= 0.15:
        print(f"\n  PASS  —  MOIRAI-Large paths contain tradeable signal.")
        print(f"  → Proceed to Approach 2 (path-to-trade simulation)")
    elif best_r >= 0.10:
        print(f"\n  MARGINAL  —  Weak signal detected.")
        print(f"  → Try Approach 3 (S3 consensus filter) or 4 (feature augmentation)")
    else:
        print(f"\n  FAIL  —  MOIRAI-Large paths lack tradeable signal at these thresholds.")
        print(f"  → Pivot to Approach 5 (Kronos OHLCV paths)")
    print(f"{'='*70}")

    # ── 6. Charts ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.suptitle(
        f'MOIRAI-1.1-Large Path Quality Validation\n'
        f'{len(df)} predictions | {df["ticker"].nunique()} tickers | '
        f'Spearman r = {r_main:+.4f}',
        fontsize=14, fontweight='bold',
    )

    # Chart 1: Scatter spike_prob_3pct vs actual
    ax = axes[0, 0]
    ax.scatter(df['spike_prob_3pct'], df['actual_max_return'] * 100,
               alpha=0.3, s=8, c='steelblue')
    ax.set_xlabel('MOIRAI spike_prob_3pct')
    ax.set_ylabel('Actual max return (%)')
    ax.set_title(f'Signal vs Actual  (r = {r_main:+.3f})')
    ax.axhline(0, color='r', linestyle='--', alpha=0.4)

    # Chart 2: Per-ticker Spearman r
    ax = axes[0, 1]
    colors = ['#2e7d32' if r > 0.15 else '#ff8f00' if r > 0 else '#c62828'
              for r in ticker_df['r']]
    ax.barh(ticker_df['Ticker'], ticker_df['r'], color=colors)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.axvline(0.15, color='green', linestyle='--', alpha=0.5, label='r = 0.15')
    ax.set_title('Per-Ticker Spearman r')
    ax.legend(fontsize=9)

    # Chart 3: Quintile avg actual max return
    ax = axes[0, 2]
    q_labels = quint['quintile'].astype(str).values
    q_vals = quint['Avg_actual_max'].values * 100
    bar_colors = ['#c62828', '#ef6c00', '#f9a825', '#558b2f', '#1b5e20'][:len(q_vals)]
    ax.bar(q_labels, q_vals, color=bar_colors)
    ax.set_ylabel('Avg actual max return (%)')
    ax.set_title('Actual Return by MOIRAI Quintile')
    for j, v in enumerate(q_vals):
        ax.text(j, v + 0.05, f'{v:.2f}%', ha='center', fontsize=9)

    # Chart 4: Correlation heatmap
    ax = axes[1, 0]
    corr_cols = ['spike_prob_2pct', 'spike_prob_3pct', 'expected_max_return',
                 'direction_pct', 'actual_max_return', 'actual_terminal_return']
    corr_matrix = df[corr_cols].corr(method='spearman')
    short_labels = ['sp2%', 'sp3%', 'E[max]', 'dir%', 'act_max', 'act_term']
    im = ax.imshow(corr_matrix.values, cmap='RdYlGn', vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(len(short_labels)))
    ax.set_yticks(range(len(short_labels)))
    ax.set_xticklabels(short_labels, rotation=45, ha='right')
    ax.set_yticklabels(short_labels)
    for ii in range(len(short_labels)):
        for jj in range(len(short_labels)):
            ax.text(jj, ii, f'{corr_matrix.values[ii, jj]:.2f}',
                    ha='center', va='center', fontsize=8)
    ax.set_title('Spearman Correlation Matrix')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Chart 5: Monthly r bars
    ax = axes[1, 1]
    if monthly_stats:
        mr_df = pd.DataFrame(monthly_stats)
        bar_c = ['#2e7d32' if r > 0.15 else '#ff8f00' if r > 0.10 else '#c62828'
                 for r in mr_df['r']]
        ax.bar(mr_df['Month'], mr_df['r'], color=bar_c)
        ax.axhline(0.15, color='green', linestyle='--', alpha=0.5, label='0.15')
        ax.axhline(0.10, color='orange', linestyle='--', alpha=0.5, label='0.10')
        ax.set_ylabel('Spearman r')
        ax.set_title('Monthly Signal Stability')
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, 'Not enough monthly data', transform=ax.transAxes,
                ha='center', va='center')

    # Chart 6: Expected vs actual max return
    ax = axes[1, 2]
    ax.scatter(df['expected_max_return'] * 100, df['actual_max_return'] * 100,
               alpha=0.3, s=8, c='darkorange')
    ax.set_xlabel('MOIRAI expected max return (%)')
    ax.set_ylabel('Actual max return (%)')
    ax.set_title(f'Expected vs Actual Max  (r = {r_exp:+.3f})')
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], 'r--', alpha=0.5)

    plt.tight_layout()
    chart_path = f'{LOCAL_RESULTS_DIR}/moirai_validation.png'
    os.makedirs(LOCAL_RESULTS_DIR, exist_ok=True)
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nChart saved to {chart_path}")

    return df, ticker_df


df, ticker_stats = validate_path_quality()


# ==============================================================================
# CELL 5: UPLOAD PATHS + RESULTS TO GCS
# ==============================================================================

def upload_paths_to_gcs():
    """Upload .npy paths and summary to GCS for persistence."""
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    count = 0

    for fname in os.listdir(LOCAL_PATHS_DIR):
        if not fname.endswith('.npy'):
            continue
        local_path = f"{LOCAL_PATHS_DIR}/{fname}"
        blob_path = f"{GCS_MOIRAI_PATHS}/{fname}"
        blob = bucket.blob(blob_path)
        if blob.exists() and blob.size == os.path.getsize(local_path):
            continue
        blob.upload_from_filename(local_path)
        count += 1

    summary_local = f'{LOCAL_RESULTS_DIR}/path_validation_summary.parquet'
    if os.path.exists(summary_local):
        blob = bucket.blob(f'{GCS_MOIRAI_PATHS}/path_validation_summary.parquet')
        blob.upload_from_filename(summary_local)

    print(f"Uploaded {count} .npy files + summary to gs://{BUCKET_NAME}/{GCS_MOIRAI_PATHS}/")


upload_paths_to_gcs()
