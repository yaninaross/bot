# ==============================================================================
# MOIRAI-1.1-Large Path Quality Validation — LONG + SHORT, Dual Entry Times
# ==============================================================================
# Cell 0: pip install (~30s, no conflicts)
# Cell 1: Config + helpers
# Cell 2: Load model from GCS (~10-15s, weights pre-cached)
# Cell 3: Generate paths + compute LONG & SHORT actuals for all tickers × 2 entries
# Cell 4: Validate path quality — Spearman r for LONG and SHORT separately
# Cell 5: Upload paths to GCS for persistence
# ==============================================================================
#
# Scoring logic:
#   Model scores at score_minute (context window ends here).
#   Trade enters at entry_minute = score_minute + 15 (gap for order placement).
#   Actual returns computed over 60 bars after entry_minute.
#
#   Session 1: score 6:15 AM PT → enter 6:30 AM PT
#   Session 2: score 6:45 AM PT → enter 7:00 AM PT
# ==============================================================================

# ==============================================================================
# CELL 0: PIP INSTALL (~30 seconds, no torch download, no conflicts)
# ==============================================================================

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

BUCKET_NAME        = os.environ.get("GCS_BUCKET", "")
GCS_MOIRAI_WEIGHTS = 'Algos/MOIRAI/weights'
GCS_MOIRAI_PATHS   = 'Algos/MOIRAI/paths_v2'
LOCAL_WEIGHTS_DIR   = '/tmp/moirai_weights'
LOCAL_PATHS_DIR     = '/tmp/moirai_paths_v2'
LOCAL_RESULTS_DIR   = '/tmp/moirai_results_v2'

MODEL_SIZE         = 'large'   # 'small' (14M) | 'base' (91M) | 'large' (311M)
HF_REPO            = f'Salesforce/moirai-1.1-R-{MODEL_SIZE}'
CONTEXT_LENGTH     = 240       # 4 hours of 1-min bars
PREDICTION_LENGTH  = 75        # 15 min gap + 60 min trading window
NUM_SAMPLES        = 100       # 100 sample paths per prediction
ENTRY_GAP          = 15        # bars between scoring and trade entry
TRADE_WINDOW       = 60        # bars of forward returns after entry

MASTER_INFERENCE_GCS = 'Data/Inference_v2/{ticker}_master_inference.parquet'

# Score 15 min before entry to allow trade placement
SCORE_CONFIGS = [
    {'score_minute': 375, 'entry_minute': 390, 'label': '630'},  # entry 6:30 AM PT
    {'score_minute': 405, 'entry_minute': 420, 'label': '700'},  # entry 7:00 AM PT
]

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
# CELL 3: GENERATE PATHS + COMPUTE LONG & SHORT ACTUALS
# ==============================================================================

def prepare_windows(df, score_minute, entry_minute, date_start, date_end):
    """Extract context windows at score_minute, actual returns from entry_minute.

    Context ends at score_minute. Actual LONG/SHORT returns computed over
    TRADE_WINDOW bars starting from entry_minute.
    """
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

    gap = entry_minute - score_minute
    total_fwd = gap + TRADE_WINDOW

    windows = []
    max_i = len(df_pt) - total_fwd
    for i in range(CONTEXT_LENGTH, max_i):
        if mins[i] != score_minute:
            continue
        if dates[i] < ds or dates[i] > de:
            continue

        ctx = closes[i - CONTEXT_LENGTH + 1 : i + 1].astype(np.float32)
        if len(ctx) != CONTEXT_LENGTH or np.isnan(ctx).any():
            continue

        entry_idx = i + gap
        entry_price = float(closes[entry_idx])
        if entry_price <= 0:
            continue

        fwd_h = highs[entry_idx+1 : entry_idx+1+TRADE_WINDOW]
        fwd_l = lows[entry_idx+1 : entry_idx+1+TRADE_WINDOW]
        fwd_c = closes[entry_idx+1 : entry_idx+1+TRADE_WINDOW]
        if len(fwd_h) != TRADE_WINDOW:
            continue

        windows.append({
            'score_datetime': df_pt.index[i],
            'entry_datetime': df_pt.index[entry_idx],
            'entry_price': entry_price,
            'context': ctx,
            # LONG actuals: max upside from entry
            'actual_max_return': float((fwd_h.max() - entry_price) / entry_price),
            # SHORT actuals: max downside from entry (positive = price dropped)
            'actual_max_drop': float((entry_price - fwd_l.min()) / entry_price),
            'actual_terminal_return': float((fwd_c[-1] - entry_price) / entry_price),
        })

    return windows


def generate_paths_for_ticker(pipeline, ticker, score_minute, entry_minute,
                               entry_label, date_start, date_end):
    """Generate 100 sample paths + compute LONG & SHORT metrics for one ticker."""
    from gluonts.dataset.pandas import PandasDataset

    df = load_master(ticker)
    if df is None:
        return pd.DataFrame()

    windows = prepare_windows(df, score_minute, entry_minute, date_start, date_end)
    if not windows:
        print(f"  {ticker} [{entry_label}]: 0 valid windows in [{date_start}, {date_end}]")
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

            # Path metrics computed on post-gap window only (bars ENTRY_GAP to end)
            trade_paths = paths[:, ENTRY_GAP:]   # (NUM_SAMPLES, TRADE_WINDOW)
            returns_up = (trade_paths - ep) / ep   # positive = price went up
            returns_dn = (ep - trade_paths) / ep   # positive = price went down

            max_up_per_path = returns_up.max(axis=1)    # best LONG return per path
            max_dn_per_path = returns_dn.max(axis=1)    # best SHORT return per path

            results.append({
                'score_datetime': w['score_datetime'],
                'entry_datetime': w['entry_datetime'],
                'entry_label': entry_label,
                'ticker': ticker,
                'entry_price': ep,
                'paths': paths,
                # LONG path metrics
                'long_spike_prob_2pct': float(np.mean(max_up_per_path >= 0.02)),
                'long_spike_prob_3pct': float(np.mean(max_up_per_path >= 0.03)),
                'long_spike_prob_4pct': float(np.mean(max_up_per_path >= 0.04)),
                'long_spike_prob_5pct': float(np.mean(max_up_per_path >= 0.05)),
                'long_expected_max': float(max_up_per_path.mean()),
                'long_direction_pct': float(np.mean(returns_up[:, -1] > 0)),
                'long_risk_ratio': float(max_up_per_path.mean() /
                                         max(abs(max_dn_per_path.mean()), 1e-8)),
                'long_path_spread': float(np.std(max_up_per_path)),
                # SHORT path metrics
                'short_dump_prob_2pct': float(np.mean(max_dn_per_path >= 0.02)),
                'short_dump_prob_3pct': float(np.mean(max_dn_per_path >= 0.03)),
                'short_dump_prob_4pct': float(np.mean(max_dn_per_path >= 0.04)),
                'short_dump_prob_5pct': float(np.mean(max_dn_per_path >= 0.05)),
                'short_expected_max_drop': float(max_dn_per_path.mean()),
                'short_direction_pct': float(np.mean(returns_dn[:, -1] > 0)),
                'short_risk_ratio': float(max_dn_per_path.mean() /
                                          max(abs(max_up_per_path.mean()), 1e-8)),
                'short_path_spread': float(np.std(max_dn_per_path)),
                # Ground truth
                'actual_max_return': w['actual_max_return'],
                'actual_max_drop': w['actual_max_drop'],
                'actual_terminal_return': w['actual_terminal_return'],
            })

    print(f"  {ticker} [{entry_label}]: {len(results)} predictions")
    return pd.DataFrame(results)


def _upload_summary_to_gcs(summary_path):
    """Push summary parquet to GCS for crash-resilient resume."""
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f'{GCS_MOIRAI_PATHS}/path_validation_v2_summary.parquet')
    blob.upload_from_filename(summary_path)


def _download_summary_from_gcs(summary_path):
    """Try to resume from GCS if local /tmp/ is gone (runtime restart)."""
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f'{GCS_MOIRAI_PATHS}/path_validation_v2_summary.parquet')
    if blob.exists():
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        blob.download_to_filename(summary_path)
        return True
    return False


def batch_generate_all_tickers(tickers=None, date_start=None, date_end=None,
                                save_npy=True):
    """Generate paths for all tickers × all SCORE_CONFIGS. Incremental save."""
    tickers = tickers or TICKERS
    date_start = date_start or DATE_START
    date_end = date_end or DATE_END

    pipeline = load_moirai_from_gcs()

    os.makedirs(LOCAL_RESULTS_DIR, exist_ok=True)
    os.makedirs(LOCAL_PATHS_DIR, exist_ok=True)
    summary_path = f'{LOCAL_RESULTS_DIR}/path_validation_v2_summary.parquet'

    all_results = []
    done_keys = set()
    if not os.path.exists(summary_path):
        if _download_summary_from_gcs(summary_path):
            print(f"  Resumed summary from GCS")
    if os.path.exists(summary_path):
        prev = pd.read_parquet(summary_path)
        done_keys = set(zip(prev['ticker'], prev['entry_label']))
        all_results.append(prev)
        print(f"  Resuming: {len(done_keys)} (ticker, entry) combos already done")

    total_combos = len(tickers) * len(SCORE_CONFIGS)
    combo_i = 0

    for ticker in tickers:
        for cfg in SCORE_CONFIGS:
            combo_i += 1
            key = (ticker, cfg['label'])
            if key in done_keys:
                continue

            print(f"\n[{combo_i}/{total_combos}] {ticker} entry={cfg['label']}")
            ticker_df = generate_paths_for_ticker(
                pipeline, ticker,
                cfg['score_minute'], cfg['entry_minute'], cfg['label'],
                date_start, date_end,
            )
            if ticker_df.empty:
                continue

            if save_npy:
                for _, row in ticker_df.iterrows():
                    dt_str = row['score_datetime'].strftime('%Y%m%d_%H%M')
                    fname = f"{ticker}_{cfg['label']}_{dt_str}.npy"
                    np.save(f"{LOCAL_PATHS_DIR}/{fname}", row['paths'])

            all_results.append(ticker_df)

            master = pd.concat(all_results, ignore_index=True)
            summary_cols = [c for c in master.columns if c != 'paths']
            master[summary_cols].to_parquet(summary_path)
            _upload_summary_to_gcs(summary_path)
            print(f"  Saved: {len(master)} predictions, "
                  f"{master['ticker'].nunique()} tickers → local + GCS")

            gc.collect()

    if not all_results:
        print("No predictions generated for any ticker.")
        return pd.DataFrame()

    master = pd.concat(all_results, ignore_index=True)
    master.to_parquet(summary_path, index=False)
    _upload_summary_to_gcs(summary_path)
    print(f"\n{'='*60}")
    print(f"Total: {len(master)} predictions across "
          f"{master['ticker'].nunique()} tickers × {len(SCORE_CONFIGS)} entries")
    print(f"Summary → {summary_path} (uploaded to GCS)")
    print(f".npy paths → {LOCAL_PATHS_DIR}/")
    return master


master = batch_generate_all_tickers()


# ==============================================================================
# CELL 4: VALIDATE PATH QUALITY — LONG & SHORT
# ==============================================================================

def validate_path_quality(summary_path=None):
    """Validate MOIRAI-Large signal quality for LONG and SHORT separately.

    LONG:  Spearman r between long_spike_prob_3pct and actual_max_return
    SHORT: Spearman r between short_dump_prob_3pct and actual_max_drop

    Decision thresholds (per direction):
        r >= 0.15  →  PASS
        0.10 <= r  →  MARGINAL
        r <  0.10  →  FAIL
    """
    from scipy.stats import spearmanr
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['figure.dpi'] = 120

    if summary_path is None:
        summary_path = f'{LOCAL_RESULTS_DIR}/path_validation_v2_summary.parquet'
    df = pd.read_parquet(summary_path)
    df['score_datetime'] = pd.to_datetime(df['score_datetime'])
    df['entry_datetime'] = pd.to_datetime(df['entry_datetime'])

    print(f"Loaded {len(df)} predictions for {df['ticker'].nunique()} tickers")
    print(f"Entry sessions: {df['entry_label'].unique().tolist()}")
    print(f"Date range: {df['score_datetime'].min().date()} → "
          f"{df['score_datetime'].max().date()}")

    # ── Analyze each direction ───────────────────────────────────────────────
    DIRECTIONS = {
        'LONG': {
            'metrics': ['long_spike_prob_2pct', 'long_spike_prob_3pct',
                        'long_spike_prob_4pct', 'long_spike_prob_5pct',
                        'long_expected_max', 'long_direction_pct',
                        'long_risk_ratio', 'long_path_spread'],
            'primary': 'long_spike_prob_3pct',
            'actual': 'actual_max_return',
            'secondary': 'long_expected_max',
        },
        'SHORT': {
            'metrics': ['short_dump_prob_2pct', 'short_dump_prob_3pct',
                        'short_dump_prob_4pct', 'short_dump_prob_5pct',
                        'short_expected_max_drop', 'short_direction_pct',
                        'short_risk_ratio', 'short_path_spread'],
            'primary': 'short_dump_prob_3pct',
            'actual': 'actual_max_drop',
            'secondary': 'short_expected_max_drop',
        },
    }

    all_ticker_stats = {}
    all_correlations = {}

    for direction, cfg in DIRECTIONS.items():
        print(f"\n{'='*70}")
        print(f"  {direction} SIGNAL VALIDATION")
        print(f"{'='*70}")

        # Overall correlations
        print(f"\nSPEARMAN CORRELATIONS  vs  {cfg['actual']}")
        print(f"{'-'*60}")
        correlations = {}
        for m in cfg['metrics']:
            if m not in df.columns:
                continue
            r, p = spearmanr(df[m], df[cfg['actual']])
            correlations[m] = (r, p)
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"  {m:35s}  r = {r:+.4f}   p = {p:.2e}  {sig}")
        all_correlations[direction] = correlations

        # Per-entry-session breakdown
        for label in sorted(df['entry_label'].unique()):
            sub = df[df['entry_label'] == label]
            primary = cfg['primary']
            if primary not in sub.columns:
                continue
            r, p = spearmanr(sub[primary], sub[cfg['actual']])
            print(f"  Entry {label}: {primary} r = {r:+.4f}  (n={len(sub)})")

        # Per-ticker breakdown
        print(f"\nPER-TICKER  Spearman r  ({cfg['primary']}  vs  {cfg['actual']})")
        print(f"{'-'*60}")

        ticker_stats = []
        primary = cfg['primary']
        actual = cfg['actual']
        for t in sorted(df['ticker'].unique()):
            tdf = df[df['ticker'] == t]
            if len(tdf) < 10 or primary not in tdf.columns:
                continue
            r, p = spearmanr(tdf[primary], tdf[actual])
            q5_mask = tdf[primary] >= tdf[primary].quantile(0.8)
            q1_mask = tdf[primary] <= tdf[primary].quantile(0.2)
            ticker_stats.append({
                'Ticker': t,
                'N': len(tdf),
                'r': r,
                'p': p,
                'Avg_signal': tdf[primary].mean(),
                'Avg_actual': tdf[actual].mean(),
                'Q5_actual': tdf.loc[q5_mask, actual].mean() if q5_mask.sum() > 0 else np.nan,
                'Q1_actual': tdf.loc[q1_mask, actual].mean() if q1_mask.sum() > 0 else np.nan,
            })

        ticker_df = pd.DataFrame(ticker_stats).sort_values('r', ascending=False)
        fmt = {'r': '{:+.3f}'.format, 'p': '{:.3f}'.format, 'Avg_signal': '{:.3f}'.format,
               'Avg_actual': '{:.4f}'.format, 'Q5_actual': '{:.4f}'.format,
               'Q1_actual': '{:.4f}'.format}
        print(ticker_df.to_string(index=False, formatters=fmt))
        print(f"\n  Positive r: {(ticker_df['r'] > 0).sum()}/{len(ticker_df)}")
        print(f"  r > 0.15:   {(ticker_df['r'] > 0.15).sum()}/{len(ticker_df)}")
        all_ticker_stats[direction] = ticker_df

        # Quintile analysis
        print(f"\nQUINTILE ANALYSIS  ({cfg['primary']})")
        print(f"{'-'*60}")
        try:
            df[f'{direction}_quintile'] = pd.qcut(
                df[primary], 5,
                labels=['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)'],
                duplicates='drop')
        except ValueError:
            df[f'{direction}_quintile'] = pd.cut(
                df[primary], 5,
                labels=['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)'])

        quint = df.groupby(f'{direction}_quintile', observed=True).agg(
            N=(actual, 'count'),
            Avg_signal=(primary, 'mean'),
            Avg_actual=(actual, 'mean'),
            Med_actual=(actual, 'median'),
        ).reset_index()
        print(quint.to_string(index=False, float_format='%.4f'))

        q5_avg = quint.loc[quint[f'{direction}_quintile'] == 'Q5(high)', 'Avg_actual']
        q1_avg = quint.loc[quint[f'{direction}_quintile'] == 'Q1(low)', 'Avg_actual']
        if len(q5_avg) > 0 and len(q1_avg) > 0:
            lift = q5_avg.values[0] - q1_avg.values[0]
            print(f"\n  Q5 - Q1 lift: {lift:+.4f}  "
                  f"({'signal separates' if lift > 0 else 'NO separation'})")

        # Verdict
        if primary in correlations:
            r_main = correlations[primary][0]
        else:
            r_main = 0.0
        best_metric = max(correlations, key=lambda k: correlations[k][0]) if correlations else primary
        best_r = correlations[best_metric][0] if correlations else 0.0

        print(f"\n  {direction} VERDICT:")
        print(f"  Primary ({primary}): r = {r_main:+.4f}")
        print(f"  Best ({best_metric}): r = {best_r:+.4f}")
        if best_r >= 0.15:
            print(f"  → PASS — tradeable {direction} signal")
        elif best_r >= 0.10:
            print(f"  → MARGINAL — weak {direction} signal")
        else:
            print(f"  → FAIL — no {direction} signal at these thresholds")

    # ── Charts: 4 rows × 2 cols (LONG left, SHORT right) ────────────────────
    fig, axes = plt.subplots(4, 2, figsize=(20, 24))
    fig.suptitle(
        f'MOIRAI-1.1-Large Path Quality — LONG & SHORT\n'
        f'{len(df)} predictions | {df["ticker"].nunique()} tickers | '
        f'entries: {", ".join(df["entry_label"].unique())}',
        fontsize=14, fontweight='bold',
    )

    for col, (direction, cfg) in enumerate(DIRECTIONS.items()):
        primary = cfg['primary']
        actual = cfg['actual']
        secondary = cfg['secondary']
        tdf = all_ticker_stats[direction]
        corrs = all_correlations[direction]
        r_main = corrs.get(primary, (0, 1))[0]
        r_sec = corrs.get(secondary, (0, 1))[0]

        # Row 1: Scatter signal vs actual
        ax = axes[0, col]
        ax.scatter(df[primary], df[actual] * 100,
                   alpha=0.3, s=8, c='steelblue' if direction == 'LONG' else 'crimson')
        ax.set_xlabel(f'MOIRAI {primary}')
        ax.set_ylabel(f'{actual} (%)')
        ax.set_title(f'{direction}: Signal vs Actual  (r = {r_main:+.3f})')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.4)

        # Row 2: Per-ticker Spearman r
        ax = axes[1, col]
        colors = ['#2e7d32' if r > 0.15 else '#ff8f00' if r > 0 else '#c62828'
                  for r in tdf['r']]
        ax.barh(tdf['Ticker'], tdf['r'], color=colors)
        ax.axvline(0, color='k', linewidth=0.5)
        ax.axvline(0.15, color='green', linestyle='--', alpha=0.5, label='r = 0.15')
        ax.set_title(f'{direction}: Per-Ticker Spearman r')
        ax.legend(fontsize=9)

        # Row 3: Quintile avg actual
        ax = axes[2, col]
        qcol = f'{direction}_quintile'
        if qcol in df.columns:
            quint = df.groupby(qcol, observed=True).agg(
                Avg_actual=(actual, 'mean')).reset_index()
            q_labels = quint[qcol].astype(str).values
            q_vals = quint['Avg_actual'].values * 100
            bar_colors = ['#c62828', '#ef6c00', '#f9a825', '#558b2f', '#1b5e20'][:len(q_vals)]
            ax.bar(q_labels, q_vals, color=bar_colors)
            ax.set_ylabel(f'Avg {actual} (%)')
            ax.set_title(f'{direction}: Actual by MOIRAI Quintile')
            for j, v in enumerate(q_vals):
                ax.text(j, v + 0.05, f'{v:.2f}%', ha='center', fontsize=9)

        # Row 4: Expected vs actual
        ax = axes[3, col]
        ax.scatter(df[secondary] * 100, df[actual] * 100,
                   alpha=0.3, s=8, c='darkorange' if direction == 'LONG' else 'purple')
        ax.set_xlabel(f'MOIRAI {secondary} (%)')
        ax.set_ylabel(f'{actual} (%)')
        ax.set_title(f'{direction}: Expected vs Actual  (r = {r_sec:+.3f})')
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], 'r--', alpha=0.5)

    plt.tight_layout()
    chart_path = f'{LOCAL_RESULTS_DIR}/moirai_validation_v2.png'
    os.makedirs(LOCAL_RESULTS_DIR, exist_ok=True)
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nChart saved to {chart_path}")

    return df, all_ticker_stats


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

    summary_local = f'{LOCAL_RESULTS_DIR}/path_validation_v2_summary.parquet'
    if os.path.exists(summary_local):
        blob = bucket.blob(f'{GCS_MOIRAI_PATHS}/path_validation_v2_summary.parquet')
        blob.upload_from_filename(summary_local)

    chart_local = f'{LOCAL_RESULTS_DIR}/moirai_validation_v2.png'
    if os.path.exists(chart_local):
        blob = bucket.blob(f'{GCS_MOIRAI_PATHS}/moirai_validation_v2.png')
        blob.upload_from_filename(chart_local)

    print(f"Uploaded {count} .npy files + summary + chart to "
          f"gs://{BUCKET_NAME}/{GCS_MOIRAI_PATHS}/")


upload_paths_to_gcs()
