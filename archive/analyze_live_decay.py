"""
analyze_live_decay.py — Model Decay Analysis with Live June 2026 Data
=====================================================================
Extends the training-time analysis (analyze_predictive_decay.py, May 28)
which used scoring CSVs and found:
  - 170 decay: -0.05pp/bucket (essentially flat)
  - 160 decay: -0.42pp/bucket (moderate)
  - Half-life: ~36 days (average, dragged by 160)
  - DID NOT INCLUDE JUNE DATA

This script uses actual live BQ predictions + OHLCV outcomes to answer:
  Has model precision degraded in June 2026?

For each (date, ticker), computes:
  - Model's ensemble probability prediction (from BQ)
  - Actual outcome: did price spike >= 8% from entry? (from OHLCV)
Then measures calibration, precision, and decay over time.

Tables:
  1: Bi-Weekly Precision Decay Curve
  2: Monthly Calibration (predicted prob vs actual spike rate)
  3: GO Signal Precision by Month
  4: Missed Spikes — actual spikes the model said NO to
  5: Per-Ticker April-May vs June
  6: Probability Distribution Shift
  7: Statistical Significance Tests
  SYNTHESIS: Model Decay Verdict

Run in Colab: paste as cell, runs immediately.
"""

import gc, sys, subprocess, warnings, ctypes
from datetime import datetime, timedelta
import numpy as np, pandas as pd, pytz
from scipy import stats as sp_stats

warnings.filterwarnings('ignore')

def _release_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

try:
    from google.cloud import bigquery
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "google-cloud-storage", "google-cloud-bigquery",
                           "pyarrow", "gcsfs", "--quiet"])
    from google.cloud import bigquery

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass


# ==============================================================================
# CONFIG
# ==============================================================================
BQ_TABLE      = f"{GCP_PROJECT}.scoring_logs.live_predictions"
BUCKET_NAME   = os.environ.get("GCS_BUCKET", "")
INFERENCE_DIR = "Data/Inference_v2"

START_DATE   = '2026-04-01'
END_DATE     = '2026-06-09'

SPIKE_THRESHOLD = 0.08
EOD_MINUTE      = 780

ALL_TICKERS = [
    'ASTS', 'BE', 'CLSK', 'HUT', 'IREN',
    'CIFR', 'MVIS', 'PLUG', 'RCAT',
    'ALAB', 'BBAI', 'FCEL', 'JOBY', 'KEEL', 'OKLO',
    'APLD', 'LUNR', 'RDW', 'RIOT',
]

TICKER_EXP = {
    'ASTS': 152, 'BE': 152, 'CLSK': 152, 'HUT': 152, 'IREN': 152,
    'CIFR': 160, 'MVIS': 160, 'PLUG': 160, 'RCAT': 160,
    'ALAB': 170, 'BBAI': 170, 'FCEL': 170, 'JOBY': 170, 'KEEL': 170, 'OKLO': 170,
    'APLD': 185, 'LUNR': 185, 'RDW': 185, 'RIOT': 185,
}

TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

PST = pytz.timezone('America/Los_Angeles')
W = 120


# ==============================================================================
# STEP 1: PULL ALL BQ PREDICTIONS (Apr-Jun)
# ==============================================================================
print(f"{'='*W}")
print(f"  MODEL DECAY ANALYSIS — LIVE DATA (Apr–Jun 2026)")
print(f"  Period: {START_DATE} → {END_DATE}  |  Spike threshold: {SPIKE_THRESHOLD:.0%}")
print(f"  Tickers: {len(ALL_TICKERS)}  |  Original analysis (May 28): 170 = -0.05pp/bucket")
print(f"{'='*W}")

bq_client = bigquery.Client()

ticker_list = ', '.join(f"'{t}'" for t in ALL_TICKERS)
q = f"""
SELECT
    score_date,
    ticker,
    experiment,
    seed,
    prob,
    threshold,
    signal,
    tp_pct,
    sl_pct,
    entry_minute,
    close AS entry_close,
    filter_status,
    run_timestamp
FROM `{BQ_TABLE}`
WHERE score_date >= '{START_DATE}'
  AND score_date <= '{END_DATE}'
  AND ticker IN ({ticker_list})
ORDER BY score_date, ticker, seed
"""

print(f"\n  Loading BQ scoring logs...")
df_raw = bq_client.query(q).to_dataframe()
print(f"  Raw rows: {len(df_raw):,}")

if len(df_raw) == 0:
    print("  ERROR: No scoring logs found. Check BQ table and date range.")
    raise SystemExit(0)

df_raw['score_date'] = pd.to_datetime(df_raw['score_date']).dt.strftime('%Y-%m-%d')
df_raw['prob'] = df_raw['prob'].fillna(0).astype(float)
df_raw['entry_minute'] = df_raw['entry_minute'].fillna(380).astype(int)
df_raw['entry_close'] = df_raw['entry_close'].fillna(0).astype(float)
df_raw['tp_pct'] = df_raw['tp_pct'].fillna(0).astype(float)
df_raw['sl_pct'] = df_raw['sl_pct'].fillna(0).astype(float)
df_raw['threshold'] = df_raw['threshold'].fillna(0.5).astype(float)


# ==============================================================================
# STEP 2: AGGREGATE TO ENSEMBLE PREDICTIONS
# ==============================================================================
print(f"\n  Aggregating to ensemble predictions per (date, ticker)...")

ensembles = []
for (date, ticker), grp in df_raw.groupby(['score_date', 'ticker']):
    ens_prob = grp['prob'].mean()
    threshold = grp['threshold'].iloc[-1]
    entry_min = int(grp['entry_minute'].iloc[-1])
    entry_close = grp['entry_close'].iloc[-1]
    tp_pct = grp['tp_pct'].iloc[-1]
    sl_pct = grp['sl_pct'].iloc[-1]
    experiment = grp['experiment'].iloc[-1]

    go_ok = grp[(grp['signal'] == 'GO') & (grp['filter_status'] == 'OK')]
    go_blocked = grp[(grp['signal'] == 'GO') & (grp['filter_status'] != 'OK')]

    if len(go_ok) > 0:
        final_signal = 'GO'
        filter_status = 'OK'
    elif len(go_blocked) > 0:
        final_signal = 'GO'
        filter_status = 'BLOCKED'
    else:
        final_signal = 'NO'
        filter_status = 'N/A'

    n_seeds = grp['seed'].nunique()

    ensembles.append({
        'score_date': date, 'ticker': ticker, 'experiment': experiment,
        'ens_prob': ens_prob, 'threshold': threshold,
        'final_signal': final_signal, 'filter_status': filter_status,
        'entry_minute': entry_min, 'entry_close': entry_close,
        'tp_pct': tp_pct, 'sl_pct': sl_pct, 'n_seeds': n_seeds,
    })

df_ens = pd.DataFrame(ensembles)
n_go = ((df_ens['final_signal'] == 'GO') & (df_ens['filter_status'] == 'OK')).sum()
print(f"  Ensemble predictions: {len(df_ens):,}")
print(f"  GO + OK (traded): {n_go}")
print(f"  Trading days: {df_ens['score_date'].nunique()}")

del df_raw
_release_memory()


# ==============================================================================
# STEP 3: LOAD OHLCV & COMPUTE ACTUAL OUTCOMES
# ==============================================================================
print(f"\n  Loading OHLCV data...")

start_ts = pd.Timestamp(START_DATE, tz='UTC') - pd.Timedelta(days=1)
end_ts   = pd.Timestamp(END_DATE, tz='UTC') + pd.Timedelta(days=2)

ticker_ohlcv = {}
for ticker in ALL_TICKERS:
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    df_raw = None
    uri = f"gs://{BUCKET_NAME}/{INFERENCE_DIR}/{clean}_master_inference.parquet"
    try:
        df_raw = pd.read_parquet(uri)
    except Exception:
        print(f"    {ticker}: NOT FOUND ({uri})")
        continue
    df_raw = df_raw.reset_index()
    dt_cols = [c for c in df_raw.columns if pd.api.types.is_datetime64_any_dtype(df_raw[c])]
    if dt_cols:
        df_raw['Datetime'] = pd.to_datetime(df_raw[dt_cols[0]], utc=True)
    else:
        df_raw['Datetime'] = pd.to_datetime(df_raw[df_raw.columns[0]], unit='ms', utc=True)
    df_raw = df_raw.set_index('Datetime').sort_index()
    df_raw = df_raw[~df_raw.index.duplicated(keep='first')]
    std_map = {'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'}
    lower_map = {k.lower(): v for k, v in std_map.items()}
    df_raw.columns = [lower_map.get(str(c).strip().lower(), str(c).strip()) for c in df_raw.columns]
    df_raw = df_raw[(df_raw.index >= start_ts) & (df_raw.index <= end_ts)]

    df_pst = df_raw.copy()
    df_pst.index = df_pst.index.tz_convert(PST)
    ticker_ohlcv[ticker] = df_pst
    print(f"    {ticker}: {len(df_pst):,} bars")
    del df_raw
    _release_memory()

print(f"  Loaded OHLCV for {len(ticker_ohlcv)}/{len(ALL_TICKERS)} tickers")
_release_memory()


def compute_spike_outcome(ticker, date_str, entry_minute):
    """Check if price spiked >= SPIKE_THRESHOLD from entry bar."""
    if ticker not in ticker_ohlcv:
        return None, None, 0
    ohlcv = ticker_ohlcv[ticker]
    date = pd.Timestamp(date_str).date()
    day_bars = ohlcv[ohlcv.index.date == date]
    if day_bars.empty:
        return None, None, 0

    day_minutes = day_bars.index.hour * 60 + day_bars.index.minute
    after_entry = day_bars[day_minutes >= entry_minute]
    if after_entry.empty or after_entry.iloc[0]['Close'] <= 0:
        return None, None, 0

    entry_price = after_entry.iloc[0]['Close']
    post = after_entry.iloc[1:] if len(after_entry) > 1 else after_entry
    if post.empty:
        return entry_price, 0.0, len(after_entry)

    max_gain = (post['High'].max() / entry_price - 1)
    return entry_price, max_gain, len(after_entry)


print(f"\n  Computing spike outcomes for {len(df_ens)} predictions...")
ep_list, mg_list, nb_list = [], [], []
for _, row in df_ens.iterrows():
    ep, mg, nb = compute_spike_outcome(row['ticker'], row['score_date'],
                                        row['entry_minute'])
    ep_list.append(ep)
    mg_list.append(mg)
    nb_list.append(nb)

df_ens['entry_price_ohlcv'] = ep_list
df_ens['max_gain'] = mg_list
df_ens['n_bars'] = nb_list

df_ens['spike_happened'] = (df_ens['max_gain'] >= SPIKE_THRESHOLD).astype(int)
df_ens['has_data'] = df_ens['max_gain'].notna()

valid = df_ens[df_ens['has_data']].copy()
print(f"  Predictions with OHLCV data: {len(valid)}/{len(df_ens)}")
print(f"  Actual spikes (≥{SPIKE_THRESHOLD:.0%}): {valid['spike_happened'].sum()}")
print(f"  Base spike rate: {valid['spike_happened'].mean():.1%}")

valid['date_ts'] = pd.to_datetime(valid['score_date'])
valid['month'] = valid['date_ts'].dt.strftime('%Y-%m')
valid['month_label'] = valid['date_ts'].dt.strftime('%b')
min_date = valid['date_ts'].min()
valid['days_since'] = (valid['date_ts'] - min_date).dt.days
valid['bucket'] = valid['days_since'] // 14

traded = valid[(valid['final_signal'] == 'GO') & (valid['filter_status'] == 'OK')].copy()
go_all = valid[valid['final_signal'] == 'GO'].copy()
no_sig = valid[valid['final_signal'] == 'NO'].copy()


# ==============================================================================
# TABLE 1: BI-WEEKLY PRECISION DECAY CURVE
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 1: PRECISION DECAY CURVE — Bi-Weekly Buckets")
print(f"  GO+OK (traded) signal precision over time")
print(f"  Compare to original: 170 = -0.05pp/bucket, 160 = -0.42pp/bucket")
print(f"{'='*W}")

print(f"\n  {'Bucket':>6} {'Period':>16} {'Predictions':>11} {'GO+OK':>6} {'Spikes':>7} "
      f"{'Precision':>10} {'BaseRate':>9} {'Lift':>6}")
print(f"  {'─'*6} {'─'*16} {'─'*11} {'─'*6} {'─'*7} {'─'*10} {'─'*9} {'─'*6}")

bucket_data = []
for b in sorted(valid['bucket'].unique()):
    bkt_all = valid[valid['bucket'] == b]
    bkt_go = traded[traded['bucket'] == b]

    dates = bkt_all['score_date'].unique()
    period_label = f"{min(dates)[:5]}–{max(dates)[:5]}"

    n_pred = len(bkt_all)
    n_go = len(bkt_go)
    n_spikes = bkt_all['spike_happened'].sum()
    base_rate = bkt_all['spike_happened'].mean()

    if n_go >= 3:
        prec = bkt_go['spike_happened'].mean()
        lift = prec / base_rate if base_rate > 0 else 0
        bucket_data.append({'bucket': b, 'precision': prec, 'n_go': n_go,
                            'base_rate': base_rate})
        print(f"  {b:>6} {period_label:>16} {n_pred:>11} {n_go:>6} "
              f"{int(n_spikes):>7} {prec:>9.1%} {base_rate:>8.1%} {lift:>5.1f}x")
    else:
        print(f"  {b:>6} {period_label:>16} {n_pred:>11} {n_go:>6} "
              f"{int(n_spikes):>7} {'—':>10} {base_rate:>8.1%} {'—':>6}")

if len(bucket_data) >= 3:
    x = np.array([d['bucket'] for d in bucket_data])
    y = np.array([d['precision'] for d in bucket_data])
    slope, intercept, r_val, p_val, _ = sp_stats.linregress(x, y)
    trend = ("DECAY" if slope < -0.01 and p_val < 0.20 else
             "RISING" if slope > 0.01 and p_val < 0.20 else "STABLE")
    print(f"\n  Live decay rate: {slope*100:+.2f}pp/bucket (R²={r_val**2:.3f}, p={p_val:.3f}) → {trend}")
    print(f"  Original (May 28): 170 = -0.05pp/bucket, 160 = -0.42pp/bucket")
    live_slope = slope
else:
    print(f"\n  Not enough buckets for regression")
    live_slope = None


# ==============================================================================
# TABLE 2: MONTHLY CALIBRATION
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 2: MONTHLY CALIBRATION — Predicted Prob vs Actual Spike Rate")
print(f"  All predictions (GO + NO), binned by ensemble probability")
print(f"{'='*W}")

prob_bins = [(0, 0.20, '<20%'), (0.20, 0.40, '20-40%'), (0.40, 0.60, '40-60%'),
             (0.60, 0.80, '60-80%'), (0.80, 1.01, '80%+')]
months = sorted(valid['month'].unique())

hdr = f"  {'Prob Bucket':>12}"
for m in months:
    hdr += f" {'Rate(N)':>12}"
print(f"\n  {'':>12}", end='')
for m in months:
    label = valid[valid['month'] == m]['month_label'].iloc[0]
    print(f" {label:>12}", end='')
print()
print(f"  {'─'*12}" + f" {'─'*12}" * len(months))

for lo, hi, label in prob_bins:
    line = f"  {label:>12}"
    for m in months:
        mdf = valid[(valid['month'] == m) &
                     (valid['ens_prob'] >= lo) & (valid['ens_prob'] < hi)]
        if len(mdf) >= 3:
            rate = mdf['spike_happened'].mean()
            line += f" {rate:>5.0%}({len(mdf):>4})"
        else:
            line += f" {'—':>12}"
    print(line)

print(f"\n  Perfect calibration: actual rate should ≈ midpoint of prob bucket.")
print(f"  If June high-prob bins have lower actual rate → model is over-confident.")


# ==============================================================================
# TABLE 3: GO SIGNAL PRECISION BY MONTH
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 3: GO SIGNAL PRECISION BY MONTH")
print(f"{'='*W}")

print(f"\n  {'Month':>8} {'Traded':>7} {'Hits':>5} {'Precision':>10} {'AvgProb':>8} "
      f"{'AvgGain':>8} {'AvgDrop':>8} {'SpikesTotal':>12}")
print(f"  {'─'*8} {'─'*7} {'─'*5} {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*12}")

month_prec = {}
for m in months:
    m_traded = traded[traded['month'] == m]
    m_all = valid[valid['month'] == m]
    n = len(m_traded)
    if n == 0:
        continue

    hits = m_traded['spike_happened'].sum()
    prec = m_traded['spike_happened'].mean()
    avg_prob = m_traded['ens_prob'].mean()
    avg_gain = m_traded['max_gain'].mean() * 100
    avg_drop = m_traded[m_traded['max_gain'] < 0]['max_gain'].mean() * 100 if (m_traded['max_gain'] < 0).any() else 0
    total_spikes = m_all['spike_happened'].sum()
    month_prec[m] = {'n': n, 'hits': int(hits), 'prec': prec}

    label = m_all['month_label'].iloc[0]
    print(f"  {label:>8} {n:>7} {int(hits):>5} {prec:>9.1%} {avg_prob:>7.1%} "
          f"{avg_gain:>+7.1f}% {avg_drop:>+7.1f}% {int(total_spikes):>12}")

if len(month_prec) >= 2:
    ms = sorted(month_prec.keys())
    early = [month_prec[m] for m in ms[:-1]]
    late = month_prec[ms[-1]]
    early_hits = sum(m['hits'] for m in early)
    early_n = sum(m['n'] for m in early)
    early_prec = early_hits / early_n if early_n > 0 else 0
    print(f"\n  Apr-May combined: {early_prec:.1%} ({early_hits}/{early_n})")
    print(f"  June:             {late['prec']:.1%} ({late['hits']}/{late['n']})")
    print(f"  Delta:            {(late['prec'] - early_prec)*100:+.1f}pp")


# ==============================================================================
# TABLE 4: MISSED SPIKES — Model said NO but ≥8% spike happened
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 4: MISSED SPIKES — Model Said NO, Spike ≥{SPIKE_THRESHOLD:.0%} Happened")
print(f"{'='*W}")

missed = no_sig[no_sig['spike_happened'] == 1].copy()
print(f"\n  Total missed spikes: {len(missed)}")

if len(missed) > 0:
    print(f"\n  {'Date':>10} {'Ticker':<7} {'Prob':>6} {'Thresh':>7} {'MaxGain':>8} {'Exp':>6}")
    print(f"  {'─'*10} {'─'*7} {'─'*6} {'─'*7} {'─'*8} {'─'*6}")
    for _, row in missed.sort_values('score_date').iterrows():
        print(f"  {row['score_date']:>10} {row['ticker']:<7} "
              f"{row['ens_prob']:>5.1%} {row['threshold']:>6.1%} "
              f"{row['max_gain']*100:>+7.1f}% {row['experiment']:>6}")

    missed_by_month = missed.groupby('month').size()
    print(f"\n  Missed spikes by month:")
    for m in months:
        n_missed = missed_by_month.get(m, 0)
        total_no = len(no_sig[no_sig['month'] == m])
        total_spikes_no = no_sig[(no_sig['month'] == m) & (no_sig['spike_happened'] == 1)].shape[0]
        print(f"    {m}: {n_missed} missed spikes (out of {total_no} NO signals)")
else:
    print(f"  No missed spikes — model caught all ≥{SPIKE_THRESHOLD:.0%} moves.")


# ==============================================================================
# TABLE 5: PER-TICKER — April-May vs June
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 5: PER-TICKER PRECISION — April-May vs June")
print(f"  Only tickers with ≥2 traded signals in both periods shown")
print(f"{'='*W}")

jun_month = [m for m in months if m.startswith('2026-06')]
pre_months = [m for m in months if not m.startswith('2026-06')]

print(f"\n  {'Ticker':<7} {'Exp':>5} {'AprMay':>8} {'(N)':>5} {'June':>8} {'(N)':>5} "
      f"{'Delta':>7} {'Assessment':>12}")
print(f"  {'─'*7} {'─'*5} {'─'*8} {'─'*5} {'─'*8} {'─'*5} {'─'*7} {'─'*12}")

ticker_decay_count = 0
ticker_stable_count = 0
for ticker in sorted(ALL_TICKERS):
    t_pre = traded[(traded['ticker'] == ticker) & (traded['month'].isin(pre_months))]
    t_jun = traded[(traded['ticker'] == ticker) & (traded['month'].isin(jun_month))]

    exp = TICKER_EXP.get(ticker, '?')

    if len(t_pre) >= 2 and len(t_jun) >= 1:
        pre_prec = t_pre['spike_happened'].mean()
        jun_prec = t_jun['spike_happened'].mean()
        delta = jun_prec - pre_prec
        if delta < -0.20:
            assess = "DEGRADED"
            ticker_decay_count += 1
        elif delta > 0.20:
            assess = "IMPROVED"
        else:
            assess = "STABLE"
            ticker_stable_count += 1
        print(f"  {ticker:<7} {exp:>5} {pre_prec:>7.0%} ({len(t_pre):>3}) "
              f"{jun_prec:>7.0%} ({len(t_jun):>3}) {delta*100:>+6.0f}pp {assess:>12}")
    elif len(t_pre) >= 2:
        pre_prec = t_pre['spike_happened'].mean()
        print(f"  {ticker:<7} {exp:>5} {pre_prec:>7.0%} ({len(t_pre):>3}) "
              f"{'—':>8} ({len(t_jun):>3}) {'—':>7} {'NO JUN DATA':>12}")
    elif len(t_jun) >= 1:
        jun_prec = t_jun['spike_happened'].mean()
        print(f"  {ticker:<7} {exp:>5} {'—':>8} ({len(t_pre):>3}) "
              f"{jun_prec:>7.0%} ({len(t_jun):>3}) {'—':>7} {'NO APR-MAY':>12}")

if ticker_decay_count + ticker_stable_count > 0:
    print(f"\n  Degraded: {ticker_decay_count}  Stable: {ticker_stable_count}")


# ==============================================================================
# TABLE 6: PROBABILITY DISTRIBUTION SHIFT
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 6: PROBABILITY DISTRIBUTION SHIFT")
print(f"  Has the model's output probability distribution changed?")
print(f"{'='*W}")

print(f"\n  {'Month':>8} {'MeanProb':>9} {'MedianP':>8} {'P>50%':>6} {'P>60%':>6} "
      f"{'P>70%':>6} {'P>80%':>6} {'N':>6}")
print(f"  {'─'*8} {'─'*9} {'─'*8} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")

for m in months:
    mdf = valid[valid['month'] == m]
    label = mdf['month_label'].iloc[0]
    mean_p = mdf['ens_prob'].mean()
    med_p = mdf['ens_prob'].median()
    p50 = (mdf['ens_prob'] > 0.50).mean()
    p60 = (mdf['ens_prob'] > 0.60).mean()
    p70 = (mdf['ens_prob'] > 0.70).mean()
    p80 = (mdf['ens_prob'] > 0.80).mean()
    print(f"  {label:>8} {mean_p:>8.1%} {med_p:>7.1%} {p50:>5.0%} {p60:>5.0%} "
          f"{p70:>5.0%} {p80:>5.0%} {len(mdf):>6}")

print(f"\n  If prob distribution shifted DOWN → model sees fewer opportunities")
print(f"  If prob distribution is SAME but precision dropped → calibration broke")


# ==============================================================================
# TABLE 7: STATISTICAL SIGNIFICANCE
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 7: STATISTICAL SIGNIFICANCE TESTS")
print(f"  Is the June precision decline statistically significant?")
print(f"{'='*W}")

traded_pre = traded[traded['month'].isin(pre_months)]
traded_jun = traded[traded['month'].isin(jun_month)]

if len(traded_pre) >= 5 and len(traded_jun) >= 3:
    pre_hits = int(traded_pre['spike_happened'].sum())
    pre_n = len(traded_pre)
    jun_hits = int(traded_jun['spike_happened'].sum())
    jun_n = len(traded_jun)

    pre_prec = pre_hits / pre_n
    jun_prec = jun_hits / jun_n

    print(f"\n  Apr-May: {pre_hits}/{pre_n} = {pre_prec:.1%} precision")
    print(f"  June:    {jun_hits}/{jun_n} = {jun_prec:.1%} precision")

    table = np.array([[pre_hits, pre_n - pre_hits],
                      [jun_hits, jun_n - jun_hits]])
    odds_ratio, fisher_p = sp_stats.fisher_exact(table, alternative='greater')
    print(f"\n  Fisher's exact test (one-sided: Apr-May > June):")
    print(f"    Odds ratio: {odds_ratio:.2f}")
    print(f"    p-value:    {fisher_p:.4f}")

    if fisher_p < 0.05:
        print(f"    → SIGNIFICANT at 5% level — June precision is statistically lower")
    elif fisher_p < 0.10:
        print(f"    → MARGINAL at 10% level — suggestive but not conclusive")
    else:
        print(f"    → NOT SIGNIFICANT — cannot rule out random variance")

    z_pre = pre_prec
    se = np.sqrt(z_pre * (1 - z_pre) / jun_n)
    ci_lo = z_pre - 1.96 * se
    ci_hi = z_pre + 1.96 * se
    print(f"\n  If true rate = Apr-May rate ({pre_prec:.1%}), expected June range:")
    print(f"    95% CI: [{ci_lo:.1%}, {ci_hi:.1%}] (N={jun_n})")
    print(f"    Actual June: {jun_prec:.1%}")
    if jun_prec < ci_lo:
        print(f"    → June is BELOW the expected range")
    else:
        print(f"    → June is WITHIN the expected range (variance)")

    n_sims = 10000
    sim_hits = np.random.binomial(jun_n, pre_prec, n_sims)
    sim_prec = sim_hits / jun_n
    p_worse = (sim_prec <= jun_prec).mean()
    print(f"\n  Monte Carlo simulation (10K draws at base rate {pre_prec:.1%}, N={jun_n}):")
    print(f"    P(precision ≤ {jun_prec:.1%}) = {p_worse:.4f}")
    if p_worse < 0.05:
        print(f"    → Very unlikely by chance alone — suggests real degradation")
    elif p_worse < 0.15:
        print(f"    → Unlikely but possible — could be bad luck + small sample")
    else:
        print(f"    → Within normal variance — no evidence of degradation")
else:
    print(f"\n  Insufficient data for significance tests")
    print(f"  Apr-May traded: {len(traded_pre)}, June traded: {len(traded_jun)}")
    fisher_p = None
    p_worse = None


# ==============================================================================
# TABLE 8: BY EXPERIMENT FAMILY
# ==============================================================================
print(f"\n{'='*W}")
print(f"  TABLE 8: DECAY BY EXPERIMENT FAMILY")
print(f"  Original analysis: 170 = -0.05pp/bucket, 160 = -0.42pp/bucket")
print(f"{'='*W}")

exp_families = {152: '152 (13T)', 160: '160 (13T)', 170: '170 (31T)', 185: '185 (31T)'}

print(f"\n  {'Family':>12} {'AprMay Prec':>12} {'(N)':>5} {'June Prec':>10} {'(N)':>5} "
      f"{'Delta':>7} {'Status':>10}")
print(f"  {'─'*12} {'─'*12} {'─'*5} {'─'*10} {'─'*5} {'─'*7} {'─'*10}")

for exp_id, label in sorted(exp_families.items()):
    tickers_in_exp = [t for t, e in TICKER_EXP.items() if e == exp_id]
    t_pre = traded_pre[traded_pre['ticker'].isin(tickers_in_exp)]
    t_jun = traded_jun[traded_jun['ticker'].isin(tickers_in_exp)]

    if len(t_pre) >= 2:
        pre_p = t_pre['spike_happened'].mean()
        pre_label = f"{pre_p:.0%}"
    else:
        pre_p = None
        pre_label = "—"

    if len(t_jun) >= 1:
        jun_p = t_jun['spike_happened'].mean()
        jun_label = f"{jun_p:.0%}"
    else:
        jun_p = None
        jun_label = "—"

    if pre_p is not None and jun_p is not None:
        delta = (jun_p - pre_p) * 100
        status = "DECAY" if delta < -15 else ("OK" if delta > -15 else "FLAT")
        print(f"  {label:>12} {pre_label:>12} ({len(t_pre):>3}) "
              f"{jun_label:>10} ({len(t_jun):>3}) {delta:>+6.0f}pp {status:>10}")
    else:
        print(f"  {label:>12} {pre_label:>12} ({len(t_pre):>3}) "
              f"{jun_label:>10} ({len(t_jun):>3}) {'—':>7} {'—':>10}")


# ==============================================================================
# SYNTHESIS: MODEL DECAY VERDICT
# ==============================================================================
print(f"\n{'='*W}")
print(f"  SYNTHESIS — MODEL DECAY VERDICT")
print(f"{'='*W}")

print(f"\n  EVIDENCE:")

print(f"\n  1. PRECISION TREND")
if live_slope is not None:
    print(f"     Live decay rate: {live_slope*100:+.2f}pp per 2-week bucket")
    print(f"     Original (training CSVs, May 28): 170 = -0.05pp/bucket")
    if abs(live_slope) > 0.10:
        print(f"     → Live decay is FASTER than historical — concerning")
    elif abs(live_slope) > 0.03:
        print(f"     → Live decay is MODERATE — worth monitoring")
    else:
        print(f"     → Live decay is MINIMAL — consistent with historical")

print(f"\n  2. STATISTICAL SIGNIFICANCE")
if fisher_p is not None:
    if fisher_p < 0.05:
        print(f"     Fisher p = {fisher_p:.4f} — SIGNIFICANT decline")
    elif fisher_p < 0.10:
        print(f"     Fisher p = {fisher_p:.4f} — MARGINAL (suggestive, not conclusive)")
    else:
        print(f"     Fisher p = {fisher_p:.4f} — NOT SIGNIFICANT (could be variance)")

if p_worse is not None:
    print(f"     Monte Carlo P(this bad or worse) = {p_worse:.4f}")

print(f"\n  3. SAMPLE SIZE CONTEXT")
print(f"     June has only {len(traded_jun)} traded signals vs {len(traded_pre)} in Apr-May")
if len(traded_jun) < 15:
    print(f"     → Very small sample — high variance expected")
    print(f"     → At true 60% WR, N={len(traded_jun)}: 95% CI = "
          f"[{max(0, 0.6 - 1.96*np.sqrt(0.6*0.4/max(len(traded_jun),1))):.0%}, "
          f"{min(1, 0.6 + 1.96*np.sqrt(0.6*0.4/max(len(traded_jun),1))):.0%}]")

print(f"\n  4. CALIBRATION")
has_calib_issue = False
for m in jun_month:
    mdf = valid[(valid['month'] == m) & (valid['ens_prob'] >= 0.60)]
    if len(mdf) >= 3:
        actual = mdf['spike_happened'].mean()
        if actual < 0.30:
            print(f"     June high-prob signals (≥60%): actual spike rate = {actual:.0%}")
            print(f"     → Model is OVER-CONFIDENT — predicts high prob but spikes don't happen")
            has_calib_issue = True
        else:
            print(f"     June high-prob signals (≥60%): actual spike rate = {actual:.0%} — OK")

if not has_calib_issue:
    print(f"     No calibration breakdown detected in June")

print(f"\n  5. CORRELATION-ADJUSTED VIEW")
print(f"     June 'independent' failures may really be 2-3 correlated events:")
corr_dates = traded_jun.groupby('score_date').agg(
    n_trades=('ticker', 'size'),
    n_hits=('spike_happened', 'sum'),
    avg_gain=('max_gain', 'mean')
).reset_index()
n_down_days = (corr_dates['avg_gain'] < 0).sum()
n_up_days = (corr_dates['avg_gain'] >= 0).sum()
print(f"     June trading days: {len(corr_dates)} ({n_up_days} net positive, {n_down_days} net negative)")
if len(corr_dates) > 0:
    for _, cd in corr_dates.iterrows():
        emoji = "✓" if cd['avg_gain'] >= 0 else "✗"
        print(f"       {cd['score_date']}: {int(cd['n_trades'])} trades, "
              f"{int(cd['n_hits'])} hits, avg {cd['avg_gain']*100:+.1f}% {emoji}")
    effective_n = len(corr_dates)
    print(f"     Effective independent samples: ~{effective_n} (days), not {len(traded_jun)} (trades)")

print(f"\n  ╔══════════════════════════════════════════════════════════╗")
print(f"  ║  VERDICT: {{to be determined from output above}}          ║")
print(f"  ║                                                          ║")
print(f"  ║  Key question: is the June decline larger than what      ║")
print(f"  ║  you'd expect from ~{len(corr_dates) if len(corr_dates) > 0 else '?'} independent observations?          ║")
print(f"  ║                                                          ║")
print(f"  ║  If Fisher p < 0.05 AND calibration broken:              ║")
print(f"  ║    → Model decay is CONFIRMED — consider retraining      ║")
print(f"  ║                                                          ║")
print(f"  ║  If Fisher p > 0.10 AND calibration intact:              ║")
print(f"  ║    → Variance / correlated bad luck — hold course        ║")
print(f"  ║                                                          ║")
print(f"  ║  If mixed signals:                                       ║")
print(f"  ║    → Insufficient data — add risk controls (cap-2),      ║")
print(f"  ║      wait 2 more weeks for more June data, recheck       ║")
print(f"  ╚══════════════════════════════════════════════════════════╝")

print(f"\n{'='*W}")
