"""
analyze_opening_patterns.py — Opening Session Pattern Classification (2026)
============================================================================
Classifies the first 2.5 hours of trading (-30m pre-open to +2h post-open)
into distinct pattern types. Assesses whether patterns are predictable
enough to train a classifier.

Window:  6:00 AM PST (-30m) → 8:30 AM PST (+2h)
Ref:     Market open at 6:30 AM PST (9:30 AM ET)
Data:    2026 only, all available tickers in GCS

Patterns (10 types):
  FLAT            — Range < 1%, no direction
  STEADY_RISE     — Smooth uptrend, never dips much
  STEADY_DECLINE  — Smooth downtrend, never spikes much
  DIP_RECOVERY    — Drops >1.5% then recovers above open (V-shape)
  SPIKE_HOLD      — Spikes >1.5% early/mid and holds near high
  SPIKE_FADE      — Spikes >1.5% then gives back >50% of gains
  SPIKE_REVERSAL  — Spikes >1.5% then crashes below open
  SELLOFF         — Drops >1.5% and stays near lows
  SELLOFF_BOUNCE  — Drops >1.5%, partial recovery but still negative
  CHOPPY          — Everything else (mixed, no clear pattern)

Output tables:
  1. Raw metric distributions (percentiles)
  2. Overall pattern distribution
  3. Monthly evolution (Jan → Jun 2026)
  4. Per-sector breakdown
  5. Per-ticker pattern profiles (dominant pattern, entropy)
  6. Pattern quality metrics (avg return, range, continuation)
  7. Average price paths per pattern (sparkline + checkpoints)
  8. Gap size → pattern cross-tabulation
  9. Pattern transition matrix (today → tomorrow)
  10. Trainability assessment

Colab-ready. Uses google.cloud.storage API (not FUSE).
"""

import os, gc, sys, subprocess, warnings, ctypes
from collections import defaultdict
import numpy as np, pandas as pd, pytz

warnings.filterwarnings('ignore')

def _release_memory():
    gc.collect(); gc.collect()
    try: ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception: pass

try:
    from google.cloud import storage as gcs_storage
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
        "google-cloud-storage", "pyarrow", "gcsfs", "--quiet"])
    from google.cloud import storage as gcs_storage

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

# ==============================================================================
# CONFIG
# ==============================================================================
BUCKET_NAME = os.environ.get("GCS_BUCKET", "")
OHLCV_DIR = "Data/Inference"
INFERENCE_DIR = "Data/Inference_v2"

DATA_START = '2025-12-20'      # buffer for prev_close on Jan 1
ANALYSIS_START = '2026-01-01'
ANALYSIS_END = '2026-06-30'
MONTHS = ['2026-01', '2026-02', '2026-03', '2026-04', '2026-05', '2026-06']
MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']

PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

# ── Time constants (PST minutes from midnight) ──
PRE_OPEN_MIN    = 360   # 6:00 AM PST — 30 min before open
MKT_OPEN_MIN    = 390   # 6:30 AM PST — market open
SESSION_END_MIN = 510   # 8:30 AM PST — +2h after open
MKT_CLOSE_MIN   = 780   # 1:00 PM PST — market close
WINDOW_LEN      = SESSION_END_MIN - MKT_OPEN_MIN  # 120 minutes

# ── Checkpoints (minutes relative to market open) ──
CHECKPOINT_REL = [-30, -15, 0, 5, 10, 15, 30, 45, 60, 90, 120]
CP_COLS   = ['cp_m30','cp_m15','cp_0','cp_p5','cp_p10','cp_p15',
             'cp_p30','cp_p45','cp_p60','cp_p90','cp_p120']
CP_LABELS = ['-30m','-15m','Open','+5m','+10m','+15m',
             '+30m','+45m','+60m','+90m','+120m']

# ── Pattern classification thresholds ──
FLAT_THRESH  = 0.01    # 1% total range → FLAT
MOVE_THRESH  = 0.015   # 1.5% excursion → significant move
HOLD_RATIO   = 0.50    # held >50% of peak → SPIKE_HOLD
FADE_RATIO   = 0.50    # gave back >50% → SPIKE_FADE
RECOVER_RATIO = 0.30   # recovered >30% of dip → DIP_RECOVERY
EARLY_PCT    = 0.50    # first 50% of session
LATE_PCT     = 0.75    # first 75% of session

# ── Gap bins for cross-tab ──
GAP_BINS   = [-999, -3, -1.5, -0.5, 0.5, 1.5, 3, 999]
GAP_LABELS = ['<-3%', '-3/-1.5', '-1.5/-0.5', '-0.5/+0.5',
              '+0.5/+1.5', '+1.5/+3', '>+3%']

# ── Pattern order (for consistent display) ──
PATTERN_ORDER = [
    'FLAT', 'STEADY_RISE', 'STEADY_DECLINE', 'DIP_RECOVERY',
    'SPIKE_HOLD', 'SPIKE_FADE', 'SPIKE_REVERSAL',
    'SELLOFF', 'SELLOFF_BOUNCE', 'CHOPPY',
]

# ==============================================================================
# TICKER GROUPS
# ==============================================================================
GROUPS = {
    'deployed': [
        'HUT', 'FLNC', 'CIFR', 'CLSK', 'KEEL', 'RCAT', 'RIOT', 'APLD', 'WULF',
    ],
    'crypto': [
        'IREN', 'BTBT', 'MARA', 'ALAB', 'CORZ', 'COIN', 'MSTR', 'HOOD', 'BGDE',
    ],
    'defense_space': [
        'ASTS', 'LUNR', 'BBAI', 'RKLB', 'SPIR', 'BKSY', 'MNTS',
        'KTOS', 'AVAV', 'NNDM', 'MVIS',
    ],
    'energy_nuclear': [
        'PLUG', 'BE', 'FCEL', 'OKLO', 'SMR',
    ],
    'growth_evtol': [
        'JOBY', 'ACHR', 'RDW', 'PL', 'VOYG', 'MP',
    ],
    'large_cap': [
        'SMCI', 'AMD', 'NVDA', 'TSLA', 'PLTR', 'SOUN', 'VRT', 'VST', 'MU',
        'MOD', 'POWL', 'NBIS',
    ],
    'defense_prime': [
        'GD', 'GE', 'GEV', 'LHX', 'LMT', 'NOC', 'RTX',
    ],
    'commodity': [
        'BHP', 'CAT', 'CCJ', 'CEG', 'COP', 'CVX', 'XOM', 'FCX',
    ],
    'utility_infra': [
        'DUK', 'EQIX', 'ETN', 'LIN', 'NEE',
    ],
    'benchmark': [
        'SPY', 'QQQ', 'IWM', 'TLT',
    ],
}

TICKER_TO_GROUP = {}
ALL_TICKERS = []
for grp, tickers in GROUPS.items():
    for t in tickers:
        TICKER_TO_GROUP[t] = grp
        ALL_TICKERS.append(t)

gcs_client = gcs_storage.Client()


# ==============================================================================
# DATA LOADING
# ==============================================================================
def heal_data(df):
    df = df.resample('1min').asfreq()
    for c in ['Open', 'High', 'Low', 'Close']:
        if c in df.columns:
            df[c] = df[c].ffill()
    for c in ['Volume']:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df


def load_ohlcv_gcs(ticker):
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    for dir_path in [INFERENCE_DIR, OHLCV_DIR]:
        for suffix in ['_master_inference.parquet', '_1m_full_historical_data.parquet']:
            uri = f"gs://{BUCKET_NAME}/{dir_path}/{clean}{suffix}"
            try:
                df = pd.read_parquet(uri)
                df = df.reset_index()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
                if dt_cols:
                    df['Datetime'] = pd.to_datetime(df[dt_cols[0]], utc=True)
                else:
                    num_col = next((c for c in df.columns
                                   if c in ('Datetime','Date','Time','Timestamp','Level_0')),
                                  df.columns[0])
                    df['Datetime'] = pd.to_datetime(df[num_col], unit='ms', utc=True)
                df = df.set_index('Datetime').sort_index()
                df = df[~df.index.duplicated(keep='first')]
                std_map = {'open':'Open','high':'High','low':'Low',
                           'close':'Close','volume':'Volume'}
                lower_map = {k.lower():v for k,v in std_map.items()}
                df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip())
                              for c in df.columns]
                start = pd.Timestamp(DATA_START, tz='UTC')
                end = pd.Timestamp(ANALYSIS_END, tz='UTC') + pd.Timedelta(days=2)
                df = df[(df.index >= start) & (df.index <= end)]
                if df.empty:
                    continue
                df = heal_data(df)
                df.index = df.index.tz_convert(PST)
                return df
            except Exception:
                continue
    return None


# ==============================================================================
# PATTERN CLASSIFICATION
# ==============================================================================
def classify_pattern(max_up, max_down, end_ret, time_peak, time_trough,
                     range_pct, window_len):
    early = window_len * EARLY_PCT
    late  = window_len * LATE_PCT

    if range_pct < FLAT_THRESH:
        return 'FLAT'

    had_spike = max_up > MOVE_THRESH
    had_dip   = abs(max_down) > MOVE_THRESH

    if had_dip and time_trough <= late and end_ret > 0 and end_ret > abs(max_down) * RECOVER_RATIO:
        return 'DIP_RECOVERY'

    if had_spike and time_peak <= late and end_ret < -MOVE_THRESH * 0.3:
        return 'SPIKE_REVERSAL'

    if had_spike and time_peak <= late and end_ret > max_up * HOLD_RATIO:
        return 'SPIKE_HOLD'

    if had_spike and 0 <= end_ret <= max_up * (1 - HOLD_RATIO):
        return 'SPIKE_FADE'

    if had_dip and end_ret < max_down * 0.6:
        return 'SELLOFF'

    if had_dip and end_ret < 0 and end_ret > max_down * 0.4:
        return 'SELLOFF_BOUNCE'

    if end_ret > MOVE_THRESH * 0.5 and abs(max_down) < max(end_ret * 0.3, 0.003):
        return 'STEADY_RISE'

    if end_ret < -MOVE_THRESH * 0.5 and max_up < max(abs(end_ret) * 0.3, 0.003):
        return 'STEADY_DECLINE'

    return 'CHOPPY'


# ==============================================================================
# SESSION EXTRACTION
# ==============================================================================
def extract_ticker_sessions(df, ticker, group):
    records = []
    all_dates = sorted(set(df.index.date))
    prev_close = None
    analysis_start_date = pd.Timestamp(ANALYSIS_START).date()

    for date in all_dates:
        day = df[df.index.date == date]
        mins = np.array(day.index.hour * 60 + day.index.minute)

        mkt_mask = (mins >= MKT_OPEN_MIN) & (mins <= MKT_CLOSE_MIN)
        mkt_bars = day[mkt_mask]
        real_count = (mkt_bars['Volume'] > 0).sum() if len(mkt_bars) > 0 else 0

        if real_count < 30:
            real_bars = mkt_bars[mkt_bars['Volume'] > 0]
            if len(real_bars) > 0:
                prev_close = real_bars.iloc[-1]['Close']
            continue

        session = day[(mins >= PRE_OPEN_MIN) & (mins <= SESSION_END_MIN)]
        if len(session) < 30:
            eod = mkt_bars[mkt_bars['Volume'] > 0]
            if len(eod) > 0:
                prev_close = eod.iloc[-1]['Close']
            continue

        s_mins = np.array(session.index.hour * 60 + session.index.minute)
        post_open_mask = s_mins >= MKT_OPEN_MIN
        post_open = session[post_open_mask]

        if len(post_open) < 20:
            eod = mkt_bars[mkt_bars['Volume'] > 0]
            if len(eod) > 0:
                prev_close = eod.iloc[-1]['Close']
            continue

        open_price = post_open.iloc[0]['Open']
        if pd.isna(open_price) or open_price <= 0:
            eod = mkt_bars[mkt_bars['Volume'] > 0]
            if len(eod) > 0:
                prev_close = eod.iloc[-1]['Close']
            continue

        session_vol = post_open['Volume'].sum()
        if session_vol <= 0:
            prev_close = open_price
            continue

        gap = ((open_price - prev_close) / prev_close) if (prev_close and prev_close > 0) else 0.0

        path = []
        for cp in CHECKPOINT_REL:
            target = MKT_OPEN_MIN + cp
            matches = np.where(s_mins == target)[0]
            if len(matches) > 0:
                cp_price = session.iloc[matches[0]]['Close']
            else:
                diffs = np.abs(s_mins - target)
                nearest = np.argmin(diffs)
                cp_price = session.iloc[nearest]['Close']
            path.append((cp_price - open_price) / open_price)

        po_mins = np.array(post_open.index.hour * 60 + post_open.index.minute)
        returns_from_open = (post_open['Close'].values - open_price) / open_price

        max_up   = float(np.max(returns_from_open))
        max_down = float(np.min(returns_from_open))
        peak_idx   = int(np.argmax(returns_from_open))
        trough_idx = int(np.argmin(returns_from_open))
        time_peak   = int(po_mins[peak_idx]) - MKT_OPEN_MIN
        time_trough = int(po_mins[trough_idx]) - MKT_OPEN_MIN
        end_ret = float(returns_from_open[-1])

        range_pct = (post_open['High'].max() - post_open['Low'].min()) / open_price

        eod_bars = day[np.array(day.index.hour * 60 + day.index.minute) == MKT_CLOSE_MIN]
        if len(eod_bars) > 0 and eod_bars.iloc[-1]['Close'] > 0:
            eod_price = eod_bars.iloc[-1]['Close']
            sess_end_price = post_open.iloc[-1]['Close']
            continuation = (eod_price - sess_end_price) / sess_end_price if sess_end_price > 0 else 0.0
            full_day_ret = (eod_price - open_price) / open_price
        else:
            continuation = 0.0
            full_day_ret = end_ret

        five_min_closes = post_open['Close'].iloc[::5]
        five_rets = five_min_closes.pct_change().dropna().values
        dir_changes = int((np.abs(np.diff(np.sign(five_rets))) > 1).sum()) if len(five_rets) > 3 else 0

        pattern = classify_pattern(max_up, max_down, end_ret,
                                   time_peak, time_trough, range_pct, WINDOW_LEN)

        if date >= analysis_start_date:
            record = {
                'ticker': ticker, 'group': group,
                'date': date, 'month': date.strftime('%Y-%m'),
                'gap_pct': gap * 100,
                'open_price': open_price,
                'max_up': max_up * 100,
                'max_down': max_down * 100,
                'time_peak': time_peak,
                'time_trough': time_trough,
                'end_ret': end_ret * 100,
                'range_pct': range_pct * 100,
                'dir_changes': dir_changes,
                'continuation': continuation * 100,
                'full_day_ret': full_day_ret * 100,
                'pattern': pattern,
            }
            for i, cp in enumerate(CHECKPOINT_REL):
                record[CP_COLS[i]] = path[i] * 100
            records.append(record)

        real_eod = mkt_bars[mkt_bars['Volume'] > 0]
        if len(real_eod) > 0:
            prev_close = real_eod.iloc[-1]['Close']
        else:
            prev_close = post_open.iloc[-1]['Close']

    return records


# ==============================================================================
# SPARKLINE HELPER
# ==============================================================================
SPARK_CHARS = '▁▂▃▄▅▆▇█'

def sparkline(values):
    if not values or len(values) < 2:
        return ''
    v_min, v_max = min(values), max(values)
    v_range = v_max - v_min
    if v_range < 1e-9:
        return SPARK_CHARS[4] * len(values)
    chars = []
    for v in values:
        idx = int((v - v_min) / v_range * 7)
        chars.append(SPARK_CHARS[max(0, min(7, idx))])
    return ''.join(chars)


def shannon_entropy(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * np.log2(p) for p in probs)


# ==============================================================================
# MAIN — LOAD DATA & EXTRACT SESSIONS
# ==============================================================================
W = 120
print("=" * W)
print(f"  OPENING SESSION PATTERN ANALYSIS — 2026")
print("=" * W)
print(f"  Window: 6:00 AM → 8:30 AM PST (−30m to +2h from open)")
print(f"  Period: {ANALYSIS_START} → {ANALYSIS_END}")
print(f"  Tickers: {len(ALL_TICKERS)} across {len(GROUPS)} groups")
print(f"  Thresholds: flat<{FLAT_THRESH*100:.0f}%  move>{MOVE_THRESH*100:.1f}%")
print()

print(f"  Loading OHLCV data from GCS...")
all_records = []
loaded_tickers = []
failed_tickers = []

for ticker in ALL_TICKERS:
    df = load_ohlcv_gcs(ticker)
    if df is None:
        failed_tickers.append(ticker)
        print(f"    {ticker:>8}: MISSING")
        continue
    records = extract_ticker_sessions(df, ticker, TICKER_TO_GROUP[ticker])
    if records:
        all_records.extend(records)
        loaded_tickers.append(ticker)
        print(f"    {ticker:>8}: {len(records):>4} days")
    else:
        failed_tickers.append(ticker)
        print(f"    {ticker:>8}: no valid sessions")
    del df
    _release_memory()

print(f"\n  Loaded {len(loaded_tickers)}/{len(ALL_TICKERS)} tickers, "
      f"{len(all_records):,} ticker-day observations")

if not all_records:
    print("  No data — exiting.")
    sys.exit(0)

df_r = pd.DataFrame(all_records)
df_r['date'] = pd.to_datetime(df_r['date'])
df_r['month_label'] = df_r['month'].map(dict(zip(MONTHS, MONTH_LABELS)))

os.makedirs('Results', exist_ok=True)
_release_memory()


# ==============================================================================
# TABLE 1: RAW METRIC DISTRIBUTIONS
# ==============================================================================
print()
print("=" * W)
print(f"  TABLE 1: RAW METRIC DISTRIBUTIONS (all {len(df_r):,} observations)")
print("=" * W)
metric_cols_raw = ['gap_pct', 'max_up', 'max_down', 'end_ret', 'range_pct',
                   'continuation', 'full_day_ret', 'dir_changes']
metric_labels_raw = ['Gap%', 'MaxUp%', 'MaxDn%', 'EndRet%', 'Range%',
                     'Contin%', 'FullDay%', 'DirChg']
pctiles = [5, 25, 50, 75, 95]

print(f"  {'Metric':<10} {'Mean':>8} {'Std':>8}  " +
      '  '.join(f'P{p}' for p in pctiles))
print(f"  {'─'*10} {'─'*8} {'─'*8}  " +
      '  '.join('─'*8 for _ in pctiles))

for col, label in zip(metric_cols_raw, metric_labels_raw):
    vals = df_r[col]
    parts = [f"{vals.mean():>8.2f}", f"{vals.std():>8.2f}"]
    for p in pctiles:
        parts.append(f"{vals.quantile(p/100):>8.2f}")
    print(f"  {label:<10} {'  '.join(parts)}")


# ==============================================================================
# TABLE 2: OVERALL PATTERN DISTRIBUTION
# ==============================================================================
print()
print("=" * W)
print(f"  TABLE 2: OVERALL PATTERN DISTRIBUTION")
print("=" * W)

total = len(df_r)
print(f"  {'Pattern':<18} {'Count':>6} {'%':>7}  {'AvgEnd%':>8} {'AvgRange%':>10} "
      f"{'AvgMaxUp%':>10} {'AvgMaxDn%':>10} {'AvgContin%':>11}")
print(f"  {'─'*18} {'─'*6} {'─'*7}  {'─'*8} {'─'*10} {'─'*10} {'─'*10} {'─'*11}")

for pat in PATTERN_ORDER:
    sub = df_r[df_r['pattern'] == pat]
    n = len(sub)
    if n == 0:
        continue
    pct = n / total * 100
    print(f"  {pat:<18} {n:>6} {pct:>6.1f}%  {sub['end_ret'].mean():>+8.2f} "
          f"{sub['range_pct'].mean():>10.2f} {sub['max_up'].mean():>+10.2f} "
          f"{sub['max_down'].mean():>+10.2f} {sub['continuation'].mean():>+11.2f}")


# ==============================================================================
# TABLE 3: MONTHLY PATTERN EVOLUTION
# ==============================================================================
print()
print("=" * W)
print(f"  TABLE 3: MONTHLY PATTERN EVOLUTION (% of days)")
print("=" * W)

header = f"  {'Pattern':<18}"
for ml in MONTH_LABELS:
    header += f" {ml:>6}"
header += f"  {'Trend':>10}"
print(header)
print(f"  {'─'*18}" + " ─────" * len(MONTH_LABELS) + f"  {'─'*10}")

for pat in PATTERN_ORDER:
    line = f"  {pat:<18}"
    month_pcts = []
    for ms in MONTHS:
        month_total = len(df_r[df_r['month'] == ms])
        month_count = len(df_r[(df_r['month'] == ms) & (df_r['pattern'] == pat)])
        pct = month_count / month_total * 100 if month_total > 0 else 0
        month_pcts.append(pct)
        line += f" {pct:>5.1f}%"
    if len(month_pcts) >= 2 and month_pcts[-1] > 0:
        early_avg = np.mean(month_pcts[:3]) if len(month_pcts) >= 3 else month_pcts[0]
        late_avg = np.mean(month_pcts[-2:])
        delta = late_avg - early_avg
        if delta > 3:
            trend = f"↑ +{delta:.0f}pp"
        elif delta < -3:
            trend = f"↓ {delta:.0f}pp"
        else:
            trend = "→ stable"
    else:
        trend = ""
    line += f"  {trend:>10}"
    print(line)

print()
for ms, ml in zip(MONTHS, MONTH_LABELS):
    n = len(df_r[df_r['month'] == ms])
    print(f"  {ml}: {n} obs", end="  ")
print()


# ==============================================================================
# TABLE 4: PER-SECTOR BREAKDOWN
# ==============================================================================
print()
print("=" * W)
print(f"  TABLE 4: DOMINANT PATTERNS BY SECTOR")
print("=" * W)

for grp_name in GROUPS:
    sub = df_r[df_r['group'] == grp_name]
    if sub.empty:
        continue
    n = len(sub)
    n_tickers = sub['ticker'].nunique()

    pat_counts = sub['pattern'].value_counts()
    top3 = pat_counts.head(3)
    top3_str = '  '.join(f"{p}({c/n*100:.0f}%)" for p, c in top3.items())

    print(f"  {grp_name:<16} ({n_tickers:>2} tickers, {n:>5} obs)  "
          f"AvgRange={sub['range_pct'].mean():.2f}%  AvgEnd={sub['end_ret'].mean():+.2f}%")
    print(f"    Top: {top3_str}")


# ==============================================================================
# TABLE 5: PER-TICKER PATTERN PROFILES
# ==============================================================================
print()
print("=" * W)
print(f"  TABLE 5: PER-TICKER PATTERN PROFILES (sorted by entropy — most predictable first)")
print("=" * W)
print(f"  {'Ticker':<8} {'Group':<16} {'Days':>4} {'Entropy':>8} {'Dom.Pattern':<18} "
      f"{'%Dom':>5}  Top-3 Patterns")
print(f"  {'─'*8} {'─'*16} {'─'*4} {'─'*8} {'─'*18} {'─'*5}  {'─'*50}")

ticker_entropy = []
for ticker in sorted(df_r['ticker'].unique()):
    sub = df_r[df_r['ticker'] == ticker]
    if len(sub) < 20:
        continue
    counts = [len(sub[sub['pattern'] == p]) for p in PATTERN_ORDER]
    ent = shannon_entropy(counts)

    pat_vc = sub['pattern'].value_counts()
    dom_pat = pat_vc.index[0]
    dom_pct = pat_vc.iloc[0] / len(sub) * 100

    top3 = pat_vc.head(3)
    top3_str = '  '.join(f"{p}({c/len(sub)*100:.0f}%)" for p, c in top3.items())

    group = sub['group'].iloc[0]
    ticker_entropy.append((ent, ticker, group, len(sub), dom_pat, dom_pct, top3_str))

ticker_entropy.sort(key=lambda x: x[0])
for ent, ticker, group, n_days, dom_pat, dom_pct, top3_str in ticker_entropy:
    print(f"  {ticker:<8} {group:<16} {n_days:>4} {ent:>8.2f} {dom_pat:<18} {dom_pct:>4.0f}%  {top3_str}")


# ==============================================================================
# TABLE 6: PATTERN QUALITY METRICS
# ==============================================================================
print()
print("=" * W)
print(f"  TABLE 6: PATTERN QUALITY METRICS")
print("=" * W)
print(f"  {'Pattern':<18} {'N':>5} {'AvgEnd':>8} {'MedEnd':>8} {'AvgFull':>8} "
      f"{'MedFull':>8} {'AvgCont':>8} {'WR_2h':>6} {'WR_EOD':>6}")
print(f"  {'─'*18} {'─'*5} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*6}")

for pat in PATTERN_ORDER:
    sub = df_r[df_r['pattern'] == pat]
    n = len(sub)
    if n == 0:
        continue
    wr_2h  = (sub['end_ret'] > 0).mean() * 100
    wr_eod = (sub['full_day_ret'] > 0).mean() * 100
    print(f"  {pat:<18} {n:>5} {sub['end_ret'].mean():>+8.2f} {sub['end_ret'].median():>+8.2f} "
          f"{sub['full_day_ret'].mean():>+8.2f} {sub['full_day_ret'].median():>+8.2f} "
          f"{sub['continuation'].mean():>+8.2f} {wr_2h:>5.0f}% {wr_eod:>5.0f}%")

print()
print(f"  WR_2h  = % of days with positive return at session end (+2h)")
print(f"  WR_EOD = % of days with positive return at market close (1 PM PST)")
print(f"  AvgCont = avg return from session end to market close (pattern continuation)")


# ==============================================================================
# TABLE 7: AVERAGE PRICE PATHS PER PATTERN (sparkline + checkpoints)
# ==============================================================================
print()
print("=" * W)
print(f"  TABLE 7: AVERAGE PRICE PATHS PER PATTERN")
print("=" * W)
print(f"  Checkpoints: {' '.join(f'{l:>6}' for l in CP_LABELS)}")
print()

for pat in PATTERN_ORDER:
    sub = df_r[df_r['pattern'] == pat]
    n = len(sub)
    if n == 0:
        continue
    pct = n / total * 100
    avg_path = [sub[c].mean() for c in CP_COLS]
    med_path = [sub[c].median() for c in CP_COLS]
    spark = sparkline(avg_path)

    print(f"  {pat:<18} (N={n:>4}, {pct:>5.1f}%)  {spark}")
    print(f"    Avg:  {' '.join(f'{v:>+6.2f}' for v in avg_path)}")
    print(f"    Med:  {' '.join(f'{v:>+6.2f}' for v in med_path)}")
    print()


# ==============================================================================
# TABLE 8: GAP SIZE → PATTERN CROSS-TAB
# ==============================================================================
print()
print("=" * W)
print(f"  TABLE 8: GAP SIZE → PATTERN CROSS-TABULATION (row %)")
print("=" * W)

df_r['gap_bin'] = pd.cut(df_r['gap_pct'], bins=GAP_BINS, labels=GAP_LABELS)

header = f"  {'Gap Bin':<14} {'N':>5}"
for pat in PATTERN_ORDER:
    header += f" {pat[:7]:>7}"
print(header)
print(f"  {'─'*14} {'─'*5}" + " ─────" * len(PATTERN_ORDER) + "─")

for gap_label in GAP_LABELS:
    sub = df_r[df_r['gap_bin'] == gap_label]
    n = len(sub)
    if n < 10:
        continue
    line = f"  {gap_label:<14} {n:>5}"
    for pat in PATTERN_ORDER:
        pat_n = len(sub[sub['pattern'] == pat])
        pct = pat_n / n * 100
        line += f" {pct:>6.1f}%"
    print(line)

print()
print(f"  Key question: Do certain gap sizes predict certain patterns?")
print(f"  Look for columns where % varies strongly across gap bins.")


# ==============================================================================
# TABLE 9: PATTERN TRANSITION MATRIX (today → tomorrow, same ticker)
# ==============================================================================
print()
print("=" * W)
print(f"  TABLE 9: PATTERN TRANSITION MATRIX (today's pattern → tomorrow's)")
print("=" * W)
print(f"  Shows P(tomorrow=col | today=row) as %. Read rows as 'given today is X...'")
print()

transitions = defaultdict(lambda: defaultdict(int))
for ticker in df_r['ticker'].unique():
    tk = df_r[df_r['ticker'] == ticker].sort_values('date')
    pats = tk['pattern'].values
    dates = tk['date'].values
    for i in range(len(pats) - 1):
        if (dates[i+1] - dates[i]).astype('timedelta64[D]').astype(int) <= 5:
            transitions[pats[i]][pats[i+1]] += 1

active_pats = [p for p in PATTERN_ORDER if sum(transitions[p].values()) >= 10]

header = f"  {'From \\ To':<16}"
for p in active_pats:
    header += f" {p[:7]:>7}"
print(header)
print(f"  {'─'*16}" + " ─────" * len(active_pats) + "─")

for from_pat in active_pats:
    row_total = sum(transitions[from_pat].values())
    if row_total < 10:
        continue
    line = f"  {from_pat:<16}"
    for to_pat in active_pats:
        pct = transitions[from_pat][to_pat] / row_total * 100
        if pct > 15:
            line += f" {pct:>6.0f}%"
        else:
            line += f" {pct:>6.1f}"
    print(line)

print()
print(f"  Look for: diagonal dominance (pattern persistence) or "
      f"off-diagonal hotspots (predictable transitions)")


# ==============================================================================
# TABLE 10: TRAINABILITY ASSESSMENT
# ==============================================================================
print()
print("=" * W)
print(f"  TABLE 10: TRAINABILITY ASSESSMENT")
print("=" * W)

# 1. Entropy analysis
all_entropies = [e for e, *_ in ticker_entropy]
print(f"\n  1. PATTERN ENTROPY (lower = more predictable)")
print(f"     Mean entropy:   {np.mean(all_entropies):.2f} bits")
print(f"     Median entropy: {np.median(all_entropies):.2f} bits")
print(f"     Min entropy:    {min(all_entropies):.2f} bits ({ticker_entropy[0][1]})")
print(f"     Max entropy:    {max(all_entropies):.2f} bits ({ticker_entropy[-1][1]})")
print(f"     Max possible:   {np.log2(len(PATTERN_ORDER)):.2f} bits ({len(PATTERN_ORDER)} patterns)")
print(f"     → Avg information = {(1 - np.mean(all_entropies) / np.log2(len(PATTERN_ORDER))) * 100:.0f}% "
      f"reduction from uniform")

# 2. Ticker-specific predictability
low_ent = [x for x in ticker_entropy if x[0] < np.median(all_entropies)]
print(f"\n  2. MOST PREDICTABLE TICKERS (below-median entropy)")
print(f"     {len(low_ent)} tickers with entropy < {np.median(all_entropies):.2f}:")
for ent, ticker, group, n_days, dom_pat, dom_pct, _ in low_ent[:10]:
    print(f"       {ticker:<8} entropy={ent:.2f}  dominant={dom_pat}({dom_pct:.0f}%)")

# 3. Gap → pattern information
print(f"\n  3. GAP → PATTERN PREDICTIVE POWER")
gap_groups = df_r.groupby('gap_bin')['pattern'].apply(
    lambda x: shannon_entropy(x.value_counts().values))
unconditional_ent = shannon_entropy(df_r['pattern'].value_counts().values)
conditional_ent = gap_groups.mean()
info_gain = unconditional_ent - conditional_ent
print(f"     Unconditional entropy:   {unconditional_ent:.3f} bits")
print(f"     Conditional (given gap): {conditional_ent:.3f} bits")
print(f"     Information gain:        {info_gain:.3f} bits ({info_gain/unconditional_ent*100:.1f}% reduction)")
if info_gain > 0.1:
    print(f"     → Gap size IS informative for pattern prediction")
else:
    print(f"     → Gap size has LIMITED predictive value for pattern type")

# 4. Temporal stability
print(f"\n  4. TEMPORAL STABILITY (month-to-month pattern distribution)")
month_dists = {}
for ms, ml in zip(MONTHS, MONTH_LABELS):
    sub = df_r[df_r['month'] == ms]
    if len(sub) > 0:
        dist = np.array([len(sub[sub['pattern'] == p]) / len(sub) for p in PATTERN_ORDER])
        month_dists[ml] = dist

if len(month_dists) >= 2:
    months_list = list(month_dists.keys())
    corrs = []
    for i in range(len(months_list) - 1):
        c = np.corrcoef(month_dists[months_list[i]], month_dists[months_list[i+1]])[0, 1]
        corrs.append(c)
        print(f"     {months_list[i]} → {months_list[i+1]} distribution correlation: {c:.3f}")
    print(f"     Average consecutive-month correlation: {np.mean(corrs):.3f}")
    if np.mean(corrs) > 0.8:
        print(f"     → STABLE: Pattern distributions are consistent across months")
    elif np.mean(corrs) > 0.5:
        print(f"     → MODERATE: Some regime shifts between months")
    else:
        print(f"     → UNSTABLE: Pattern mix changes significantly month-to-month")

# 5. Continuation predictive value
print(f"\n  5. CONTINUATION VALUE (does opening pattern predict rest of day?)")
print(f"     {'Pattern':<18} {'2h_ret':>8} {'EOD_ret':>8} {'Contin':>8} {'Same_dir':>9}")
print(f"     {'─'*18} {'─'*8} {'─'*8} {'─'*8} {'─'*9}")
for pat in PATTERN_ORDER:
    sub = df_r[df_r['pattern'] == pat]
    if len(sub) < 20:
        continue
    same_dir = ((sub['end_ret'] > 0) == (sub['continuation'] > 0)).mean() * 100
    print(f"     {pat:<18} {sub['end_ret'].mean():>+8.2f} {sub['full_day_ret'].mean():>+8.2f} "
          f"{sub['continuation'].mean():>+8.2f} {same_dir:>8.0f}%")

print(f"\n     Same_dir = % of days where continuation has same sign as 2h return")
print(f"     Values near 50% = no continuation signal. >60% = pattern persists. <40% = mean reverts.")

# 6. Overall verdict
print(f"\n  {'─'*70}")
print(f"  OVERALL TRAINABILITY VERDICT:")
n_patterns = len([p for p in PATTERN_ORDER if len(df_r[df_r['pattern'] == p]) >= 20])
dominant_pct = df_r['pattern'].value_counts().iloc[0] / total * 100
print(f"    Active patterns: {n_patterns}/{len(PATTERN_ORDER)}")
print(f"    Dominant pattern: {df_r['pattern'].value_counts().index[0]} ({dominant_pct:.0f}%)")
print(f"    Avg ticker entropy: {np.mean(all_entropies):.2f} / {np.log2(len(PATTERN_ORDER)):.2f} max")
print(f"    Gap info gain: {info_gain:.3f} bits")

if np.mean(all_entropies) < 2.0 and n_patterns >= 5:
    print(f"\n    → TRAINABLE: Pattern distribution is non-uniform enough for classification.")
    print(f"      Recommend: Start with {n_patterns}-class classifier. Consider merging")
    print(f"      rare patterns (<5%) into CHOPPY for cleaner training signal.")
elif np.mean(all_entropies) < 2.5:
    print(f"\n    → PROMISING: Moderate structure exists. May need to merge similar patterns")
    print(f"      or use per-ticker/per-sector models instead of universal.")
else:
    print(f"\n    → CHALLENGING: High entropy — patterns are close to uniformly distributed.")
    print(f"      Consider: fewer pattern categories, ticker-specific models, or")
    print(f"      additional features (VIX, sector momentum) to improve separability.")


# ==============================================================================
# SAVE RESULTS
# ==============================================================================
save_cols = [c for c in df_r.columns if c != 'gap_bin']
df_r[save_cols].to_csv('Results/opening_patterns_2026.csv', index=False)
print(f"\n  Results saved to Results/opening_patterns_2026.csv")

print()
print("=" * W)
print(f"  Analysis complete — {len(df_r):,} observations, {len(loaded_tickers)} tickers,")
print(f"  {len([p for p in PATTERN_ORDER if len(df_r[df_r['pattern']==p]) > 0])} active patterns.")
print(f"  Next: If trainable, create exp_220_opening_pattern_train.py")
print("=" * W)
