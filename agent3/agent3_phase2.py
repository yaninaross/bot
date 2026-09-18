# %% Agent 3 — Phase 2 Investigation
# Deeper analysis of A6 sustained_rise and C1 reversal strategies.
# Loads enriched trade lists from Phase 1 cache (phase1_cache.pkl).
# If cache not found, re-runs Phase 1 data collection (~11 min).
#
# Groups:
#   G1: A6 sustained_rise subsegmentation (S1-S6)
#   G2: C1 reversal subsegmentation (C1-S1 through C1-S6)
#   G3: Combination strategies (COMBO-1,2,3)
#   G4: New ideas (N1-N6)
#
# Colab-cell friendly. Config vars at top.

import os, sys, time, copy, pickle
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd
import pytz

try:
    from tabulate import tabulate
except ImportError:
    import subprocess; subprocess.check_call([sys.executable, "-m", "pip", "install", "tabulate", "--quiet", "--break-system-packages"])
    from tabulate import tabulate

import matplotlib
try:
    get_ipython()
except NameError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from google.colab import auth
    try: auth.authenticate_user()
    except: pass
except: pass

ET = pytz.timezone('America/New_York')

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

BACKTEST_START = '2025-09-15'
BACKTEST_END   = '2026-09-12'
OOS_CUTOFF     = '2026-01-01'

INITIAL_CAPITAL = 100_000

# Phase 2 broadened thresholds (for parameter sweeps)
P2_C_DECLINE_MIN = 0.10   # loosest decline threshold tested
P2_C_GAP_MIN     = 0.02   # loosest gap threshold tested
P2_MAX_HOLD      = 20     # extended holds for C1-S4

_HOLIDAYS = {
    '2025-01-01','2025-01-20','2025-02-17','2025-04-18','2025-05-26',
    '2025-06-19','2025-07-04','2025-09-01','2025-11-27','2025-12-25',
    '2026-01-01','2026-01-19','2026-02-16','2026-04-03','2026-05-25',
    '2026-06-19','2026-07-03','2026-09-07','2026-11-26','2026-12-25',
    '2027-01-01','2027-01-18','2027-02-15','2027-03-26','2027-05-31',
    '2027-06-18','2027-07-05','2027-09-06','2027-11-25','2027-12-24',
}


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS (same as Phase 1)
# ═══════════════════════════════════════════════════════════════════════════════

def _is_td(d):
    if isinstance(d, str): d = datetime.strptime(d, '%Y-%m-%d').date()
    return d.weekday() < 5 and d.strftime('%Y-%m-%d') not in _HOLIDAYS

def _next_n_td(d, n=5):
    if isinstance(d, str): d = datetime.strptime(d, '%Y-%m-%d').date()
    out, d = [], d + timedelta(days=1)
    for _ in range(n * 3 + 10):
        if _is_td(d):
            out.append(d)
            if len(out) >= n: return out
        d += timedelta(days=1)
    return out

def _prior_n_td(d, n=20):
    if isinstance(d, str): d = datetime.strptime(d, '%Y-%m-%d').date()
    out = []
    d = d - timedelta(days=1)
    for _ in range(n * 3 + 10):
        if _is_td(d):
            out.append(d)
            if len(out) >= n: return list(reversed(out))
        d -= timedelta(days=1)
    return list(reversed(out))


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS (same as Phase 1)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(pnl_list, label=''):
    if not pnl_list:
        return {'label':label,'n':0,'avg':0,'med':0,'std':0,'wr':0,
                'sharpe':0,'pf':0,'ci_lo':0,'ci_hi':0,'trimmed_mean':0}
    a = np.array(pnl_list)
    n = len(a); avg = float(np.mean(a)); med = float(np.median(a))
    std = float(np.std(a, ddof=1)) if n > 1 else 0
    wr = float((a > 0).mean())
    sharpe = avg / std if std > 0 else 0
    se = std / np.sqrt(n) if n > 0 else 0
    w = a[a > 0]; l = a[a < 0]
    pf = float(w.sum() / abs(l.sum())) if len(l) > 0 and l.sum() != 0 else float('inf')
    p5, p95 = np.percentile(a, 5), np.percentile(a, 95)
    trimmed = a[(a >= p5) & (a <= p95)]
    tm = float(np.mean(trimmed)) if len(trimmed) > 0 else avg
    max_loss = float(a.min()) if n > 0 else 0
    return {'label':label,'n':n,'avg':avg,'med':med,'std':std,'wr':wr,
            'sharpe':sharpe,'pf':pf,'ci_lo':avg-1.96*se,'ci_hi':avg+1.96*se,
            'trimmed_mean':tm,'max_loss':max_loss}

def metrics_row(m, extra=[]):
    ci_flag = '✓' if m['ci_lo'] > 0 else '✗'
    return [m['label'], m['n'], f"{m['avg']:+.2f}%", f"{m['med']:+.2f}%",
            f"{m['trimmed_mean']:+.2f}%", f"{m['wr']*100:.0f}%",
            f"{m['sharpe']:.3f}", f"{m['pf']:.2f}",
            f"[{m['ci_lo']:+.2f},{m['ci_hi']:+.2f}] {ci_flag}"] + extra

def metrics_headers(extra=[]):
    return ['Label','n','Avg','Median','TrimMean','WR','Sharpe','PF','CI 95%'] + extra


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD PHASE 1 DATA
# ═══════════════════════════════════════════════════════════════════════════════

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

cache_path = os.path.join(BASE_DIR, 'phase1_cache.pkl')

print(f"\n{'='*80}")
print(f"  AGENT 3 — PHASE 2 INVESTIGATION")
print(f"  {BACKTEST_START} → {BACKTEST_END}  |  OOS cutoff: {OOS_CUTOFF}")
print(f"{'='*80}")

if os.path.exists(cache_path):
    print(f"\n  Loading Phase 1 cache from {cache_path}...")
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    trades_a = cache['trades_a']
    trades_b = cache['trades_b']
    trades_c = cache['trades_c']
    dates = cache['dates']
    _gd_cache = cache.get('_gd_cache', {})
    _ticker_ref = cache.get('_ticker_ref', {})
    print(f"  → Loaded: A={len(trades_a)}, B={len(trades_b)}, C={len(trades_c)} trades, "
          f"{len(dates)} dates, {len(_gd_cache)} daily cache entries")
else:
    print(f"\n  ⚠ Phase 1 cache not found at {cache_path}")
    print(f"  Run long_investigation.py first to generate phase1_cache.pkl")
    print(f"  Or run Phase 1 in the same notebook session.")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 ENRICHMENT — add features needed for new experiments
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n  Phase 2 enrichment...")

n_dates = len(dates)

# ── Extended hold periods for C trades (T+2 through T+20) ──
extended_outcome_dates = set()
for t in trades_c:
    nxt = _next_n_td(t['scan_date'], P2_MAX_HOLD)
    for k, d in enumerate(nxt):
        ds = d.strftime('%Y-%m-%d')
        t[f't{k+1}_date'] = ds
        extended_outcome_dates.add(ds)

# Fetch any missing outcome dates
to_fetch = sorted(d for d in extended_outcome_dates if d not in _gd_cache)
if to_fetch:
    import requests, re
    POLYGON_API_KEY = os.environ["POLYGON_API_KEY"]
    _BASE = "https://api.polygon.io"
    _TK_RE = re.compile(r'^[A-Z]{1,5}$')
    def _pg(url, params=None):
        p = dict(params or {}); p['apiKey'] = POLYGON_API_KEY
        for attempt in range(5):
            try:
                r = requests.get(url, params=p, timeout=30)
                if r.status_code == 429:
                    time.sleep(12 * (attempt + 1)); continue
                r.raise_for_status(); return r.json()
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                time.sleep(5 * (2 ** attempt))
        r = requests.get(url, params=p, timeout=60); r.raise_for_status(); return r.json()

    print(f"  Fetching {len(to_fetch)} additional outcome dates...")
    for j, od in enumerate(to_fetch):
        try:
            data = _pg(f"{_BASE}/v2/aggs/grouped/locale/us/market/stocks/{od}",
                       {"adjusted":"true","include_otc":"false"})
            out = {}
            for r in data.get('results', []):
                tk = r.get('T','')
                if _TK_RE.match(tk):
                    out[tk] = {'o':float(r.get('o',0)),'h':float(r.get('h',0)),
                               'l':float(r.get('l',0)),'c':float(r.get('c',0)),'v':int(r.get('v',0))}
            _gd_cache[od] = out
        except Exception as e:
            print(f"    WARN: {od}: {e}")
        if (j+1) % 20 == 0: print(f"    {j+1}/{len(to_fetch)}")

# Compute extended PnL for C trades
for t in trades_c:
    ep = t['entry_price']
    tk = t['ticker']
    for offset in range(1, P2_MAX_HOLD + 1):
        col = f'pnl_t{offset}'
        if t.get(col) is not None:
            continue
        date_key = t.get(f't{offset}_date', '')
        dd = _gd_cache.get(date_key, {}).get(tk, {})
        t[col] = round((dd['c'] - ep) / ep * 100, 3) if dd.get('c') else None

# ── Bounce score for A trades (S2) ──
for t in trades_a:
    gp = t.get('gap_pct', 0)
    pat = t.get('open_pattern', '')
    rise_30m = t.get('_rise_30m')
    recovery = t.get('_recovery')
    if rise_30m is None or recovery is None:
        rise_30m = 0
        recovery = 0
    if abs(gp) > 0 and rise_30m > 0:
        recovery_pct = recovery / rise_30m if rise_30m > 0 else 0
        t['bounce_score'] = (rise_30m / abs(gp)) * recovery_pct
    else:
        t['bounce_score'] = 0

# ── Day of week for all trades ──
for trades in [trades_a, trades_c]:
    for t in trades:
        dt = datetime.strptime(t['scan_date'], '%Y-%m-%d')
        t['day_of_week'] = dt.strftime('%A')
        t['is_monday'] = dt.weekday() == 0
        t['is_friday'] = dt.weekday() == 4

# ── SPY regime for all trades ──
for trades in [trades_a, trades_c]:
    for t in trades:
        prior_5d = _prior_n_td(t['scan_date'], 5)
        if len(prior_5d) >= 5:
            spy_today = _gd_cache.get(t['scan_date'], {}).get('SPY', {})
            spy_5d_ago = _gd_cache.get(prior_5d[0].strftime('%Y-%m-%d'), {}).get('SPY', {})
            if spy_today.get('c') and spy_5d_ago.get('c') and spy_5d_ago['c'] > 0:
                t['spy_5d_return'] = (spy_today['c'] - spy_5d_ago['c']) / spy_5d_ago['c']
            else:
                t['spy_5d_return'] = None
        else:
            t['spy_5d_return'] = None
        if t['spy_5d_return'] is not None:
            if t['spy_5d_return'] > 0.02:
                t['spy_regime'] = 'bull'
            elif t['spy_5d_return'] < -0.02:
                t['spy_regime'] = 'bear'
            else:
                t['spy_regime'] = 'flat'
        else:
            t['spy_regime'] = None

# ── Prior trend context for A trades (N2) ──
for t in trades_a:
    prior_5d = _prior_n_td(t['scan_date'], 6)
    if len(prior_5d) >= 6:
        close_6d_ago = _gd_cache.get(prior_5d[0].strftime('%Y-%m-%d'), {}).get(t['ticker'], {})
        close_1d_ago = _gd_cache.get(prior_5d[-1].strftime('%Y-%m-%d'), {}).get(t['ticker'], {})
        if close_6d_ago.get('c') and close_1d_ago.get('c') and close_6d_ago['c'] > 0:
            t['prior_5d_return'] = (close_1d_ago['c'] - close_6d_ago['c']) / close_6d_ago['c']
        else:
            t['prior_5d_return'] = None
    else:
        t['prior_5d_return'] = None

# ── Compute prior decline returns for C trades at multiple lookback periods ──
for t in trades_c:
    tk = t['ticker']
    prev_c = t['prior_close']
    for lb_days in [3, 5, 7, 10, 15]:
        prior = _prior_n_td(t['scan_date'], lb_days)
        if len(prior) >= lb_days:
            dd = _gd_cache.get(prior[0].strftime('%Y-%m-%d'), {}).get(tk, {})
            if dd.get('c') and dd['c'] > 0:
                t[f'decline_{lb_days}d'] = (prev_c - dd['c']) / dd['c']
            else:
                t[f'decline_{lb_days}d'] = None
        else:
            t[f'decline_{lb_days}d'] = None

# ── Alternative entry prices for A6 trades (S4) ──
# S4a: open price (already stored as 'open_price')
# S4b: local_low entry (already stored as 'entry_price')
# S4c: 10:30 AM entry — need bar 60 close from minute bars
# Since minute bars aren't stored in cache, compute S4 PnLs from available data
for t in trades_a:
    ep_open = t.get('open_price')
    if ep_open and ep_open > 0:
        for offset in range(1, 6):
            col_src = f'pnl_t{offset}'
            col_dst = f'pnl_open_t{offset}'
            date_key = t.get(f't{offset}_date', '')
            dd = _gd_cache.get(date_key, {}).get(t['ticker'], {})
            t[col_dst] = round((dd['c'] - ep_open) / ep_open * 100, 3) if dd.get('c') else None
    # pnl_60m gives us the 10:30 AM price (bar 60 = 60 min after open)
    bar60_close = None
    if t.get('pnl_60m') is not None and t.get('entry_price'):
        bar60_close = t['entry_price'] * (1 + t['pnl_60m'] / 100)
    if bar60_close and bar60_close > 0:
        t['entry_price_1030'] = bar60_close
        for offset in range(1, 6):
            date_key = t.get(f't{offset}_date', '')
            dd = _gd_cache.get(date_key, {}).get(t['ticker'], {})
            t[f'pnl_1030_t{offset}'] = round((dd['c'] - bar60_close) / bar60_close * 100, 3) if dd.get('c') else None

# ── Prior 5-day decline for A trades (for N3 overlap detection) ──
for t in trades_a:
    tk = t['ticker']
    prev_c = t['prior_close']
    prior_5d = _prior_n_td(t['scan_date'], 5)
    if len(prior_5d) >= 5:
        dd = _gd_cache.get(prior_5d[0].strftime('%Y-%m-%d'), {}).get(tk, {})
        if dd.get('c') and dd['c'] > 0:
            t['a_decline_5d'] = (prev_c - dd['c']) / dd['c']
        else:
            t['a_decline_5d'] = None
    else:
        t['a_decline_5d'] = None

# ── Sector from ticker reference ──
for trades in [trades_a, trades_c]:
    for t in trades:
        ref = _ticker_ref.get(t['ticker'], {})
        t['sector'] = ref.get('type', 'UNK')

# Rebuild bounce_score properly from PnL data
# rise_30m and recovery aren't stored directly — recompute from existing fields
# We know: pnl_30m = (close_30m - entry) / entry * 100
#           pnl_60m = (close_60m - entry) / entry * 100
# For sustained_rise classification:
#   rise_30m = (high_30m - open) / |open|
#   recovery = (close_60m - open) / |open|
# These were computed from raw bars in Phase 1 but not stored.
# Instead, approximate bounce_score from available fields:
for t in trades_a:
    gp = abs(t.get('gap_pct', 0))
    pnl30 = t.get('pnl_30m')
    pnl60 = t.get('pnl_60m')
    pat = t.get('open_pattern', '')
    if pat == 'sustained_rise' and pnl30 is not None and pnl60 is not None and gp > 0:
        rise_proxy = max(pnl30, 0) / 100
        recovery_proxy = pnl60 / 100 if pnl60 > 0 else 0
        t['bounce_score'] = (rise_proxy / gp) * (recovery_proxy / rise_proxy if rise_proxy > 0 else 0)
    elif pat in ('rise_then_drop', 'mild_rise', 'continued_drop'):
        t['bounce_score'] = 0.0
    else:
        t['bounce_score'] = 0.0

print(f"  → Enrichment complete")

# ── Build base sets ──
a6_sr = [t for t in trades_a if t.get('open_pattern') == 'sustained_rise']
c1_all = [t for t in trades_c if t.get('hyp_c_type', '').startswith('C1') or t.get('hyp_c_type') == 'C1+C2']

print(f"  → A6 sustained_rise: {len(a6_sr)} trades")
print(f"  → C1 reversal: {len(c1_all)} trades")


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: Run filtered experiment
# ═══════════════════════════════════════════════════════════════════════════════

def run_filtered(trades, pnl_key, filters, section_title, label_prefix=''):
    print(f"\n  ── {section_title} ──")
    results = OrderedDict()
    rows = []
    for fname, ffn in filters:
        subset = [t for t in trades if ffn(t)]
        pnl = [t[pnl_key] for t in subset if t.get(pnl_key) is not None]
        key = f'{label_prefix}{fname}' if label_prefix else fname
        m = compute_metrics(pnl, key)
        results[key] = m
        rows.append(metrics_row(m))
    if rows:
        print(tabulate(rows, headers=metrics_headers(), tablefmt='simple'))
    return results


def tpd_str(n, n_dates):
    return f"{n/n_dates:.2f}" if n_dates > 0 else "0"


# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
#
#  GROUP 1: A6 SUSTAINED_RISE SUBSEGMENTATION
#
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

all_results = OrderedDict()

print(f"\n\n{'='*80}")
print(f"  GROUP 1: A6 SUSTAINED_RISE SUBSEGMENTATION")
print(f"  Base: {len(a6_sr)} sustained_rise trades")
print(f"{'='*80}")

# ── S1: Gap Magnitude Within sustained_rise ──
print(f"\n{'─'*60}")
print(f"  S1: GAP MAGNITUDE WITHIN SUSTAINED_RISE")
print(f"{'─'*60}")

s1_filters = [
    ('S1a_D1_-5to-8%',   lambda t: -0.08 <= t['gap_pct'] < -0.05),
    ('S1b_D2_-8to-12%',  lambda t: -0.12 <= t['gap_pct'] < -0.08),
    ('S1c_D3_-12to-20%', lambda t: -0.20 <= t['gap_pct'] < -0.12),
]

for hold, pkey in [('T+2','pnl_t2'), ('T+3','pnl_t3')]:
    r = run_filtered(a6_sr, pkey, s1_filters, f'S1 — {hold}', f'S1_{hold}_')
    all_results.update(r)

# ── S2: Bounce Strength Score ──
print(f"\n{'─'*60}")
print(f"  S2: BOUNCE STRENGTH SCORE (continuous)")
print(f"{'─'*60}")

bs_values = [t['bounce_score'] for t in a6_sr if t.get('bounce_score') is not None and t['bounce_score'] > 0]
_bs_p33 = 0.0
_bs_p66 = 0.0
if bs_values:
    p33 = np.percentile(bs_values, 33)
    p66 = np.percentile(bs_values, 66)
    _bs_p33 = p33
    _bs_p66 = p66
    print(f"  Bounce score distribution: min={min(bs_values):.3f}, P33={p33:.3f}, P66={p66:.3f}, max={max(bs_values):.3f}")
    s2_filters = [
        ('S2_low',  lambda t, lo=0, hi=p33: t.get('bounce_score', 0) > 0 and t['bounce_score'] <= hi),
        ('S2_mid',  lambda t, lo=p33, hi=p66: lo < t.get('bounce_score', 0) <= hi),
        ('S2_high', lambda t, lo=p66: t.get('bounce_score', 0) > lo),
    ]
    for hold, pkey in [('T+2','pnl_t2'), ('T+3','pnl_t3')]:
        r = run_filtered(a6_sr, pkey, s2_filters, f'S2 — {hold}', f'S2_{hold}_')
        all_results.update(r)
else:
    print(f"  ⚠ No bounce scores available — S2 skipped")

# ── S3: Volume on Bounce Day ──
print(f"\n{'─'*60}")
print(f"  S3: VOLUME ON BOUNCE DAY")
print(f"{'─'*60}")

s3_filters = [
    ('S3a_vol>=2x', lambda t: t.get('vol_ratio') is not None and t['vol_ratio'] >= 2.0),
    ('S3b_vol<2x',  lambda t: t.get('vol_ratio') is not None and t['vol_ratio'] < 2.0),
]
for hold, pkey in [('T+2','pnl_t2'), ('T+3','pnl_t3')]:
    r = run_filtered(a6_sr, pkey, s3_filters, f'S3 — {hold}', f'S3_{hold}_')
    all_results.update(r)

# ── S4: Entry Point Comparison ──
print(f"\n{'─'*60}")
print(f"  S4: ENTRY POINT COMPARISON")
print(f"  S4a=OPEN (9:30), S4b=LOCAL_LOW (current), S4c=10:30 AM")
print(f"{'─'*60}")

for hold_label, offset in [('T+2', 2), ('T+3', 3)]:
    print(f"\n  ── {hold_label} ──")
    rows = []
    for entry_label, pnl_col in [('S4a_OPEN', f'pnl_open_t{offset}'),
                                  ('S4b_LOCAL_LOW', f'pnl_t{offset}'),
                                  ('S4c_1030', f'pnl_1030_t{offset}')]:
        pnl = [t[pnl_col] for t in a6_sr if t.get(pnl_col) is not None]
        key = f'S4_{hold_label}_{entry_label}'
        m = compute_metrics(pnl, key)
        all_results[key] = m
        flag = ' ⚠ lookahead' if 'OPEN' in entry_label else ''
        rows.append(metrics_row(m, [flag]))
    print(tabulate(rows, headers=metrics_headers(['Note']), tablefmt='simple'))

# ── S5: Day of Week Effect ──
print(f"\n{'─'*60}")
print(f"  S5: DAY OF WEEK EFFECT")
print(f"{'─'*60}")

s5_filters = [
    ('S5_Monday',  lambda t: t.get('day_of_week') == 'Monday'),
    ('S5_Tuesday', lambda t: t.get('day_of_week') == 'Tuesday'),
    ('S5_WedThu',  lambda t: t.get('day_of_week') in ('Wednesday', 'Thursday')),
    ('S5_Friday',  lambda t: t.get('day_of_week') == 'Friday'),
]
for hold, pkey in [('T+2','pnl_t2')]:
    r = run_filtered(a6_sr, pkey, s5_filters, f'S5 — {hold}', f'S5_{hold}_')
    all_results.update(r)

# ── S6: CS_ONLY ──
print(f"\n{'─'*60}")
print(f"  S6: SUSTAINED_RISE + CS_ONLY")
print(f"{'─'*60}")

s6_filters = [
    ('S6_CS_only',   lambda t: t.get('is_cs', False)),
    ('S6_all',       lambda t: True),
]
for hold, pkey in [('T+2','pnl_t2'), ('T+3','pnl_t3')]:
    r = run_filtered(a6_sr, pkey, s6_filters, f'S6 — {hold}', f'S6_{hold}_')
    all_results.update(r)


# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
#
#  GROUP 2: C1 REVERSAL SUBSEGMENTATION
#
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print(f"  GROUP 2: C1 REVERSAL SUBSEGMENTATION")
print(f"  Base: {len(c1_all)} C1 reversal trades")
print(f"{'='*80}")

# ── C1-S1: Decline Threshold Sweep ──
# Phase 1 collected C candidates with ≥15% decline.
# For -10% and -12% thresholds, we'd need MORE candidates (looser).
# Since we only have ≥15%, we can only test TIGHTER thresholds (≥20%, ≥25%)
# and report the current ≥15% as baseline.
print(f"\n{'─'*60}")
print(f"  C1-S1: DECLINE THRESHOLD SWEEP (5-day)")
print(f"  Note: Phase 1 collected with ≥15% threshold.")
print(f"  Tighter thresholds tested from existing data.")
print(f"{'─'*60}")

c1s1_filters = [
    ('C1c_15%_baseline', lambda t: True),
    ('C1d_20%',  lambda t: t.get('decline_5d') is not None and t['decline_5d'] <= -0.20),
    ('C1e_25%',  lambda t: t.get('decline_5d') is not None and t['decline_5d'] <= -0.25),
    ('C1f_30%',  lambda t: t.get('decline_5d') is not None and t['decline_5d'] <= -0.30),
]
for hold, pkey in [('T+3','pnl_t3'), ('T+5','pnl_t5')]:
    r = run_filtered(c1_all, pkey, c1s1_filters, f'C1-S1 — {hold}', f'C1S1_{hold}_')
    all_results.update(r)

# ── C1-S2: Gap-Up Threshold Sweep ──
# Phase 1 collected with gap ≥3%. Can only test TIGHTER (≥5%, ≥8%).
print(f"\n{'─'*60}")
print(f"  C1-S2: GAP-UP THRESHOLD SWEEP")
print(f"  Note: Phase 1 collected with ≥3% gap threshold.")
print(f"{'─'*60}")

c1s2_filters = [
    ('C1_gap3_baseline', lambda t: True),
    ('C1_gap5',  lambda t: t.get('gap_pct', 0) >= 0.05),
    ('C1_gap8',  lambda t: t.get('gap_pct', 0) >= 0.08),
    ('C1_gap10', lambda t: t.get('gap_pct', 0) >= 0.10),
]
for hold, pkey in [('T+3','pnl_t3'), ('T+5','pnl_t5')]:
    r = run_filtered(c1_all, pkey, c1s2_filters, f'C1-S2 — {hold}', f'C1S2_{hold}_')
    all_results.update(r)

# ── C1-S3: Pattern Confirmation (C1 + sustained_rise) ──
# Need to classify C1 trades' first-60-min pattern.
# C1 trades enter at open. The pattern classifier needs rth_bars which aren't
# stored. But we can approximate: if a C1 stock is ALSO in trades_a with
# sustained_rise on the same date, it qualifies.
# More precisely: check if any A trade on same ticker+date has sustained_rise.
print(f"\n{'─'*60}")
print(f"  C1-S3: C1 + SUSTAINED_RISE PATTERN CONFIRMATION")
print(f"{'─'*60}")

a6_sr_keys = {(t['ticker'], t['scan_date']) for t in a6_sr}
for t in c1_all:
    t['has_sustained_rise'] = (t['ticker'], t['scan_date']) in a6_sr_keys

c1s3_filters = [
    ('C1_SR_yes',  lambda t: t.get('has_sustained_rise', False)),
    ('C1_SR_no',   lambda t: not t.get('has_sustained_rise', False)),
    ('C1_all',     lambda t: True),
]
for hold, pkey in [('T+3','pnl_t3'), ('T+5','pnl_t5')]:
    r = run_filtered(c1_all, pkey, c1s3_filters, f'C1-S3 — {hold}', f'C1S3_{hold}_')
    all_results.update(r)

n_sr = sum(1 for t in c1_all if t.get('has_sustained_rise'))
print(f"\n  Note: {n_sr}/{len(c1_all)} C1 trades also have sustained_rise pattern")

# ── C1-S4: Hold Period Granularity ──
print(f"\n{'─'*60}")
print(f"  C1-S4: HOLD PERIOD SWEEP (T+2 through T+20)")
print(f"{'─'*60}")

c1s4_holds = [(f'T+{k}', f'pnl_t{k}') for k in [2,3,4,5,6,7,10,15,20]]
c1s4_results = OrderedDict()
rows = []
for hold_label, pnl_key in c1s4_holds:
    pnl = [t[pnl_key] for t in c1_all if t.get(pnl_key) is not None]
    key = f'C1S4_{hold_label}'
    m = compute_metrics(pnl, key)
    c1s4_results[key] = m
    all_results[key] = m
    rows.append(metrics_row(m, [tpd_str(m['n'], n_dates)]))
print(tabulate(rows, headers=metrics_headers(['TPD']), tablefmt='simple'))

# ── C1-S5: Sector Analysis ──
print(f"\n{'─'*60}")
print(f"  C1-S5: SECTOR / ASSET TYPE ANALYSIS")
print(f"{'─'*60}")

sectors = defaultdict(list)
for t in c1_all:
    if t.get('pnl_t3') is not None:
        sectors[t.get('sector', 'UNK')].append(t['pnl_t3'])
rows = []
for sec in sorted(sectors, key=lambda s: -np.mean(sectors[s])):
    a = np.array(sectors[sec])
    se = np.std(a, ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0
    rows.append([sec, len(a), f"{np.mean(a):+.2f}%", f"{(a>0).mean()*100:.0f}%",
                 f"[{np.mean(a)-1.96*se:+.2f},{np.mean(a)+1.96*se:+.2f}]"])
print(tabulate(rows, headers=['Type','n','Avg T+3','WR','CI 95%'], tablefmt='simple'))

# ── C1-S6: Lookback Period Variation ──
print(f"\n{'─'*60}")
print(f"  C1-S6: LOOKBACK PERIOD VARIATION")
print(f"  Testing different decline lookback periods")
print(f"{'─'*60}")

# For this, we use the decline_Xd fields computed during enrichment
c1s6_configs = [
    ('C1_3d_-10%',  'decline_3d',  -0.10),
    ('C1_5d_-15%',  'decline_5d',  -0.15),
    ('C1_7d_-18%',  'decline_7d',  -0.18),
    ('C1_10d_-20%', 'decline_10d', -0.20),
    ('C1_15d_-25%', 'decline_15d', -0.25),
]

for hold, pkey in [('T+3','pnl_t3'), ('T+5','pnl_t5')]:
    print(f"\n  ── {hold} ──")
    rows = []
    for lbl, decline_col, threshold in c1s6_configs:
        subset = [t for t in c1_all if t.get(decline_col) is not None and t[decline_col] <= threshold]
        pnl = [t[pkey] for t in subset if t.get(pkey) is not None]
        key = f'C1S6_{hold}_{lbl}'
        m = compute_metrics(pnl, key)
        all_results[key] = m
        rows.append(metrics_row(m, [str(m['n'])]))
    print(tabulate(rows, headers=metrics_headers(['n_pass']), tablefmt='simple'))


# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
#
#  GROUP 3: COMBINATION STRATEGIES
#
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print(f"  GROUP 3: COMBINATION STRATEGIES")
print(f"{'='*80}")

# ── COMBO-2: A6 + C1 combined portfolio ──
print(f"\n{'─'*60}")
print(f"  COMBO-2: A6 + C1 COMBINED PORTFOLIO")
print(f"{'─'*60}")

# Check overlap
a6_keys = {(t['ticker'], t['scan_date']) for t in a6_sr}
c1_keys = {(t['ticker'], t['scan_date']) for t in c1_all}
overlap = a6_keys & c1_keys
a6_dates = {t['scan_date'] for t in a6_sr}
c1_dates = {t['scan_date'] for t in c1_all}
both_dates = a6_dates & c1_dates

print(f"  A6 trades: {len(a6_sr)} across {len(a6_dates)} unique dates")
print(f"  C1 trades: {len(c1_all)} across {len(c1_dates)} unique dates")
print(f"  Ticker+date overlap: {len(overlap)} (same stock, same day)")
print(f"  Date overlap: {len(both_dates)} days with both A6 and C1 trades")

# Build combined daily PnL
# A6 uses T+2, C1 uses T+3
daily_pnl_a6 = defaultdict(list)
for t in a6_sr:
    if t.get('pnl_t2') is not None:
        daily_pnl_a6[t['scan_date']].append(t['pnl_t2'])

daily_pnl_c1 = defaultdict(list)
for t in c1_all:
    if t.get('pnl_t3') is not None:
        daily_pnl_c1[t['scan_date']].append(t['pnl_t3'])

# Combined: each day, combine all A6 and C1 trades with equal weight
combined_pnl_all = []
combo_daily = []
capital = INITIAL_CAPITAL
for dt in dates:
    a6_pnls = daily_pnl_a6.get(dt, [])
    c1_pnls = daily_pnl_c1.get(dt, [])
    all_day = a6_pnls + c1_pnls
    if all_day:
        n_trades = len(all_day)
        pos_size = capital / n_trades
        day_dollar = sum(pos_size * (r / 100) for r in all_day)
        combined_pnl_all.extend(all_day)
        combo_daily.append({'date': dt, 'n': n_trades, 'dollar': day_dollar,
                            'n_a6': len(a6_pnls), 'n_c1': len(c1_pnls)})

# Metrics
m_a6 = compute_metrics([t['pnl_t2'] for t in a6_sr if t.get('pnl_t2') is not None], 'A6_SR_T+2')
m_c1 = compute_metrics([t['pnl_t3'] for t in c1_all if t.get('pnl_t3') is not None], 'C1_T+3')
m_combo = compute_metrics(combined_pnl_all, 'COMBO_A6+C1')

print(f"\n  ── Individual vs Combined ──")
rows = [metrics_row(m_a6, [tpd_str(m_a6['n'], n_dates)]),
        metrics_row(m_c1, [tpd_str(m_c1['n'], n_dates)]),
        metrics_row(m_combo, [tpd_str(m_combo['n'], n_dates)])]
print(tabulate(rows, headers=metrics_headers(['TPD']), tablefmt='simple'))

# Monthly combined dollar PnL
if combo_daily:
    monthly = defaultdict(lambda: {'dollar': 0, 'trades': 0, 'days': 0})
    for cd in combo_daily:
        mo = cd['date'][:7]
        monthly[mo]['dollar'] += cd['dollar']
        monthly[mo]['trades'] += cd['n']
        monthly[mo]['days'] += 1
    print(f"\n  ── COMBO Monthly P&L (${INITIAL_CAPITAL:,.0f} capital) ──")
    rows = []
    cum = 0
    for mo in sorted(monthly):
        md = monthly[mo]
        cum += md['dollar']
        rows.append([mo, md['trades'], md['days'], f"${md['dollar']:+,.0f}", f"${cum:+,.0f}"])
    print(tabulate(rows, headers=['Month','Trades','Active Days','Month $','Cumulative $'], tablefmt='simple'))

all_results['COMBO_A6_C1'] = m_combo

# ── COMBO-3: C1 + D2 Tier (C1 stocks with -8% to -12% gap on reversal day) ──
print(f"\n{'─'*60}")
print(f"  COMBO-3: C1 + GAP MAGNITUDE ON REVERSAL DAY")
print(f"{'─'*60}")

combo3_filters = [
    ('C1_gapD1_3-8%',   lambda t: 0.03 <= t.get('gap_pct', 0) < 0.08),
    ('C1_gapD2_8-12%',  lambda t: 0.08 <= t.get('gap_pct', 0) < 0.12),
    ('C1_gapD3_12%+',   lambda t: t.get('gap_pct', 0) >= 0.12),
]
for hold, pkey in [('T+3','pnl_t3'), ('T+5','pnl_t5')]:
    r = run_filtered(c1_all, pkey, combo3_filters, f'COMBO-3 — {hold}', f'COMBO3_{hold}_')
    all_results.update(r)


# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
#
#  GROUP 4: NEW IDEAS
#
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print(f"  GROUP 4: NEW IDEAS")
print(f"{'='*80}")

# ── N1: Market Regime Filter ──
print(f"\n{'─'*60}")
print(f"  N1: MARKET REGIME FILTER (SPY 5-day return)")
print(f"{'─'*60}")

regime_filters = [
    ('bull_SPY>+2%',  lambda t: t.get('spy_regime') == 'bull'),
    ('flat_SPY±2%',   lambda t: t.get('spy_regime') == 'flat'),
    ('bear_SPY<-2%',  lambda t: t.get('spy_regime') == 'bear'),
]

print(f"\n  A6 sustained_rise by regime:")
for hold, pkey in [('T+2','pnl_t2')]:
    r = run_filtered(a6_sr, pkey, regime_filters, f'N1-A6 — {hold}', f'N1A6_{hold}_')
    all_results.update(r)

print(f"\n  C1 reversal by regime:")
for hold, pkey in [('T+3','pnl_t3')]:
    r = run_filtered(c1_all, pkey, regime_filters, f'N1-C1 — {hold}', f'N1C1_{hold}_')
    all_results.update(r)

# ── N2: Prior Trend Context ──
print(f"\n{'─'*60}")
print(f"  N2: PRIOR TREND CONTEXT (5-day return before gap-down)")
print(f"{'─'*60}")

n2_filters = [
    ('prior_up_>+5%',   lambda t: t.get('prior_5d_return') is not None and t['prior_5d_return'] > 0.05),
    ('prior_flat_±5%',  lambda t: t.get('prior_5d_return') is not None and -0.05 <= t['prior_5d_return'] <= 0.05),
    ('prior_down_<-5%', lambda t: t.get('prior_5d_return') is not None and t['prior_5d_return'] < -0.05),
]
for hold, pkey in [('T+2','pnl_t2')]:
    r = run_filtered(a6_sr, pkey, n2_filters, f'N2-A6 — {hold}', f'N2A6_{hold}_')
    all_results.update(r)

# ── N3: Gap-Down + Prior 5-Day Decline Overlap ──
print(f"\n{'─'*60}")
print(f"  N3: GAP-DOWN + PRIOR 5d DECLINE (A6 + C1 overlap)")
print(f"  Stock declined ≥15% over 5d, THEN gapped down ≥5%,")
print(f"  AND shows sustained_rise in first 60 min")
print(f"{'─'*60}")

n3_trades = [t for t in a6_sr if t.get('a_decline_5d') is not None and t['a_decline_5d'] <= -0.15]
n3_trades_10 = [t for t in a6_sr if t.get('a_decline_5d') is not None and t['a_decline_5d'] <= -0.10]
print(f"  Found: {len(n3_trades)} trades with ≥15% prior 5d decline + gap-down + sustained_rise")
print(f"  Found: {len(n3_trades_10)} trades with ≥10% prior 5d decline + gap-down + sustained_rise")

for hold, pkey in [('T+2','pnl_t2'), ('T+3','pnl_t3'), ('T+5','pnl_t5')]:
    pnl_15 = [t[pkey] for t in n3_trades if t.get(pkey) is not None]
    pnl_10 = [t[pkey] for t in n3_trades_10 if t.get(pkey) is not None]
    m15 = compute_metrics(pnl_15, f'N3_15%_{hold}')
    m10 = compute_metrics(pnl_10, f'N3_10%_{hold}')
    all_results[f'N3_15%_{hold}'] = m15
    all_results[f'N3_10%_{hold}'] = m10

    if pnl_15 or pnl_10:
        rows = []
        if pnl_15: rows.append(metrics_row(m15))
        if pnl_10: rows.append(metrics_row(m10))
        print(f"\n  ── N3 — {hold} ──")
        print(tabulate(rows, headers=metrics_headers(), tablefmt='simple'))

# ── N5: Weekly Pattern — Monday Reversal ──
print(f"\n{'─'*60}")
print(f"  N5: DAY-OF-WEEK EFFECT FOR C1")
print(f"{'─'*60}")

n5_filters = [
    ('Monday',   lambda t: t.get('day_of_week') == 'Monday'),
    ('Tuesday',  lambda t: t.get('day_of_week') == 'Tuesday'),
    ('WedThu',   lambda t: t.get('day_of_week') in ('Wednesday', 'Thursday')),
    ('Friday',   lambda t: t.get('day_of_week') == 'Friday'),
]
for hold, pkey in [('T+3','pnl_t3')]:
    r = run_filtered(c1_all, pkey, n5_filters, f'N5-C1 — {hold}', f'N5C1_{hold}_')
    all_results.update(r)

# ── N6: T+1 Confirmation Rule for C1 ──
print(f"\n{'─'*60}")
print(f"  N6: T+1 CONFIRMATION MANAGEMENT RULE (C1)")
print(f"  N6a: Always hold to T+3 (current)")
print(f"  N6b: Exit T+1 if T+1 close < entry; hold to T+3 if T+1 > entry")
print(f"{'─'*60}")

n6a_pnl = [t['pnl_t3'] for t in c1_all if t.get('pnl_t3') is not None]
n6b_pnl = []
n6b_exit_early = 0
n6b_hold = 0
for t in c1_all:
    p1 = t.get('pnl_t1')
    p3 = t.get('pnl_t3')
    if p1 is None:
        continue
    if p1 < 0:
        n6b_pnl.append(p1)
        n6b_exit_early += 1
    elif p3 is not None:
        n6b_pnl.append(p3)
        n6b_hold += 1

m_n6a = compute_metrics(n6a_pnl, 'N6a_always_T+3')
m_n6b = compute_metrics(n6b_pnl, 'N6b_managed')
all_results['N6a_always_T+3'] = m_n6a
all_results['N6b_managed'] = m_n6b

rows = [metrics_row(m_n6a), metrics_row(m_n6b)]
print(tabulate(rows, headers=metrics_headers(), tablefmt='simple'))
print(f"\n  N6b: {n6b_exit_early} early exits at T+1, {n6b_hold} held to T+3")
print(f"  Management value: {m_n6b['avg'] - m_n6a['avg']:+.2f}% per trade")


# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
#
#  OOS VALIDATION (Phase 2 protocol)
#
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_p2_trade_list(exp_id):
    """Return (trade_list, filter_fn) for Phase 2 experiments."""
    eid = exp_id

    # Group 1: A6 SR subsegmentation — base is a6_sr
    if eid.startswith('S1_') or eid.startswith('S2_') or eid.startswith('S3_') or \
       eid.startswith('S4_') or eid.startswith('S5_') or eid.startswith('S6_'):
        return a6_sr

    # Group 2: C1 subsegmentation
    if 'C1S1_' in eid or 'C1S2_' in eid or 'C1S3_' in eid or \
       'C1S4_' in eid or 'C1S6_' in eid:
        return c1_all

    # Combo
    if 'COMBO3_' in eid:
        return c1_all
    if 'COMBO_A6_C1' in eid:
        return None  # combined — need special handling

    # Group 4
    if 'N1A6_' in eid: return a6_sr
    if 'N1C1_' in eid: return c1_all
    if 'N2A6_' in eid: return a6_sr
    if 'N3_' in eid: return a6_sr
    if 'N5C1_' in eid: return c1_all
    if 'N6' in eid: return c1_all

    return None


def _resolve_p2_filter(exp_id):
    """Return filter function for a Phase 2 experiment."""
    eid = exp_id

    # S1 filters
    if 'S1a_D1' in eid: return lambda t: -0.08 <= t['gap_pct'] < -0.05
    if 'S1b_D2' in eid: return lambda t: -0.12 <= t['gap_pct'] < -0.08
    if 'S1c_D3' in eid: return lambda t: -0.20 <= t['gap_pct'] < -0.12

    # S2 filters
    if 'S2_low' in eid:  return lambda t: t.get('bounce_score', 0) > 0 and t['bounce_score'] <= _bs_p33
    if 'S2_mid' in eid:  return lambda t: _bs_p33 < t.get('bounce_score', 0) <= _bs_p66
    if 'S2_high' in eid: return lambda t: t.get('bounce_score', 0) > _bs_p66

    # S3 filters
    if 'S3a_vol>=2x' in eid: return lambda t: t.get('vol_ratio') is not None and t['vol_ratio'] >= 2.0
    if 'S3b_vol<2x' in eid:  return lambda t: t.get('vol_ratio') is not None and t['vol_ratio'] < 2.0

    # S4 — entry variants (pnl_key changes, not filter)
    if 'S4a_OPEN' in eid:      return lambda t: True
    if 'S4b_LOCAL_LOW' in eid: return lambda t: True
    if 'S4c_1030' in eid:      return lambda t: True

    # S5 filters
    if 'Monday' in eid:  return lambda t: t.get('day_of_week') == 'Monday'
    if 'Tuesday' in eid: return lambda t: t.get('day_of_week') == 'Tuesday'
    if 'WedThu' in eid:  return lambda t: t.get('day_of_week') in ('Wednesday', 'Thursday')
    if 'Friday' in eid:  return lambda t: t.get('day_of_week') == 'Friday'

    # S6 filters
    if 'S6_CS_only' in eid: return lambda t: t.get('is_cs', False)
    if 'S6_all' in eid:     return lambda t: True

    # C1-S1
    if 'C1c_15%' in eid: return lambda t: True
    if 'C1d_20%' in eid: return lambda t: t.get('decline_5d') is not None and t['decline_5d'] <= -0.20
    if 'C1e_25%' in eid: return lambda t: t.get('decline_5d') is not None and t['decline_5d'] <= -0.25
    if 'C1f_30%' in eid: return lambda t: t.get('decline_5d') is not None and t['decline_5d'] <= -0.30

    # C1-S2
    if 'C1_gap3' in eid:  return lambda t: True
    if 'C1_gap5' in eid:  return lambda t: t.get('gap_pct', 0) >= 0.05
    if 'C1_gap8' in eid:  return lambda t: t.get('gap_pct', 0) >= 0.08
    if 'C1_gap10' in eid: return lambda t: t.get('gap_pct', 0) >= 0.10

    # C1-S3
    if 'C1_SR_yes' in eid: return lambda t: t.get('has_sustained_rise', False)
    if 'C1_SR_no' in eid:  return lambda t: not t.get('has_sustained_rise', False)
    if 'C1_all' in eid:    return lambda t: True

    # COMBO-3
    if 'gapD1_3-8%' in eid:  return lambda t: 0.03 <= t.get('gap_pct', 0) < 0.08
    if 'gapD2_8-12%' in eid: return lambda t: 0.08 <= t.get('gap_pct', 0) < 0.12
    if 'gapD3_12%' in eid:   return lambda t: t.get('gap_pct', 0) >= 0.12

    # N1
    if 'bull' in eid:  return lambda t: t.get('spy_regime') == 'bull'
    if 'flat' in eid and 'SPY' in eid: return lambda t: t.get('spy_regime') == 'flat'
    if 'bear' in eid:  return lambda t: t.get('spy_regime') == 'bear'

    # N2
    if 'prior_up' in eid:   return lambda t: t.get('prior_5d_return') is not None and t['prior_5d_return'] > 0.05
    if 'prior_flat' in eid: return lambda t: t.get('prior_5d_return') is not None and -0.05 <= t['prior_5d_return'] <= 0.05
    if 'prior_down' in eid: return lambda t: t.get('prior_5d_return') is not None and t['prior_5d_return'] < -0.05

    # N3
    if 'N3_15%' in eid: return lambda t: t.get('a_decline_5d') is not None and t['a_decline_5d'] <= -0.15
    if 'N3_10%' in eid: return lambda t: t.get('a_decline_5d') is not None and t['a_decline_5d'] <= -0.10

    # C1-S6 lookback period filters
    if 'C1_3d_-10%' in eid:  return lambda t: t.get('decline_3d') is not None and t['decline_3d'] <= -0.10
    if 'C1_5d_-15%' in eid:  return lambda t: t.get('decline_5d') is not None and t['decline_5d'] <= -0.15
    if 'C1_7d_-18%' in eid:  return lambda t: t.get('decline_7d') is not None and t['decline_7d'] <= -0.18
    if 'C1_10d_-20%' in eid: return lambda t: t.get('decline_10d') is not None and t['decline_10d'] <= -0.20
    if 'C1_15d_-25%' in eid: return lambda t: t.get('decline_15d') is not None and t['decline_15d'] <= -0.25

    # N5 — same as day-of-week
    # N6 — special handling

    return lambda t: True


def _resolve_pnl_key(exp_id):
    """Extract PnL key from experiment ID."""
    # S4 entry variants — must check BEFORE generic T+N extraction
    if 'S4a_OPEN' in exp_id:
        for p in exp_id.split('_'):
            if p.startswith('T+'):
                return f"pnl_open_t{p.replace('T+','')}"
    if 'S4c_1030' in exp_id:
        for p in exp_id.split('_'):
            if p.startswith('T+'):
                return f"pnl_1030_t{p.replace('T+','')}"
    # Generic T+N
    for p in exp_id.split('_'):
        if p.startswith('T+') or p in ('EOD', '30m', '60m'):
            if '+' in p:
                return f"pnl_t{p.replace('T+','')}"
            return f"pnl_{p.lower()}"
    return None


print(f"\n\n{'='*80}")
print(f"  OOS VALIDATION — Phase 2 Protocol")
print(f"  IS: {BACKTEST_START} → {OOS_CUTOFF}  |  OOS: {OOS_CUTOFF} → {BACKTEST_END}")
print(f"{'='*80}")

# Identify experiments passing full-period pension criteria
passing = OrderedDict()
for k, m in all_results.items():
    if m.get('n', 0) >= 10 and m.get('ci_lo', -999) > 0 and m.get('sharpe', 0) >= 0.15 and m.get('wr', 0) >= 0.58:
        passing[k] = m

print(f"\n  {len(passing)} experiments pass full-period criteria (CI>0, Sharpe≥0.15, WR≥58%)")
if not passing:
    print(f"  No experiments qualify for OOS validation.")
else:
    for k in passing:
        print(f"    {k}: avg={passing[k]['avg']:+.2f}%, Sharpe={passing[k]['sharpe']:.3f}, WR={passing[k]['wr']*100:.0f}%")

oos_results = OrderedDict()

for exp_id in passing:
    trade_list = _resolve_p2_trade_list(exp_id)
    if trade_list is None:
        continue
    filt = _resolve_p2_filter(exp_id)
    pnl_key = _resolve_pnl_key(exp_id)
    if pnl_key is None:
        continue

    filtered = [t for t in trade_list if filt(t)]
    is_trades = [t for t in filtered if t['scan_date'] < OOS_CUTOFF and t.get(pnl_key) is not None]
    oos_trades = [t for t in filtered if t['scan_date'] >= OOS_CUTOFF and t.get(pnl_key) is not None]

    is_pnl = [t[pnl_key] for t in is_trades]
    oos_pnl = [t[pnl_key] for t in oos_trades]

    m_is = compute_metrics(is_pnl, f'{exp_id}_IS')
    m_oos = compute_metrics(oos_pnl, f'{exp_id}_OOS')

    # Pension criteria for both IS and OOS
    pension_ready = (
        m_is.get('ci_lo', -1) > 0 and m_oos.get('ci_lo', -1) > 0
        and m_is.get('sharpe', 0) >= 0.15 and m_oos.get('sharpe', 0) >= 0.15
        and m_is.get('wr', 0) >= 0.58 and m_oos.get('wr', 0) >= 0.58
    )
    # Degradation check
    degradation_ok = m_oos['avg'] >= 0.6 * m_is['avg'] if m_is['avg'] > 0 else True

    if pension_ready and degradation_ok:
        status = '★ PENSION READY'
    elif m_oos.get('ci_lo', -1) > 0 and m_oos.get('sharpe', 0) >= 0.15:
        status = '◇ NEAR MISS (IS fails)'
    elif m_is['avg'] > 0 and m_oos['avg'] < 0.6 * m_is['avg']:
        status = '✗ DEGRADED'
    else:
        status = '✗ Not pension ready'

    print(f"\n  {exp_id}:")
    print(f"    IS  ({BACKTEST_START}→{OOS_CUTOFF}): n={m_is['n']}, avg={m_is['avg']:+.2f}%, "
          f"Sharpe={m_is['sharpe']:.3f}, WR={m_is['wr']*100:.0f}%, CI [{m_is['ci_lo']:+.2f},{m_is['ci_hi']:+.2f}]")
    print(f"    OOS ({OOS_CUTOFF}→{BACKTEST_END}): n={m_oos['n']}, avg={m_oos['avg']:+.2f}%, "
          f"Sharpe={m_oos['sharpe']:.3f}, WR={m_oos['wr']*100:.0f}%, CI [{m_oos['ci_lo']:+.2f},{m_oos['ci_hi']:+.2f}]")
    if m_is['avg'] > 0:
        deg_pct = (m_oos['avg'] / m_is['avg'] - 1) * 100 if m_is['avg'] != 0 else 0
        print(f"    IS→OOS degradation: {deg_pct:+.0f}%")
    print(f"    → {status}")

    oos_results[exp_id] = {'is': m_is, 'oos': m_oos, 'status': status,
                           'pension_ready': 'PENSION READY' in status}


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print(f"  MASTER SUMMARY TABLE — ALL PHASE 2 EXPERIMENTS")
print(f"{'='*80}")

rows = []
for k, m in all_results.items():
    if m.get('n', 0) < 3:
        continue
    ci_flag = '✓' if m['ci_lo'] > 0 else '✗'
    pension = ''
    if m['n'] >= 10 and m['ci_lo'] > 0 and m['sharpe'] >= 0.15 and m['wr'] >= 0.58:
        pension = '★'
    oos = oos_results.get(k, {})
    oos_status = oos.get('status', '')[:12] if oos else ''
    rows.append([k, m['n'], tpd_str(m['n'], n_dates), f"{m['avg']:+.2f}%",
                 f"{m['wr']*100:.0f}%", f"{m['sharpe']:.3f}",
                 f"[{m['ci_lo']:+.2f},{m['ci_hi']:+.2f}] {ci_flag}",
                 pension, oos_status])

print(tabulate(rows, headers=['ID','n','TPD','Avg','WR','Sharpe','CI 95%','Pass','OOS'],
               tablefmt='simple'))


# ═══════════════════════════════════════════════════════════════════════════════
# KEY QUESTIONS
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*80}")
print(f"  KEY QUESTIONS — ANSWERS")
print(f"{'='*80}")

print(f"\n  Q1: Does D2 tier + sustained_rise improve OOS metrics?")
s1b = all_results.get('S1_T+2_S1b_D2_-8to-12%', {})
a6_base = m_a6
if s1b.get('n', 0) > 0:
    print(f"      S1b (D2+SR): n={s1b['n']}, avg={s1b['avg']:+.2f}%, Sharpe={s1b['sharpe']:.3f}")
    print(f"      A6 baseline: n={a6_base['n']}, avg={a6_base['avg']:+.2f}%, Sharpe={a6_base['sharpe']:.3f}")
    print(f"      → {'YES — better per-trade' if s1b['sharpe'] > a6_base['sharpe'] else 'NO — similar or worse'}"
          f" but n drops from {a6_base['n']} to {s1b['n']}")
else:
    print(f"      → INSUFFICIENT DATA")

print(f"\n  Q2: Does C1 decline threshold matter?")
for lbl, key in [('-15%', 'C1S1_T+3_C1c_15%_baseline'), ('-20%', 'C1S1_T+3_C1d_20%'),
                 ('-25%', 'C1S1_T+3_C1e_25%')]:
    m = all_results.get(key, {})
    if m.get('n', 0) > 0:
        print(f"      {lbl}: n={m['n']}, avg={m['avg']:+.2f}%, Sharpe={m['sharpe']:.3f}")

print(f"\n  Q3: Does C1 + sustained_rise have CI > 0 in both IS and OOS?")
c1sr = oos_results.get('C1S3_T+3_C1_SR_yes', {})
if c1sr:
    print(f"      IS:  n={c1sr['is']['n']}, avg={c1sr['is']['avg']:+.2f}%, CI [{c1sr['is']['ci_lo']:+.2f},{c1sr['is']['ci_hi']:+.2f}]")
    print(f"      OOS: n={c1sr['oos']['n']}, avg={c1sr['oos']['avg']:+.2f}%, CI [{c1sr['oos']['ci_lo']:+.2f},{c1sr['oos']['ci_hi']:+.2f}]")
    print(f"      → {c1sr['status']}")
else:
    c1sr_full = all_results.get('C1S3_T+3_C1_SR_yes', {})
    if c1sr_full.get('n', 0) > 0:
        print(f"      Full period: n={c1sr_full['n']}, avg={c1sr_full['avg']:+.2f}%, Sharpe={c1sr_full['sharpe']:.3f}")
        print(f"      → Did not qualify for OOS validation")
    else:
        print(f"      → INSUFFICIENT DATA")

print(f"\n  Q4: Are A6 and C1 uncorrelated (fire on different days)?")
print(f"      Ticker+date overlap: {len(overlap)}")
print(f"      Date overlap (both active): {len(both_dates)} / {len(a6_dates | c1_dates)} active dates")
pct_overlap = len(both_dates) / len(a6_dates | c1_dates) * 100 if a6_dates | c1_dates else 0
print(f"      → {'LOW correlation' if pct_overlap < 30 else 'MODERATE correlation'} ({pct_overlap:.0f}% date overlap)")

print(f"\n  Q5: Does market regime explain when strategies fail?")
for strat, prefix in [('A6', 'N1A6_T+2'), ('C1', 'N1C1_T+3')]:
    for regime in ['bull', 'flat', 'bear']:
        key = f'{prefix}_{regime}_SPY>+2%' if regime == 'bull' else \
              f'{prefix}_{regime}_SPY<-2%' if regime == 'bear' else \
              f'{prefix}_flat_SPY±2%'
        m = all_results.get(key, {})
        if m.get('n', 0) > 0:
            print(f"      {strat} {regime}: n={m['n']}, avg={m['avg']:+.2f}%, Sharpe={m['sharpe']:.3f}")

print(f"\n  Q6: Does T+1 management improve C1 T+3?")
print(f"      N6a (always hold): avg={m_n6a['avg']:+.2f}%, Sharpe={m_n6a['sharpe']:.3f}")
print(f"      N6b (managed):     avg={m_n6b['avg']:+.2f}%, Sharpe={m_n6b['sharpe']:.3f}")
mgmt_value = m_n6b['avg'] - m_n6a['avg']
print(f"      → Management value: {mgmt_value:+.2f}% per trade")
print(f"      → {'YES — worth the complexity' if mgmt_value > 0.5 else 'NO — marginal or negative'}")

print(f"\n  Q7: Expected monthly PnL at $100K?")
if combo_daily:
    total_dollar = sum(cd['dollar'] for cd in combo_daily)
    n_months = len(set(cd['date'][:7] for cd in combo_daily))
    monthly_avg = total_dollar / n_months if n_months > 0 else 0
    total_trades = sum(cd['n'] for cd in combo_daily)
    print(f"      Combined A6+C1 portfolio:")
    print(f"      Total trades: {total_trades} over {n_months} months")
    print(f"      Total dollar P&L: ${total_dollar:+,.0f}")
    print(f"      Avg monthly P&L: ${monthly_avg:+,.0f}")
    print(f"      Avg monthly return: {monthly_avg/INITIAL_CAPITAL*100:+.1f}%")
    print(f"      Trades per month: {total_trades/n_months:.1f}")


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n  Generating charts...")

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Agent 3 — Phase 2 Investigation Summary', fontsize=13, fontweight='bold')

# 1. Group comparison — Sharpe
ax = axes[0, 0]
group_keys = [k for k, m in all_results.items() if m.get('n', 0) >= 10 and m.get('ci_lo', -99) > 0]
group_keys = sorted(group_keys, key=lambda k: -all_results[k]['sharpe'])[:15]
if group_keys:
    sharpes = [all_results[k]['sharpe'] for k in group_keys]
    colors = ['#27ae60' if s >= 0.15 else '#f39c12' for s in sharpes]
    ax.barh(range(len(group_keys)), sharpes, color=colors, alpha=0.8)
    ax.set_yticks(range(len(group_keys)))
    ax.set_yticklabels(group_keys, fontsize=6)
    ax.axvline(0.15, color='#e74c3c', ls='--', lw=1, label='Pension threshold')
    ax.set_xlabel('Sharpe')
    ax.set_title('CI-Confirmed Experiments (Sharpe)')
    ax.invert_yaxis()
    ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 2. C1-S4 hold period curve
ax = axes[0, 1]
c1s4_keys = [f'C1S4_T+{k}' for k in [2,3,4,5,6,7,10,15,20]]
holds_x = [2,3,4,5,6,7,10,15,20]
avgs_y = []
sharpes_y = []
for k in c1s4_keys:
    m = all_results.get(k, {})
    avgs_y.append(m.get('avg', 0))
    sharpes_y.append(m.get('sharpe', 0))
ax.plot(holds_x, avgs_y, 'b-o', linewidth=2, markersize=6, label='Avg PnL (%)')
ax2 = ax.twinx()
ax2.plot(holds_x, sharpes_y, 'r--s', linewidth=1.5, markersize=5, label='Sharpe', alpha=0.7)
ax.set_xlabel('Hold Period (trading days)')
ax.set_ylabel('Avg PnL (%)', color='blue')
ax2.set_ylabel('Sharpe', color='red')
ax.set_title('C1 Reversal: Hold Period Curve')
ax.axhline(0, color='#333', lw=0.5)
ax.legend(loc='upper left', fontsize=8)
ax2.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

# 3. Combined portfolio equity curve
ax = axes[1, 0]
if combo_daily:
    eq = [INITIAL_CAPITAL]
    eq_dates_plot = []
    for cd in combo_daily:
        eq.append(eq[-1] + cd['dollar'])
        eq_dates_plot.append(datetime.strptime(cd['date'], '%Y-%m-%d'))
    eq_dates_plot = [eq_dates_plot[0] - timedelta(days=1)] + eq_dates_plot
    ax.plot(eq_dates_plot, eq, color='#2c3e50', linewidth=2)
    ax.fill_between(eq_dates_plot, INITIAL_CAPITAL, eq, alpha=0.15,
                    where=[e >= INITIAL_CAPITAL for e in eq], color='#27ae60')
    ax.fill_between(eq_dates_plot, INITIAL_CAPITAL, eq, alpha=0.15,
                    where=[e < INITIAL_CAPITAL for e in eq], color='#e74c3c')
    ax.axhline(INITIAL_CAPITAL, color='#999', ls='--', lw=0.8)
    ax.set_title(f'Combined A6+C1 Portfolio (${INITIAL_CAPITAL:,.0f} → ${eq[-1]:,.0f})')
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.grid(True, alpha=0.3)

# 4. OOS comparison
ax = axes[1, 1]
oos_keys = list(oos_results.keys())[:10]
if oos_keys:
    is_avgs = [oos_results[k]['is']['avg'] for k in oos_keys]
    oos_avgs = [oos_results[k]['oos']['avg'] for k in oos_keys]
    x = np.arange(len(oos_keys))
    w = 0.35
    ax.barh(x - w/2, is_avgs, w, label='IS', color='#3498db', alpha=0.7)
    ax.barh(x + w/2, oos_avgs, w, label='OOS', color='#e74c3c', alpha=0.7)
    ax.set_yticks(x)
    ax.set_yticklabels(oos_keys, fontsize=6)
    ax.axvline(0, color='#333', lw=0.8)
    ax.set_xlabel('Avg PnL (%)')
    ax.set_title('IS vs OOS Comparison')
    ax.invert_yaxis()
    ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.94])
p2_chart = os.path.join(BASE_DIR, 'agent3_phase2_summary.png')
plt.savefig(p2_chart, dpi=150, bbox_inches='tight')
plt.show()
print(f"  Chart saved: {p2_chart}")


# ═══════════════════════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════════════════════

pension_ready = [k for k, v in oos_results.items() if v.get('pension_ready')]
near_miss = [k for k, v in oos_results.items() if 'NEAR MISS' in v.get('status', '')]

print(f"\n\n{'='*80}")
print(f"  PHASE 2 INVESTIGATION COMPLETE")
print(f"  Experiments run: {len(all_results)}")
print(f"  Full-period passing: {len(passing)}")
print(f"  OOS validated: {len(oos_results)}")
print(f"  PENSION READY: {len(pension_ready)} — {', '.join(pension_ready) if pension_ready else 'None'}")
print(f"  NEAR MISS: {len(near_miss)} — {', '.join(near_miss) if near_miss else 'None'}")
print(f"{'='*80}")
