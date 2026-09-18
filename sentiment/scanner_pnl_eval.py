# %% ── Scanner PnL Evaluation — Intraday Return Analysis ─────────────────────
#
# Queries today's scanner predictions from BQ (premarket + discovery),
# fetches actual 1-min Polygon price data, and evaluates directional
# returns at 30/60 minute horizons.
#
# Entry time: each signal enters at max(scan_time_et, market_open).
# Pre-open signals enter at 9:30 ET open; post-open signals enter at
# the next minute bar after their scan_time_pt converted to ET.
#
# Features:
#   - Per-signal entry time (no lookahead bias)
#   - 2.5% stop loss per trade (checked bar-by-bar via highs/lows)
#   - Equal-weight portfolio allocation
#   - Minute-by-minute return curves (relative to each trade's entry)
#   - One position per ticker: earliest selected signal wins
#
# Colab-ready, config vars at top.

import os, sys, time, json, warnings
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
import pytz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings('ignore')

try:
    from google.cloud import bigquery
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "google-cloud-bigquery", "--quiet"])
    from google.cloud import bigquery

try:
    from polygon import RESTClient as PolygonClient
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "polygon-api-client", "--quiet"])
    from polygon import RESTClient as PolygonClient

try:
    from google.colab import auth
    try:
        auth.authenticate_user()
    except Exception:
        pass
except (ImportError, Exception):
    pass


# ==============================================================================
# CONFIG
# ==============================================================================
EVAL_DATE = '2026-09-10'
POLYGON_API_KEY = os.environ["POLYGON_API_KEY"]
BQ_PREMARKET_TABLE = f"{GCP_PROJECT}.scoring_logs.premarket_scanner"
BQ_DISCOVERY_TABLE = f"{GCP_PROJECT}.scoring_logs.discovery_scanner"

HORIZONS_MIN = [30, 60]
INTRADAY_MINUTES = 60
MARKET_OPEN_ET = "09:30"
MARKET_CLOSE_ET = "16:00"
STOP_LOSS_PCT = 2.5

ET = pytz.timezone('America/New_York')
PT = pytz.timezone('America/Los_Angeles')

OUTPUT_DIR = "/tmp"


# ==============================================================================
# FETCH PREDICTIONS FROM BQ
# ==============================================================================
print(f"\n{'='*100}")
print(f"  SCANNER PnL EVALUATION — {EVAL_DATE}")
print(f"  Horizons: {HORIZONS_MIN} min | Stop loss: {STOP_LOSS_PCT}% | Entry: at signal time or {MARKET_OPEN_ET} ET")
print(f"{'='*100}")

bq_client = bigquery.Client()

def fetch_predictions(table, scanner_name):
    if scanner_name == 'premarket':
        q = f"""
        SELECT
            ticker, direction, confidence, remaining_move_pct,
            move_already_absorbed_pct AS move_already_priced_pct,
            expected_move_pct, composite_score AS tradability_score,
            catalyst_type, catalyst, selected, scanner_version,
            scan_time_pt, gate_reason, move_is_exhausted,
            scan_score, 0 AS catalyst_strength, 0.0 AS scan_move_score,
            rank_in_direction,
            COALESCE(stage2_model, '') AS stage2_model
        FROM `{table}`
        WHERE scan_date = '{EVAL_DATE}'
          AND direction IN ('LONG', 'SHORT')
        ORDER BY direction, rank_in_direction
        """
    else:
        q = f"""
        SELECT
            ticker, direction, confidence, remaining_move_pct,
            move_already_priced_pct,
            expected_move_pct, tradability_score,
            catalyst_type, catalyst, selected, scanner_version,
            scan_time_pt, gate_reason, move_is_exhausted,
            0 AS scan_score, catalyst_strength, scan_move_score,
            rank_in_direction,
            COALESCE(stage2_model, '') AS stage2_model
        FROM `{table}`
        WHERE scan_date = '{EVAL_DATE}'
          AND direction IN ('LONG', 'SHORT')
        ORDER BY direction, rank_in_direction
        """
    df = bq_client.query(q).to_dataframe()
    df['scanner'] = scanner_name
    return df

print(f"\n  Fetching predictions from BQ...")
df_pre = fetch_predictions(BQ_PREMARKET_TABLE, 'premarket')
df_disc = fetch_predictions(BQ_DISCOVERY_TABLE, 'discovery')

print(f"    Premarket: {len(df_pre)} signals ({df_pre['selected'].sum()} selected)")
print(f"    Discovery: {len(df_disc)} signals ({df_disc['selected'].sum()} selected)")

df_all = pd.concat([df_pre, df_disc], ignore_index=True)


# ==============================================================================
# PARSE SCAN TIME → ENTRY TIME (no lookahead)
# ==============================================================================
def _parse_scan_time(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    if len(s) == 5 and ':' in s:
        return s
    if len(s) >= 5 and s[2] == ':':
        return s[:5]
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S'):
        try:
            return datetime.strptime(s[:19], fmt).strftime('%H:%M')
        except (ValueError, IndexError):
            continue
    return s[:5] if len(s) >= 5 else None

df_all['scan_time_hhmm'] = df_all['scan_time_pt'].apply(_parse_scan_time)
df_all = df_all[df_all['scan_time_hhmm'].notna()].copy()

def _scan_pt_to_entry_et(scan_pt_hhmm):
    """Convert PT scan time to ET entry time, floored at market open."""
    pt_dt = PT.localize(datetime.strptime(f"{EVAL_DATE} {scan_pt_hhmm}", '%Y-%m-%d %H:%M'))
    et_dt = pt_dt.astimezone(ET)
    et_hhmm = et_dt.strftime('%H:%M')
    if et_hhmm < MARKET_OPEN_ET:
        return MARKET_OPEN_ET
    if et_hhmm >= MARKET_CLOSE_ET:
        return None
    return et_hhmm

df_all['entry_time_et'] = df_all['scan_time_hhmm'].apply(_scan_pt_to_entry_et)
n_past_close = df_all['entry_time_et'].isna().sum()
if n_past_close > 0:
    print(f"    Dropped {n_past_close} signals after market close")
df_all = df_all[df_all['entry_time_et'].notna()].copy()

# ── Per-ticker dedup: earliest selected signal wins ──────────────────────────
# For each ticker, find the first run where it was selected.
# If never selected, keep the first appearance (for comparison).
df_first_sel = (df_all[df_all['selected'] == True]
                .sort_values('scan_time_hhmm')
                .drop_duplicates('ticker', keep='first'))
tickers_with_sel = set(df_first_sel['ticker'])

df_first_unseen = (df_all[~df_all['ticker'].isin(tickers_with_sel)]
                   .sort_values('scan_time_hhmm')
                   .drop_duplicates('ticker', keep='first'))

df_all = pd.concat([df_first_sel, df_first_unseen]).reset_index(drop=True)

n_pre_open = (df_all['entry_time_et'] == MARKET_OPEN_ET).sum()
n_post_open = (df_all['entry_time_et'] != MARKET_OPEN_ET).sum()
print(f"    After dedup: {len(df_all)} unique tickers "
      f"({n_pre_open} pre-open → {MARKET_OPEN_ET}, {n_post_open} intraday entry)")

for _, r in df_all.sort_values('entry_time_et').iterrows():
    sel_mark = '✓' if r['selected'] else '·'
    print(f"      {sel_mark} {r['ticker']:<6} {r['direction']:<6} "
          f"scan={r['scan_time_hhmm']} PT → entry={r['entry_time_et']} ET  "
          f"({r['scanner']})")

if df_all.empty:
    print(f"\n  No predictions found for {EVAL_DATE}. Exiting.")
    sys.exit(0)


# ==============================================================================
# FETCH POLYGON PRICE DATA — all 1-min bars per ticker
# ==============================================================================
poly = PolygonClient(POLYGON_API_KEY)

def fetch_minute_bars(ticker, eval_date):
    """Fetch all 1-min bars for the day. Returns (bar_map, prev_close) or (None, None)."""
    try:
        bars = list(poly.get_aggs(
            ticker=ticker, multiplier=1, timespan="minute",
            from_=eval_date, to=eval_date, limit=500,
        ))
        if not bars:
            return None, None

        bar_map = {}
        for b in bars:
            ts = pd.Timestamp(b.timestamp, unit='ms', tz='UTC').tz_convert(ET)
            key = ts.strftime('%H:%M')
            bar_map[key] = {
                'open': float(b.open),
                'high': float(b.high),
                'low': float(b.low),
                'close': float(b.close),
                'volume': int(b.volume),
            }

        prev_close = None
        try:
            from_date = (datetime.strptime(eval_date, '%Y-%m-%d') - timedelta(days=5)).strftime('%Y-%m-%d')
            daily = list(poly.get_aggs(
                ticker=ticker, multiplier=1, timespan="day",
                from_=from_date, to=eval_date, limit=10,
            ))
            ref_dt = datetime.strptime(eval_date, '%Y-%m-%d')
            prev = [b for b in daily
                    if pd.Timestamp(b.timestamp, unit='ms', tz='UTC').date() < ref_dt.date()]
            if prev:
                prev_close = float(prev[-1].close)
        except Exception:
            pass

        return bar_map, prev_close

    except Exception as e:
        print(f"      [ERROR] {ticker}: {e}")
        return None, None


tickers = df_all['ticker'].unique().tolist()
print(f"\n  Fetching Polygon 1-min bars for {len(tickers)} tickers...")

bar_data = {}
for i, ticker in enumerate(tickers):
    bar_map, prev_close = fetch_minute_bars(ticker, EVAL_DATE)
    if bar_map:
        bar_data[ticker] = {'bars': bar_map, 'prev_close': prev_close}
        n_bars = len(bar_map)
        first_bar = min(bar_map.keys())
        last_bar = max(bar_map.keys())
        print(f"    {i+1:>3}/{len(tickers)} {ticker:<6} {n_bars} bars ({first_bar}–{last_bar})")
    else:
        print(f"    {i+1:>3}/{len(tickers)} {ticker:<6} NO DATA")

print(f"\n  Got price data for {len(bar_data)}/{len(tickers)} tickers")


# ==============================================================================
# COMPUTE DIRECTIONAL PnL — per-signal entry time
# ==============================================================================
records = []

for _, row in df_all.iterrows():
    ticker = row['ticker']
    direction = row['direction']
    entry_hhmm = row['entry_time_et']

    td = bar_data.get(ticker)
    if td is None:
        continue

    bar_map = td['bars']
    prev_close = td['prev_close']

    # Find entry bar
    actual_entry_hhmm = entry_hhmm
    entry_bar = bar_map.get(entry_hhmm)
    if not entry_bar:
        candidates = sorted(k for k in bar_map if k >= entry_hhmm)
        if not candidates:
            continue
        actual_entry_hhmm = candidates[0]
        entry_bar = bar_map[actual_entry_hhmm]

    entry_price = entry_bar['open']
    entry_h, entry_m_val = map(int, actual_entry_hhmm.split(':'))

    sign = 1.0 if direction == 'LONG' else -1.0

    raw_curve = {}
    sl_curve = {}
    stopped_out = False
    stop_minute = None

    for m in range(INTRADAY_MINUTES + 1):
        total_min = entry_h * 60 + entry_m_val + m
        target_h = total_min // 60
        target_m = total_min % 60
        target_hhmm = f"{target_h:02d}:{target_m:02d}"

        if target_hhmm >= MARKET_CLOSE_ET:
            break

        bar = bar_map.get(target_hhmm)
        if bar is None:
            if m > 0 and (m - 1) in raw_curve:
                raw_curve[m] = raw_curve[m - 1]
                if stopped_out:
                    sl_curve[m] = -STOP_LOSS_PCT
                else:
                    sl_curve[m] = raw_curve[m]
            continue

        ret = ((bar['close'] - entry_price) / entry_price) * 100
        raw_pnl = ret * sign
        raw_curve[m] = round(raw_pnl, 4)

        if stopped_out:
            sl_curve[m] = -STOP_LOSS_PCT
            continue

        if direction == 'LONG':
            worst_in_bar = ((bar['low'] - entry_price) / entry_price) * 100
        else:
            worst_in_bar = -((bar['high'] - entry_price) / entry_price) * 100

        if worst_in_bar <= -STOP_LOSS_PCT:
            stopped_out = True
            stop_minute = m
            sl_curve[m] = -STOP_LOSS_PCT
        else:
            sl_curve[m] = round(raw_pnl, 4)

    if not raw_curve:
        continue

    # EOD return
    all_bars_sorted = sorted(bar_map.items(), key=lambda x: x[0])
    eod_price = all_bars_sorted[-1][1]['close'] if all_bars_sorted else entry_price
    eod_ret = round(((eod_price - entry_price) / entry_price) * 100 * sign, 2)

    gap_pct = 0.0
    if prev_close:
        gap_pct = round(((entry_price - prev_close) / prev_close) * 100, 2)

    # Minutes available for this trade (may be < INTRADAY_MINUTES if entered late)
    max_m_available = max(raw_curve.keys()) if raw_curve else 0

    rec = {
        'ticker': ticker,
        'scanner': row['scanner'],
        'direction': direction,
        'confidence': int(row['confidence']),
        'remaining_move_pct': float(row.get('remaining_move_pct', 0)),
        'tradability_score': float(row.get('tradability_score', 0)),
        'catalyst_type': str(row.get('catalyst_type', '')),
        'selected': bool(row['selected']) if pd.notna(row.get('selected')) else False,
        'gate_reason': '' if pd.isna(row.get('gate_reason')) else str(row['gate_reason']),
        'move_is_exhausted': bool(row['move_is_exhausted']) if pd.notna(row.get('move_is_exhausted')) else False,
        'scan_time_pt': str(row.get('scan_time_hhmm', '') or ''),
        'entry_time_et': actual_entry_hhmm,
        'entry_price': entry_price,
        'gap_pct': gap_pct,
        'rank': int(row.get('rank_in_direction', 0)),
        'stage2_model': str(row.get('stage2_model', '')),
        'raw_curve': raw_curve,
        'sl_curve': sl_curve,
        'stopped_out': stopped_out,
        'stop_minute': stop_minute,
        'max_minutes': max_m_available,
    }

    for h in HORIZONS_MIN:
        if h <= max_m_available:
            rec[f'pnl_{h}m'] = raw_curve.get(h, 0)
            rec[f'sl_pnl_{h}m'] = sl_curve.get(h, 0)
        else:
            rec[f'pnl_{h}m'] = np.nan
            rec[f'sl_pnl_{h}m'] = np.nan

    rec['pnl_eod'] = eod_ret
    rec['sl_pnl_eod'] = -STOP_LOSS_PCT if stopped_out else eod_ret

    for h in HORIZONS_MIN:
        rec[f'win_{h}m'] = rec[f'pnl_{h}m'] > 0 if pd.notna(rec[f'pnl_{h}m']) else np.nan
        rec[f'sl_win_{h}m'] = rec[f'sl_pnl_{h}m'] > 0 if pd.notna(rec[f'sl_pnl_{h}m']) else np.nan

    records.append(rec)

df_pnl = pd.DataFrame(records)

if df_pnl.empty:
    print(f"\n  No PnL records computed. Exiting.")
    sys.exit(0)


# ==============================================================================
# PORTFOLIO PnL — equal-weight across all concurrent positions
# ==============================================================================
def build_portfolio_curve(df, curve_col='sl_curve'):
    port = {}
    for m in range(INTRADAY_MINUTES + 1):
        vals = [r[curve_col].get(m) for _, r in df.iterrows()
                if r[curve_col].get(m) is not None]
        if vals:
            port[m] = np.mean(vals)
    return port

df_sel = df_pnl[df_pnl['selected'] == True].copy()
df_not_sel = df_pnl[df_pnl['selected'] == False].copy()
df_long = df_pnl[df_pnl['direction'] == 'LONG'].copy()
df_short = df_pnl[df_pnl['direction'] == 'SHORT'].copy()
df_gated = df_pnl[
    (df_pnl['gate_reason'] != '') & (df_pnl['gate_reason'] != 'nan')
].copy()
df_ungated = df_pnl[
    (df_pnl['gate_reason'] == '') | (df_pnl['gate_reason'] == 'nan')
].copy()


# ==============================================================================
# TEXT ANALYSIS & OUTPUT
# ==============================================================================
print(f"\n\n{'='*100}")
print(f"  ╔══════════════════════════════════════════════════════════════════╗")
print(f"  ║  SCANNER PnL RESULTS — {EVAL_DATE} (SL={STOP_LOSS_PCT}%){'':>23}║")
print(f"  ╚══════════════════════════════════════════════════════════════════╝")
print(f"{'='*100}")


def print_stats(df, label, indent="  "):
    n = len(df)
    if n == 0:
        print(f"{indent}{label}: no trades")
        return
    n_stopped = df['stopped_out'].sum()
    for h in HORIZONS_MIN:
        sl_col = f'sl_pnl_{h}m'
        valid = df[sl_col].dropna()
        n_valid = len(valid)
        if n_valid == 0:
            print(f"{indent}{label} @ {h:>2}m: n={n:>3}  (no data at this horizon)")
            continue
        raw_col = f'pnl_{h}m'
        raw_valid = df[raw_col].dropna()
        raw_avg = raw_valid.mean() if len(raw_valid) > 0 else 0
        sl_avg = valid.mean()
        sl_wr = (valid > 0).sum() / n_valid * 100
        n_trunc = n - n_valid
        trunc_note = f"  ({n_trunc} truncated)" if n_trunc > 0 else ""
        print(f"{indent}{label} @ {h:>2}m: n={n_valid:>3}  "
              f"raw={raw_avg:+.2f}%  SL={sl_avg:+.2f}%  WR={sl_wr:.0f}%  "
              f"stopped={n_stopped}/{n}{trunc_note}")
    if 'sl_pnl_eod' in df.columns:
        eod = df['sl_pnl_eod'].dropna()
        if len(eod) > 0:
            print(f"{indent}{label} @ EOD: avg={eod.mean():+.2f}%  "
                  f"WR={((eod > 0).sum() / len(eod) * 100):.0f}%")
    print()


print(f"\n  ── ALL PREDICTIONS ──")
print_stats(df_pnl, "ALL")

print(f"  ── SELECTED vs NOT SELECTED ──")
print_stats(df_sel, "SELECTED")
print_stats(df_not_sel, "NOT_SEL")

print(f"  ── BY DIRECTION ──")
print_stats(df_long, "LONG")
print_stats(df_short, "SHORT")

print(f"  ── BY SCANNER ──")
for scanner in df_pnl['scanner'].unique():
    df_s = df_pnl[df_pnl['scanner'] == scanner]
    print_stats(df_s, scanner.upper())

print(f"  ── BY CATALYST TYPE ──")
for cat in sorted(df_pnl['catalyst_type'].unique()):
    if not cat or cat == 'nan':
        continue
    df_c = df_pnl[df_pnl['catalyst_type'] == cat]
    if len(df_c) >= 2:
        print_stats(df_c, cat.upper())

if len(df_gated) > 0:
    print(f"  ── GATED vs UNGATED ──")
    print_stats(df_ungated, "UNGATED")
    print_stats(df_gated, "GATED")

# ── Portfolio PnL (equal-weight, SL-adjusted) ──
print(f"\n  ── PORTFOLIO PnL (equal-weight, {STOP_LOSS_PCT}% SL) ──")
n_sel_trades = len(df_sel)
if n_sel_trades > 0:
    port_sel = build_portfolio_curve(df_sel, 'sl_curve')
    for h in HORIZONS_MIN:
        p = port_sel.get(h, 0)
        print(f"    SELECTED portfolio @ {h}m: {p:+.3f}% per-trade avg  "
              f"(= portfolio return with 1/{n_sel_trades} allocation each)")
    port_eod = df_sel['sl_pnl_eod'].mean()
    print(f"    SELECTED portfolio @ EOD: {port_eod:+.3f}%")
else:
    print(f"    No selected trades")

n_all = len(df_pnl)
port_all = build_portfolio_curve(df_pnl, 'sl_curve')
for h in HORIZONS_MIN:
    p = port_all.get(h, 0)
    print(f"    ALL portfolio @ {h}m: {p:+.3f}%  (n={n_all})")


# ── Per-ticker detail table ──
print(f"\n  ── PER-TICKER RESULTS (SL={STOP_LOSS_PCT}%) ──")
header = (f"  {'Ticker':<7} {'Dir':<6} {'Scan':<10} {'Run':>5} {'Entry':>5} {'Conf':>4} {'Sel':>4} "
          f"{'Gap':>6} {'30m':>7} {'SL30':>7} {'60m':>7} {'SL60':>7} "
          f"{'EOD':>7} {'Stop':>5} {'Catalyst':<20}")
print(header)
print(f"  {'─'*145}")

df_sorted = df_pnl.sort_values('sl_pnl_60m', ascending=False, na_position='last')
for _, r in df_sorted.iterrows():
    sel_str = '✓' if r['selected'] else '·'
    stop_str = f"@{r['stop_minute']:.0f}m" if r['stopped_out'] else '—'
    cat = str(r.get('catalyst_type', ''))[:18]
    gate_tag = ' [G]' if r.get('gate_reason') and r['gate_reason'] not in ('', 'nan') else ''
    eod_str = f"{r.get('sl_pnl_eod', 0):+.1f}%"
    run_time = str(r.get('scan_time_pt', ''))
    entry_time = str(r.get('entry_time_et', ''))

    pnl_30 = f"{r['pnl_30m']:+6.1f}%" if pd.notna(r.get('pnl_30m')) else '   n/a'
    sl_30 = f"{r['sl_pnl_30m']:+6.1f}%" if pd.notna(r.get('sl_pnl_30m')) else '   n/a'
    pnl_60 = f"{r['pnl_60m']:+6.1f}%" if pd.notna(r.get('pnl_60m')) else '   n/a'
    sl_60 = f"{r['sl_pnl_60m']:+6.1f}%" if pd.notna(r.get('sl_pnl_60m')) else '   n/a'

    print(f"  {r['ticker']:<7} {r['direction']:<6} {r['scanner']:<10} "
          f"{run_time:>5} {entry_time:>5} {r['confidence']:>4} {sel_str:>4} "
          f"{r['gap_pct']:>+5.1f}% "
          f"{pnl_30:>7} "
          f"{sl_30:>7} "
          f"{pnl_60:>7} "
          f"{sl_60:>7} "
          f"{eod_str:>7} "
          f"{stop_str:>5} "
          f"{cat}{gate_tag}")

# Print horizon coverage
for h in HORIZONS_MIN:
    n_valid = df_pnl[f'sl_pnl_{h}m'].notna().sum()
    n_trunc = len(df_pnl) - n_valid
    if n_trunc > 0:
        print(f"\n  ⚠ {n_trunc}/{len(df_pnl)} trades entered too late for {h}m horizon")


# ==============================================================================
# CHARTS — Page 1: Time-series return curves
# ==============================================================================
fig_path_ts = f"{OUTPUT_DIR}/scanner_pnl_timeseries_{EVAL_DATE}.png"

fig, axes = plt.subplots(3, 2, figsize=(20, 18))
fig.suptitle(f'Scanner PnL — Intraday Return Curves — {EVAL_DATE}\n'
             f'Entry at signal time (pre-open → {MARKET_OPEN_ET} ET) | '
             f'SL={STOP_LOSS_PCT}% | Equal-weight',
             fontsize=13, fontweight='bold')

minutes = list(range(INTRADAY_MINUTES + 1))

def plot_avg_curve(ax, df_subset, label, color, ls='-', lw=2, curve_col='sl_curve'):
    curve = build_portfolio_curve(df_subset, curve_col)
    xs = sorted(curve.keys())
    ys = [curve[x] for x in xs]
    ax.plot(xs, ys, color=color, linestyle=ls, linewidth=lw, label=label)
    return curve

def style_time_axis(ax, title):
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.axhline(y=-STOP_LOSS_PCT, color='red', linewidth=0.5, linestyle=':', alpha=0.5)
    ax.axvline(x=30, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axvline(x=60, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.set_xlabel('Minutes from entry')
    ax.set_ylabel('Avg PnL %')
    ax.set_title(title)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, INTRADAY_MINUTES)

# 1. All predictions: raw vs SL-adjusted
ax = axes[0, 0]
plot_avg_curve(ax, df_pnl, f'Raw (n={len(df_pnl)})', '#3498db', curve_col='raw_curve')
plot_avg_curve(ax, df_pnl, f'SL {STOP_LOSS_PCT}% (n={len(df_pnl)})', '#e74c3c', curve_col='sl_curve')
if len(df_sel) > 0:
    plot_avg_curve(ax, df_sel, f'Selected SL (n={len(df_sel)})', '#2ecc71', lw=2.5, curve_col='sl_curve')
style_time_axis(ax, 'All Predictions: Raw vs Stop-Loss Adjusted')

# 2. Selected vs Not Selected (SL-adjusted)
ax = axes[0, 1]
if len(df_sel) > 0:
    plot_avg_curve(ax, df_sel, f'Selected (n={len(df_sel)})', '#2ecc71', lw=2.5)
if len(df_not_sel) > 0:
    plot_avg_curve(ax, df_not_sel, f'Not Selected (n={len(df_not_sel)})', '#e74c3c')
if len(df_gated) > 0:
    plot_avg_curve(ax, df_gated, f'Gated (n={len(df_gated)})', '#95a5a6', ls='--')
style_time_axis(ax, 'Selected vs Not Selected vs Gated (SL-adj)')

# 3. By direction (SL-adjusted)
ax = axes[1, 0]
if len(df_long) > 0:
    plot_avg_curve(ax, df_long, f'LONG (n={len(df_long)})', '#27ae60', lw=2)
if len(df_short) > 0:
    plot_avg_curve(ax, df_short, f'SHORT (n={len(df_short)})', '#c0392b', lw=2)
style_time_axis(ax, 'LONG vs SHORT (SL-adjusted)')

# 4. By scanner (SL-adjusted)
ax = axes[1, 1]
scanner_colors = {'premarket': '#3498db', 'discovery': '#e67e22'}
for scanner in df_pnl['scanner'].unique():
    df_s = df_pnl[df_pnl['scanner'] == scanner]
    c = scanner_colors.get(scanner, '#666666')
    plot_avg_curve(ax, df_s, f'{scanner} (n={len(df_s)})', c, lw=2)
    df_s_sel = df_s[df_s['selected'] == True]
    if len(df_s_sel) > 0:
        plot_avg_curve(ax, df_s_sel, f'{scanner} sel (n={len(df_s_sel)})', c, ls='--', lw=1.5)
style_time_axis(ax, 'By Scanner (solid=all, dashed=selected)')

# 5. By catalyst type — top types by count (SL-adjusted)
ax = axes[2, 0]
cat_counts = df_pnl['catalyst_type'].value_counts()
top_cats = [c for c in cat_counts.index if c and c != 'nan'][:6]
cat_colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#e67e22', '#1abc9c']
for cat, color in zip(top_cats, cat_colors):
    df_c = df_pnl[df_pnl['catalyst_type'] == cat]
    if len(df_c) >= 2:
        plot_avg_curve(ax, df_c, f'{cat} (n={len(df_c)})', color, lw=1.5)
style_time_axis(ax, 'By Catalyst Type (SL-adjusted)')

# 6. Individual ticker curves for SELECTED trades only
ax = axes[2, 1]
if len(df_sel) > 0:
    for _, r in df_sel.iterrows():
        curve = r['sl_curve']
        xs = sorted(curve.keys())
        ys = [curve[x] for x in xs]
        c = '#27ae60' if r['direction'] == 'LONG' else '#c0392b'
        ax.plot(xs, ys, color=c, linewidth=0.8, alpha=0.5)
        if xs:
            ax.annotate(r['ticker'], xy=(xs[-1], ys[-1]), fontsize=6,
                       alpha=0.7, ha='left')
    plot_avg_curve(ax, df_sel, f'Avg selected (n={len(df_sel)})', 'black', lw=2.5)
else:
    ax.text(0.5, 0.5, 'No selected trades', transform=ax.transAxes,
            ha='center', va='center', fontsize=14)
style_time_axis(ax, 'Individual Selected Trades (SL-adjusted)')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(fig_path_ts, dpi=150, bbox_inches='tight')
print(f"\n  Time-series chart saved: {fig_path_ts}")
plt.close()


# ==============================================================================
# CHARTS — Page 2: Bar charts & scatter
# ==============================================================================
fig_path_bars = f"{OUTPUT_DIR}/scanner_pnl_bars_{EVAL_DATE}.png"

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle(f'Scanner PnL — Summary Charts — {EVAL_DATE}\n'
             f'Entry at signal time | SL={STOP_LOSS_PCT}%',
             fontsize=13, fontweight='bold')

# 1. PnL distribution by horizon (SL-adjusted)
ax = axes[0, 0]
for h in HORIZONS_MIN:
    vals = df_pnl[f'sl_pnl_{h}m'].dropna()
    if len(vals) > 0:
        ax.hist(vals, bins=20, alpha=0.5, label=f'{h}m (avg={vals.mean():+.2f}%, n={len(vals)})')
ax.axvline(x=0, color='black', linewidth=1, linestyle='--')
ax.axvline(x=-STOP_LOSS_PCT, color='red', linewidth=1, linestyle=':', alpha=0.5)
ax.set_xlabel('PnL % (SL-adjusted)')
ax.set_ylabel('Count')
ax.set_title('PnL Distribution (SL-adjusted)')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Selected vs not selected bar chart (SL-adjusted)
ax = axes[0, 1]
x_pos = np.arange(len(HORIZONS_MIN))
width = 0.35
sel_means = [df_sel[f'sl_pnl_{h}m'].dropna().mean() if len(df_sel) > 0 else 0 for h in HORIZONS_MIN]
not_means = [df_not_sel[f'sl_pnl_{h}m'].dropna().mean() if len(df_not_sel) > 0 else 0 for h in HORIZONS_MIN]
sel_means = [0 if pd.isna(x) else x for x in sel_means]
not_means = [0 if pd.isna(x) else x for x in not_means]
n_sel_valid = [df_sel[f'sl_pnl_{h}m'].notna().sum() if len(df_sel) > 0 else 0 for h in HORIZONS_MIN]
n_not_valid = [df_not_sel[f'sl_pnl_{h}m'].notna().sum() if len(df_not_sel) > 0 else 0 for h in HORIZONS_MIN]
bars1 = ax.bar(x_pos - width/2, sel_means, width, label=f'Selected (n={len(df_sel)})', color='#2ecc71')
bars2 = ax.bar(x_pos + width/2, not_means, width, label=f'Not Selected (n={len(df_not_sel)})', color='#e74c3c', alpha=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels([f'{h}m' for h in HORIZONS_MIN])
ax.set_ylabel('Avg PnL %')
ax.set_title('Selected vs Not Selected (SL-adj)')
ax.legend()
ax.axhline(y=0, color='black', linewidth=0.8)
ax.grid(True, alpha=0.3, axis='y')
for bar_group in [bars1, bars2]:
    for bar in bar_group:
        h = bar.get_height()
        ax.annotate(f'{h:+.2f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

# 3. LONG vs SHORT bar chart (SL-adjusted)
ax = axes[0, 2]
long_means = [df_long[f'sl_pnl_{h}m'].dropna().mean() if len(df_long) > 0 else 0 for h in HORIZONS_MIN]
short_means = [df_short[f'sl_pnl_{h}m'].dropna().mean() if len(df_short) > 0 else 0 for h in HORIZONS_MIN]
long_means = [0 if pd.isna(x) else x for x in long_means]
short_means = [0 if pd.isna(x) else x for x in short_means]
bars1 = ax.bar(x_pos - width/2, long_means, width, label=f'LONG (n={len(df_long)})', color='#27ae60')
bars2 = ax.bar(x_pos + width/2, short_means, width, label=f'SHORT (n={len(df_short)})', color='#c0392b')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'{h}m' for h in HORIZONS_MIN])
ax.set_ylabel('Avg PnL %')
ax.set_title('LONG vs SHORT (SL-adj)')
ax.legend()
ax.axhline(y=0, color='black', linewidth=0.8)
ax.grid(True, alpha=0.3, axis='y')
for bar_group in [bars1, bars2]:
    for bar in bar_group:
        h = bar.get_height()
        ax.annotate(f'{h:+.2f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

# 4. Per-ticker PnL at 60m (horizontal bar chart, SL-adjusted)
ax = axes[1, 0]
df_60m = df_pnl.dropna(subset=['sl_pnl_60m']).sort_values('sl_pnl_60m', ascending=True)
if len(df_60m) > 0:
    colors = ['#27ae60' if x > 0 else '#c0392b' for x in df_60m['sl_pnl_60m']]
    y_pos = range(len(df_60m))
    tick_fontsize = max(4, min(7, 200 // max(len(df_60m), 1)))
    ax.barh(y_pos, df_60m['sl_pnl_60m'], color=colors, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f"{'✓' if r['selected'] else '·'}{r['ticker']} ({r['direction'][0]}) @{r['entry_time_et']}"
         for _, r in df_60m.iterrows()],
        fontsize=tick_fontsize)
    ax.set_xlabel('PnL % (SL-adjusted)')
    ax.set_title(f'Per-Ticker PnL @ 60m (SL={STOP_LOSS_PCT}%)')
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.axvline(x=-STOP_LOSS_PCT, color='red', linewidth=0.5, linestyle=':', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='x')
else:
    ax.text(0.5, 0.5, 'No trades with 60m data', transform=ax.transAxes,
            ha='center', va='center', fontsize=12)

# 5. Confidence vs actual PnL scatter (60m, SL-adjusted)
ax = axes[1, 1]
df_scatter = df_pnl.dropna(subset=['sl_pnl_60m'])
for direction, color, marker in [('LONG', '#27ae60', '^'), ('SHORT', '#c0392b', 'v')]:
    mask = df_scatter['direction'] == direction
    df_d = df_scatter[mask]
    sel_mask = df_d['selected'] == True
    ax.scatter(df_d[sel_mask]['confidence'], df_d[sel_mask]['sl_pnl_60m'],
              c=color, marker=marker, s=80, alpha=0.8, edgecolors='black', linewidths=0.5,
              label=f'{direction} (sel)')
    ax.scatter(df_d[~sel_mask]['confidence'], df_d[~sel_mask]['sl_pnl_60m'],
              c=color, marker=marker, s=40, alpha=0.3,
              label=f'{direction} (not)')
    for _, r in df_d[sel_mask].iterrows():
        ax.annotate(r['ticker'], xy=(r['confidence'], r['sl_pnl_60m']),
                   fontsize=6, alpha=0.7, xytext=(3, 3), textcoords='offset points')
ax.axhline(y=0, color='black', linewidth=0.8)
ax.axhline(y=-STOP_LOSS_PCT, color='red', linewidth=0.5, linestyle=':', alpha=0.3)
ax.set_xlabel('Confidence (1-10)')
ax.set_ylabel('PnL @ 60m % (SL-adj)')
ax.set_title('Confidence vs Actual PnL')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 6. Catalyst type performance (60m avg PnL, SL-adjusted)
ax = axes[1, 2]
df_cat_valid = df_pnl.dropna(subset=['sl_pnl_60m', 'sl_win_60m'])
if len(df_cat_valid) > 0:
    cat_stats = df_cat_valid.groupby('catalyst_type').agg(
        avg_pnl=('sl_pnl_60m', 'mean'),
        count=('sl_pnl_60m', 'count'),
        wr=('sl_win_60m', 'mean'),
    ).reset_index()
    cat_stats = cat_stats[cat_stats['count'] >= 1].sort_values('avg_pnl', ascending=True)
    colors_cat = ['#27ae60' if x > 0 else '#c0392b' for x in cat_stats['avg_pnl']]
    y_pos_cat = range(len(cat_stats))
    bars = ax.barh(y_pos_cat, cat_stats['avg_pnl'], color=colors_cat, height=0.6)
    ax.set_yticks(y_pos_cat)
    ax.set_yticklabels([f"{r['catalyst_type']} (n={r['count']})" for _, r in cat_stats.iterrows()],
                       fontsize=9)
    ax.set_xlabel('Avg PnL @ 60m % (SL-adj)')
    ax.set_title('PnL by Catalyst Type (SL-adj)')
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')
    for bar, (_, r) in zip(bars, cat_stats.iterrows()):
        w = bar.get_width()
        ax.annotate(f'WR={r["wr"]:.0%}', xy=(w, bar.get_y() + bar.get_height()/2),
                   xytext=(5 if w >= 0 else -5, 0), textcoords='offset points',
                   ha='left' if w >= 0 else 'right', va='center', fontsize=8)

plt.tight_layout()
plt.savefig(fig_path_bars, dpi=150, bbox_inches='tight')
print(f"  Summary chart saved: {fig_path_bars}")
plt.close()


# ==============================================================================
# SUMMARY STATISTICS
# ==============================================================================
print(f"\n\n{'='*100}")
print(f"  SUMMARY (SL={STOP_LOSS_PCT}%)")
print(f"{'='*100}")

n_total = len(df_pnl)
n_sel_count = len(df_sel)
n_stopped = df_pnl['stopped_out'].sum()
n_sel_stopped = df_sel['stopped_out'].sum() if n_sel_count > 0 else 0

print(f"\n  Trades: {n_total} total, {n_sel_count} selected, {n_stopped} stopped out")
if n_sel_count > 0:
    print(f"  Selected stopped: {n_sel_stopped}/{n_sel_count}")

entry_times = df_pnl['entry_time_et'].value_counts().sort_index()
print(f"\n  Entry times:")
for et, cnt in entry_times.items():
    print(f"    {et} ET: {cnt} trades")

for h in HORIZONS_MIN:
    raw_col = f'pnl_{h}m'
    sl_col = f'sl_pnl_{h}m'
    valid_all = df_pnl[sl_col].dropna()
    valid_sel = df_sel[sl_col].dropna() if n_sel_count > 0 else pd.Series(dtype=float)
    n_valid_all = len(valid_all)
    n_valid_sel = len(valid_sel)

    raw_avg = df_pnl[raw_col].dropna().mean() if n_valid_all > 0 else 0
    sl_avg = valid_all.mean() if n_valid_all > 0 else 0
    sl_sel_avg = valid_sel.mean() if n_valid_sel > 0 else 0
    wr_all = (valid_all > 0).mean() * 100 if n_valid_all > 0 else 0
    wr_sel = (valid_sel > 0).mean() * 100 if n_valid_sel > 0 else 0
    se_all = valid_all.std() / np.sqrt(n_valid_all) if n_valid_all > 1 else float('inf')
    se_sel = valid_sel.std() / np.sqrt(n_valid_sel) if n_valid_sel > 1 else float('inf')

    n_trunc_all = n_total - n_valid_all
    n_trunc_sel = n_sel_count - n_valid_sel
    trunc_all = f"  ({n_trunc_all} truncated)" if n_trunc_all > 0 else ""
    trunc_sel = f"  ({n_trunc_sel} truncated)" if n_trunc_sel > 0 else ""

    print(f"\n  @ {h}m horizon:")
    print(f"    ALL  (n={n_valid_all}): raw={raw_avg:+.2f}%  SL={sl_avg:+.2f}% ±{se_all:.2f}% SE  WR={wr_all:.0f}%{trunc_all}")
    print(f"    SEL  (n={n_valid_sel}):  SL={sl_sel_avg:+.2f}% ±{se_sel:.2f}% SE  WR={wr_sel:.0f}%{trunc_sel}")
    if n_valid_sel > 0:
        print(f"    Portfolio (1/{n_sel_count} per trade): {sl_sel_avg:+.3f}%")

print(f"\n  ── TOP 5 WINNERS @ 60m (SL-adj) ──")
top5 = df_pnl.dropna(subset=['sl_pnl_60m']).nlargest(5, 'sl_pnl_60m')
for _, r in top5.iterrows():
    sel = '✓SEL' if r['selected'] else ''
    print(f"    {r['ticker']:<6} {r['direction']:<6} {r['sl_pnl_60m']:>+6.2f}%  "
          f"entry={r['entry_time_et']}  conf={r['confidence']}  {r['catalyst_type']}  {sel}")

print(f"\n  ── TOP 5 LOSERS @ 60m (SL-adj) ──")
bot5 = df_pnl.dropna(subset=['sl_pnl_60m']).nsmallest(5, 'sl_pnl_60m')
for _, r in bot5.iterrows():
    sel = '✓SEL' if r['selected'] else ''
    gate = f"  GATE:{r['gate_reason'][:30]}" if r['gate_reason'] and r['gate_reason'] != 'nan' else ''
    stop = f"  stop@{r['stop_minute']:.0f}m" if r['stopped_out'] else ''
    print(f"    {r['ticker']:<6} {r['direction']:<6} {r['sl_pnl_60m']:>+6.2f}%  "
          f"entry={r['entry_time_et']}  conf={r['confidence']}  {r['catalyst_type']}  {sel}{gate}{stop}")

if len(df_gated) > 0 and len(df_ungated) > 0:
    print(f"\n  ── GATE EFFECTIVENESS @ 60m (SL-adj) ──")
    gate_valid = df_gated['sl_pnl_60m'].dropna()
    ungate_valid = df_ungated['sl_pnl_60m'].dropna()
    if len(gate_valid) > 0 and len(ungate_valid) > 0:
        gate_avg = gate_valid.mean()
        ungate_avg = ungate_valid.mean()
        delta = ungate_avg - gate_avg
        print(f"    Ungated avg: {ungate_avg:+.2f}%  (n={len(ungate_valid)})")
        print(f"    Gated avg:   {gate_avg:+.2f}%  (n={len(gate_valid)})")
        print(f"    Gate value:  {delta:+.2f}%  "
              f"({'gates SAVED money' if delta > 0 else 'gates COST money'})")

print(f"\n  Charts: {fig_path_ts}")
print(f"          {fig_path_bars}")

print(f"\n{'='*100}")
print(f"  DONE — {len(df_pnl)} predictions evaluated")
print(f"{'='*100}\n")
