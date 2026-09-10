# ── Pattern Library Population Script ──────────────────────────────────────────
# Run daily after market close. Fetches scanner signals from BQ, computes actual
# PnL from Polygon, displays results, and lets you select which patterns to add
# to the BQ pattern library (scanner_patterns) for Stage 2 few-shot retrieval.
#
# Usage: Run cells top-to-bottom in Colab. Fill in PATTERNS_TO_ADD in Cell 5.
# ──────────────────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
# CELL 1: CONFIG
# ══════════════════════════════════════════════════════════════════════════════

EVAL_DATE = '2026-09-09'  # <── change to the date you want to evaluate

ENTRY_DELAY_MINUTES = 30  # entry = scan_time + 30 min (time to review signals)
HOLD_MINUTES  = 60        # compute return at entry + N minutes
EOD_TIME_PT   = '12:55'   # 12:55 PM PT = 3:55 PM ET (near close)
MARKET_OPEN_PT = '06:30'  # earliest possible entry (market open)

GCP_PROJECT   = 'ys-time-series'
BQ_PREMARKET  = 'ys-time-series.scoring_logs.premarket_scanner'
BQ_DISCOVERY  = 'ys-time-series.scoring_logs.discovery_scanner'
BQ_PATTERNS   = 'ys-time-series.scoring_logs.scanner_patterns'

POLYGON_API_KEY = '2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs'

# ══════════════════════════════════════════════════════════════════════════════
# CELL 2: FETCH SCANNER SIGNALS FROM BQ
# ══════════════════════════════════════════════════════════════════════════════

import pandas as pd
from google.cloud import bigquery

bq_client = bigquery.Client(project=GCP_PROJECT)

query = f"""
SELECT
    'premarket' AS source,
    scan_date, scan_time_pt, ticker, direction, confidence,
    CAST(composite_score AS FLOAT64) AS score,
    catalyst_type, catalyst, reasoning,
    CAST(expected_move_pct AS FLOAT64) AS expected_move_pct,
    CAST(remaining_move_pct AS FLOAT64) AS remaining_move_pct,
    selected, gate_reason,
    rank_in_direction
FROM `{BQ_PREMARKET}`
WHERE scan_date = '{EVAL_DATE}'
  AND direction IN ('LONG', 'SHORT')

UNION ALL

SELECT
    'discovery' AS source,
    scan_date, scan_time_pt, ticker, direction, confidence,
    CAST(tradability_score AS FLOAT64) AS score,
    catalyst_type, catalyst, reasoning,
    CAST(expected_move_pct AS FLOAT64) AS expected_move_pct,
    CAST(remaining_move_pct AS FLOAT64) AS remaining_move_pct,
    selected, gate_reason,
    rank_in_direction
FROM `{BQ_DISCOVERY}`
WHERE scan_date = '{EVAL_DATE}'
  AND direction IN ('LONG', 'SHORT')
"""

signals_df = bq_client.query(query).to_dataframe()
print(f"Loaded {len(signals_df)} signals for {EVAL_DATE}")
print(f"  Sources: {signals_df['source'].value_counts().to_dict()}")
print(f"  Directions: {signals_df['direction'].value_counts().to_dict()}")

# Deduplicate: same ticker+direction across scanners → keep highest score
signals_df = signals_df.sort_values('score', ascending=False).drop_duplicates(
    subset=['ticker', 'direction'], keep='first'
).reset_index(drop=True)
print(f"  After dedup: {len(signals_df)} unique ticker+direction signals")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 3: FETCH ACTUAL PRICES FROM POLYGON
# ══════════════════════════════════════════════════════════════════════════════

import pytz
from datetime import datetime, timedelta

try:
    from polygon import RESTClient
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                           'polygon-api-client', '--quiet'])
    from polygon import RESTClient

poly = RESTClient(POLYGON_API_KEY)
ET = pytz.timezone('America/New_York')
PT = pytz.timezone('America/Los_Angeles')

market_open_pt = PT.localize(datetime.strptime(f'{EVAL_DATE} {MARKET_OPEN_PT}', '%Y-%m-%d %H:%M'))
eod_dt_pt = PT.localize(datetime.strptime(f'{EVAL_DATE} {EOD_TIME_PT}', '%Y-%m-%d %H:%M'))
eod_dt_et = eod_dt_pt.astimezone(ET)

print(f"Market open: {market_open_pt.strftime('%I:%M %p PT')}")
print(f"Entry delay: +{ENTRY_DELAY_MINUTES}m after scan_time")
print(f"Hold:        +{HOLD_MINUTES}m after entry")
print(f"EOD:         {eod_dt_et.strftime('%I:%M %p ET')}")


def _compute_entry_time_et(scan_time_pt_str):
    """Compute entry time = max(scan_time + delay, market_open) in ET."""
    scan_dt_pt = PT.localize(
        datetime.strptime(f'{EVAL_DATE} {scan_time_pt_str}', '%Y-%m-%d %H:%M'))
    entry_dt_pt = scan_dt_pt + timedelta(minutes=ENTRY_DELAY_MINUTES)
    if entry_dt_pt < market_open_pt:
        entry_dt_pt = market_open_pt
    return entry_dt_pt.astimezone(ET)


# Pre-fetch all 1-min bars per ticker (avoid duplicate Polygon calls)
_bars_cache = {}

def _get_bars(ticker):
    if ticker not in _bars_cache:
        try:
            bars = list(poly.get_aggs(
                ticker=ticker, multiplier=1, timespan='minute',
                from_=EVAL_DATE, to=EVAL_DATE, limit=500,
            ))
            ts_bars = []
            for b in bars:
                ts = pd.Timestamp(b.timestamp, unit='ms', tz='UTC').tz_convert(ET)
                ts_bars.append({
                    'ts': ts, 'open': float(b.open), 'high': float(b.high),
                    'low': float(b.low), 'close': float(b.close),
                    'volume': int(getattr(b, 'volume', 0) or 0),
                })
            if ts_bars:
                _bars_cache[ticker] = pd.DataFrame(ts_bars).set_index('ts').sort_index()
            else:
                _bars_cache[ticker] = None
        except Exception as e:
            print(f"  [ERROR] bars for {ticker}: {e}")
            _bars_cache[ticker] = None
    return _bars_cache[ticker]


def fetch_actual_pnl(ticker, direction, scan_time_pt_str):
    """Compute PnL using entry = scan_time + delay, hold = entry + 60m."""
    df = _get_bars(ticker)
    if df is None or df.empty:
        return None

    entry_dt_et = _compute_entry_time_et(scan_time_pt_str)
    hold_dt_et = entry_dt_et + timedelta(minutes=HOLD_MINUTES)

    # Entry price: first bar at or after entry time
    entry_mask = df.index >= entry_dt_et
    if not entry_mask.any():
        return None
    entry_price = df.loc[entry_mask].iloc[0]['open']

    # Hold price: bar closest to entry + hold_minutes
    hold_mask = df.index >= hold_dt_et
    if hold_mask.any():
        hold_price = df.loc[hold_mask].iloc[0]['close']
    else:
        hold_price = df.iloc[-1]['close']

    # EOD price: last bar before EOD cutoff
    eod_mask = df.index <= eod_dt_et
    if eod_mask.any():
        eod_price = df.loc[eod_mask].iloc[-1]['close']
    else:
        eod_price = df.iloc[-1]['close']

    # MFE/MAE during hold window (entry to EOD)
    hold_window = df.loc[(df.index >= entry_dt_et) & (df.index <= eod_dt_et)]
    if hold_window.empty:
        return None

    if direction == 'LONG':
        mfe_price = hold_window['high'].max()
        mae_price = hold_window['low'].min()
        sign = 1
    else:
        mfe_price = hold_window['low'].min()
        mae_price = hold_window['high'].max()
        sign = -1

    ret_60m = sign * (hold_price - entry_price) / entry_price * 100
    ret_eod = sign * (eod_price - entry_price) / entry_price * 100
    mfe_pct = sign * (mfe_price - entry_price) / entry_price * 100
    mae_pct = sign * (mae_price - entry_price) / entry_price * 100

    return {
        'entry_time_et': entry_dt_et.strftime('%I:%M %p'),
        'entry_price': round(entry_price, 2),
        'hold_price': round(hold_price, 2),
        'eod_price': round(eod_price, 2),
        'ret_60m': round(ret_60m, 2),
        'ret_eod': round(ret_eod, 2),
        'mfe_pct': round(mfe_pct, 2),
        'mae_pct': round(mae_pct, 2),
        'outcome': 'WIN' if ret_60m > 1.0 else ('LOSS' if ret_60m < -1.0 else 'FLAT'),
    }


results = []
for _, row in signals_df.iterrows():
    ticker = row['ticker']
    direction = row['direction']
    scan_time = row.get('scan_time_pt', MARKET_OPEN_PT)
    print(f"  {ticker:<6} ({direction:<5} scan={scan_time})...", end='')
    pnl = fetch_actual_pnl(ticker, direction, scan_time)
    if pnl:
        print(f" entry@{pnl['entry_time_et']}=${pnl['entry_price']}, "
              f"60m={pnl['ret_60m']:+.2f}%, MFE={pnl['mfe_pct']:+.2f}%, "
              f"EOD={pnl['ret_eod']:+.2f}% → {pnl['outcome']}")
        results.append({**row.to_dict(), **pnl})
    else:
        print(f" NO DATA (market closed or no bars)")
        results.append({**row.to_dict(), 'ret_60m': None, 'outcome': 'NO_DATA'})

results_df = pd.DataFrame(results)
print(f"\n{len(results)} signals evaluated, "
      f"{len([r for r in results if r.get('ret_60m') is not None])} with price data")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 4: SUMMARY TABLE + CHART
# ══════════════════════════════════════════════════════════════════════════════

import matplotlib.pyplot as plt
import numpy as np

display_cols = ['ticker', 'direction', 'source', 'scan_time_pt', 'entry_time_et',
                'selected', 'catalyst_type', 'confidence', 'score',
                'ret_60m', 'mfe_pct', 'mae_pct', 'ret_eod', 'outcome', 'gate_reason']

summary = results_df[[c for c in display_cols if c in results_df.columns]].copy()
summary = summary.sort_values('ret_60m', ascending=False, na_position='last')

print("\n" + "=" * 100)
print(f"  SCANNER EVALUATION — {EVAL_DATE}")
print("=" * 100)

has_data = summary[summary['ret_60m'].notna()]
if len(has_data) > 0:
    avg_pnl = has_data['ret_60m'].mean()
    win_rate = (has_data['ret_60m'] > 1.0).mean() * 100
    avg_mfe = has_data['mfe_pct'].mean() if 'mfe_pct' in has_data else 0

    selected = has_data[has_data['selected'] == True]
    rejected = has_data[has_data['selected'] == False]

    print(f"\n  ALL signals:     n={len(has_data):>3}  avg 60m={avg_pnl:+.2f}%  "
          f"WR={win_rate:.0f}%  avg MFE={avg_mfe:+.2f}%")
    if len(selected) > 0:
        print(f"  SELECTED:        n={len(selected):>3}  avg 60m={selected['ret_60m'].mean():+.2f}%  "
              f"WR={(selected['ret_60m'] > 1.0).mean()*100:.0f}%  "
              f"avg MFE={selected['mfe_pct'].mean():+.2f}%")
    if len(rejected) > 0:
        print(f"  NOT SELECTED:    n={len(rejected):>3}  avg 60m={rejected['ret_60m'].mean():+.2f}%  "
              f"WR={(rejected['ret_60m'] > 1.0).mean()*100:.0f}%  "
              f"avg MFE={rejected['mfe_pct'].mean():+.2f}%")

    for dir_label in ['LONG', 'SHORT']:
        sub = has_data[has_data['direction'] == dir_label]
        if len(sub) > 0:
            print(f"  {dir_label:>5} only:      n={len(sub):>3}  avg 60m={sub['ret_60m'].mean():+.2f}%  "
                  f"WR={(sub['ret_60m'] > 1.0).mean()*100:.0f}%")

    # Per catalyst type
    print(f"\n  By catalyst type:")
    for cat, grp in has_data.groupby('catalyst_type'):
        print(f"    {str(cat):<16} n={len(grp):>2}  avg={grp['ret_60m'].mean():+.2f}%  "
              f"WR={(grp['ret_60m'] > 1.0).mean()*100:.0f}%")

print(f"\n{'─' * 100}")
for _, r in summary.iterrows():
    sel_tag = '✅' if r.get('selected') else '⛔'
    ret = f"{r['ret_60m']:+.2f}%" if pd.notna(r.get('ret_60m')) else 'N/A'
    mfe = f"{r['mfe_pct']:+.2f}%" if pd.notna(r.get('mfe_pct')) else ''
    gate = f" [{r['gate_reason']}]" if r.get('gate_reason') else ''
    scan_t = str(r.get('scan_time_pt', ''))[:5]
    entry_t = str(r.get('entry_time_et', ''))[:8]
    print(f"  {sel_tag} {r['ticker']:<6} {r['direction']:<5} scan={scan_t} entry={entry_t:<8} "
          f"{str(r.get('catalyst_type','')):<14} conf={r.get('confidence',0)}  "
          f"score={r.get('score',0):>6.1f}  60m={ret:>8}  MFE={mfe:>8}{gate}")
print(f"{'─' * 100}")

# ── Chart ──
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart: PnL by ticker
plot_data = has_data.sort_values('ret_60m', ascending=True)
colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in plot_data['ret_60m']]
labels = [f"{r['ticker']} ({r['direction'][0]})" for _, r in plot_data.iterrows()]
axes[0].barh(range(len(plot_data)), plot_data['ret_60m'], color=colors)
axes[0].set_yticks(range(len(plot_data)))
axes[0].set_yticklabels(labels, fontsize=8)
axes[0].axvline(x=0, color='black', linewidth=0.5)
axes[0].set_xlabel('Return at 60m (%)')
axes[0].set_title(f'Scanner Signal PnL — {EVAL_DATE}')

# Scatter: MFE vs 60m return
if 'mfe_pct' in has_data.columns:
    sel_mask = has_data['selected'] == True
    axes[1].scatter(has_data.loc[sel_mask, 'mfe_pct'],
                    has_data.loc[sel_mask, 'ret_60m'],
                    c='blue', label='Selected', alpha=0.7, s=60)
    axes[1].scatter(has_data.loc[~sel_mask, 'mfe_pct'],
                    has_data.loc[~sel_mask, 'ret_60m'],
                    c='gray', label='Not Selected', alpha=0.5, s=40)
    for _, r in has_data.iterrows():
        axes[1].annotate(r['ticker'], (r['mfe_pct'], r['ret_60m']),
                        fontsize=7, alpha=0.7)
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].axvline(x=0, color='black', linewidth=0.5)
    axes[1].plot([has_data['mfe_pct'].min(), has_data['mfe_pct'].max()],
                 [has_data['mfe_pct'].min(), has_data['mfe_pct'].max()],
                 'k--', alpha=0.3, label='MFE=Return (no giveback)')
    axes[1].set_xlabel('MFE (%)')
    axes[1].set_ylabel('60m Return (%)')
    axes[1].set_title('MFE vs Realized Return')
    axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(f'/tmp/scanner_eval_{EVAL_DATE}.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"  Chart saved: /tmp/scanner_eval_{EVAL_DATE}.png")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 5: SELECT PATTERNS TO ADD
# ══════════════════════════════════════════════════════════════════════════════
# Review the table above and fill in patterns worth remembering.
# Each entry needs: ticker, lesson, and pattern_rule.
# secondary_ticker is optional (for pair trades like deal acquirer/target).

PATTERNS_TO_ADD = [
    # {
    #     'ticker': 'TTAN',
    #     'pattern_type': 'earnings_guidance_cut',
    #     'lesson': 'Earnings beat + guidance cut = two-wave selloff. Never predict LONG bounce.',
    #     'pattern_rule': 'earnings + beat + guidance_cut → SHORT only',
    #     'secondary_ticker': None,
    # },
    # {
    #     'ticker': 'SIG',
    #     'pattern_type': 'earnings_beat_continuation',
    #     'lesson': 'Strong earnings beat + gap-up > 15% still has 3-5% continuation at open.',
    #     'pattern_rule': 'earnings + beat + gap_up > 10% + no guidance_cut → LONG continuation',
    #     'secondary_ticker': None,
    # },
]

print(f"Patterns to add: {len(PATTERNS_TO_ADD)}")
for p in PATTERNS_TO_ADD:
    print(f"  • {p['ticker']}: {p['pattern_type']} — {p['lesson'][:80]}")

# ══════════════════════════════════════════════════════════════════════════════
# CELL 6: PUSH SELECTED PATTERNS TO BQ
# ══════════════════════════════════════════════════════════════════════════════

import json, uuid

if not PATTERNS_TO_ADD:
    print("No patterns to add. Fill in PATTERNS_TO_ADD in Cell 5 and re-run this cell.")
else:
    # Ensure table exists
    try:
        bq_client.query(f"""
        CREATE TABLE IF NOT EXISTS `{BQ_PATTERNS}` (
            pattern_id          STRING,
            event_date          STRING,
            pattern_type        STRING,
            catalyst_type       STRING,
            tickers             STRING,
            primary_ticker      STRING,
            secondary_ticker    STRING,
            catalyst_summary    STRING,
            primary_direction   STRING,
            primary_gap_pct     FLOAT64,
            primary_ret_60m     FLOAT64,
            primary_mfe_pct     FLOAT64,
            primary_eod_pct     FLOAT64,
            primary_outcome     STRING,
            secondary_direction STRING,
            secondary_gap_pct   FLOAT64,
            secondary_ret_60m   FLOAT64,
            secondary_eod_pct   FLOAT64,
            secondary_outcome   STRING,
            lesson              STRING,
            pattern_rule        STRING,
            created_at          STRING
        )""").result()
    except Exception:
        pass

    rows_to_insert = []
    for p in PATTERNS_TO_ADD:
        ticker = p['ticker']
        match = results_df[results_df['ticker'] == ticker]
        if match.empty:
            print(f"  ⚠ {ticker} not found in today's signals, skipping")
            continue

        row = match.iloc[0]
        sec_ticker = p.get('secondary_ticker')
        sec_data = {}
        if sec_ticker:
            sec_match = results_df[results_df['ticker'] == sec_ticker]
            if not sec_match.empty:
                sec_data = sec_match.iloc[0].to_dict()

        tickers_str = ticker
        if sec_ticker:
            tickers_str = f"{ticker},{sec_ticker}"

        pattern_row = {
            'pattern_id': str(uuid.uuid4())[:8],
            'event_date': EVAL_DATE,
            'pattern_type': p['pattern_type'],
            'catalyst_type': str(row.get('catalyst_type', '')),
            'tickers': tickers_str,
            'primary_ticker': ticker,
            'secondary_ticker': sec_ticker or '',
            'catalyst_summary': str(row.get('catalyst', ''))[:500],
            'primary_direction': row.get('direction', ''),
            'primary_gap_pct': float(row.get('expected_move_pct', 0) or 0),
            'primary_ret_60m': float(row.get('ret_60m', 0) or 0),
            'primary_mfe_pct': float(row.get('mfe_pct', 0) or 0),
            'primary_eod_pct': float(row.get('ret_eod', 0) or 0),
            'primary_outcome': row.get('outcome', 'UNKNOWN'),
            'secondary_direction': sec_data.get('direction', ''),
            'secondary_gap_pct': float(sec_data.get('expected_move_pct', 0) or 0),
            'secondary_ret_60m': float(sec_data.get('ret_60m', 0) or 0),
            'secondary_eod_pct': float(sec_data.get('ret_eod', 0) or 0),
            'secondary_outcome': sec_data.get('outcome', ''),
            'lesson': p['lesson'],
            'pattern_rule': p.get('pattern_rule', ''),
            'created_at': datetime.now(pytz.utc).isoformat(),
        }
        rows_to_insert.append(pattern_row)
        print(f"  ✓ {ticker} ({row.get('direction')}) → {p['pattern_type']}: "
              f"60m={row.get('ret_60m', 'N/A')}, outcome={row.get('outcome', '?')}")

    if rows_to_insert:
        errors = bq_client.insert_rows_json(BQ_PATTERNS, rows_to_insert)
        if errors:
            print(f"  ✗ BQ insert errors: {errors}")
        else:
            print(f"\n  ✓ Inserted {len(rows_to_insert)} patterns → {BQ_PATTERNS}")

    # Verify
    count = bq_client.query(f"SELECT COUNT(*) AS n FROM `{BQ_PATTERNS}`").to_dataframe()
    print(f"  Pattern library total: {count['n'].iloc[0]} patterns")
