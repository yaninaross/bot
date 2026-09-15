# %% ── Scanner Investigation — FP, FN, Missed Opportunities, Patterns ────────
#
# Open-ended analysis of today's scanner performance:
#   1. False Positives: selected signals that lost money
#   2. False Negatives: discovered but not selected, yet moved big
#   3. Missed opportunities: big movers today not even discovered
#   4. Pattern analysis: what do losers have in common?
#   5. Gate analysis: did gates catch the right things?
#
# Colab-ready, config vars at top.

import os, sys, time, json, warnings
from datetime import datetime, timedelta
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import pytz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
POLYGON_API_KEY = "YOUR_POLYGON_API_KEY"
BQ_PREMARKET = "YOUR_GCP_PROJECT.scoring_logs.premarket_scanner"
BQ_DISCOVERY = "YOUR_GCP_PROJECT.scoring_logs.discovery_scanner"
STOP_LOSS_PCT = 2.5
ENTRY_TIME_ET = "09:30"
FP_THRESHOLD = -1.0
TP_THRESHOLD = 1.5
BIG_MOVE_PCT = 5.0

ET = pytz.timezone('America/New_York')
PT = pytz.timezone('America/Los_Angeles')

poly = PolygonClient(POLYGON_API_KEY)
bq = bigquery.Client()

W = 100

print(f"\n{'='*W}")
print(f"  SCANNER INVESTIGATION — {EVAL_DATE}")
print(f"  FP / FN / Missed Opportunities / Pattern Analysis")
print(f"{'='*W}")


# ==============================================================================
# 1. FETCH ALL BQ ROWS (including scan-only NEUTRAL)
# ==============================================================================
print(f"\n  Phase 1: Fetching ALL scanner rows from BQ...")

def fetch_all_rows(table, scanner_name):
    q = f"""
    SELECT
        ticker, direction, confidence, selected,
        catalyst_type, catalyst, gate_reason, move_is_exhausted,
        remaining_move_pct, expected_move_pct, scanner_version,
        scan_time_pt, stage2_attempted, stage2_model,
        stage2_error, claude_error, rank_in_direction,
        COALESCE(reasoning, '') as reasoning
    FROM `{table}`
    WHERE scan_date = '{EVAL_DATE}'
    ORDER BY scan_time_pt DESC, direction, rank_in_direction
    """
    df = bq.query(q).to_dataframe()
    df['scanner'] = scanner_name
    return df

df_pre_all = fetch_all_rows(BQ_PREMARKET, 'premarket')
df_disc_all = fetch_all_rows(BQ_DISCOVERY, 'discovery')

latest_pre_time = df_pre_all['scan_time_pt'].max() if len(df_pre_all) > 0 else None
latest_disc_time = df_disc_all['scan_time_pt'].max() if len(df_disc_all) > 0 else None

df_pre = df_pre_all[df_pre_all['scan_time_pt'] == latest_pre_time].copy()
df_disc = df_disc_all[df_disc_all['scan_time_pt'] == latest_disc_time].copy()

print(f"    Premarket latest run ({latest_pre_time}): {len(df_pre)} rows")
print(f"      LONG: {len(df_pre[df_pre['direction']=='LONG'])}, "
      f"SHORT: {len(df_pre[df_pre['direction']=='SHORT'])}, "
      f"NEUTRAL: {len(df_pre[df_pre['direction']=='NEUTRAL'])}")
print(f"    Discovery latest run ({latest_disc_time}): {len(df_disc)} rows")
print(f"      LONG: {len(df_disc[df_disc['direction']=='LONG'])}, "
      f"SHORT: {len(df_disc[df_disc['direction']=='SHORT'])}, "
      f"NEUTRAL: {len(df_disc[df_disc['direction']=='NEUTRAL'])}")

df_combined = pd.concat([df_pre, df_disc], ignore_index=True)
all_discovered_tickers = set(df_combined['ticker'].unique())
print(f"    Total unique tickers discovered: {len(all_discovered_tickers)}")


# ==============================================================================
# 2. FETCH PRICE DATA FOR ALL DISCOVERED TICKERS
# ==============================================================================
print(f"\n  Phase 2: Fetching Polygon intraday data for all {len(all_discovered_tickers)} tickers...")

def get_intraday_returns(ticker, eval_date):
    try:
        bars = list(poly.get_aggs(
            ticker=ticker, multiplier=1, timespan="minute",
            from_=eval_date, to=eval_date, limit=500,
        ))
        if not bars:
            return None

        bar_map = {}
        for b in bars:
            ts = pd.Timestamp(b.timestamp, unit='ms', tz='UTC').tz_convert(ET)
            key = ts.strftime('%H:%M')
            bar_map[key] = {
                'open': float(b.open), 'high': float(b.high),
                'low': float(b.low), 'close': float(b.close),
                'volume': int(b.volume), 'ts': ts,
            }

        entry_bar = bar_map.get(ENTRY_TIME_ET)
        if not entry_bar:
            candidates = sorted(
                [(k, v) for k, v in bar_map.items() if k >= ENTRY_TIME_ET],
                key=lambda x: x[0]
            )
            if candidates:
                entry_bar = candidates[0][1]
            else:
                return None

        entry_price = entry_bar['open']
        entry_dt = datetime.strptime(f"{eval_date} {ENTRY_TIME_ET}", '%Y-%m-%d %H:%M')
        entry_dt = ET.localize(entry_dt)

        rth_bars = {k: v for k, v in bar_map.items() if k >= ENTRY_TIME_ET and k <= '16:00'}
        sorted_bars = sorted(rth_bars.values(), key=lambda b: b['ts'])

        if not sorted_bars:
            return None

        mfe_price = max(b['high'] for b in sorted_bars)
        mae_price = min(b['low'] for b in sorted_bars)

        for target_key in ['10:00', '10:30']:
            bar = bar_map.get(target_key)

        ret_30m = 0
        ret_60m = 0
        for m, label in [(30, 'ret_30m'), (60, 'ret_60m')]:
            tgt = (entry_dt + timedelta(minutes=m)).strftime('%H:%M')
            if tgt in bar_map:
                ret = ((bar_map[tgt]['close'] - entry_price) / entry_price) * 100
            else:
                cands = sorted([(k, v) for k, v in bar_map.items()
                               if k <= tgt and k >= ENTRY_TIME_ET], key=lambda x: x[0])
                ret = ((cands[-1][1]['close'] - entry_price) / entry_price) * 100 if cands else 0
            if label == 'ret_30m': ret_30m = ret
            else: ret_60m = ret

        eod_price = sorted_bars[-1]['close']
        eod_ret = ((eod_price - entry_price) / entry_price) * 100

        prev_close = None
        gap_pct = 0
        try:
            from_date = (datetime.strptime(eval_date, '%Y-%m-%d') - timedelta(days=5)).strftime('%Y-%m-%d')
            daily = list(poly.get_aggs(ticker=ticker, multiplier=1, timespan="day",
                                        from_=from_date, to=eval_date, limit=10))
            ref_dt = datetime.strptime(eval_date, '%Y-%m-%d')
            prev = [b for b in daily if pd.Timestamp(b.timestamp, unit='ms', tz='UTC').date() < ref_dt.date()]
            if prev:
                prev_close = float(prev[-1].close)
                gap_pct = ((entry_price - prev_close) / prev_close) * 100
        except Exception:
            pass

        mfe_pct = ((mfe_price - entry_price) / entry_price) * 100
        mae_pct = ((mae_price - entry_price) / entry_price) * 100

        return {
            'entry_price': round(entry_price, 2),
            'ret_30m': round(ret_30m, 2),
            'ret_60m': round(ret_60m, 2),
            'ret_eod': round(eod_ret, 2),
            'mfe_pct': round(mfe_pct, 2),
            'mae_pct': round(mae_pct, 2),
            'gap_pct': round(gap_pct, 2),
            'prev_close': prev_close,
            'eod_price': round(eod_price, 2),
        }
    except Exception as e:
        return None

price_cache = {}
tickers_list = sorted(all_discovered_tickers)
for i, t in enumerate(tickers_list):
    r = get_intraday_returns(t, EVAL_DATE)
    if r:
        price_cache[t] = r
    if (i + 1) % 20 == 0:
        print(f"    {i+1}/{len(tickers_list)} fetched...")

print(f"    Got price data for {len(price_cache)}/{len(tickers_list)} tickers")


# ==============================================================================
# 3. FIND BIG MOVERS NOT DISCOVERED — Polygon gainers/losers
# ==============================================================================
print(f"\n  Phase 3: Finding big movers the scanners missed...")

big_movers = []

try:
    import requests
    url = "https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/gainers"
    params = {'apiKey': POLYGON_API_KEY}
    resp = requests.get(url, params=params, timeout=15)
    if resp.status_code == 200:
        for t in resp.json().get('tickers', []):
            ticker = t.get('ticker', '')
            day = t.get('todaysChangePerc', 0)
            price = t.get('updated_aggregate', {}).get('c', 0) or t.get('lastTrade', {}).get('p', 0)
            vol = t.get('updated_aggregate', {}).get('v', 0) or 0
            prev = t.get('prevDay', {}).get('c', 0) or 0
            if abs(day) >= BIG_MOVE_PCT and price >= 1.0:
                big_movers.append({
                    'ticker': ticker, 'change_pct': round(day, 2),
                    'price': round(price, 2), 'volume': int(vol),
                    'prev_close': round(prev, 2), 'prev_volume': 0,
                    'discovered': ticker in all_discovered_tickers,
                })

    url2 = "https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/losers"
    resp2 = requests.get(url2, params=params, timeout=15)
    if resp2.status_code == 200:
        for t in resp2.json().get('tickers', []):
            ticker = t.get('ticker', '')
            day = t.get('todaysChangePerc', 0)
            price = t.get('updated_aggregate', {}).get('c', 0) or t.get('lastTrade', {}).get('p', 0)
            vol = t.get('updated_aggregate', {}).get('v', 0) or 0
            prev = t.get('prevDay', {}).get('c', 0) or 0
            if abs(day) >= BIG_MOVE_PCT and price >= 1.0:
                big_movers.append({
                    'ticker': ticker, 'change_pct': round(day, 2),
                    'price': round(price, 2), 'volume': int(vol),
                    'prev_close': round(prev, 2), 'prev_volume': 0,
                    'discovered': ticker in all_discovered_tickers,
                })

    big_movers = list({m['ticker']: m for m in big_movers}.values())
    big_movers.sort(key=lambda x: abs(x['change_pct']), reverse=True)
    print(f"    Found {len(big_movers)} stocks with |change| >= {BIG_MOVE_PCT}% today")
    print(f"    Of those, {sum(1 for m in big_movers if m['discovered'])} were discovered by scanners")
    print(f"    MISSED: {sum(1 for m in big_movers if not m['discovered'])} not discovered at all")
except Exception as e:
    print(f"    Gainers/losers fetch failed: {e}")
    big_movers = []


# ==============================================================================
# 4. CLASSIFY ALL SIGNALS
# ==============================================================================
print(f"\n  Phase 4: Classifying all signals...")

records = []
for _, row in df_combined.iterrows():
    ticker = row['ticker']
    direction = row['direction']
    prices = price_cache.get(ticker)

    if prices is None:
        continue

    sign = 1.0 if direction == 'LONG' else (-1.0 if direction == 'SHORT' else 0)

    if direction == 'NEUTRAL':
        long_60m = prices['ret_60m']
        short_60m = -prices['ret_60m']
        best_dir = 'LONG' if long_60m > short_60m else 'SHORT'
        best_pnl = max(long_60m, short_60m)
        rec = {
            'ticker': ticker,
            'scanner': row['scanner'],
            'direction': 'NEUTRAL',
            'selected': False,
            'stage2_attempted': bool(pd.notna(row.get('stage2_attempted')) and row.get('stage2_attempted')),
            'confidence': 0,
            'catalyst_type': str(row.get('catalyst_type', '') or ''),
            'catalyst': str(row.get('catalyst', '') or '')[:100],
            'gate_reason': str(row.get('gate_reason', '') or ''),
            'reasoning': str(row.get('reasoning', '') or '')[:200],
            'move_is_exhausted': bool(pd.notna(row.get('move_is_exhausted')) and row.get('move_is_exhausted')),
            'gap_pct': prices['gap_pct'],
            'ret_30m': prices['ret_30m'],
            'ret_60m': prices['ret_60m'],
            'ret_eod': prices['ret_eod'],
            'mfe_pct': prices['mfe_pct'],
            'mae_pct': prices['mae_pct'],
            'pnl_60m': 0,
            'sl_pnl_60m': 0,
            'best_dir': best_dir,
            'best_pnl_60m': round(best_pnl, 2),
            'entry_price': prices['entry_price'],
        }
    else:
        raw_60m = prices['ret_60m'] * sign
        raw_30m = prices['ret_30m'] * sign
        raw_eod = prices['ret_eod'] * sign

        dir_mfe = prices['mfe_pct'] if direction == 'LONG' else -prices['mae_pct']
        dir_mae = prices['mae_pct'] if direction == 'LONG' else -prices['mfe_pct']

        sl_60m = max(raw_60m, -STOP_LOSS_PCT) if dir_mae <= -STOP_LOSS_PCT else raw_60m
        if dir_mae <= -STOP_LOSS_PCT:
            sl_60m = -STOP_LOSS_PCT

        rec = {
            'ticker': ticker,
            'scanner': row['scanner'],
            'direction': direction,
            'selected': bool(pd.notna(row.get('selected')) and row.get('selected')),
            'stage2_attempted': bool(pd.notna(row.get('stage2_attempted')) and row.get('stage2_attempted')),
            'confidence': int(row['confidence']) if pd.notna(row.get('confidence')) else 0,
            'catalyst_type': str(row.get('catalyst_type', '') or ''),
            'catalyst': str(row.get('catalyst', '') or '')[:100],
            'gate_reason': str(row.get('gate_reason', '') or ''),
            'reasoning': str(row.get('reasoning', '') or '')[:200],
            'move_is_exhausted': bool(pd.notna(row.get('move_is_exhausted')) and row.get('move_is_exhausted')),
            'gap_pct': prices['gap_pct'],
            'ret_30m': prices['ret_30m'],
            'ret_60m': prices['ret_60m'],
            'ret_eod': prices['ret_eod'],
            'mfe_pct': prices['mfe_pct'],
            'mae_pct': prices['mae_pct'],
            'pnl_60m': round(raw_60m, 2),
            'sl_pnl_60m': round(sl_60m, 2),
            'dir_mfe': round(dir_mfe, 2),
            'dir_mae': round(dir_mae, 2),
            'best_dir': direction,
            'best_pnl_60m': round(raw_60m, 2),
            'entry_price': prices['entry_price'],
        }

    records.append(rec)

df = pd.DataFrame(records)

df_deduped = df.sort_values('confidence', ascending=False).drop_duplicates(
    subset=['ticker', 'direction', 'scanner'], keep='first').reset_index(drop=True)

df_signals = df_deduped[df_deduped['direction'].isin(['LONG', 'SHORT'])].copy()
df_neutral = df_deduped[df_deduped['direction'] == 'NEUTRAL'].copy()

df_selected = df_signals[df_signals['selected'] == True].copy()
df_not_selected = df_signals[df_signals['selected'] == False].copy()

df_selected['label'] = df_selected['sl_pnl_60m'].apply(
    lambda x: 'TP' if x >= TP_THRESHOLD else ('FP' if x <= FP_THRESHOLD else 'MARGINAL'))
df_not_selected['label'] = 'NOT_SEL'

print(f"    Signals with direction: {len(df_signals)} (sel={len(df_selected)}, not_sel={len(df_not_selected)})")
print(f"    Scan-only (NEUTRAL): {len(df_neutral)}")


# ==============================================================================
# 5. ANALYSIS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  ╔══════════════════════════════════════════════════════════════════╗")
print(f"  ║  INVESTIGATION RESULTS — {EVAL_DATE}{'':>34}║")
print(f"  ╚══════════════════════════════════════════════════════════════════╝")
print(f"{'='*W}")

# ── 5A: FALSE POSITIVES — selected signals that lost money ──
print(f"\n  ═══ FALSE POSITIVES (Selected, PnL@60m <= {FP_THRESHOLD}%) ═══")
fps = df_selected[df_selected['label'] == 'FP'].sort_values('sl_pnl_60m')
if len(fps) == 0:
    print(f"    None — all selected signals above {FP_THRESHOLD}%")
else:
    for _, r in fps.iterrows():
        print(f"\n    ✗ {r['ticker']} {r['direction']} ({r['scanner']}) — SL PnL: {r['sl_pnl_60m']:+.2f}%")
        print(f"      conf={r['confidence']}/10  gap={r['gap_pct']:+.1f}%  "
              f"raw_60m={r['pnl_60m']:+.2f}%  MFE={r.get('dir_mfe', 0):+.2f}%  MAE={r.get('dir_mae', 0):+.2f}%")
        print(f"      catalyst_type: {r['catalyst_type']}")
        print(f"      catalyst: {r['catalyst'][:90]}")
        if r['reasoning']:
            print(f"      reasoning: {r['reasoning'][:120]}")

    print(f"\n    FP PATTERN ANALYSIS:")
    fp_cats = Counter(fps['catalyst_type'])
    fp_dirs = Counter(fps['direction'])
    fp_scanners = Counter(fps['scanner'])
    fp_gaps = fps['gap_pct']
    fp_confs = fps['confidence']
    print(f"      Catalyst types: {dict(fp_cats)}")
    print(f"      Directions: {dict(fp_dirs)}")
    print(f"      Scanners: {dict(fp_scanners)}")
    print(f"      Avg gap: {fp_gaps.mean():+.1f}%  Avg conf: {fp_confs.mean():.1f}")
    print(f"      Exhausted: {fps['move_is_exhausted'].sum()}/{len(fps)}")

# ── 5B: TRUE POSITIVES — selected signals that made money ──
print(f"\n  ═══ TRUE POSITIVES (Selected, PnL@60m >= {TP_THRESHOLD}%) ═══")
tps = df_selected[df_selected['label'] == 'TP'].sort_values('sl_pnl_60m', ascending=False)
if len(tps) == 0:
    print(f"    None above {TP_THRESHOLD}%")
else:
    for _, r in tps.iterrows():
        print(f"    ✓ {r['ticker']} {r['direction']} ({r['scanner']}) — SL PnL: {r['sl_pnl_60m']:+.2f}%")
        print(f"      conf={r['confidence']}/10  gap={r['gap_pct']:+.1f}%  "
              f"catalyst: {r['catalyst_type']} — {r['catalyst'][:70]}")

    print(f"\n    TP PATTERN ANALYSIS:")
    tp_cats = Counter(tps['catalyst_type'])
    tp_dirs = Counter(tps['direction'])
    tp_scanners = Counter(tps['scanner'])
    print(f"      Catalyst types: {dict(tp_cats)}")
    print(f"      Directions: {dict(tp_dirs)}")
    print(f"      Scanners: {dict(tp_scanners)}")
    print(f"      Avg gap: {tps['gap_pct'].mean():+.1f}%  Avg conf: {tps['confidence'].mean():.1f}")

# ── 5C: MARGINAL — selected but near zero ──
print(f"\n  ═══ MARGINAL (Selected, between {FP_THRESHOLD}% and {TP_THRESHOLD}%) ═══")
marginals = df_selected[df_selected['label'] == 'MARGINAL'].sort_values('sl_pnl_60m', ascending=False)
for _, r in marginals.iterrows():
    print(f"    ~ {r['ticker']} {r['direction']} ({r['scanner']}) — SL PnL: {r['sl_pnl_60m']:+.2f}%  "
          f"conf={r['confidence']}  {r['catalyst_type']}")

# ── 5D: FALSE NEGATIVES — not selected but would have been profitable ──
print(f"\n  ═══ FALSE NEGATIVES (Not Selected, but PnL@60m >= {TP_THRESHOLD}%) ═══")
print(f"  (These are missed opportunities — tickers the scanners saw but didn't select)")
fns = df_not_selected[df_not_selected['pnl_60m'] >= TP_THRESHOLD].sort_values('pnl_60m', ascending=False)
if len(fns) == 0:
    print(f"    None — no unselected signals above {TP_THRESHOLD}%")
else:
    for _, r in fns.iterrows():
        gate = f"  GATE: {r['gate_reason'][:50]}" if r['gate_reason'] else '  (no gate — low conf/rank)'
        s2 = '(Stage2)' if r['stage2_attempted'] else '(scan-only)'
        print(f"\n    ★ {r['ticker']} {r['direction']} ({r['scanner']}) {s2} — PnL: {r['pnl_60m']:+.2f}%")
        print(f"      conf={r['confidence']}/10  gap={r['gap_pct']:+.1f}%  EOD={r['ret_eod']:+.1f}%")
        print(f"      {r['catalyst_type']}: {r['catalyst'][:80]}")
        print(f"      Why missed:{gate}")

    print(f"\n    FN PATTERN ANALYSIS:")
    fn_cats = Counter(fns['catalyst_type'])
    fn_gates = Counter(r for r in fns['gate_reason'] if r)
    print(f"      Catalyst types: {dict(fn_cats)}")
    print(f"      Total FN value (avg PnL@60m): {fns['pnl_60m'].mean():+.2f}%")
    print(f"      Gate reasons blocking these:")
    for gate, count in Counter(
        [g.split(';')[0].strip() if g else 'NO_GATE (low rank/conf)' for g in fns['gate_reason']]
    ).most_common():
        print(f"        {gate}: {count}")

# ── 5E: SCAN-ONLY MOVERS — discovered in Stage 1 but never deep-analyzed ──
print(f"\n  ═══ SCAN-ONLY BIG MOVERS (Discovered but no Stage 2) ═══")
print(f"  (Tickers found in Stage 1 but dropped before deep analysis)")
neutral_movers = df_neutral[df_neutral['ret_60m'].abs() >= BIG_MOVE_PCT].sort_values(
    'ret_60m', key=abs, ascending=False)
if len(neutral_movers) == 0:
    neutral_movers = df_neutral[df_neutral['ret_60m'].abs() >= 3.0].sort_values(
        'ret_60m', key=abs, ascending=False).head(10)
    if len(neutral_movers) > 0:
        print(f"  (No scan-only tickers moved >{BIG_MOVE_PCT}%, showing top movers >3%)")

if len(neutral_movers) == 0:
    print(f"    No scan-only tickers with significant moves")
else:
    for _, r in neutral_movers.iterrows():
        ideal_dir = 'LONG' if r['ret_60m'] > 0 else 'SHORT'
        print(f"    {r['ticker']:<6} gap={r['gap_pct']:+.1f}%  60m={r['ret_60m']:+.1f}%  "
              f"EOD={r['ret_eod']:+.1f}%  ideal={ideal_dir}  "
              f"cat={r['catalyst_type']}  ({r['scanner']})")

# ── 5F: TOTALLY MISSED — big movers not even discovered ──
print(f"\n  ═══ TOTALLY MISSED — Big Movers Not Discovered ═══")
missed = [m for m in big_movers if not m['discovered']]
missed_top = sorted(missed, key=lambda x: abs(x['change_pct']), reverse=True)[:25]
if not missed_top:
    print(f"    No missed big movers (all were discovered)")
else:
    print(f"    Top {len(missed_top)} missed movers (|change| >= {BIG_MOVE_PCT}%):")
    print(f"    {'Ticker':<7} {'Chg%':>7} {'Price':>8} {'Volume':>12} {'Dir':<6}")
    print(f"    {'─'*50}")
    for m in missed_top:
        d = 'LONG' if m['change_pct'] > 0 else 'SHORT'
        print(f"    {m['ticker']:<7} {m['change_pct']:>+6.1f}% ${m['price']:>7.2f} "
              f"{m['volume']:>11,} {d}")

# ── 5G: GATE ANALYSIS — which gates are helping vs hurting ──
print(f"\n  ═══ GATE ANALYSIS ═══")
gated = df_signals[df_signals['gate_reason'] != ''].copy()
ungated = df_signals[df_signals['gate_reason'] == ''].copy()

if len(gated) > 0:
    gate_types = defaultdict(list)
    for _, r in gated.iterrows():
        gates = r['gate_reason'].split(';')
        primary = gates[0].strip().split('(')[0].strip().split(' ')[0] if gates else 'UNKNOWN'
        gate_types[primary].append(r['pnl_60m'])

    print(f"\n    {'Gate Type':<30} {'N':>4} {'Avg PnL':>8} {'WR':>6} {'Verdict'}")
    print(f"    {'─'*65}")
    for gate, pnls in sorted(gate_types.items(), key=lambda x: np.mean(x[1])):
        avg = np.mean(pnls)
        wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        verdict = '✓ GOOD GATE' if avg < 0 else '✗ GATE COSTS $'
        print(f"    {gate:<30} {len(pnls):>4} {avg:>+7.2f}% {wr:>5.0f}% {verdict}")

    print(f"\n    Overall: ungated avg={ungated['pnl_60m'].mean():+.2f}% (n={len(ungated)}), "
          f"gated avg={gated['pnl_60m'].mean():+.2f}% (n={len(gated)})")

# ── 5H: PATTERN ANALYSIS — what do losers have in common? ──
print(f"\n  ═══ PATTERN ANALYSIS — Common Traits of Losers ═══")
all_with_direction = df_signals.copy()
losers = all_with_direction[all_with_direction['pnl_60m'] < -2.0].copy()
winners = all_with_direction[all_with_direction['pnl_60m'] > 2.0].copy()

if len(losers) > 0 and len(winners) > 0:
    print(f"\n    Comparing {len(winners)} winners (>+2%) vs {len(losers)} losers (<-2%):")
    print(f"\n    {'Feature':<30} {'Winners':>12} {'Losers':>12} {'Delta':>10}")
    print(f"    {'─'*65}")

    comparisons = [
        ('Avg confidence', winners['confidence'].mean(), losers['confidence'].mean()),
        ('Avg |gap|', winners['gap_pct'].abs().mean(), losers['gap_pct'].abs().mean()),
        ('Avg gap (signed)', winners['gap_pct'].mean(), losers['gap_pct'].mean()),
        ('% exhausted', winners['move_is_exhausted'].mean()*100, losers['move_is_exhausted'].mean()*100),
        ('% gated', (winners['gate_reason'] != '').mean()*100, (losers['gate_reason'] != '').mean()*100),
        ('Avg entry price', winners['entry_price'].mean(), losers['entry_price'].mean()),
    ]
    for feat, w_val, l_val in comparisons:
        delta = w_val - l_val
        print(f"    {feat:<30} {w_val:>11.2f} {l_val:>11.2f} {delta:>+9.2f}")

    print(f"\n    Catalyst type breakdown:")
    print(f"    {'Type':<20} {'Winners':>10} {'Losers':>10} {'Net':>8}")
    print(f"    {'─'*50}")
    all_cats = set(list(winners['catalyst_type'].unique()) + list(losers['catalyst_type'].unique()))
    for cat in sorted(all_cats):
        n_w = len(winners[winners['catalyst_type'] == cat])
        n_l = len(losers[losers['catalyst_type'] == cat])
        if n_w + n_l >= 1:
            print(f"    {cat:<20} {n_w:>10} {n_l:>10} {n_w - n_l:>+7}")

    print(f"\n    Direction breakdown:")
    for d in ['LONG', 'SHORT']:
        n_w = len(winners[winners['direction'] == d])
        n_l = len(losers[losers['direction'] == d])
        print(f"      {d}: {n_w} winners, {n_l} losers")

    print(f"\n    Scanner breakdown:")
    for s in ['premarket', 'discovery']:
        n_w = len(winners[winners['scanner'] == s])
        n_l = len(losers[losers['scanner'] == s])
        print(f"      {s}: {n_w} winners, {n_l} losers")

# ── 5I: BIGGEST LOSERS DEEP DIVE ──
print(f"\n  ═══ BIGGEST LOSERS — DEEP DIVE ═══")
biggest_losers = all_with_direction.nsmallest(10, 'pnl_60m')
for _, r in biggest_losers.iterrows():
    sel = '✓SEL' if r['selected'] else '    '
    gate = f"  GATE: {r['gate_reason'][:40]}" if r['gate_reason'] else ''
    print(f"\n    {sel} {r['ticker']} {r['direction']} ({r['scanner']}) — PnL: {r['pnl_60m']:+.2f}%  "
          f"(SL: {r['sl_pnl_60m']:+.2f}%)")
    print(f"        conf={r['confidence']}  gap={r['gap_pct']:+.1f}%  "
          f"MFE={r.get('dir_mfe', 0):+.1f}%  MAE={r.get('dir_mae', 0):+.1f}%  EOD={r['ret_eod']:+.1f}%")
    print(f"        {r['catalyst_type']}: {r['catalyst'][:80]}")
    if r['reasoning']:
        print(f"        reason: {r['reasoning'][:100]}")
    print(f"        {gate}")


# ==============================================================================
# 6. CHARTS
# ==============================================================================
fig_path = f"/tmp/scanner_investigation_{EVAL_DATE}.png"

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle(f'Scanner Investigation — {EVAL_DATE}\n'
             f'FP/FN/Missed Analysis', fontsize=14, fontweight='bold')

# 1. Selected: TP vs FP vs Marginal
ax = axes[0, 0]
labels_count = df_selected['label'].value_counts()
colors_map = {'TP': '#27ae60', 'FP': '#c0392b', 'MARGINAL': '#f39c12'}
ax.bar(range(len(labels_count)), labels_count.values,
       color=[colors_map.get(l, '#999') for l in labels_count.index])
ax.set_xticks(range(len(labels_count)))
ax.set_xticklabels([f"{l}\n(n={v})" for l, v in labels_count.items()])
ax.set_ylabel('Count')
ax.set_title(f'Selected Signal Outcomes\n(TP>={TP_THRESHOLD}%, FP<={FP_THRESHOLD}%)')
ax.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(labels_count.values):
    ax.annotate(str(v), xy=(i, v), ha='center', va='bottom', fontsize=14, fontweight='bold')

# 2. PnL distribution: selected vs gated vs scan-only
ax = axes[0, 1]
if len(df_selected) > 0:
    ax.hist(df_selected['sl_pnl_60m'], bins=15, alpha=0.6, color='#2ecc71', label=f'Selected (n={len(df_selected)})')
gated_with_dir = df_signals[df_signals['gate_reason'] != '']
if len(gated_with_dir) > 0:
    ax.hist(gated_with_dir['pnl_60m'], bins=15, alpha=0.4, color='#e74c3c', label=f'Gated (n={len(gated_with_dir)})')
ax.axvline(x=0, color='black', linewidth=1, linestyle='--')
ax.set_xlabel('PnL @ 60m %')
ax.set_ylabel('Count')
ax.set_title('Selected vs Gated PnL Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Catalyst type: win rate heatmap
ax = axes[0, 2]
cat_data = df_signals.groupby('catalyst_type').agg(
    n=('pnl_60m', 'count'),
    avg=('pnl_60m', 'mean'),
    wr=('pnl_60m', lambda x: (x > 0).mean() * 100),
).reset_index()
cat_data = cat_data[cat_data['n'] >= 2].sort_values('avg')
colors_c = ['#27ae60' if x > 0 else '#c0392b' for x in cat_data['avg']]
bars = ax.barh(range(len(cat_data)), cat_data['avg'], color=colors_c)
ax.set_yticks(range(len(cat_data)))
ax.set_yticklabels([f"{r['catalyst_type']} (n={r['n']}, WR={r['wr']:.0f}%)"
                     for _, r in cat_data.iterrows()], fontsize=9)
ax.set_xlabel('Avg PnL @ 60m %')
ax.set_title('All Signals by Catalyst Type')
ax.axvline(x=0, color='black', linewidth=0.8)
ax.grid(True, alpha=0.3, axis='x')

# 4. Gap vs PnL scatter with classification
ax = axes[1, 0]
for label, color, marker, s in [
    ('selected_win', '#27ae60', '^', 80),
    ('selected_lose', '#c0392b', 'v', 80),
    ('gated', '#95a5a6', 'x', 40),
]:
    if label == 'selected_win':
        mask = (df_signals['selected']) & (df_signals['pnl_60m'] > 0)
    elif label == 'selected_lose':
        mask = (df_signals['selected']) & (df_signals['pnl_60m'] <= 0)
    else:
        mask = df_signals['gate_reason'] != ''
    subset = df_signals[mask]
    ax.scatter(subset['gap_pct'], subset['pnl_60m'], c=color, marker=marker,
              s=s, alpha=0.6, label=f'{label} (n={len(subset)})')
    if label != 'gated':
        for _, r in subset.iterrows():
            ax.annotate(r['ticker'], xy=(r['gap_pct'], r['pnl_60m']),
                       fontsize=5, alpha=0.6, xytext=(2, 2), textcoords='offset points')
ax.axhline(y=0, color='black', linewidth=0.8)
ax.axvline(x=0, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
ax.set_xlabel('Gap %')
ax.set_ylabel('PnL @ 60m %')
ax.set_title('Gap vs Outcome (selected labeled)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 5. FN value — what we missed
ax = axes[1, 1]
if len(fns) > 0:
    fn_sorted = fns.sort_values('pnl_60m', ascending=True)
    bars = ax.barh(range(len(fn_sorted)), fn_sorted['pnl_60m'], color='#e67e22', height=0.6)
    ax.set_yticks(range(len(fn_sorted)))
    ax.set_yticklabels([f"{r['ticker']} {r['direction'][0]} ({r['scanner'][:4]})"
                         for _, r in fn_sorted.iterrows()], fontsize=8)
    ax.set_xlabel('PnL @ 60m % (would have captured)')
    ax.set_title(f'False Negatives — Missed Profits\n(n={len(fns)}, avg={fns["pnl_60m"].mean():+.2f}%)')
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')
else:
    ax.text(0.5, 0.5, 'No False Negatives', transform=ax.transAxes,
            ha='center', va='center', fontsize=14)
    ax.set_title('False Negatives')

# 6. Biggest totally-missed movers
ax = axes[1, 2]
if missed_top:
    missed_show = missed_top[:15]
    missed_chgs = [m['change_pct'] for m in missed_show]
    missed_tickers = [m['ticker'] for m in missed_show]
    colors_m = ['#27ae60' if x > 0 else '#c0392b' for x in missed_chgs]
    ax.barh(range(len(missed_show)), missed_chgs, color=colors_m, height=0.6)
    ax.set_yticks(range(len(missed_show)))
    ax.set_yticklabels(missed_tickers, fontsize=8)
    ax.set_xlabel('Day Change %')
    ax.set_title(f'Totally Missed Movers (not discovered)\n'
                 f'(top {len(missed_show)} of {len(missed)})')
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')
else:
    ax.text(0.5, 0.5, 'All big movers were discovered', transform=ax.transAxes,
            ha='center', va='center', fontsize=14)
    ax.set_title('Totally Missed Movers')

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\n  Investigation chart saved: {fig_path}")
plt.close()


# ==============================================================================
# 7. SUMMARY
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  INVESTIGATION SUMMARY")
print(f"{'='*W}")

n_selected = len(df_selected)
n_tp = len(tps)
n_fp = len(fps)
n_marginal = len(marginals)
n_fn = len(fns)
n_missed = len(missed)

print(f"""
  Selected signals:     {n_selected}
    True Positives:     {n_tp} (>={TP_THRESHOLD}%)
    False Positives:    {n_fp} (<={FP_THRESHOLD}%)
    Marginal:           {n_marginal}

  False Negatives:      {n_fn} (discovered but not selected, >={TP_THRESHOLD}%)
  Totally missed:       {n_missed} (big movers not even discovered)

  Selected avg PnL:     {df_selected['sl_pnl_60m'].mean():+.2f}% (SL-adj)
  Gate effectiveness:   {(ungated['pnl_60m'].mean() - gated['pnl_60m'].mean()):+.2f}%/trade saved
""")

if n_fn > 0:
    print(f"  ⚠ KEY CONCERN: {n_fn} false negatives — profitable signals left on the table")
    print(f"    Avg missed profit: {fns['pnl_60m'].mean():+.2f}%")
    print(f"    Top missed: {', '.join(fns.nlargest(3, 'pnl_60m')['ticker'].tolist())}")

if n_fp > 0:
    print(f"  ⚠ FP CONCERN: {n_fp} selected signals lost >{abs(FP_THRESHOLD)}%")
    print(f"    Worst FP: {fps.iloc[0]['ticker']} {fps.iloc[0]['direction']} {fps.iloc[0]['sl_pnl_60m']:+.2f}%")

print(f"\n{'='*W}")
print(f"  DONE")
print(f"{'='*W}\n")
