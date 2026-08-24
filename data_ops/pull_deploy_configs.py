"""
pull_deploy_configs.py — Generate asset_configs for daily_score_combined.py from BQ
===================================================================================
Queries wf_optimization_results, picks best config per ticker by composite score,
groups by experiment family, and prints ready-to-paste Python config blocks.

Run this after any WF optimization completes, then paste output into
daily_score_combined.py to update the asset_configs.
"""

import math
from google.cloud import bigquery
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)

MODEL_VERSION = 'robust_wf_v11'
TEST_START = '2026-04-01'
# S8_mix asymmetric filter: one period >= 1.5% AND other >= 1.0%
ASYM_HIGH = 0.015
ASYM_LOW  = 0.010

FAMILY_MAP = {
    '152': {'model_id': '152_XSECTOR', 'label': 'EXP_152', 'seed_offset': 0},
    '160': {'model_id': '160_XSECTOR', 'label': 'EXP_160', 'seed_offset': 0},
    '170': {'model_id': '175_XSECTOR', 'label': 'EXP_170', 'seed_offset': 17501},
    '185': {'model_id': '185_XSECTOR', 'label': 'EXP_185', 'seed_offset': 0},
}

bq = bigquery.Client()

q = f'''
WITH base AS (
  SELECT *,
    CASE WHEN (val_a_count + val_b_count) > 0
      THEN (val_a_avg_pnl * val_a_count + val_b_avg_pnl * val_b_count) / (val_a_count + val_b_count)
      ELSE 0 END AS wavg_pnl,
    CASE WHEN (val_a_count + val_b_count) > 0
      THEN (val_a_win_rate * val_a_count + val_b_win_rate * val_b_count) / (val_a_count + val_b_count)
      ELSE 0 END AS combined_wr,
    CASE WHEN val_a_avg_pnl > 0
      THEN GREATEST(1.0 - ABS(val_b_avg_pnl - val_a_avg_pnl) / val_a_avg_pnl, 0.2)
      ELSE 0.2 END AS consistency,
    EXP(-0.5 * POW(((val_a_count / GREATEST(TIMESTAMP_DIFF(
        PARSE_TIMESTAMP('%Y-%m-%d', val_a_end), PARSE_TIMESTAMP('%Y-%m-%d', val_a_start), DAY) / 30.44, 0.5)
      + val_b_count / GREATEST(TIMESTAMP_DIFF(
        PARSE_TIMESTAMP('%Y-%m-%d', val_b_end), PARSE_TIMESTAMP('%Y-%m-%d', val_b_start), DAY) / 30.44, 0.5)
      ) / 2.0 - 3.0) / 2.5, 2)) AS nq3
  FROM `ys-time-series.scoring_logs.wf_optimization_results`
  WHERE model_version = '{MODEL_VERSION}'
    AND survived_a = TRUE
    AND ((val_a_avg_pnl >= {ASYM_HIGH} AND val_b_avg_pnl >= {ASYM_LOW})
      OR (val_a_avg_pnl >= {ASYM_LOW} AND val_b_avg_pnl >= {ASYM_HIGH}))
    AND val_a_win_rate >= 0.50
    AND val_b_win_rate >= 0.50
    AND val_b_start = '{TEST_START}'
),
scored AS (
  SELECT *,
    POW(wavg_pnl, 2) * POW(combined_wr, 2) * nq3 * consistency * LOG(1 + val_a_count + val_b_count, 2) AS composite
  FROM base
),
ranked AS (
  SELECT *,
    ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY composite DESC) AS rn
  FROM scored
)
SELECT * FROM ranked WHERE rn = 1
ORDER BY composite DESC
'''

df = bq.query(q).to_dataframe()

print(f"{'='*80}")
print(f"pull_deploy_configs.py — Best WF configs from BQ")
print(f"  model_version = {MODEL_VERSION}")
print(f"  filter = S8_mix (asym {ASYM_HIGH*100:.1f}%/{ASYM_LOW*100:.1f}%)")
print(f"  {len(df)} tickers qualified")
print(f"{'='*80}")

print(f"\n{'TICKER':<8} {'EXPERIMENT':<25} {'SEED':>5} {'MODE':<16} {'PROB':>5} {'ENTRY':>6} "
      f"{'PNL':>8} {'WR':>7} {'N':>4} {'COMP':>8}")
print('-' * 110)
for _, r in df.iterrows():
    print(f"{r['ticker']:<8} {r['experiment']:<25} {r['seed']:>5} {r['tpsl_mode']:<16} "
          f"{r['prob_thresh']:>5.2f} {r['entry_minute']:>6.0f} "
          f"{r['val_b_avg_pnl']*100:>7.2f}% {r['val_b_win_rate']*100:>6.1f}% "
          f"{r['val_b_count']:>3} {r['composite']:>8.4f}")

def get_family(experiment):
    base = experiment.split('_')[0]
    if base in FAMILY_MAP:
        return base
    return None

def format_config(row):
    mode = row['tpsl_mode']
    family = get_family(row['experiment'])
    info = FAMILY_MAP.get(family, {})
    seed_offset = info.get('seed_offset', 0)

    seed = int(row['seed'])
    is_ensemble = seed == -1
    if not is_ensemble and seed_offset > 0:
        seed = seed_offset + seed

    vix = row['vix_threshold'] if pd.notna(row['vix_threshold']) else 999.0
    tlt = row['tlt_crash'] if pd.notna(row['tlt_crash']) else -999.0
    spy = row['spy_ignore'] if pd.notna(row['spy_ignore']) else 999.0
    vol = int(row['vol_filter_pct']) if pd.notna(row['vol_filter_pct']) else 0

    val_n = int(row['val_b_count'])
    val_wr = row['val_b_win_rate']
    val_pnl = row['val_b_avg_pnl']

    if val_n >= 10:
        conf = 'HIGH'
        note = f'N={val_n} trades, strong sample'
    elif val_n >= 5:
        conf = 'MED'
        note = f'N={val_n} trades, moderate sample'
    else:
        conf = 'LOW'
        note = f'N={val_n} trades, small sample'

    lines = []
    lines.append(f"        '{row['ticker']}': {{")

    if is_ensemble:
        lines.append(f"            'seed': -1, 'is_ensemble': True, "
                     f"'prob_threshold': {row['prob_thresh']:.3f}, "
                     f"'entry_minute': {int(row['entry_minute'])},")
    else:
        lines.append(f"            'seed': {seed}, "
                     f"'prob_threshold': {row['prob_thresh']:.3f}, "
                     f"'entry_minute': {int(row['entry_minute'])},")

    lines.append(f"            'tpsl_mode': '{mode}',")

    if mode in ('static', 'regime_scaled'):
        lines.append(f"            'tp_pct': {row['tp_pct']:.2f}, "
                     f"'sl_pct': {row['sl_pct']:.2f},")
    elif mode in ('atr', 'atr_adaptive'):
        tp_m = row['tp_mult'] if pd.notna(row['tp_mult']) else 2.0
        sl_m = row['sl_mult'] if pd.notna(row['sl_mult']) else 1.0
        lines.append(f"            'tp_pct': 0.0, 'sl_pct': 0.0,")
        lines.append(f"            'tp_mult': {tp_m}, 'sl_mult': {sl_m},")
    elif mode in ('trailing_decay', 'trail'):
        sl_m = row['sl_mult'] if pd.notna(row['sl_mult']) else 2.0
        ts_m = row['trail_start_mult'] if pd.notna(row['trail_start_mult']) else 2.0
        te_m = row['trail_end_mult'] if pd.notna(row['trail_end_mult']) else 1.0
        lines.append(f"            'tp_pct': 0.0, 'sl_pct': 0.0,")
        lines.append(f"            'sl_mult': {sl_m}, "
                     f"'trail_start_mult': {ts_m}, 'trail_end_mult': {te_m},")

    lines.append(f"            'vix_threshold': {vix}, "
                 f"'tlt_crash': {tlt}, 'spy_ignore': {spy},")
    lines.append(f"            'vol_filter_pct': {vol},")
    lines.append(f"            'confidence': '{conf}', "
                 f"'confidence_note': '{note}',")
    lines.append(f"            'stats': {{'win_rate': '{val_wr*100:.0f}%', "
                 f"'avg_pnl': '{val_pnl*100:.1f}%', 'composite': '{row['composite']:.4f}'}},")
    lines.append(f"        }},")

    return '\n'.join(lines)


grouped = {}
for _, row in df.iterrows():
    family = get_family(row['experiment'])
    if family is None:
        continue
    if family not in grouped:
        grouped[family] = []
    grouped[family].append(row)

for family_key in ['152', '160', '170', '185']:
    if family_key not in grouped:
        print(f"\n# --- {FAMILY_MAP[family_key]['label']}: no qualifying tickers ---")
        continue

    info = FAMILY_MAP[family_key]
    rows = grouped[family_key]

    print(f"\n\n{'#'*80}")
    print(f"# {info['label']} — {info['model_id']}  ({len(rows)} tickers)")
    print(f"# Paste into the 'asset_configs' dict of {info['label']}")
    print(f"{'#'*80}")
    print(f"    'asset_configs': {{")
    for row in sorted(rows, key=lambda r: r['ticker']):
        print(format_config(row))
    print(f"    }},")

print(f"\n\n{'='*80}")
print(f"TOTAL: {len(df)} tickers across {len(grouped)} experiment families")
print(f"{'='*80}")
