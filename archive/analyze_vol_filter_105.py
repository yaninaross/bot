"""
analyze_vol_filter_105.py — Compare Exp-105 WF results: vol_filter_pct vs old MVB
==================================================================================
Queries BQ for the two most recent 105_DEFENSE runs and shows whether
the percentile-based volume filter added value.

Run: python3 analyze_vol_filter_105.py
"""

import pandas as pd
import numpy as np
from google.cloud import bigquery as bq
import math

BQ_TABLE = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"

client = bq.Client()


def _safe_bool(val, default=False):
    try:
        return bool(val)
    except (ValueError, TypeError):
        return default


def _safe_float(val, default=0.0):
    try:
        v = float(val)
        return default if pd.isna(v) else v
    except (ValueError, TypeError):
        return default


def _safe_int(val, default=0):
    try:
        v = float(val)
        return default if pd.isna(v) else int(v)
    except (ValueError, TypeError):
        return default


def composite_score(avg_pnl, win_rate, n, min_n=3):
    if n < min_n or avg_pnl <= 0:
        return 0.0
    return avg_pnl * win_rate * math.sqrt(n) * math.log2(1 + n)


# ──────────────────────────────────────────────────────────────────────
# 1. Get the recent 105_DEFENSE runs
# ──────────────────────────────────────────────────────────────────────
runs_q = f"""
SELECT run_id,
       MIN(run_timestamp) AS run_start,
       COUNT(*) AS row_count,
       COUNTIF(vol_filter_pct IS NOT NULL AND vol_filter_pct > 0) AS vf_nonzero
FROM `{BQ_TABLE}`
WHERE experiment = '105_DEFENSE'
GROUP BY run_id
ORDER BY run_start DESC
LIMIT 10
"""
runs_df = client.query(runs_q).to_dataframe()

if len(runs_df) == 0:
    print("No 105_DEFENSE runs found in BQ.")
    exit()

print("=" * 100)
print("  RECENT 105_DEFENSE RUNS IN BQ")
print("=" * 100)

new_run_id = runs_df.iloc[0]['run_id']
baseline_df = runs_df[runs_df['vf_nonzero'] == 0]
old_run_id = baseline_df.iloc[0]['run_id'] if len(baseline_df) > 0 else None

for _, r in runs_df.iterrows():
    if r['run_id'] == new_run_id:
        label = "  <-- NEW"
    elif r['run_id'] == old_run_id:
        label = "  <-- BASELINE (no vol_filter)"
    else:
        label = ""
    vf_tag = f"  VF>0: {int(r['vf_nonzero'])}" if r['vf_nonzero'] > 0 else "  (MVB only)"
    print(f"  {r['run_id'][:8]}...  {r['run_start']}  rows={r['row_count']:>4}{vf_tag}{label}")

# ──────────────────────────────────────────────────────────────────────
# 2. Load both runs
# ──────────────────────────────────────────────────────────────────────
def load_run(run_id):
    q = f"SELECT * FROM `{BQ_TABLE}` WHERE run_id = '{run_id}' ORDER BY ticker, seed"
    df = client.query(q).to_dataframe()
    if 'vol_filter_pct' not in df.columns:
        df['vol_filter_pct'] = 0.0
    df['vol_filter_pct'] = df['vol_filter_pct'].fillna(0).astype(int)
    return df

df_new = load_run(new_run_id)
df_old = load_run(old_run_id) if old_run_id else pd.DataFrame()

print(f"\n  NEW run: {new_run_id[:8]}...  {len(df_new)} rows, {df_new['ticker'].nunique()} tickers")
if not df_old.empty:
    print(f"  OLD run: {old_run_id[:8]}...  {len(df_old)} rows, {df_old['ticker'].nunique()} tickers")

# ──────────────────────────────────────────────────────────────────────
# 3. Volume filter distribution
# ──────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 100}")
print(f"  VOLUME FILTER DISTRIBUTION — NEW RUN")
print(f"{'=' * 100}")
for vpct in sorted(df_new['vol_filter_pct'].unique()):
    n = len(df_new[df_new['vol_filter_pct'] == vpct])
    print(f"  VF={int(vpct):>3}%:  {n:>4} seeds  ({n/len(df_new)*100:.1f}%)")
print(f"\n  Total seeds with VF > 0: {len(df_new[df_new['vol_filter_pct'] > 0])}/{len(df_new)}")

vf_nonzero = df_new[df_new['vol_filter_pct'] > 0]
if len(vf_nonzero) > 0:
    print(f"\n  {'Ticker':<7} {'Seed':>5} {'VF%':>4} {'Entry':>6} {'Prob':>6} "
          f"{'OPT PnL':>8} {'OPT WR':>7} {'OPT N':>6} {'CS':>8} "
          f"{'VAL-A':>8} {'VA WR':>6} {'VA N':>5} {'A':>5}")
    print(f"  {'─' * 98}")
    for _, r in vf_nonzero.sort_values(['ticker', 'seed']).iterrows():
        pa = "PASS" if _safe_bool(r.get('survived_a')) else "FAIL"
        cs = composite_score(_safe_float(r['opt_avg_pnl']),
                             _safe_float(r['opt_win_rate']),
                             _safe_int(r['opt_count']))
        print(f"  {r['ticker']:<7} {int(r['seed']):>5} {int(r['vol_filter_pct']):>3}% "
              f"{r.get('entry_time', ''):>6} {r['prob_thresh']:>6.3f} "
              f"{r['opt_avg_pnl']:>7.2%} {r['opt_win_rate']:>6.1%} {int(r['opt_count']):>6} "
              f"{cs:>8.4f} "
              f"{_safe_float(r.get('val_a_avg_pnl')):>7.2%} "
              f"{_safe_float(r.get('val_a_win_rate')):>5.1%} "
              f"{_safe_int(r.get('val_a_count')):>5} {pa:>5}")

# ──────────────────────────────────────────────────────────────────────
# 4. Side-by-side comparison (only tickers in both runs)
# ──────────────────────────────────────────────────────────────────────
if not df_old.empty:
    df_new['_cs'] = df_new.apply(lambda r: composite_score(
        _safe_float(r['opt_avg_pnl']), _safe_float(r['opt_win_rate']),
        _safe_int(r['opt_count'])), axis=1)
    df_old['_cs'] = df_old.apply(lambda r: composite_score(
        _safe_float(r['opt_avg_pnl']), _safe_float(r['opt_win_rate']),
        _safe_int(r['opt_count'])), axis=1)

    best_new = df_new.loc[df_new.groupby('ticker')['_cs'].idxmax()].copy()
    best_old = df_old.loc[df_old.groupby('ticker')['_cs'].idxmax()].copy()

    common_tickers = sorted(set(best_new['ticker']) & set(best_old['ticker']))
    new_only = sorted(set(best_new['ticker']) - set(best_old['ticker']))
    old_only = sorted(set(best_old['ticker']) - set(best_new['ticker']))

    if common_tickers:
        W = 160
        print(f"\n\n{'=' * W}")
        print(f"  SIDE-BY-SIDE: NEW (vol_filter_pct) vs OLD (min_vol_bars)")
        print(f"  Best seed selected by COMPOSITE SCORE = avg_pnl × win_rate × log2(1+N)")
        print(f"  {len(common_tickers)} tickers in both runs")
        print(f"{'=' * W}")
        print(f"  {'Ticker':<7} {'VF%':>4} "
              f"{'|':>1} {'OLD CS':>7} {'NEW CS':>7} "
              f"{'|':>1} {'OLD Avg':>8} {'NEW Avg':>8} {'dAvg':>7} "
              f"{'|':>1} {'OLD WR':>7} {'NEW WR':>7} "
              f"{'|':>1} {'OLD N':>5} {'NEW N':>5} "
              f"{'|':>1} {'OLD VA':>8} {'NEW VA':>8} {'dVA':>7} "
              f"{'|':>1} {'O WR':>5} {'N WR':>5} "
              f"{'|':>1} {'O N':>4} {'N N':>4} "
              f"{'|':>1} {'A':>5} {'B':>5}")
        print(f"  {'─' * (W - 2)}")

        sum_old_cs = sum_new_cs = 0.0
        sum_old_opt = sum_new_opt = 0.0
        sum_old_va = sum_new_va = 0.0
        sum_old_n = sum_new_n = 0
        sum_old_va_n = sum_new_va_n = 0
        old_pass_a = new_pass_a = 0
        old_pass_b = new_pass_b = 0
        improved = degraded = same = 0

        for t in common_tickers:
            nr = best_new[best_new['ticker'] == t].iloc[0]
            oldr = best_old[best_old['ticker'] == t].iloc[0]

            vf = _safe_int(nr.get('vol_filter_pct'))
            n_opt = _safe_float(nr['opt_avg_pnl'])
            o_opt = _safe_float(oldr['opt_avg_pnl'])
            n_wr = _safe_float(nr['opt_win_rate'])
            o_wr = _safe_float(oldr['opt_win_rate'])
            n_cnt = _safe_int(nr['opt_count'])
            o_cnt = _safe_int(oldr['opt_count'])
            n_cs = composite_score(n_opt, n_wr, n_cnt)
            o_cs = composite_score(o_opt, o_wr, o_cnt)
            n_va = _safe_float(nr.get('val_a_avg_pnl'))
            o_va = _safe_float(oldr.get('val_a_avg_pnl'))
            n_va_wr = _safe_float(nr.get('val_a_win_rate'))
            o_va_wr = _safe_float(oldr.get('val_a_win_rate'))
            n_va_n = _safe_int(nr.get('val_a_count'))
            o_va_n = _safe_int(oldr.get('val_a_count'))
            n_pa = _safe_bool(nr.get('survived_a'))
            o_pa = _safe_bool(oldr.get('survived_a'))
            n_pb = _safe_bool(nr.get('survived_b'))
            o_pb = _safe_bool(oldr.get('survived_b'))

            d_opt = n_opt - o_opt
            d_va = n_va - o_va

            sum_old_cs += o_cs; sum_new_cs += n_cs
            sum_old_opt += o_opt; sum_new_opt += n_opt
            sum_old_va += o_va; sum_new_va += n_va
            sum_old_n += o_cnt; sum_new_n += n_cnt
            sum_old_va_n += o_va_n; sum_new_va_n += n_va_n
            if o_pa: old_pass_a += 1
            if n_pa: new_pass_a += 1
            if o_pb: old_pass_b += 1
            if n_pb: new_pass_b += 1

            if d_va > 0.001: improved += 1
            elif d_va < -0.001: degraded += 1
            else: same += 1

            d_opt_s = f"{d_opt:>+6.2%}" if abs(d_opt) > 0.0001 else "     -"
            d_va_s  = f"{d_va:>+6.2%}" if abs(d_va) > 0.0001 else "     -"
            a_tag = "PASS" if n_pa else "FAIL"
            b_tag = "PASS" if n_pb else "FAIL"

            print(f"  {t:<7} {vf:>3}% "
                  f"| {o_cs:>7.4f} {n_cs:>7.4f} "
                  f"| {o_opt:>7.2%} {n_opt:>7.2%} {d_opt_s} "
                  f"| {o_wr:>6.1%} {n_wr:>6.1%} "
                  f"| {o_cnt:>5} {n_cnt:>5} "
                  f"| {o_va:>7.2%} {n_va:>7.2%} {d_va_s} "
                  f"| {o_va_wr:>5.1%} {n_va_wr:>5.1%} "
                  f"| {o_va_n:>4} {n_va_n:>4} "
                  f"| {a_tag:>5} {b_tag:>5}")

        nc = len(common_tickers)
        print(f"  {'─' * (W - 2)}")
        print(f"  AVERAGES ({nc} tickers):")
        print(f"    Composite Score: OLD={sum_old_cs/nc:.4f}  NEW={sum_new_cs/nc:.4f}  "
              f"delta={sum_new_cs/nc - sum_old_cs/nc:+.4f}")
        print(f"    OPT Avg PnL:  OLD={sum_old_opt/nc:.2%}  NEW={sum_new_opt/nc:.2%}  "
              f"delta={sum_new_opt/nc - sum_old_opt/nc:+.2%}")
        print(f"    VAL-A Avg PnL: OLD={sum_old_va/nc:.2%}  NEW={sum_new_va/nc:.2%}  "
              f"delta={sum_new_va/nc - sum_old_va/nc:+.2%}")
        print(f"    OPT Total N:  OLD={sum_old_n}  NEW={sum_new_n}  delta={sum_new_n - sum_old_n:+d}")
        print(f"    VAL-A Total N: OLD={sum_old_va_n}  NEW={sum_new_va_n}  delta={sum_new_va_n - sum_old_va_n:+d}")
        print(f"    PASS-A: OLD={old_pass_a}/{nc}  NEW={new_pass_a}/{nc}")
        print(f"    PASS-B: OLD={old_pass_b}/{nc}  NEW={new_pass_b}/{nc}")
        print(f"\n  VAL-A direction: {improved} improved, {degraded} degraded, {same} unchanged")

    # ──────────────────────────────────────────────────────────────────
    # 5. VF>0 deep dive — compare same ticker/seed with and without VF
    # ──────────────────────────────────────────────────────────────────
    vf_seeds = df_new[df_new['vol_filter_pct'] > 0][['ticker', 'seed', 'vol_filter_pct']].copy()
    if len(vf_seeds) > 0 and not df_old.empty:
        print(f"\n\n{'=' * 135}")
        print(f"  DEEP DIVE: Same ticker+seed, OLD (VF=0) vs NEW (VF>0)")
        print(f"{'=' * 135}")
        print(f"  {'Ticker':<7} {'Seed':>5} {'VF%':>4} "
              f"{'|':>1} {'OLD CS':>7} {'NEW CS':>7} "
              f"{'|':>1} {'OLD OPT':>8} {'NEW OPT':>8} "
              f"{'|':>1} {'OLD WR':>7} {'NEW WR':>7} "
              f"{'|':>1} {'OLD N':>5} {'NEW N':>5} "
              f"{'|':>1} {'OLD VA':>8} {'NEW VA':>8} {'dVA':>7} "
              f"{'|':>1} {'O VN':>4} {'N VN':>4} "
              f"{'Verdict':>9}")
        print(f"  {'─' * 133}")

        dd_better = dd_worse = dd_same = 0
        for _, vs in vf_seeds.iterrows():
            t, s, vf = vs['ticker'], vs['seed'], vs['vol_filter_pct']
            new_row = df_new[(df_new['ticker'] == t) & (df_new['seed'] == s)]
            old_row = df_old[(df_old['ticker'] == t) & (df_old['seed'] == s)]
            if new_row.empty or old_row.empty:
                continue
            nr = new_row.iloc[0]
            oldr = old_row.iloc[0]

            n_opt = _safe_float(nr['opt_avg_pnl'])
            o_opt = _safe_float(oldr['opt_avg_pnl'])
            n_va = _safe_float(nr.get('val_a_avg_pnl'))
            o_va = _safe_float(oldr.get('val_a_avg_pnl'))
            n_wr = _safe_float(nr['opt_win_rate'])
            o_wr = _safe_float(oldr['opt_win_rate'])
            n_cnt = _safe_int(nr['opt_count'])
            o_cnt = _safe_int(oldr['opt_count'])
            n_cs = composite_score(n_opt, n_wr, n_cnt)
            o_cs = composite_score(o_opt, o_wr, o_cnt)
            n_va_n = _safe_int(nr.get('val_a_count'))
            o_va_n = _safe_int(oldr.get('val_a_count'))

            d_va = n_va - o_va
            verdict = "BETTER" if d_va > 0.001 else "WORSE" if d_va < -0.001 else "SAME"
            if d_va > 0.001: dd_better += 1
            elif d_va < -0.001: dd_worse += 1
            else: dd_same += 1

            print(f"  {t:<7} {int(s):>5} {int(vf):>3}% "
                  f"| {o_cs:>7.4f} {n_cs:>7.4f} "
                  f"| {o_opt:>7.2%} {n_opt:>7.2%} "
                  f"| {o_wr:>6.1%} {n_wr:>6.1%} "
                  f"| {o_cnt:>5} {n_cnt:>5} "
                  f"| {o_va:>7.2%} {n_va:>7.2%} {d_va:>+6.2%} "
                  f"| {o_va_n:>4} {n_va_n:>4} "
                  f"{verdict:>9}")

        print(f"  {'─' * 133}")
        print(f"  Deep dive VAL-A: {dd_better} BETTER, {dd_worse} WORSE, {dd_same} SAME")

    # ──────────────────────────────────────────────────────────────────
    # 6. Filtered trades analysis — what did the filter remove?
    # ──────────────────────────────────────────────────────────────────
    if len(vf_seeds) > 0 and not df_old.empty:
        QUALITY_THRESH = 0.02
        W = 145
        print(f"\n\n{'=' * W}")
        print(f"  FILTERED TRADES ANALYSIS: What trades did the volume filter remove?")
        print(f"  Quality threshold = {QUALITY_THRESH:.0%} avg PnL per trade")
        print(f"  GOOD = removed sub-par trades (avg PnL < {QUALITY_THRESH:.0%})")
        print(f"  BAD  = removed quality trades (avg PnL >= {QUALITY_THRESH:.0%})")
        print(f"{'=' * W}")
        print(f"  {'Ticker':<7} {'Seed':>5} {'VF%':>4} "
              f"{'│':>1} {'── OPT FILTERED ──':>20} "
              f"{'│':>1} {'────── VAL-A FILTERED ──────':>30} "
              f"{'│':>1} {'Verdict':>8}")
        print(f"  {'':7} {'':>5} {'':>4} "
              f"{'│':>1} {'dN':>4} {'Avg PnL':>9} {'dWR':>7} "
              f"{'│':>1} {'dN':>4} {'Avg PnL':>9} {'dWR':>7} {'TotPnL':>8} "
              f"{'│':>1}")
        print(f"  {'─' * (W - 2)}")

        seen = set()
        opt_good = opt_bad = opt_neutral = 0
        va_good = va_bad = va_neutral = va_skip = 0
        total_va_filtered_pnl = 0.0
        total_va_filtered_n = 0

        for _, vs in vf_seeds.sort_values(['ticker', 'seed']).iterrows():
            t, s, vf = vs['ticker'], vs['seed'], vs['vol_filter_pct']
            key = (t, int(s))
            if key in seen:
                continue
            seen.add(key)

            new_row = df_new[(df_new['ticker'] == t) & (df_new['seed'] == s)]
            old_row = df_old[(df_old['ticker'] == t) & (df_old['seed'] == s)]
            if new_row.empty or old_row.empty:
                continue
            nr = new_row.iloc[0]
            oldr = old_row.iloc[0]

            # OPT window
            o_n = _safe_int(oldr['opt_count'])
            n_n = _safe_int(nr['opt_count'])
            o_avg = _safe_float(oldr['opt_avg_pnl'])
            n_avg = _safe_float(nr['opt_avg_pnl'])
            o_wr = _safe_float(oldr['opt_win_rate'])
            n_wr = _safe_float(nr['opt_win_rate'])

            opt_dn = o_n - n_n
            if opt_dn > 0:
                opt_filt_total = o_n * o_avg - n_n * n_avg
                opt_filt_avg = opt_filt_total / opt_dn
                opt_filt_wins = round(o_n * o_wr - n_n * n_wr)
                opt_filt_wr = opt_filt_wins / opt_dn if opt_dn > 0 else 0
                opt_d_wr = opt_filt_wr - o_wr
            else:
                opt_filt_avg = 0.0
                opt_d_wr = 0.0

            if opt_dn > 0 and opt_filt_avg < QUALITY_THRESH:
                opt_good += 1
            elif opt_dn > 0 and opt_filt_avg >= QUALITY_THRESH:
                opt_bad += 1
            else:
                opt_neutral += 1

            # VAL-A window
            o_va_n = _safe_int(oldr.get('val_a_count'))
            n_va_n = _safe_int(nr.get('val_a_count'))
            o_va_avg = _safe_float(oldr.get('val_a_avg_pnl'))
            n_va_avg = _safe_float(nr.get('val_a_avg_pnl'))
            o_va_wr = _safe_float(oldr.get('val_a_win_rate'))
            n_va_wr = _safe_float(nr.get('val_a_win_rate'))

            va_dn = o_va_n - n_va_n
            if va_dn > 0:
                va_filt_total = o_va_n * o_va_avg - n_va_n * n_va_avg
                va_filt_avg = va_filt_total / va_dn
                va_filt_wins = round(o_va_n * o_va_wr - n_va_n * n_va_wr)
                va_filt_wr = va_filt_wins / va_dn if va_dn > 0 else 0
                va_d_wr = va_filt_wr - o_va_wr
                total_va_filtered_pnl += va_filt_total
                total_va_filtered_n += va_dn
                verdict = "GOOD" if va_filt_avg < QUALITY_THRESH else "BAD"
                if va_filt_avg < QUALITY_THRESH: va_good += 1
                else: va_bad += 1
            elif va_dn == 0:
                va_filt_avg = 0.0
                va_filt_total = 0.0
                va_d_wr = 0.0
                verdict = "NO CHG"
                va_skip += 1
            else:
                va_filt_avg = 0.0
                va_filt_total = 0.0
                va_d_wr = 0.0
                verdict = "N/A"
                va_skip += 1

            opt_s = (f"{opt_dn:>+3d}  {opt_filt_avg:>+8.2%} {opt_d_wr:>+6.1%}"
                     if opt_dn > 0 else f"{'  0':>4}  {'  n/a':>8} {'  n/a':>6}")
            va_s = (f"{va_dn:>+3d}  {va_filt_avg:>+8.2%} {va_d_wr:>+6.1%} {va_filt_total:>+7.2%}"
                    if va_dn > 0 else f"{'  0':>4}  {'  n/a':>8} {'  n/a':>6} {'  n/a':>7}")

            print(f"  {t:<7} {int(s):>5} {int(vf):>3}% "
                  f"│ {opt_s} "
                  f"│ {va_s} "
                  f"│ {verdict:>8}")

        print(f"  {'─' * (W - 2)}")
        n_unique = len(seen)
        print(f"  OPT  filtered trades: {opt_good} GOOD (removed <{QUALITY_THRESH:.0%}), "
              f"{opt_bad} BAD (removed >={QUALITY_THRESH:.0%}), {opt_neutral} N/A")
        print(f"  VAL-A filtered trades: {va_good} GOOD (removed <{QUALITY_THRESH:.0%}), "
              f"{va_bad} BAD (removed >={QUALITY_THRESH:.0%}), {va_skip} no change")
        if total_va_filtered_n > 0:
            avg_filtered = total_va_filtered_pnl / total_va_filtered_n
            print(f"\n  VAL-A filtered trades aggregate:")
            print(f"    Total filtered trades: {total_va_filtered_n}")
            print(f"    Total PnL of filtered trades: {total_va_filtered_pnl:+.2%}")
            print(f"    Avg PnL of filtered trades: {avg_filtered:+.2%}  "
                  f"(quality threshold: {QUALITY_THRESH:.0%})")
            qual = ("FILTER IS WORKING — removed trades averaged below quality threshold"
                    if avg_filtered < QUALITY_THRESH
                    else "FILTER IS HURTING — removed trades averaged above quality threshold")
            print(f"    >> {qual}")

    # ──────────────────────────────────────────────────────────────────
    # 7. Tickers only in one run
    # ──────────────────────────────────────────────────────────────────
    if old_only:
        print(f"\n  Tickers only in OLD run ({len(old_only)}): {', '.join(old_only)}")
    if new_only:
        print(f"  Tickers only in NEW run ({len(new_only)}): {', '.join(new_only)}")

else:
    print("\n  No previous 105_DEFENSE run found for comparison.")

print(f"\n{'=' * 100}")
print("  Done.")
