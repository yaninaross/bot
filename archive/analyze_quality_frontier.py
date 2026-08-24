"""
analyze_quality_frontier.py — Quality vs Quantity Efficient Frontier
=====================================================================
Deep-dives into the tradeoff between per-trade quality (avg PnL) and
portfolio total return. Builds the efficient frontier, analyzes trade-day
overlap, measures per-ticker robustness, and proposes a portfolio-aware
composite scoring approach.

Run in Colab: paste as cell. Requires: pandas, numpy, google-cloud-bigquery.
"""

import pandas as pd
import numpy as np

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

from google.cloud import bigquery

# ── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_VERSION = 'robust_wf_v8'
TEST_START = '2026-04-01'
BQ_TABLE = "ys-time-series.scoring_logs.wf_optimization_results"
MIN_TRADES = 2
W = 120

# Crypto tickers — these trade on correlated days (BTC-driven)
CRYPTO_TICKERS = {'BTBT', 'CIFR', 'CLSK', 'COIN', 'CORZ', 'HUT', 'IREN', 'KEEL', 'RIOT', 'WULF', 'MARA'}
# ── END CONFIG ──────────────────────────────────────────────────────────────

client = bigquery.Client()

# ==============================================================================
# STEP 1: QUERY BQ DATA
# ==============================================================================
print(f"{'='*W}")
print(f"  QUALITY vs QUANTITY FRONTIER — {MODEL_VERSION}")
print(f"  TEST period: {TEST_START} onwards")
print(f"{'='*W}")

q = f"""
SELECT ticker, experiment, seed, tpsl_mode, prob_thresh,
       val_a_avg_pnl, val_a_win_rate, val_a_count,
       val_b_avg_pnl, val_b_win_rate, val_b_count,
       tp_pct, sl_pct, tp_mult, sl_mult,
       survived_a, survived_b, is_ensemble
FROM `{BQ_TABLE}`
WHERE model_version = '{MODEL_VERSION}' AND survived_a = TRUE AND val_b_start = '{TEST_START}'
"""
df = client.query(q).to_dataframe()
print(f"  Loaded {len(df)} configs, {df['ticker'].nunique()} tickers")

# ==============================================================================
# STEP 2: BEST CONFIG PER TICKER
# ==============================================================================
eligible = df[df['val_b_count'] >= MIN_TRADES].copy()
eligible['total_pnl'] = eligible['val_b_avg_pnl'] * eligible['val_b_count']

best_per_ticker = eligible.loc[eligible.groupby('ticker')['val_b_avg_pnl'].idxmax()].copy()
best_per_ticker = best_per_ticker.sort_values('val_b_avg_pnl', ascending=False).reset_index(drop=True)

print(f"\n{'='*W}")
print(f"  STEP 2: BEST CONFIG PER TICKER (ranked by val_b_avg_pnl)")
print(f"{'='*W}")
print(f"  {'#':>3} {'Ticker':>7} {'AvgPnL':>8} {'WR':>5} {'Trades':>7} {'TotalPnL':>9} {'Mode':>16} {'Exp':>20} {'SurvB':>6}")
print(f"  " + "-" * 90)

for i, (_, row) in enumerate(best_per_ticker.iterrows()):
    print(f"  {i+1:>3} {row['ticker']:>7} {row['val_b_avg_pnl']:>7.2%} {row['val_b_win_rate']:>4.0%} "
          f"{int(row['val_b_count']):>7} {row['total_pnl']:>8.2%} {row['tpsl_mode']:>16} "
          f"{row['experiment']:>20} {'Y' if row['survived_b'] else 'N':>6}")

# ==============================================================================
# STEP 3: EFFICIENT FRONTIER
# ==============================================================================
print(f"\n{'='*W}")
print(f"  STEP 3: EFFICIENT FRONTIER — Adding tickers in PnL-descending order")
print(f"{'='*W}")

cumul_trades = 0
cumul_total_pnl = 0.0
frontier = []

for i, (_, row) in enumerate(best_per_ticker.iterrows()):
    cumul_trades += int(row['val_b_count'])
    cumul_total_pnl += row['total_pnl']
    wtd_avg = cumul_total_pnl / cumul_trades if cumul_trades > 0 else 0
    n = i + 1
    frontier.append({
        'n': n, 'ticker': row['ticker'],
        'tkr_avg': row['val_b_avg_pnl'], 'tkr_cnt': int(row['val_b_count']),
        'tkr_tot': row['total_pnl'],
        'port_cnt': cumul_trades, 'port_avg': wtd_avg, 'port_tot': cumul_total_pnl,
    })

fd = pd.DataFrame(frontier)

print(f"  {'#':>3} {'Ticker':>7} | {'TkrAvg':>7} {'TkrN':>5} {'TkrTot':>8} | {'PortN':>6} {'PortAvg':>8} {'PortTot':>9}")
print(f"  " + "-" * 75)
for _, r in fd.iterrows():
    print(f"  {int(r['n']):>3} {r['ticker']:>7} | {r['tkr_avg']:>6.2%} {int(r['tkr_cnt']):>5} {r['tkr_tot']:>7.2%} | "
          f"{int(r['port_cnt']):>6} {r['port_avg']:>7.2%} {r['port_tot']:>8.2%}")

# Key inflection points
print(f"\n  --- KEY INFLECTION POINTS ---")
max_idx = fd['port_tot'].idxmax()
print(f"  Max total PnL: {int(fd.loc[max_idx, 'n'])} tickers = "
      f"{fd.loc[max_idx, 'port_tot']:.2%} total, {fd.loc[max_idx, 'port_avg']:.2%} avg, "
      f"{int(fd.loc[max_idx, 'port_cnt'])} trades")

for threshold in [0.03, 0.02, 0.015, 0.01]:
    for i, r in fd.iterrows():
        if r['port_avg'] < threshold:
            if i > 0:
                prev = fd.loc[i-1]
                print(f"  Avg < {threshold:.1%} at ticker #{int(r['n'])} ({r['ticker']}) -> "
                      f"sweet spot: {int(prev['n'])} tickers, {prev['port_avg']:.2%} avg, "
                      f"{prev['port_tot']:.2%} total, {int(prev['port_cnt'])} trades")
            break

# Marginal contribution
print(f"\n  --- MARGINAL CONTRIBUTION ---")
print(f"  {'#':>3} {'Ticker':>7} {'MargTot':>9} {'PortAvg':>8} {'Verdict':>10}")
print(f"  " + "-" * 45)
for i, r in fd.iterrows():
    mt = r['port_tot'] if i == 0 else r['port_tot'] - fd.loc[i-1, 'port_tot']
    verdict = "YES" if mt > 0 and r['port_avg'] >= 0.01 else ("MARGINAL" if mt > 0 else "NO")
    print(f"  {int(r['n']):>3} {r['ticker']:>7} {mt:>8.2%} {r['port_avg']:>7.2%} {verdict:>10}")


# ==============================================================================
# STEP 4: TRADE-DAY OVERLAP ANALYSIS
# ==============================================================================
print(f"\n{'='*W}")
print(f"  STEP 4: TRADE-DAY OVERLAP ANALYSIS")
print(f"{'='*W}")

top10 = best_per_ticker.head(10)
top10_tickers = top10['ticker'].tolist()

crypto_in_top = [t for t in top10_tickers if t in CRYPTO_TICKERS]
non_crypto_in_top = [t for t in top10_tickers if t not in CRYPTO_TICKERS]

print(f"  Top 10 tickers: {top10_tickers}")
print(f"  Crypto:     {crypto_in_top}")
print(f"  Non-crypto: {non_crypto_in_top}")

# val_b test period is ~40 trading days (Apr 1 - May 29)
total_trades_top10 = int(top10['val_b_count'].sum())
avg_trades_top10 = top10['val_b_count'].mean()

print(f"\n  Total trades across top 10: {total_trades_top10}")
print(f"  Avg trades per ticker: {avg_trades_top10:.1f}")
print(f"  Test period: ~40 trading days")
print(f"  If INDEPENDENT: {total_trades_top10} trades across many unique days")
print(f"  If CORRELATED:  ~{int(avg_trades_top10)} trades on ~{int(avg_trades_top10)} days (stacked risk)")

# Experiment diversity — same experiment = shares model weights = likely similar signal days
exp_diversity = top10.groupby('experiment')['ticker'].apply(list).to_dict()
print(f"\n  Experiment diversity (top 10):")
for exp, tks in sorted(exp_diversity.items()):
    print(f"    {exp}: {tks}")

# Crypto overlap risk
if len(crypto_in_top) >= 2:
    crypto_top = top10[top10['ticker'].isin(CRYPTO_TICKERS)]
    crypto_trades = int(crypto_top['val_b_count'].sum())
    crypto_avg = crypto_top['val_b_count'].mean()
    print(f"\n  CRYPTO OVERLAP RISK:")
    print(f"    {len(crypto_in_top)} crypto tickers, {crypto_trades} total trades")
    print(f"    BTC-driven tickers trade on SAME days -> overlap likely 50-80%")
    print(f"    Effective unique trade days: ~{int(crypto_avg * 1.3)}-{int(crypto_avg * 1.8)} (not {crypto_trades})")
    print(f"    -> Stacked drawdown risk on bad BTC days")

print(f"\n  DIVERSIFICATION BENEFIT:")
print(f"    Non-crypto: {len(non_crypto_in_top)} tickers trade on INDEPENDENT catalysts")
print(f"    True diversification comes from non-crypto tickers (ASTS, FCEL, RCAT, etc.)")
print(f"    Crypto group should be treated as ~1-2 effective positions, not {len(crypto_in_top)}")


# ==============================================================================
# STEP 5: PER-TICKER ROBUSTNESS
# ==============================================================================
print(f"\n{'='*W}")
print(f"  STEP 5: PER-TICKER ROBUSTNESS (configs with val_b_avg_pnl > thresholds)")
print(f"{'='*W}")

robustness = eligible.groupby('ticker').agg(
    total_configs=('val_b_avg_pnl', 'count'),
    configs_gt_1pct=('val_b_avg_pnl', lambda x: (x > 0.01).sum()),
    configs_gt_2pct=('val_b_avg_pnl', lambda x: (x > 0.02).sum()),
    configs_gt_3pct=('val_b_avg_pnl', lambda x: (x > 0.03).sum()),
    median_pnl=('val_b_avg_pnl', 'median'),
    best_pnl=('val_b_avg_pnl', 'max'),
    std_pnl=('val_b_avg_pnl', 'std'),
).reset_index()

robustness['pct_gt_1'] = robustness['configs_gt_1pct'] / robustness['total_configs']

# Assign robustness tier
def robustness_tier(row):
    if row['pct_gt_1'] > 0.30 and row['configs_gt_1pct'] >= 10:
        return 'ROBUST'
    elif row['configs_gt_1pct'] >= 5:
        return 'OK'
    else:
        return 'FRAGILE'

robustness['tier'] = robustness.apply(robustness_tier, axis=1)
robustness = robustness.sort_values('configs_gt_1pct', ascending=False)

print(f"  {'Ticker':>7} {'Total':>6} {'>1%':>5} {'>2%':>5} {'>3%':>5} {'%>1%':>6} {'Median':>8} {'Best':>8} {'Std':>7} {'Tier':>8}")
print(f"  " + "-" * 80)
for _, r in robustness.iterrows():
    in_top10 = " <--" if r['ticker'] in top10_tickers else ""
    print(f"  {r['ticker']:>7} {int(r['total_configs']):>6} {int(r['configs_gt_1pct']):>5} "
          f"{int(r['configs_gt_2pct']):>5} {int(r['configs_gt_3pct']):>5} {r['pct_gt_1']:>5.0%} "
          f"{r['median_pnl']:>7.2%} {r['best_pnl']:>7.2%} {r['std_pnl']:>6.2%} {r['tier']:>8}{in_top10}")

# Cross-reference: fragile tickers in top 10
print(f"\n  --- FRAGILE TICKERS IN TOP 10 (high risk of overfitting) ---")
for _, r in robustness.iterrows():
    if r['ticker'] in top10_tickers:
        print(f"    {r['ticker']:>5}: {int(r['configs_gt_1pct'])}/{int(r['total_configs'])} configs > 1% "
              f"({r['pct_gt_1']:.0%}) -> {r['tier']}")


# ==============================================================================
# STEP 6: PORTFOLIO-AWARE COMPOSITE SCORING
# ==============================================================================
print(f"\n{'='*W}")
print(f"  STEP 6: PORTFOLIO-AWARE COMPOSITE SCORING")
print(f"{'='*W}")

# Build enhanced per-ticker summary
summary = []
for _, brow in best_per_ticker.iterrows():
    tk = brow['ticker']
    rrow = robustness[robustness['ticker'] == tk]
    if rrow.empty:
        rob_factor = 0.0
        rob_tier = 'FRAGILE'
    else:
        rrow = rrow.iloc[0]
        rob_factor = rrow['pct_gt_1']
        rob_tier = rrow['tier']

    is_crypto = tk in CRYPTO_TICKERS

    # CURRENT score: avg_pnl * win_rate * log2(1+N)
    current_score = brow['val_b_avg_pnl'] * brow['val_b_win_rate'] * np.log2(1 + brow['val_b_count'])

    # NEW portfolio-aware score
    # 1. sqrt(win_rate) instead of win_rate — less penalty for 60% WR high-PnL tickers
    # 2. Robustness multiplier — rewards tickers where MANY configs work
    # 3. Crypto diversification penalty
    robustness_mult = 0.5 + 0.5 * rob_factor  # ranges 0.5 to 1.0
    div_mult = 0.7 if is_crypto else 1.0
    quality_pass = brow['val_b_avg_pnl'] >= 0.015

    port_score = (brow['val_b_avg_pnl']
                  * np.sqrt(brow['val_b_win_rate'])
                  * np.log2(1 + brow['val_b_count'])
                  * robustness_mult
                  * div_mult)

    summary.append({
        'ticker': tk,
        'avg_pnl': brow['val_b_avg_pnl'],
        'win_rate': brow['val_b_win_rate'],
        'trades': int(brow['val_b_count']),
        'total_pnl': brow['total_pnl'],
        'current_score': current_score,
        'robustness_pct': rob_factor,
        'rob_tier': rob_tier,
        'is_crypto': is_crypto,
        'port_score': port_score,
        'quality_pass': quality_pass,
    })

sdf = pd.DataFrame(summary)

# Show both rankings
print(f"\n  --- CURRENT vs PORTFOLIO-AWARE RANKING ---")
print(f"  {'CurRk':>5} {'PortRk':>6} {'Ticker':>7} {'AvgPnL':>8} {'WR':>5} {'#Tr':>4} {'TotPnL':>8} "
      f"{'CurrScore':>10} {'Rob%':>5} {'Tier':>8} {'Crypto':>7} {'PortScore':>10} {'QPass':>6}")
print(f"  " + "-" * 105)

sdf_curr = sdf.sort_values('current_score', ascending=False).reset_index(drop=True)
sdf_port = sdf.sort_values('port_score', ascending=False).reset_index(drop=True)

# Build rank maps
curr_rank = {row['ticker']: i+1 for i, (_, row) in enumerate(sdf_curr.iterrows())}
port_rank = {row['ticker']: i+1 for i, (_, row) in enumerate(sdf_port.iterrows())}

for _, r in sdf_port.iterrows():
    delta = curr_rank[r['ticker']] - port_rank[r['ticker']]
    delta_str = f"({delta:+d})" if delta != 0 else ""
    print(f"  {curr_rank[r['ticker']]:>5} {port_rank[r['ticker']]:>6} {r['ticker']:>7} "
          f"{r['avg_pnl']:>7.2%} {r['win_rate']:>4.0%} {r['trades']:>4} {r['total_pnl']:>7.2%} "
          f"{r['current_score']:>10.5f} {r['robustness_pct']:>4.0%} {r['rob_tier']:>8} "
          f"{'Y' if r['is_crypto'] else 'N':>7} {r['port_score']:>10.5f} "
          f"{'PASS' if r['quality_pass'] else 'FAIL':>6} {delta_str}")


# ==============================================================================
# GREEDY PORTFOLIO CONSTRUCTION (with crypto cap)
# ==============================================================================
print(f"\n  --- GREEDY PORTFOLIO CONSTRUCTION ---")
print(f"  Rules: quality_pass (avg_pnl >= 1.5%), max 3 crypto, stop when port avg < 1.0%")

candidates = sdf[sdf['quality_pass']].sort_values('port_score', ascending=False)
selected = []
crypto_count = 0
max_crypto = 3
port_trades = 0
port_total = 0.0

print(f"\n  {'Action':>6} {'Ticker':>7} | {'N_tk':>4} {'Trades':>7} {'PortAvg':>8} {'PortTot':>8}")
print(f"  " + "-" * 55)

for _, r in candidates.iterrows():
    if r['is_crypto'] and crypto_count >= max_crypto:
        print(f"  {'SKIP':>6} {r['ticker']:>7} | crypto cap reached (already {crypto_count} crypto)")
        continue

    # Check if adding would drop avg below 1.0%
    new_trades = port_trades + r['trades']
    new_total = port_total + r['total_pnl']
    new_avg = new_total / new_trades if new_trades > 0 else 0

    if new_avg < 0.01 and len(selected) > 0:
        print(f"  {'STOP':>6} {r['ticker']:>7} | would drop avg to {new_avg:.2%} (below 1.0% floor)")
        continue

    selected.append(r['ticker'])
    if r['is_crypto']:
        crypto_count += 1
    port_trades = new_trades
    port_total = new_total

    print(f"  {'ADD':>6} {r['ticker']:>7} | {len(selected):>4} {port_trades:>7} {new_avg:>7.2%} {port_total:>7.2%}")

port_avg = port_total / port_trades if port_trades > 0 else 0
print(f"\n  PORTFOLIO: {len(selected)} tickers = {selected}")
print(f"  Total trades: {port_trades}, Weighted avg PnL: {port_avg:.2%}, Total PnL sum: {port_total:.2%}")


# ==============================================================================
# COMPARISON TABLE: Different portfolio sizes
# ==============================================================================
print(f"\n  --- COMPARISON: NAIVE TOP-N vs PORTFOLIO-AWARE ---")
print(f"  {'Method':<20} {'Tickers':>8} {'Trades':>7} {'AvgPnL':>8} {'TotalPnL':>9} {'Crypto':>7}")
print(f"  " + "-" * 65)

# Naive top-N (PnL ranked)
for n in [5, 8, 10, 13, 17, 21]:
    if n > len(best_per_ticker):
        continue
    top_n = best_per_ticker.head(n)
    tot_t = int(top_n['val_b_count'].sum())
    tot_p = top_n['total_pnl'].sum()
    avg_p = tot_p / tot_t if tot_t > 0 else 0
    n_crypto = sum(1 for t in top_n['ticker'] if t in CRYPTO_TICKERS)
    print(f"  {'Naive top-'+str(n):<20} {n:>8} {tot_t:>7} {avg_p:>7.2%} {tot_p:>8.2%} {n_crypto:>7}")

# Portfolio-aware
sel_df = sdf[sdf['ticker'].isin(selected)]
n_crypto_sel = sum(1 for t in selected if t in CRYPTO_TICKERS)
print(f"  {'Port-aware':>20} {len(selected):>8} {port_trades:>7} {port_avg:>7.2%} {port_total:>8.2%} {n_crypto_sel:>7}")


# ==============================================================================
# FORMULA PROPOSAL
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  PROPOSED PORTFOLIO-AWARE COMPOSITE SCORING FORMULA")
print(f"{'='*W}")
print("""
  CURRENT FORMULA (per-ticker independent):
    score = avg_pnl * win_rate * log2(1 + N)
    threshold: avg_pnl > 1.8%, min N = 3

  PROPOSED FORMULA:

    Step 1 - Quality Gate:
      PASS if avg_pnl >= 1.5% AND win_rate >= 50% AND N >= 2

    Step 2 - Ticker Quality Score:
      quality = avg_pnl * sqrt(win_rate) * log2(1 + N)

      WHY sqrt(win_rate): A ticker with 3.5% avg and 60% WR is MORE valuable
      than one with 2.0% avg and 80% WR. The current formula over-penalizes
      moderate WR on high-PnL tickers.

    Step 3 - Robustness Multiplier:
      R = (configs_with_pnl_above_1pct / total_configs)   # 0 to 1
      robustness_mult = 0.5 + 0.5 * R                     # ranges 0.5 to 1.0

      WHY: Cherry-picking the single best config is fragile. If 50 out of 80
      configs produce >1% PnL, the signal is REAL. If only 2 out of 80 do,
      you're probably overfitting.

    Step 4 - Diversification Multiplier:
      For each crypto ticker already in portfolio:
        div_mult = 0.85^(n_crypto_already_selected)
      For non-crypto:
        div_mult = 1.0

      WHY: CIFR, CLSK, HUT, IREN, RIOT, etc. all trade on the SAME BTC-driven
      days. Adding the 4th crypto ticker gives ~20% marginal diversification,
      not another independent bet.

    Step 5 - Greedy Selection:
      Sort by port_score = quality * robustness_mult * div_mult
      Add ticker if:
        a) quality_gate PASSED
        b) marginal total PnL > 0 (positive expected contribution)
        c) portfolio weighted avg PnL stays above 1.0% floor
      Stop when next ticker would violate (c)

  KEY DIFFERENCES:
    1. sqrt(win_rate) rewards high-conviction, high-PnL tickers
    2. Robustness multiplier filters out overfitting
    3. Crypto cap prevents correlated-day stacking
    4. Portfolio avg floor prevents dilution death spiral
    5. 1.5% gate (vs 1.8%) lets in borderline tickers IF they help the portfolio
""")

print(f"{'='*W}")
print(f"  ANALYSIS COMPLETE")
print(f"{'='*W}")
