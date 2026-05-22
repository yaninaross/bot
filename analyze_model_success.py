"""
analyze_model_success.py — Model Success Rate Analyzer
========================================================
Queries BQ wf_optimization_results and ranks experiments by
what % of their tickers pass deployment threshold (PnL >= 1.8%).

Shows:
  TABLE 1: Per-experiment success rate (deploy-ready / total tickers)
  TABLE 2: Per-experiment breakdown (which tickers pass/fail)
  TABLE 3: Overall ranking

Run in Colab:
  from analyze_model_success import main
  main()
"""

import pandas as pd
import numpy as np

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

from google.cloud import bigquery

BQ_TABLE = "ys-time-series.scoring_logs.wf_optimization_results"
DEPLOY_PNL_THRESHOLD = 0.018
W = 110


def main():
    client = bigquery.Client()
    df = client.query(f"SELECT * FROM `{BQ_TABLE}` ORDER BY experiment, ticker, seed").to_dataframe()

    if df.empty:
        print("No data in BQ table.")
        return

    df['survived_a'] = df['survived_a'].fillna(False).astype(bool)

    experiments = sorted(df['experiment'].unique())

    # ==================================================================
    # For each experiment+ticker: find best seed, check if deploy-ready
    # ==================================================================
    results = []

    for exp in experiments:
        ed = df[df['experiment'] == exp]
        tickers = sorted(ed['ticker'].unique())

        for ticker in tickers:
            td = ed[ed['ticker'] == ticker]
            n_seeds = len(td)
            survived = td[td['survived_a']]
            pass_a = len(survived)
            pass_rate = pass_a / n_seeds if n_seeds > 0 else 0

            if len(survived) > 0:
                best = survived.loc[survived['val_a_avg_pnl'].idxmax()]
                best_pnl = best['val_a_avg_pnl']
                best_seed = int(best['seed'])
                best_wr = best['val_a_win_rate']
                best_n = int(best['val_a_count'])
                deploy_ready = best_pnl >= DEPLOY_PNL_THRESHOLD
            else:
                best_pnl = td['val_a_avg_pnl'].max()
                best_seed = 0
                best_wr = 0
                best_n = 0
                deploy_ready = False

            results.append({
                'experiment': exp,
                'ticker': ticker,
                'n_seeds': n_seeds,
                'pass_a': pass_a,
                'pass_rate': pass_rate,
                'best_pnl': best_pnl,
                'best_seed': best_seed,
                'best_wr': best_wr,
                'best_n': best_n,
                'deploy_ready': deploy_ready,
            })

    rdf = pd.DataFrame(results)

    # ==================================================================
    # TABLE 1: EXPERIMENT SUCCESS RATES
    # ==================================================================
    print(f"{'='*W}")
    print(f"  MODEL SUCCESS RATE ANALYZER")
    print(f"  Deploy threshold: PnL >= {DEPLOY_PNL_THRESHOLD:.1%}")
    print(f"{'='*W}")

    print(f"\n{'='*W}")
    print(f"  TABLE 1: EXPERIMENT SUCCESS RATES (ranked by deploy %)")
    print(f"{'='*W}")
    print(f"  {'Experiment':<20} {'Tickers':>8} {'Deploy':>7} {'Fail':>5} "
          f"{'Deploy%':>8} {'Med PnL':>8} {'Avg PassRate':>12}")
    print(f"  {'─'*20} {'─'*8} {'─'*7} {'─'*5} {'─'*8} {'─'*8} {'─'*12}")

    exp_stats = []
    for exp in experiments:
        ed = rdf[rdf['experiment'] == exp]
        n_tickers = len(ed)
        n_deploy = ed['deploy_ready'].sum()
        n_fail = n_tickers - n_deploy
        deploy_pct = n_deploy / n_tickers if n_tickers > 0 else 0
        med_pnl = ed[ed['deploy_ready']]['best_pnl'].median() if n_deploy > 0 else 0
        avg_pass_rate = ed['pass_rate'].mean()

        exp_stats.append({
            'experiment': exp,
            'n_tickers': n_tickers,
            'n_deploy': int(n_deploy),
            'n_fail': int(n_fail),
            'deploy_pct': deploy_pct,
            'med_pnl': med_pnl,
            'avg_pass_rate': avg_pass_rate,
        })

    exp_stats.sort(key=lambda x: (-x['deploy_pct'], -x['med_pnl']))

    for s in exp_stats:
        bar = '#' * int(s['deploy_pct'] * 20)
        print(f"  {s['experiment']:<20} {s['n_tickers']:>8} {s['n_deploy']:>7} "
              f"{s['n_fail']:>5} {s['deploy_pct']:>7.0%} {s['med_pnl']:>7.2%} "
              f"{s['avg_pass_rate']:>11.0%}  {bar}")

    # ==================================================================
    # TABLE 2: PER-EXPERIMENT TICKER BREAKDOWN
    # ==================================================================
    print(f"\n\n{'='*W}")
    print(f"  TABLE 2: PER-EXPERIMENT TICKER BREAKDOWN")
    print(f"{'='*W}")

    for s in exp_stats:
        exp = s['experiment']
        ed = rdf[rdf['experiment'] == exp].sort_values('best_pnl', ascending=False)

        print(f"\n  ── {exp} ({s['n_deploy']}/{s['n_tickers']} deploy, "
              f"{s['deploy_pct']:.0%} success) ──")
        print(f"  {'Ticker':<8} {'Seeds':>5} {'PassA':>7} {'BestPnL':>8} "
              f"{'WR':>5} {'N':>4} {'Seed':>6}  Status")
        print(f"  {'─'*8} {'─'*5} {'─'*7} {'─'*8} {'─'*5} {'─'*4} {'─'*6}  {'─'*10}")

        for _, r in ed.iterrows():
            if r['deploy_ready']:
                status = "DEPLOY"
            elif r['pass_a'] == 0:
                status = "DEAD"
            else:
                status = f"({r['best_pnl']:.2%})"
            print(f"  {r['ticker']:<8} {r['n_seeds']:>5} "
                  f"{r['pass_a']:>4}/{r['n_seeds']} {r['best_pnl']:>7.2%} "
                  f"{r['best_wr']:>4.0%} {r['best_n']:>4} {r['best_seed']:>6}  {status}")

    # ==================================================================
    # TABLE 3: RANKING SUMMARY
    # ==================================================================
    print(f"\n\n{'='*W}")
    print(f"  TABLE 3: MODEL RANKING")
    print(f"{'='*W}")

    for rank, s in enumerate(exp_stats, 1):
        deploy_tickers = rdf[(rdf['experiment'] == s['experiment']) & rdf['deploy_ready']]
        ticker_list = ', '.join(
            deploy_tickers.sort_values('best_pnl', ascending=False)['ticker'].tolist())
        print(f"\n  #{rank}  {s['experiment']}  —  {s['deploy_pct']:.0%} success "
              f"({s['n_deploy']}/{s['n_tickers']})")
        if ticker_list:
            print(f"       Deploy: {ticker_list}")
        print(f"       Median deploy PnL: {s['med_pnl']:.2%}, "
              f"Avg seed pass rate: {s['avg_pass_rate']:.0%}")

    print(f"\n{'='*W}")


main()
