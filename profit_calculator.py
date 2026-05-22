"""
profit_calculator.py — Strategy Return Projector (Quarterly Tax)
================================================================
Compounds per-trade returns, pays estimated taxes quarterly per IRS
schedule (Q1-Q4), then projects over the specified number of months.
"""

import math

def calculate_projection(
    initial_investment,
    months,
    avg_profit_per_trade_pct,
    trades_per_month,
    tax_rate_percentage,
):
    trades_per_quarter = trades_per_month * 3
    trades_per_year = trades_per_month * 12

    monthly_multiplier = (1 + (avg_profit_per_trade_pct / 100.0)) ** trades_per_month
    quarterly_multiplier = (1 + (avg_profit_per_trade_pct / 100.0)) ** trades_per_quarter
    annual_multiplier_pretax = (1 + (avg_profit_per_trade_pct / 100.0)) ** trades_per_year
    annual_return_pretax = (annual_multiplier_pretax - 1) * 100

    print(f"--- STRATEGY METRICS ---")
    print(f"   Avg Profit/Trade:  {avg_profit_per_trade_pct}%")
    print(f"   Trades/Month:      {trades_per_month:.1f}  ({trades_per_quarter:.0f}/quarter, {trades_per_year:.0f}/year)")
    print(f"   Annual Return:     {annual_return_pretax:,.2f}% (pre-tax, compounded)")
    print(f"   Tax Rate:          {tax_rate_percentage}% (short-term cap gains)")
    print(f"   Tax Schedule:      Quarterly estimated payments")
    print(f"   Projection:        {months} months ({months/12:.1f} years)")
    print("-" * 60)

    current_capital = initial_investment
    total_tax_paid = 0

    full_quarters = months // 3
    remaining_months = months % 3

    for q in range(1, full_quarters + 1):
        q_start = current_capital
        q_end = q_start * quarterly_multiplier
        q_profit = q_end - q_start
        q_tax = q_profit * (tax_rate_percentage / 100.0)
        current_capital = q_end - q_tax
        total_tax_paid += q_tax

        q_return = (current_capital - q_start) / q_start * 100
        print(f"   Q{q:>2}:  ${current_capital:>14,.2f}   "
              f"(+{q_return:.1f}% net)   "
              f"Tax paid: ${q_tax:>10,.2f}")

    if remaining_months > 0:
        r_start = current_capital
        r_trades = trades_per_month * remaining_months
        r_multiplier = (1 + (avg_profit_per_trade_pct / 100.0)) ** r_trades
        r_end = r_start * r_multiplier
        r_profit = r_end - r_start
        r_tax = r_profit * (tax_rate_percentage / 100.0)
        current_capital = r_end - r_tax
        total_tax_paid += r_tax

        r_return = (current_capital - r_start) / r_start * 100
        print(f"   +{remaining_months}mo: ${current_capital:>14,.2f}   "
              f"(+{r_return:.1f}% net)   "
              f"Tax paid: ${r_tax:>10,.2f}")

    return current_capital, total_tax_paid


if __name__ == '__main__':
    # --- STRATEGY PARAMETERS ---
    AVG_PROFIT       = 4.77     # Combined BT avg PnL across 27 active tickers
    TRADES_PER_MONTH = 15.4     # ~185/year / 12

    # --- INVESTMENT SETTINGS ---
    START_CASH = 220_000.0
    MONTHS     = 60             # 5 years
    TAX_RATE   = 40.0           # Short-term capital gains

    print("=" * 60)
    print("  STRATEGY RETURN PROJECTION (QUARTERLY TAX)")
    print("=" * 60)

    final_val, total_tax = calculate_projection(
        START_CASH, MONTHS, AVG_PROFIT, TRADES_PER_MONTH, TAX_RATE
    )

    print("=" * 60)
    print(f"  FINAL ACCOUNT VALUE:  ${final_val:>14,.2f}")
    print(f"  TOTAL TAX PAID:       ${total_tax:>14,.2f}")
    print(f"  TOTAL NET GAIN:       ${final_val - START_CASH:>14,.2f}")
    print("=" * 60)
