"""
profit_calculator.py — Strategy Return Projector (Daily Compound + Quarterly Tax)
==================================================================================
Daily compounding (252 trading days/year), quarterly IRS estimated tax payments,
monthly cash contributions/withdrawals, month-by-month forecast.
"""

from datetime import date
from dateutil.relativedelta import relativedelta


def calculate_projection(
    initial_investment,
    months,
    avg_profit_per_trade_pct,
    trades_per_month,
    tax_rate_percentage,
    monthly_cash_flow=0.0,
    start_date=None,
):
    TRADING_DAYS_PER_MONTH = 21

    trades_per_day = trades_per_month / TRADING_DAYS_PER_MONTH
    trades_per_year = trades_per_month * 12

    daily_multiplier = (1 + (avg_profit_per_trade_pct / 100.0)) ** trades_per_day

    annual_multiplier_pretax = daily_multiplier ** (TRADING_DAYS_PER_MONTH * 12)
    annual_return_pretax = (annual_multiplier_pretax - 1) * 100

    if start_date is None:
        start_date = date.today().replace(day=1) + relativedelta(months=1)

    print(f"--- STRATEGY METRICS ---")
    print(f"   Avg Profit/Trade:  {avg_profit_per_trade_pct}%")
    print(f"   Trades/Day:        {trades_per_day:.2f}  ({trades_per_month:.1f}/month, {trades_per_year:.0f}/year)")
    print(f"   Daily Multiplier:  {daily_multiplier:.6f}")
    print(f"   Annual Return:     {annual_return_pretax:,.2f}% (pre-tax, compounded daily)")
    print(f"   Tax Rate:          {tax_rate_percentage}% (short-term cap gains)")
    print(f"   Tax Schedule:      Quarterly estimated payments")
    print(f"   Monthly Cash Flow: ${monthly_cash_flow:+,.2f}")
    print(f"   Forecast:          {months} months  ({start_date.strftime('%b %Y')} → "
          f"{(start_date + relativedelta(months=months - 1)).strftime('%b %Y')})")
    print()

    # ── Month-by-month forecast ──
    W = 105
    print(f"   {'Month':<10} {'Date':<10} {'Account':>14}  {'Month Profit':>14}  "
          f"{'Cum Profit':>14}  {'Cum Tax':>12}  {'Contrib':>12}  {'Note'}")
    print(f"   {'-'*W}")

    current_capital = initial_investment
    total_tax_paid = 0
    total_contributed = 0
    quarter_profit = 0
    quarter_num = 0
    cumulative_profit = 0

    for month in range(1, months + 1):
        month_date = start_date + relativedelta(months=month - 1)
        month_label = month_date.strftime('%b %Y')

        current_capital += monthly_cash_flow
        total_contributed += monthly_cash_flow

        month_start_cap = current_capital
        for day in range(TRADING_DAYS_PER_MONTH):
            start_cap = current_capital
            current_capital *= daily_multiplier
            quarter_profit += current_capital - start_cap

        month_profit = current_capital - month_start_cap
        cumulative_profit += month_profit

        note = ""
        if month % 3 == 0:
            quarter_num += 1
            q_tax = max(0, quarter_profit * (tax_rate_percentage / 100.0))
            current_capital -= q_tax
            total_tax_paid += q_tax
            note = f"Q{quarter_num} tax: ${q_tax:,.2f}"
            quarter_profit = 0

        contrib_str = f"${monthly_cash_flow:+,.0f}" if monthly_cash_flow != 0 else ""

        print(f"   {month:>3}       {month_label:<10} ${current_capital:>13,.2f}  "
              f"${month_profit:>13,.2f}  ${cumulative_profit:>13,.2f}  "
              f"${total_tax_paid:>11,.2f}  {contrib_str:>12}  {note}")

    if months % 3 != 0:
        q_tax = max(0, quarter_profit * (tax_rate_percentage / 100.0))
        current_capital -= q_tax
        total_tax_paid += q_tax
        remaining = months % 3
        print(f"   {'':>10} {'':>10} ${current_capital:>13,.2f}  "
              f"{'':>15} {'':>15} "
              f"${total_tax_paid:>11,.2f}  {'':>12}  "
              f"Partial Q tax ({remaining}mo): ${q_tax:,.2f}")

    return current_capital, total_tax_paid, total_contributed


if __name__ == '__main__':
    # --- STRATEGY PARAMETERS ---
    AVG_PROFIT       = 4.77     # Combined BT avg PnL across active tickers
    TRADES_PER_MONTH = 15.4     # ~185/year / 12

    # --- INVESTMENT SETTINGS ---
    START_CASH = 220_000.0
    MONTHS     = 16             # Forecast horizon
    TAX_RATE   = 40.0           # Short-term capital gains

    # --- MONTHLY CASH FLOW ---
    MONTHLY_CASH_FLOW = 0.0     # Positive = deposit, negative = withdraw

    print("=" * 60)
    print("  STRATEGY RETURN PROJECTION")
    print("  Daily Compounding + Quarterly Tax + Monthly Cash Flow")
    print("=" * 60)

    final_val, total_tax, total_contrib = calculate_projection(
        START_CASH, MONTHS, AVG_PROFIT, TRADES_PER_MONTH, TAX_RATE, MONTHLY_CASH_FLOW
    )

    print()
    print("=" * 60)
    print(f"  FINAL ACCOUNT VALUE:  ${final_val:>14,.2f}")
    print(f"  TOTAL TAX PAID:       ${total_tax:>14,.2f}")
    print(f"  TOTAL CONTRIBUTED:    ${total_contrib:>14,.2f}")
    print(f"  TOTAL NET GAIN:       ${final_val - START_CASH - total_contrib:>14,.2f}")
    print("=" * 60)
