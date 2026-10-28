# WINNER: H038
**Strategy:** SHORT gap_up gap>=1% 17:00
**Direction:** SHORT
**Period tested:** Aug 1 - Oct 6 2026
**Avg PnL:** +2.18% | **Sharpe:** 0.627 | **n:** 53 | **WR:** 83%

## How to Trade This (Human Instructions)

### What to do the night before
- Run gap scanner at market close or pre-market
- Filter: gap_up candidates with gap >= 1%
- Universe: US equities, price >= $5.0, float >= 5M shares
- Expect 5.8 trades per week on average

### Entry
- Time: local_high
- Signal: SHORT on gap_up stocks meeting filters
- Order type: market at specified time
- Max candidates per day: 5

### Exit
- Primary: 17:00
- Average hold: 7.3 hours

### Position sizing
- $100,000 daily capital, equal-weight across day's trades
- Minimum position: $5,000

### What this strategy does NOT work on
- [Review monthly breakdown for weak periods]
- [Check worst trades for failure modes]

## Backtest Stats
| Metric | Value |
|--------|-------|
| n trades | 53 |
| Avg PnL | +2.18% |
| Median PnL | +1.53% |
| Trimmed Mean | +2.00% |
| Sharpe | 0.627 |
| Win Rate | 83% |
| Profit Factor | 6.83 |
| 95% CI | [+1.29, +3.19] |
| Max Loss | -6.26% |
| Max Gain | +13.51% |
| Max Drawdown | -0.3% |
| Trades/Week | 5.8 |
| Avg Hold | 7.3 hours |

## Top 5 Worst Trades
| Date | Ticker | Gap | PnL |
|------|--------|-----|-----|
| 2026-09-02 | DELL | +8.7% | -6.26% |
| 2026-09-11 | HOFT | +8.3% | -3.12% |
| 2026-08-10 | SDRL | +7.7% | -2.83% |
| 2026-09-17 | AXTI | +5.6% | -1.96% |
| 2026-08-26 | META | +3.6% | -1.86% |

## Top 5 Best Trades
| Date | Ticker | Gap | PnL |
|------|--------|-----|-----|
| 2026-08-21 | HUT | +4.2% | +13.51% |
| 2026-09-10 | MCFT | +5.8% | +11.50% |
| 2026-09-22 | PLNT | +2.5% | +8.90% |
| 2026-08-06 | SOUN | +25.1% | +7.79% |
| 2026-08-19 | AXTI | +2.7% | +6.75% |
