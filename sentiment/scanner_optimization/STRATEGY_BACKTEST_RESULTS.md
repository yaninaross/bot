# Strategy Backtest — Master Strategy Definition Results

**Run Date:** 2026-09-13
**Trading Dates:** 2026-09-08, 2026-09-09, 2026-09-10, 2026-09-11
**Strategies:** 10

## Strategy Definitions

| ID | Label | Direction | Gap Sign | Gap Min % | Universe | ORB | Entry | Exit |
|---|---|---|---|---|---|---|---|---|
| BASELINE_V1 | Baseline V1 (Frozen) | SHORT | positive | 3.0% | llm | Yes (SHORT) | local_high | t2_close |
| B1_WIDE_SHORT | Wide Short (No Filters) | SHORT | positive | 1.5% | all | No | local_high | t2_close |
| B2_NO_ORB | Short, No ORB | SHORT | positive | 3.0% | llm | No | local_high | t2_close |
| B3_EOD_ENTRY | Short, EOD Entry | SHORT | positive | 3.0% | llm | Yes (SHORT) | eod_close | t2_close |
| B4_T1_ENTRY | Short, T+1 Entry | SHORT | positive | 3.0% | llm | Yes (SHORT) | t1_open | t2_close |
| C1_LONG_FADE | Long Gap-Down Fade | LONG | negative | 3.0% | all | No | local_low | t2_close |
| C2_LONG_ORB | Long Gap-Down, ORB | LONG | negative | 3.0% | all | Yes (LONG) | local_low | t2_close |
| C3_EARNINGS | Earnings Bounce | LONG | negative | 3.0% | all | Yes (LONG) | local_low | t2_close |
| D1_LONG_MOM | Long Momentum | LONG | positive | 5.0% | all | Yes (LONG) | orb_break | 60m (2.5% SL) |
| D2_SHORT_MOM | Short Momentum | SHORT | negative | 5.0% | all | Yes (SHORT) | orb_break | 60m (2.5% SL) |

## Data Pipeline

```
[1/5] Building trading calendar...
    SPY daily: 9 bars (2026-08-29 to 2026-09-21)
    2026-09-08: prior=2026-09-04, T+1=2026-09-09, T+2=2026-09-10
    2026-09-09: prior=2026-09-08, T+1=2026-09-10, T+2=2026-09-11
    2026-09-10: prior=2026-09-09, T+1=2026-09-11, T+2=None [T+2 NOT AVAILABLE]
    2026-09-11: prior=2026-09-10, T+1=None, T+2=None [T+2 NOT AVAILABLE]
    Grouped daily needed for 5 dates

[2/5] Fetching grouped daily data...
    2026-09-04: 12063 tickers
    2026-09-08: 12083 tickers
    2026-09-09: 12065 tickers
    2026-09-10: 12124 tickers
    2026-09-11: 12035 tickers

[3/5] LLM-flagged universe from BQ...
    premarket: 277 rows
    discovery: 409 rows
    Total LLM-flagged: 372 unique (ticker, date)

[4/5] Building gapper universe...
    2026-09-08: 1168 gap-up, 1154 gap-down (|gap| >= 1.5%)
    2026-09-09: 875 gap-up, 878 gap-down (|gap| >= 1.5%)
    2026-09-10: 575 gap-up, 2313 gap-down (|gap| >= 1.5%)
    2026-09-11: 2093 gap-up, 572 gap-down (|gap| >= 1.5%)
    Total gappers: 9628

[5/5] Fetching minute bars...
    Fetched: 9327/9628 ok (1289s)
    Cache: 121.0 MB
```

## Strategy Results

```
Strategy             n      avg       WR    Exclusions (top)
BASELINE_V1          6    +8.39%    67%     not_llm=9442, gap=132, catalyst_exclude=30, no_orb=16
B1_WIDE_SHORT     1544    +2.23%    71%     gap=4917, no_t2=2052, short_session=1010, no_bars=105
B2_NO_ORB           12    +6.84%    75%     not_llm=9442, gap=132, catalyst_exclude=30, no_t2=12
B3_EOD_ENTRY         6    +3.30%    83%     not_llm=9442, gap=132, catalyst_exclude=30, no_orb=16
B4_T1_ENTRY          6    +2.16%    67%     not_llm=9442, gap=132, catalyst_exclude=30, no_orb=16
C1_LONG_FADE       414    -2.93%    32%     gap=7727, no_t2=838, short_session=585, no_bars=61
C2_LONG_ORB        200    -1.58%    39%     gap=7727, short_session=585, no_orb=540, no_t2=512
C3_EARNINGS          1   -10.47%     0%     gap=7727, catalyst_include=1893, no_t2=4, no_orb=3
D1_LONG_MOM         87    +0.57%    31%     gap=8817, no_orb=389, short_session=306, no_bars=19
D2_SHORT_MOM        85    -0.85%    21%     gap=8801, no_orb=392, short_session=313, no_bars=29
```

## Metrics Summary

| Strategy | n | avg | med | trim | std | Sharpe | WR | PF | max- | max+ | CI lo | CI hi | total | t/d |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASELINE_V1 | 6 | +8.39 | +1.37 | +8.39 | 19.70 | 0.426 | 67% | 10.11 | -3.69 | +48.13 | -7.37 | +24.15 | +50.3 | 1.5 |
| B1_WIDE_SHORT | 1544 | +2.23 | +2.98 | +3.20 | 30.91 | 0.072 | 71% | 1.94 | -1158.50 | +48.13 | +0.69 | +3.78 | +3449.3 | 386.0 |
| B2_NO_ORB | 12 | +6.84 | +3.94 | +3.76 | 13.58 | 0.504 | 75% | 15.80 | -3.69 | +48.13 | -0.84 | +14.52 | +82.1 | 3.0 |
| B3_EOD_ENTRY | 6 | +3.30 | +4.25 | +3.30 | 3.45 | 0.955 | 83% | 6.92 | -3.34 | +5.94 | +0.54 | +6.06 | +19.8 | 1.5 |
| B4_T1_ENTRY | 6 | +2.16 | +2.96 | +2.16 | 4.26 | 0.507 | 67% | 3.54 | -5.10 | +6.92 | -1.25 | +5.57 | +13.0 | 1.5 |
| C1_LONG_FADE | 414 | -2.93 | -3.51 | -3.22 | 13.80 | -0.212 | 32% | 0.52 | -56.45 | +114.94 | -4.26 | -1.60 | -1213.8 | 103.5 |
| C2_LONG_ORB | 200 | -1.58 | -2.35 | -2.04 | 14.41 | -0.109 | 39% | 0.70 | -43.37 | +114.94 | -3.57 | +0.42 | -315.5 | 50.0 |
| C3_EARNINGS | 1 | -10.47 | -10.47 | -10.47 | 0.00 | 0.000 | 0% | 0.00 | -10.47 | -10.47 | -10.47 | -10.47 | -10.5 | 0.2 |
| D1_LONG_MOM | 87 | +0.57 | -2.50 | -1.13 | 10.14 | 0.056 | 31% | 1.35 | -2.50 | +88.55 | -1.57 | +2.70 | +49.2 | 21.8 |
| D2_SHORT_MOM | 85 | -0.85 | -2.50 | -1.85 | 3.70 | -0.229 | 21% | 0.54 | -2.50 | +14.18 | -1.63 | -0.06 | -72.1 | 21.2 |

## Acceptance Criteria

| Strategy | ci>0 | n>=50 | d>=10 | Sh>=.3 | WR>=55 | mx>-15 | t/d>=1 | READY |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| BASELINE_V1 | - | - | - | + | + | + | + | **-** |
| B1_WIDE_SHORT | + | + | - | - | + | - | + | **-** |
| B2_NO_ORB | - | - | - | + | + | + | + | **-** |
| B3_EOD_ENTRY | + | - | - | + | + | + | + | **-** |
| B4_T1_ENTRY | - | - | - | + | + | + | + | **-** |
| C1_LONG_FADE | - | + | - | - | - | - | + | **-** |
| C2_LONG_ORB | - | + | - | - | - | - | + | **-** |
| C3_EARNINGS | - | - | - | - | - | + | - | **-** |
| D1_LONG_MOM | - | + | - | - | - | + | + | **-** |
| D2_SHORT_MOM | - | + | - | - | - | + | + | **-** |

**No strategy meets all acceptance criteria.** Primary blockers: insufficient dates (4 vs 10 required), most strategies lack n>=50.

## Per-Date Breakdown

### 2026-09-08

| Strategy | n | avg | WR | total |
|---|---:|---:|---:|---:|
| BASELINE_V1 | 6 | +8.39 | 67% | +50.3 |
| B1_WIDE_SHORT | 893 | +2.01 | 72% | +1791.7 |
| B2_NO_ORB | 12 | +6.84 | 75% | +82.1 |
| B3_EOD_ENTRY | 6 | +3.30 | 83% | +19.8 |
| B4_T1_ENTRY | 6 | +2.16 | 67% | +13.0 |
| C1_LONG_FADE | 245 | -3.17 | 30% | -777.4 |
| C2_LONG_ORB | 114 | -2.93 | 33% | -334.3 |
| C3_EARNINGS | 1 | -10.47 | 0% | -10.5 |
| D1_LONG_MOM | 31 | +2.95 | 39% | +91.5 |
| D2_SHORT_MOM | 19 | -0.81 | 16% | -15.5 |

### 2026-09-09

| Strategy | n | avg | WR | total |
|---|---:|---:|---:|---:|
| B1_WIDE_SHORT | 651 | +2.55 | 70% | +1657.6 |
| C1_LONG_FADE | 169 | -2.58 | 35% | -436.4 |
| C2_LONG_ORB | 86 | +0.22 | 47% | +18.8 |
| D1_LONG_MOM | 9 | -0.72 | 33% | -6.5 |
| D2_SHORT_MOM | 12 | -1.18 | 33% | -14.2 |

### 2026-09-10

| Strategy | n | avg | WR | total |
|---|---:|---:|---:|---:|
| D1_LONG_MOM | 15 | -0.73 | 33% | -11.0 |
| D2_SHORT_MOM | 40 | -1.44 | 15% | -57.5 |

### 2026-09-11

| Strategy | n | avg | WR | total |
|---|---:|---:|---:|---:|
| D1_LONG_MOM | 32 | -0.78 | 22% | -24.9 |
| D2_SHORT_MOM | 14 | +1.07 | 36% | +15.0 |

## Output Files

- `results/strategy_BASELINE_V1.csv` (6 trades)
- `results/strategy_B1_WIDE_SHORT.csv` (1544 trades)
- `results/strategy_B2_NO_ORB.csv` (12 trades)
- `results/strategy_B3_EOD_ENTRY.csv` (6 trades)
- `results/strategy_B4_T1_ENTRY.csv` (6 trades)
- `results/strategy_C1_LONG_FADE.csv` (414 trades)
- `results/strategy_C2_LONG_ORB.csv` (200 trades)
- `results/strategy_C3_EARNINGS.csv` (1 trade)
- `results/strategy_D1_LONG_MOM.csv` (87 trades)
- `results/strategy_D2_SHORT_MOM.csv` (85 trades)
- `results/strategy_metrics_all.json`
- `strategy_comparison.png`

**Total: 10 strategies, 2361 trades**
