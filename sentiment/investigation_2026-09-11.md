# Scanner PnL Investigation — 2026-09-11

Combined analysis of premarket + discovery scanner trades from Sep 10-11, 2026.
- **169 trades** evaluated (58 from Sep 10, 111 from Sep 11)
- **31 selected**, 138 not selected
- **2.5% stop loss** applied to all trades, checked bar-by-bar on highs/lows
- **Equal-weight capital allocation** — portfolio return = mean of individual SL-adjusted returns
- Entry at signal time or market open (09:30 ET), whichever is later
- Only trades with ≥30 min of market data included
- Alpha = trade PnL minus SPY return over the same window (isolates stock-specific edge from market direction)

---

## Overall Results

| Slice | 30m PnL | 30m WR | 60m PnL | 60m WR | 60m Alpha | n | Stopped |
|-------|---------|--------|---------|--------|-----------|---|---------|
| ALL | -0.16% | 44% | -0.19% | 38% | +0.08% | 169 | 45/169 |
| SELECTED | -0.09% | 45% | -0.22% | 32% | +0.07% | 31 | 7/31 |
| NOT SELECTED | -0.17% | 44% | -0.18% | 40% | +0.08% | 138 | 38/138 |

**Portfolio PnL (equal-weight, SL-adjusted):**
- Selected @ 30m: **-0.089%**
- Selected @ 60m: **-0.219%**
- Selected @ EOD: **-0.251%**

---

## By Direction — SHORT is the edge

| Direction | 30m PnL | 30m WR | 60m PnL | 60m WR | 60m Alpha | EOD PnL | EOD WR | n |
|-----------|---------|--------|---------|--------|-----------|---------|--------|---|
| **SHORT** | **-0.07%** | **48%** | **-0.06%** | **43%** | **+0.43%** | **+0.21%** | **51%** | 94 |
| LONG | -0.26% | 40% | -0.35% | 33% | -0.39% | -0.61% | 29% | 75 |

**Observation:** SHORT signals generate positive alpha (+0.43% at 60m) even though raw PnL is slightly negative — the scanner picks the right stocks to short, but market going up (SPY +0.06%) drags raw PnL. LONG signals underperform SPY by -0.39%, meaning they're worse than just buying the index.

---

## By Scanner — Discovery outperforms Premarket

| Scanner | 30m PnL | 60m PnL | 60m Alpha | EOD PnL | n |
|---------|---------|---------|-----------|---------|---|
| **Discovery** | **-0.10%** | **-0.15%** | **+0.21%** | **-0.13%** | 129 |
| Premarket | -0.34% | -0.30% | -0.33% | -0.25% | 40 |

**Observation:** Discovery scanner has positive alpha; premarket scanner has negative alpha. Discovery finds more actionable signals, likely because thematic search uncovers catalysts the premarket deploy-universe scan misses.

---

## By Date

| Date | 30m PnL | 60m PnL | n | Stopped | Stop Rate |
|------|---------|---------|---|---------|-----------|
| Sep 10 | -0.07% | -0.03% | 58 | 7 | 12% |
| Sep 11 | -0.20% | -0.27% | 111 | 38 | 34% |

**Observation:** Sep 11 had 3x the stop-out rate of Sep 10. Possible causes: higher intraday volatility, more aggressive entries, or weaker catalysts passing the selection filter. Sep 10 was nearly breakeven at 60m.

---

## By Catalyst Type (60m, SL-adjusted)

### Positive PnL catalyst types:
| Catalyst | 60m PnL | WR | n | Notes |
|----------|---------|-----|---|-------|
| **Regulatory** | **+0.33%** | **58%** | 26 | Best WR, consistent |
| **Offering** | **+0.66%** | **46%** | 13 | Highest raw PnL, alpha +2.56% |
| Squeeze | +0.79% | 40% | 5 | Small sample |

### Negative PnL catalyst types:
| Catalyst | 60m PnL | WR | n | Notes |
|----------|---------|-----|---|-------|
| Analyst | -0.46% | 35% | 26 | Consistently bad, large sample |
| Earnings | -0.47% | 30% | 23 | Guidance kills direction calls |
| Deal | -0.43% | 41% | 17 | Acquirer/target confusion? |
| Breaking | -0.43% | 33% | 6 | Reactive, not predictive |
| Bankruptcy | -2.50% | 0% | 2 | Both stopped out |
| Unknown | -0.51% | 33% | 30 | Missing catalyst = bad signal |

**Observation:** Offerings and regulatory events are the only reliable catalyst types. Analyst upgrades/downgrades and earnings reactions are net negative — the LLM struggles to predict direction on these information-rich catalysts. "Unknown" catalyst type (missing classification) is a red flag.

---

## Gate Effectiveness

| Group | 60m PnL | n |
|-------|---------|---|
| Ungated | -0.16% | 114 |
| Gated | -0.24% | 55 |
| **Gate value** | **+0.07%** | — |

Gates are saving money (gated trades perform worse), but the magnitude is small. Cross-scanner direction conflicts (TSLA, RXT, OKLO, APUR gated on Sep 11) are the right call.

---

## Selected Trade Performance Detail

### Top 5 Winners (60m, SL-adjusted):
| Ticker | Dir | 60m PnL | Catalyst | Selected? |
|--------|-----|---------|----------|-----------|
| EDHL | LONG | +8.44% | squeeze | No |
| LHSW | SHORT | +6.75% | unknown | No |
| BHVN | SHORT | +6.36% | regulatory | **Yes** |
| LASE | SHORT | +6.11% | offering | No |
| HOFT | SHORT | +5.88% | earnings | No |

### Top 5 Losers (60m, SL-adjusted):
| Ticker | Dir | 60m PnL | Catalyst | Selected? | Stop |
|--------|-----|---------|----------|-----------|------|
| RH | LONG | -2.50% | earnings | **Yes** | @28m |
| AENT | SHORT | -2.50% | earnings | **Yes** | @0m |
| CVX | LONG | -2.50% | breaking | **Yes** | @48m |
| HOOD | SHORT | -2.50% | analyst | **Yes** | @30m |
| ETN | SHORT | -2.50% | analyst | **Yes** | @16m |

**Observation:** The selection filter picks high-conviction trades, but 4 of the top 5 losers are selected. All losers hit the 2.5% stop. The best winners (EDHL, LHSW, LASE) were NOT selected — the selection criteria may be too conservative on unanalyzed tickers or too aggressive on high-confidence calls.

---

## Notable Trades

- **OKLO SHORT**: User made +4.45% actual. Scanner had the right direction but gated it due to cross-scanner conflict (premarket SHORT vs discovery LONG). Gate was too aggressive here.
- **CPRT SHORT**: +5.03% at 30m, correctly selected. Deal catalyst, strong performer.
- **AENT SHORT**: Stopped out immediately at open (@0m). Earnings catalyst — gapped +31.9% and the short was crushed instantly.

---

## Key Takeaways

1. **SHORT signals are the edge.** Positive alpha (+0.43%), 51% EOD WR. LONG signals are worse than buying SPY.
2. **Discovery scanner > Premarket scanner.** Both in raw PnL and alpha. Premarket scanner has negative alpha.
3. **Offering + Regulatory = only reliable catalysts.** Everything else is net negative at 60m.
4. **Earnings and analyst catalysts destroy value.** -0.47% and -0.46% avg at 60m. The LLM cannot reliably predict direction on these.
5. **Selection filter is too aggressive.** 23% stop rate on selected trades (7/31). Best winners are unselected. Consider widening the selection or reducing confidence threshold.
6. **"Unknown" catalyst type is a red flag.** 30 trades with no catalyst classification, -0.51% avg. These should be filtered or the classification improved.
7. **Cross-scanner conflicts are mostly correct** to gate, but OKLO was a missed opportunity (+4.45%).
8. **Stop loss is essential.** Without the 2.5% SL, several trades would have lost 5-19% (SURG -13.3%, TNON -9.6%, OPTT +19.3% adverse).

---

## Suggested Improvements

- **Filter out analyst and earnings catalysts** from selection (or apply higher threshold).
- **Weight SHORT signals higher** in the portfolio allocation.
- **Require catalyst_type != 'unknown'** for selection.
- **Reduce cross-scanner conflict gating** — consider keeping the signal from the scanner with higher tradability score instead of blocking both.
- **Track offering/regulatory catalyst types separately** — these may warrant higher position sizing.
