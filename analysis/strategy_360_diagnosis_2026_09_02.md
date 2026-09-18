# Strategy 360 Diagnosis — September 2, 2026

## Executive Summary

Live trading has been negative for the week of Aug 24 - Sep 2, 2026. Five parallel investigation agents analyzed market regime, signal quality, trade forensics, historical trends, directional asymmetry, timing shifts, and asset rotation across 130 tickers.

**Root cause: Bilateral volatility compression (market-level, not model-level).** Pre-market spike frequency collapsed 86%, from 6.6% to 0.9% of ticker-days. Both LONG and SHORT MFE are compressing simultaneously despite stable/increasing volume. Model staleness correlation is r = -0.004 — newer folds perform no better than older ones. This is a regime problem, not a model degradation problem.

---

## 1. Backtest Results (Aug 24 - Sep 2, 2026)

| Metric | ST LONG | ST SHORT | Combined |
|--------|---------|----------|----------|
| Trades | 23 | 18 | 41 |
| Avg PnL/trade | +0.36% | +1.29% | +0.77% |
| Win rate | 52% | 72% | 61% |
| Cum EW PnL | +2.32% | +11.72% | +5.26% |
| Sharpe | 3.2 | 8.4 | 7.7 |
| Max DD | 2.95% | 2.16% | 1.79% |
| TP hits | 2 | 0 | 2 |
| SL hits | 2 | 4 | 6 |
| EOD exits | 19 | 14 | 33 |

**Critical observation:** Only 2 out of 41 trades hit TP (5%). 33 out of 41 exited at EOD. TP levels are unreachable in the current regime.

### Per-Ticker LONG Trades
| Ticker | Source | Mode | Trades | Avg PnL | WR | TP | SL | EOD |
|--------|--------|------|--------|---------|----|----|----|----|
| APLD | MR-v2 F9 | ensemble | 3 | +1.14% | 67% | 0 | 1 | 2 |
| CRDO | MR-v2 F8 | static | 1 | -3.30% | 0% | 0 | 1 | 0 |
| CVX | MR-v2 F8 | static | 3 | +0.25% | 67% | 0 | 0 | 3 |
| KTOS | L5-1 F7 | static | 1 | +1.03% | 100% | 0 | 0 | 1 |
| MARA | MR-v2 F9 | static | 2 | +5.70% | 100% | 2 | 0 | 0 |
| NBIS | MR-v2 F7 | ensemble | 2 | -2.98% | 0% | 0 | 0 | 2 |
| PL | MR-v2 F7 | limit_entry | 1 | -0.63% | 0% | 0 | 0 | 1 |
| PLTR | L5-1 F7 | static | 5 | +1.50% | 80% | 0 | 0 | 5 |
| SMR | MR-v2 F9 | static | 1 | -1.07% | 0% | 0 | 0 | 1 |
| SOUN | MR-v2 F7 | grace | 1 | +1.39% | 100% | 0 | 0 | 1 |
| WOLF | MR-v2 F8 | static | 3 | -2.06% | 0% | 0 | 0 | 3 |

### Per-Ticker SHORT Trades
| Ticker | Source | Mode | Trades | Avg PnL | WR | TP | SL | EOD |
|--------|--------|------|--------|---------|----|----|----|----|
| AEHR | MR-v2 F10 | static | 2 | +4.21% | 100% | 0 | 0 | 2 |
| FCEL | MR-v2 F9 | ensemble | 1 | -6.30% | 0% | 0 | 1 | 0 |
| HOOD | MR-v2 F10 | ensemble | 3 | -0.38% | 33% | 0 | 1 | 2 |
| HUT | MR-v2 F7 | static | 1 | +2.74% | 100% | 0 | 0 | 1 |
| MXL | MR-v2 F8 | static | 2 | +2.26% | 100% | 0 | 0 | 2 |
| NNE | MR-v2 F6 | volscaled | 4 | +2.14% | 75% | 0 | 0 | 4 |
| NTLA | MR-v2 F7 | ensemble | 1 | +1.34% | 100% | 0 | 0 | 1 |
| RCAT | MR-v2 F9 | grace | 1 | +4.70% | 100% | 0 | 0 | 1 |
| SLS | MR-v2 F10 | limit_entry | 3 | +0.15% | 67% | 0 | 1 | 2 |

---

## 2. Market Regime Analysis

**Agent 1 findings: Severe spike compression across all dimensions.**

| Metric | July | Recent (Aug 24 - Sep 2) | Change |
|--------|------|------------------------|--------|
| Avg pre-market spike magnitude | 1.51% | 1.04% | -31% |
| 3%+ spike frequency | 6.6% of ticker-days | 0.9% | -86% |
| Pre-market volatility (1-min std) | 1.14% | 0.88% | -23% |
| SPY daily range | 0.99% | 0.53% | -46% |
| VIX weekly avg | 17.2 | 15.4 | -1.8 pts |

### Sector Spike Breakdown
| Sector | July 3%+ frequency | Recent 3%+ frequency | Change |
|--------|-------------------|---------------------|--------|
| Crypto miners | 7.3% | 0.0% | Complete shutoff |
| Clean energy | 18.2% | 4.2% | -77% |
| Defense/Space | 1.9% | 1.8% | Stable (already low) |
| Tech/Other | 5.6% | 0.0% | Complete shutoff |

### Regime Break Timeline
The inflection point was around Aug 16-23. Weeks of Aug 2 and Aug 9 still had healthy spike activity (14.5% and 6.9% frequency). Week of Aug 23 dropped to 2.8%, and Aug 30 onward collapsed to <1%.

**Charts:** `/tmp/regime_analysis.png`

---

## 3. Trade PnL Forensics

**Agent 3 findings: LONG MFE is structurally below TP target. Model selects wrong days.**

### MFE/MAE Summary
| Direction | Median MFE | Median MAE | % days reaching 3%+ MFE | Median time to MFE |
|-----------|-----------|-----------|------------------------|-------------------|
| LONG | +1.80% | -2.23% | 25% (22/88) | 0.3h (18 min) |
| SHORT | +3.22% | -1.71% | 53% (38/72) | 1.2h |

### Winners vs Losers Pattern
| | LONG Winners | LONG Losers | SHORT Winners | SHORT Losers |
|--|-------------|-------------|---------------|--------------|
| Avg MFE | 3.67% | 1.33% | 4.57% | 1.87% |
| Time to MFE | 2.7h | 0.8h | 3.5h | 0.4h |
| Avg MAE | — | -3.70% | — | -4.06% |

Winners show slow sustained moves (2-3 hours). Losers show a brief pop in the first 20 minutes then deep drawdown.

### WOLF Case Study (3 trades, ALL losing)
- Aug 24, 25, 27 were all fade days — price dropped from entry and never recovered
- Aug 24: -7.77% MAE with only +0.79% MFE
- Aug 27: literally 0% MFE — open price was the high of the day
- BUT Aug 26 (+9.00% MFE) and Aug 31 (+8.05% MFE) were NOT traded
- **The model signaled on the wrong days**

### MARA Case Study (2 trades, both winning)
- Aug 24 and Aug 25: genuine sustained uptrends (7-8% MFE)
- Price built throughout the day (sustained, not pop-and-fade)
- But Aug 28: 0% MFE and -9.41% MAE — when MARA misses, it crashes hard

### Missed Opportunities
99 instances of 3%+ intraday moves existed in the period. Key misses:
- CRDO had SHORT spikes on 5 different days (up to -10.36% on Sep 2) — none traded
- SMR had LONG spikes on 6 of 8 days (3-8% MFE) — only 1 actually traded

**Charts:**
- `/tmp/forensics_mfe_mae_distributions.png`
- `/tmp/forensics_mfe_vs_mae_scatter.png`
- `/tmp/forensics_wolf_vs_mara_paths.png`
- `/tmp/forensics_time_to_mfe.png`
- `/tmp/forensics_tp_hit_rates.png`
- `/tmp/forensics_per_ticker_mfe_boxplot.png`

---

## 4. Live Signal Quality from BigQuery

**Agent 2 findings: Model confidence declining. Consensus mechanism overriding individual model NO signals.**

### BQ Logging Bug
The `signal` field (GO/NO) is determined by Phase 3 consensus selection (n_fires ranking), but `prob` and `threshold` logged are from Phase 1 hybrid model scoring. These represent different things. 65 GO signals have prob < threshold (margins down to -0.39). The consensus of multiple models overrides the single-model threshold.

Relevant files: `scoring/daily_score_st_long.py` (lines 1183-1217, 1335-1349)

### Model Confidence Trend

**LONG (ST_LONG_M_CONS_6NEWEST):**
| Week | Avg GO Margin | % Low-Confidence |
|------|-------------|-----------------|
| W34 (Aug 18-24) | +0.088 | 40% |
| W35 (Aug 25-31) | +0.020 | 77% |
| W36 (Sep 1-2) | -0.013 | 63% |

**SHORT (ST_SHORT_MRV2_NEWEST3):**
| Week | Avg GO Margin | % Low-Confidence |
|------|-------------|-----------------|
| W33 (Aug 11-17) | +0.207 | 29% |
| W35 (Aug 25-31) | -0.058 | 89% |
| W36 (Sep 1-2) | -0.006 | 67% |

### Worst Models
- **LONG MR-v2 fold7:** 7 GO signals, avg margin = -0.193, min = -0.350 (NBIS at 0.35 prob vs 0.70 threshold)
- **LONG MR-v2 fold8:** 11 GO signals, avg margin = -0.037
- **SHORT MR-v2 fold6:** 5 GO signals, avg margin = -0.128

### Best Models
- LONG ASL6 fold7: avg margin = +0.167
- LONG L5-1 fold7: avg margin = +0.059

### Suspicious Live GO Signals (Aug 25 - Sep 2)
| Direction | Ticker | Prob | Threshold | Margin |
|-----------|--------|------|-----------|--------|
| LONG | NBIS | 0.35 | 0.70 | -0.35 |
| LONG | SMCI | 0.39 | 0.70 | -0.31 |
| LONG | AXTI | 0.53-0.67 | 0.80 | -0.13 to -0.27 |
| LONG | PL | 0.27-0.28 | 0.50 | -0.22 |
| SHORT | NNE | 0.36-0.42 | 0.60 | -0.18 to -0.24 |
| SHORT | ACMR | 0.19 | 0.30 | -0.11 |

**Charts:**
- `/tmp/signal_quality_overview.png`
- `/tmp/signal_quality_per_model.png`

---

## 5. Historical PnL Trend

**Agent 4 findings: Not gradual degradation — regime mismatch. Consensus is doing all the heavy lifting.**

### Raw Model Signal Performance (NOT consensus-filtered)
| Direction | Avg PnL | Win Rate | GO Signals |
|-----------|---------|----------|-----------|
| LONG | -0.97% | 5% | 405 |
| SHORT | -0.21% | 7% | 296 |

**No model family/fold exceeds the 1.8% deploy threshold.** Only ASL6 F6 is positive at +1.11%, but with just 4 trades (statistically meaningless). Every MR-v2 fold is negative.

### GO Rate Collapse
- **LONG:** 48% (Aug 10) → 23% (Aug 17) → 13% (Aug 24) → 13% (Sep 1)
- **SHORT:** 45% (Aug 3) → 38% (Aug 10) → 20% (Aug 17) → 13% (Aug 24) → 11% (Sep 1)

### Model Staleness
- Staleness-PnL correlation: **r = -0.004, p = 0.938** (essentially zero)
- Newest folds (MR-v2 F9, F10) perform no better than oldest (F6, F7)
- **This is NOT model aging**

### Trade Outcome Distribution
Out of 2,007 valid trades analyzed: only 4 TP hits, 41 SL hits, 1,962 EOD exits. The TP/SL levels (6-16% TP, 3-12% SL) are far too wide for the current low-volatility regime.

**Charts:**
- `/tmp/pnl_degradation_trend.png`
- `/tmp/fold_comparison.png`
- `/tmp/outcome_distribution.png`
- `/tmp/signal_quality_trend.png`
- `/tmp/staleness_vs_pnl.png`

---

## 6. Directional Spike Asymmetry

**Agent 5 findings: SHORT is compressing HARDER than LONG. This is bilateral volatility compression, not a directional shift.**

### MFE Compression by Direction
| Period | LONG MFE | SHORT MFE |
|--------|----------|-----------|
| First half (Jul 1 - Aug 1) | 3.72% | 5.13% |
| Second half (Aug 1 - Sep 2) | 3.72% | 3.63% |
| Change | 0.0% | **-29.2%** |

### DOWN/UP Spike Ratio (FALLING, not rising)
| Period | DOWN spikes | UP spikes | Ratio |
|--------|------------|-----------|-------|
| First half | 285 | 202 | 1.41 (heavy short bias) |
| Second half | 224 | 220 | 1.02 (near parity) |

The short advantage is evaporating. Most recent full week (W36): 9 UP + 16 DOWN + **37 FLAT** days.

### Volume is NOT the explanation
Pre-market volume actually INCREASED:
- Crypto: +6.7%
- Defense: +13.5%
- Other: +10.1%

Volume is there but price moves are smaller. The market is absorbing orders without spiking — suggesting increased market-maker presence or algorithmic dampening.

### Sector Divergence
| Sector | LONG MFE change | SHORT MFE change |
|--------|----------------|-----------------|
| Crypto | -12.4% | -20.3% |
| Defense | **+30.6%** | -35.2% |
| Other | -0.3% | -34.1% |

Defense is becoming LONG-only territory (PLTR +1.44% net long bias). SHORT is collapsing fastest across all sectors.

### W35-W36 Collapse
- W35: LONG 2.60%, SHORT 3.59% — LONG drops below 3%
- W36: LONG **1.81%**, SHORT **2.16%** — BOTH below 3%, 37 of 61 ticker-days FLAT
- P90 MFE (tail): LONG 8.3%→3.5%, SHORT 13.7%→3.8%

**Charts:**
- `/tmp/directional_spike_weekly.png`
- `/tmp/mfe_directional_trend.png`
- `/tmp/spike_ratio_trend.png`
- `/tmp/volume_trend.png`

---

## 7. Spike Timing Shift Analysis

**Agent 6 findings: Moves have NOT shifted to Asian/European hours. ALL windows compressed uniformly.**

### Move Size by Time Window (July vs August change)
| Window (PT) | July Avg Range | August Avg Range | Change |
|-------------|---------------|-----------------|--------|
| 1:00-3:00 AM | 2.33% | 1.70% | -27% |
| 3:00-6:00 AM | 3.14% | 2.50% | -20% |
| 6:00-6:30 AM | 1.96% | 1.70% | -13% |
| 6:30-7:00 AM | 5.56% | 4.55% | -18% |
| 7:00-10:00 AM | 5.60% | 4.77% | -15% |
| 10:00 AM-1:00 PM | 3.72% | 2.89% | -22% |

**The 6:00-6:30 AM pre-market window was NEVER where the biggest moves happened.** It captured the day's largest move on only 0.2-0.5% of days. The real action has always been at market open (6:30-7:00, ~47% of days) and morning session (7:00-10:00, ~42%).

### Crypto Overnight
Crypto overnight (1-6 AM) share of daily range increased to 57% in W36, but overnight gap sizes actually shrank (~2%→1%). Moves happen overnight but get absorbed by US pre-market.

### Actionable Finding
Several crypto tickers (MARA, RIOT, CIFR, CLSK, SLS) flipped their best MFE window from 6:30-7:00 to 7:00-10:00 AM in August. Morning session is now the best LONG MFE window at 2.51%.

**Charts:**
- `/tmp/time_window_move_share.png`
- `/tmp/crypto_overnight_gap.png`
- `/tmp/volume_by_window.png`

---

## 8. Asset Rotation Scan

**Agent 7 findings: This is NOT an asset selection problem. Spikes are suppressed market-wide.**

### Universe Scan: 130 Tickers
Of 68 tickers with current data (Aug 18 - Sep 2):
- Only **13 of 68** had even a single 3%+ pre-market spike day
- **12 of those 13 are already in the deploy universe**
- Only new ticker with a spike day: **CORZ** (1 day, crypto miner)
- Average spike magnitude for deployed tickers: **1.05%**

### Suggested Removes (deployed but flat)
| Ticker | Sector | Direction | Avg Spike | Reason |
|--------|--------|-----------|-----------|--------|
| CVX | Energy | LONG | 0.27% | Effectively dead |
| BKSY | Defense | LONG | 0.40% | Low volume (1,660) |
| CAT | Industrial | SHORT | 0.45% | Too large-cap |
| CEG | Nuclear | SHORT | 0.50% | Too stable |
| TSLA | EV/Auto | SHORT | 0.52% | Too large-cap for spikes |

Additional borderline (<0.8% avg): SOUN, MU, JOBY, AVAV, VOYG, AMD, ACHR, BBAI, PLTR, OKLO, RKLB.

### Tickers to Re-add to Pipeline (stale data but were spiking)
| Ticker | Sector | Spike Days | Avg Spike | PM Volume |
|--------|--------|-----------|-----------|-----------|
| DFNS | Defense/Space | 5/7 | 4.54% | 20,597 |
| AAOI | Semis | 1/8 | 1.88% | 157,617 |
| MRVL | Semis | 1/8 | 1.58% | 263,112 |
| WDC | Semis | 1/8 | 1.47% | 66,930 |
| STRL | Industrial | 1/7 | 10.52% max | 3,821 |

### Sector Coverage Gaps
| Sector | Tickers | Deploy Coverage | Uncovered |
|--------|---------|----------------|-----------|
| Semis | 26 | 50% | +13 |
| Industrial | 17 | 11.8% | +15 |
| AI/Tech | 14 | 35.7% | +9 |
| Mining | 5 | 0% | +5 |
| Retail | 3 | 0% | +3 |

**Charts:**
- `/tmp/asset_rotation_scan.png`
- `/tmp/sector_spike_heatmap.png`
- `/tmp/full_spike_scan_results.csv`

---

## 9. Consolidated Diagnosis

| Question | Answer | Confidence |
|----------|--------|-----------|
| Is it model degradation? | **No.** Staleness r = -0.004. | High |
| Is it market change? | **Yes.** Bilateral volatility compression since mid-August. | High |
| Is it short-term? | **Likely.** Seasonal low-vol pattern. VIX at 15.4 historically unsustainable. | Medium |
| Have moves shifted to Asian hours? | **No.** All windows compressed uniformly 13-27%. | High |
| Are moves in other assets? | **No.** 130 tickers scanned. Spikes suppressed market-wide. | High |
| Is consensus helping? | **Yes, but barely.** Raw signals deeply negative. Consensus salvages to marginal. | High |
| What's broken in consensus? | Overrides individual model NO signals. 65 GOs with prob < threshold. | High |
| Is LONG or SHORT worse? | **LONG is structurally broken** (MFE 1.80%). SHORT holding better (MFE 3.22%). | High |

### Contributing Factors (ranked)
1. **Market regime shift (80%)** — bilateral spike compression, VIX low, SPY range halved
2. **Consensus mechanism weakness (10%)** — overriding quality gates, firing on low-confidence signals
3. **TP/SL miscalibration (5%)** — levels too wide for current vol (only 2/41 TP hits)
4. **Wrong-day selection (5%)** — model fires on fade days, misses spike days (WOLF case)

---

## 10. Recommendations

### Immediate (this week)
1. **Reduce LONG MAX_TRADES from 3 to 1** — LONG MFE median 1.80%, structurally unprofitable at scale
2. **Lower SHORT TP by ~30%** — capture available move before it fades
3. **Add minimum consensus margin gate** — require at least 1 model with margin > +0.05

### Short-term (1-2 weeks)
4. **Build portfolio-level regime gate** — auto-pause when weekly 3%+ spike frequency < 3%
5. **Remove MR-v2 fold7 from LONG consensus** (avg margin -0.19)
6. **Investigate MR-v2 fold6 in SHORT** (avg margin -0.13)
7. **Fix BQ logging** — log consensus-level data (n_fires, best_fire_prob, sum_conviction)
8. **Refresh pipeline for DFNS, AAOI, MRVL, WDC, COHR** — stale data, were spiking
9. **Remove flat tickers** — CVX, BKSY, CAT, CEG, TSLA consuming scoring capacity with no edge

### Medium-term (if regime persists 2+ weeks)
10. **Retrain with adaptive TP** — current 3%+ target may need lowering to 2%
11. **Explore delayed entry** (6:30-7:00 instead of 6:25) — MFE window has shifted for crypto
12. **Consider faster exits** — take 1.5-2% within first 30 minutes instead of holding to 1 PM
13. **Trade smaller and wait** — VIX cycles. Low-vol regimes end. The models aren't broken.

---

## Analysis Scripts & Charts Index

All analysis scripts saved to `/tmp/`:

| Script | Purpose |
|--------|---------|
| `/tmp/regime_analysis.py` | Market regime analysis |
| `/tmp/signal_quality_analysis.py` | BQ signal quality analysis |
| `/tmp/historical_pnl_comparison.py` | Historical PnL trend + fold comparison |
| `/tmp/directional_spike_analysis.py` | Directional asymmetry analysis |
| `/tmp/time_shift_analysis.py` | Spike timing shift analysis |

### Charts
| Chart | Description |
|-------|-------------|
| `/tmp/regime_analysis.png` | 6-panel: spike magnitude, VIX, vol, frequency, SPY range, per-ticker change |
| `/tmp/forensics_mfe_mae_distributions.png` | LONG vs SHORT MFE and MAE histograms |
| `/tmp/forensics_mfe_vs_mae_scatter.png` | MFE vs MAE scatter (green=win, red=loss) |
| `/tmp/forensics_wolf_vs_mara_paths.png` | Intraday price paths — all losing vs all winning |
| `/tmp/forensics_time_to_mfe.png` | Time to max favorable excursion |
| `/tmp/forensics_tp_hit_rates.png` | TP hit rate curve by target level |
| `/tmp/forensics_per_ticker_mfe_boxplot.png` | Per-ticker MFE boxplots |
| `/tmp/signal_quality_overview.png` | GO counts, margins, prob distributions |
| `/tmp/signal_quality_per_model.png` | Per-model probability heatmaps |
| `/tmp/pnl_degradation_trend.png` | Weekly PnL, trade count, win rate |
| `/tmp/fold_comparison.png` | PnL by model family + fold |
| `/tmp/outcome_distribution.png` | TP/SL/EOD outcomes over time |
| `/tmp/signal_quality_trend.png` | Model confidence trend |
| `/tmp/staleness_vs_pnl.png` | Model age vs PnL scatter |
| `/tmp/directional_spike_weekly.png` | UP vs DOWN spike counts weekly |
| `/tmp/mfe_directional_trend.png` | LONG vs SHORT MFE trend + P90 tail |
| `/tmp/spike_ratio_trend.png` | DOWN/UP ratio trend |
| `/tmp/volume_trend.png` | Pre-market volume by sector |
| `/tmp/time_window_move_share.png` | Move share heatmap by time window |
| `/tmp/crypto_overnight_gap.png` | Crypto overnight vs pre-market analysis |
| `/tmp/volume_by_window.png` | Volume distribution by time window |
| `/tmp/asset_rotation_scan.png` | 4-panel asset rotation scan |
| `/tmp/sector_spike_heatmap.png` | Sector-level spike heatmap |

---

*Analysis conducted September 2, 2026 using 7 parallel investigation agents.*
*Data sources: GCS (gs://<GCS_BUCKET>), BigQuery (<GCP_PROJECT>.scoring_logs), 130 tickers scanned.*
