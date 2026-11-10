# Experiment Log

Central registry of all model experiments. Updated as experiments progress.

**Statuses:** TRAINING → SCORED → OPTIMIZED → TESTED → DEPLOYED / REJECTED / ARCHIVED / PENDING

**Rule:** Never delete rows. Rejected experiments are valuable negative results.

## Numbering Convention

- **97-149:** Legacy single-fold models (archived)
- **150-199:** LONG walkforward S3 family (deployed)
- **200-299:** LONG experimental architectures
- **400-499:** SHORT models
- **500+:** Alternative strategies (swing, etc.)
- **ST-\*:** Spike Transformer via `ra_moe/` (DeStationary Attention + FiLM)
- **MOIRAI:** Foundation model experiments

## Registry

| ID | Dir | Hypothesis | Architecture | Status | Avg PnL | Date | Notes |
|---|---|---|---|---|---|---|---|
| 97 | LONG | Crypto spike baseline, 16 teachers | Transformer, WF-1 | ARCHIVED | -- | 2025-Q3 | Superseded by 152 |
| 100 | LONG | Defense sector, 12 teachers | Transformer, WF-1 | ARCHIVED | -- | 2025-Q3 | Data drift → retrained as 103 |
| 103 | LONG | Defense retrain, aligned heal_data | Transformer, WF-1 | DEPLOYED | +6.62% LUNR | 2025-Q4 | 3 tickers live: LUNR/RKLB/KTOS |
| 104 | LONG | 3% threshold variant | Transformer, WF-1 | ARCHIVED | -- | 2025-Q4 | Subsumed by 103 |
| 105 | LONG | Expanded teacher set | Transformer, WF-1 | ARCHIVED | -- | 2025-Q4 | Subsumed by 103 |
| 106 | LONG | 8% threshold variant of 105 | Transformer, WF-1 | ARCHIVED | -- | 2025-Q4 | Subsumed by 103 |
| 110-113 | LONG | data_utils shared arch variants | Transformer, WF-1 | ARCHIVED | -- | 2025-Q4 | Superseded by walkforward |
| 150 | LONG | First walkforward, 13 teachers | Transformer, WF-6 | DEPLOYED | baseline | 2026-Q1 | S3 gate component |
| 152 | LONG | WF continuation | Transformer, WF-6 | DEPLOYED | baseline | 2026-Q1 | Fastest decay |
| 160 | LONG | WF stitched, 13 teachers | Transformer, WF-6 | DEPLOYED | baseline | 2026-Q1 | S3 gate component |
| 170-175 | LONG | WF 6-fold stitched | Transformer, WF-6 | DEPLOYED | 8x slower decay | 2026-Q1 | S3 component |
| 180-185 | LONG | WF stitched, 31 teachers | Transformer, WF-6 | DEPLOYED | **best overall** | 2026-Q2 | S3 best family |
| 197 | LONG | Variant | Transformer | ARCHIVED | -- | 2026-Q2 | -- |
| 200 | LONG | CNN compression + label smoothing | CNN-Transformer, WF-6 | REJECTED | -- | 2026-Q2 | Scoring bug, underperformed |
| 205 | LONG | Variant of 200 | CNN-Transformer | REJECTED | -- | 2026-Q2 | Same issues |
| 210 | LONG | Dip-then-rip target | Transformer, WF-3 | REJECTED | -- | 2026-Q2 | Different signal |
| 220-225 | LONG | Balanced crypto+non-crypto teachers | Transformer, WF-6 | REJECTED | -- | 2026-Q2 | Signal dilution + augmentation |
| 230-235 | LONG | Curated teachers | Transformer, WF-6 | REJECTED | -- | 2026-Q2 | Still overfit |
| 240-245 | LONG | 28 regime features | Transformer, WF-6 | PENDING | -- | 2026-Q3 | 69 total cols |
| 400-405 | SHORT | Downside dump baseline | Transformer, WF-6 | DEPLOYED | baseline | 2026-Q3 | SHORT baseline |
| 410-415 | SHORT | SHORT refined | Transformer, WF-6 | DEPLOYED | deployed | 2026-Q3 | -- |
| 420 | SHORT | 10AM entry experiment | Transformer, WF-1 | TESTED | -- | 2026-Q3 | Not deployed |
| 425 | SHORT | 10AM reversal, 2.5% dump | Transformer, WF-1 | PENDING | -- | 2026-Q3 | Optimizer created |
| 430 | SHORT | Variant | Transformer, WF-6 | TESTED | -- | 2026-Q3 | -- |
| 435 | SHORT | 10AM variant | Transformer, WF-1 | PENDING | -- | 2026-Q3 | Optimizer created |
| 500 | LONG | Morning swing two-stream | CNN+MLP hybrid, WF-3 | REJECTED | -2.94% VAL→TEST | 2026-Q3 | Not deployment-ready |
| ST L5-1 | LONG | Spike Transformer, RA-MoE | DeStationary+FiLM, F5/6/7 | DEPLOYED | +1.88%/day | 2026-Q3 | NEWEST_FIRST, 36 tickers |
| ST ASL6 | LONG | ASL variant, paired with L5-1 | Same + ASL γ_neg=6 | DEPLOYED | paired | 2026-Q3 | 6 models total with L5-1 |
| ST ASL6-645 | LONG | Entry-offset target construction | Same + offset labels | REJECTED | +0.55%/day | 2026-08 | Fewer positives, harmful |
| ST A (L5-3) | LONG | Arch variant A vs L5-1 | Variant | REJECTED | < L5-1 | 2026-08 | L5-1 v1 is ceiling |
| ST B (L5-4) | LONG | Arch variant B vs L5-1 | Variant | REJECTED | < L5-1 | 2026-08 | L5-1 v1 is ceiling |
| ST C (L5-5) | LONG | Arch variant C vs L5-1 | Variant | REJECTED | < L5-1 | 2026-08 | L5-1 v1 is ceiling |
| ST v14 | LONG | Optimizer variant | Same arch | REJECTED | < L5-1 | 2026-08 | All P3+P4 tiers failed |
| ST L5-1 SHORT | SHORT | ST L5-1 for SHORT | Same arch, F5/6/7 | DEPLOYED | +2.64% avg | 2026-Q3 | Superseded by L5-2 HYB |
| ST L5-2 SHORT | SHORT | SHORT upgrade to L5-2 | Same arch, F5/6/7 | DEPLOYED | +3.20%/day M_AGR | 2026-08 | HYB_F7_30, 26 tickers |
| ST X-ASSET | SHORT | BTC/ETH lead-lag features | Same + 47 seq, 26 reg | PENDING | +0.42% PnL lift | 2026-08 | First positive ML exp, optimizer next |
| ST MultiRes v1 | SHORT | Dual 1m+15m backbone | 2+1 blocks, 153K params | TESTED | overfitting | 2026-08 | 50K samples insufficient |
| ST MultiRes v2 | SHORT | MultiRes + 3x training data | Same arch, 150K samples | TESTED | pending opt | 2026-08 | Addresses v1 overfit |
| MOIRAI-v2 | LONG | Foundation model consensus | MOIRAI-Large | PENDING | r=+0.55 | 2026-Q2 | Consensus filter path only viable |
| TQ-33 Base | SHORT | 33 Base+Lee-Ready T&Q features on 157 T&Q stocks | M4 Log / M7 GBDT / M12 Ens, WF-20M | TESTED | +0.59%/tr ($139K 12M) | 2026-11 | Clean $10-$100 unadj baseline (`Old IC +0.010 M7`) |
| TQ-63 Naive | SHORT | Flat Top-12 selector over 33 T&Q + 30 Cross-Modal | M4 / M7 / M12, WF-20M | REJECTED | -0.45%/tr | 2026-11 | `OHLCV×T&Q` ratios displaced 6 core `T&Q` features |
| TQ-Upgraded (+8 Ortho) | SHORT | Fixed Base 20 + Top-5 Orthogonal (`Partial r\|Core8`) + Vol-Wt PM | M7 GBDT / M12 Ens, WF-20M | VERIFIED | +0.50%/tr ($197K 24M, 61.8% WR) | 2026-11 | Standalone 1-bar ML across 157 T&Q stocks (`$10-$100 Unadj`) |
| TQ-Synergy-2 (Gatekeeper) | SHORT | C2 + Catalyst + Walk-Forward T&Q Morning & `_B2` Bounce Veto | 3-Pillar Synergy (M7+M12+ORTH_Z), WF-20M | VERIFIED | **+2.16%/tr ($366K 24M, Sh 1.57, -30.8% DD)** | 2026-11 | **Vetoes 37 toxic morning (`-19.9%`) & 22 bearish `_B2` (`-22.0%`) trades** |
| TQ-Synergy-4A (Full 3-Pillar) | SHORT | Synergy-2 + Unlocked 15–30% Gaps + Non-Diluting Kelly 35% T&Q Slot-Fill | 3-Pillar Synergy (Kelly 0.70x Fill), WF-20M | DEPLOY-READY | **+0.98%/tr ($437K 24M Sh 1.59; $238K 12M Sh 1.79)** | 2026-11 | **#1 Overall on 157 T&Q stocks: beats C2+Cat (`$310K` 24M, `$183K` 12M) by +$127K** |

## Key Learnings (Negative & Positive Results)

1. **L5-1 v1 is the LONG ceiling** — 31 improvement ideas (A/B/C variants, v14, temporal decay, vol-scaled TP/SL, ensemble, ML ticker selection) all failed. Simplicity is the regularizer.
2. **6:15 AM >> 6:45 AM** — 61-95% PnL decay at later entry for both directions.
3. **ML cannot beat S3 static** — S3 +2.23% > Lasso_TOP8 +1.99% > ORACLE +2.08% for LONG ticker selection.
4. **MOIRAI standalone is marginal** — family rho +0.93/+0.97, 87% LONG edge lost to binarization, max +0.41%/trade.
5. **Entry-offset targets are harmful** — ASL6-645 reduces positive labels, model fires less, worse performance.
6. **X-ASSET is the first positive ML experiment** — BTC/ETH lead-lag features add genuine signal to crypto-correlated tickers.
7. **Orthogonal T&Q Screening (`Partial r | Core8`) vs. Flat Correlation Selection (`2026-11-09`)** — Throwing hybrid `OHLCV × T&Q` features into a single unconstrained Top-K selector displaces genuine order-flow features and induces multicollinearity with `Core 8 OHLCV`. Keeping `Base 20 (Core 8 OHLCV + Top 12 Pure T&Q)` fixed and adding a dedicated block of **Partial-Correlation-Verified Orthogonal Features** (`deep08_pm_trapped_vol_share`, `deep05_quote_int_vs_20d_norm`, `deep12_total_overhead_trapped_vol`, `new02_inst_sell_block_pressure`, `new01_retail_vs_inst_ofi_div`, `new05_passive_ask_absorption`) lifts `M7 Deep GBDT` 24M OOS compounded equity from `$81,442` to **`$196,626` (`+77.1%` Tot PnL, `61.8%` WR, `1.07` Sharpe across `157` `T&Q` stocks)**.
8. **Reverse-Split `Spread / Close` Distortion in `Master_Inference_v2` (`2026-11-09`)** — In `Master_Inference_v2`, 1-minute bar `Open/Close/Volume` are split-adjusted (`adjusted=True`), whereas `Spread` (from raw quotes) and `Total_Vol` (from raw trades) are unadjusted. On stocks that later executed `1-for-10` to `1-for-60,000` reverse splits (`VIVK`, `SDOT`, `AZI`, `BIAF`, `VEEE`, `BIYA`, `AKAN`, `MASK`, `OPTT`, `RVSN`, `IXHL`), dividing unadjusted `Spread` by split-adjusted `Close` artificially shrinks `Spread%` by `10x–60,000x`. Always restore `unadj_open_px = open_px / split_factor` and `unadj_spread_pct = spread_pct * split_factor`!
9. **3-Pillar Synergy Architecture: Domain-Trained `T&Q` ML + 3-Bar Structural Mechanics + Kelly Non-Diluting Slot Hierarchy (`2026-11-10`)** — Standalone 1-bar `T&Q` ML (`2,619` raw setups, `-0.16%` base-rate mean) trails rule-based `Deployed C2` (`189` 3-bar setups, `+1.55%` mean) because 1-bar red candles lack 3-bar structural peak confirmation (`i-3..i+3`), while training a meta-model solely on `189` C2 trades overfits small folds. By training `M7`, `M12`, `M4`, and `ORTH_Z` on the full `2,619`-setup domain pool and using their walk-forward OOS predictions (`GRAND_SYN`) to (a) veto `37` toxic morning C2 setups (`-19.9%` sum) and `22` bearish `_B2` midday bounces (`-22.0%` sum), (b) unlock `15–30%` 3-bar gaps when `GRAND_SYN >= +0.85`, and (c) fill **only idle portfolio slots (`k_C2 < 2`)** at a **Kelly-proportional `35%` slot weight (`0.70x`)**, `SYNERGY 4A` achieves **`$437,129` (`+175.8%` Tot PnL, `1.59` Sharpe)** over 24M OOS and **`$238,228` (`+102.2%`, `1.79` Sharpe)** over Last 12M — beating `Deployed C2 + Catalyst` (`$309,939` / `$182,823`) by **`+$127,190` (`+41%`)**.


