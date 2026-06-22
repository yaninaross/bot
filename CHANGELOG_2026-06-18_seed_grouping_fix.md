# Changelog: Seed Grouping & Experiment Config Fixes (2026-06-18)

## Summary

Three systemic bugs found across 10 scripts in the bot-RAG pipeline. All scripts that query BQ `wf_optimization_results` and group by seed were affected.

---

## Bug 1: `GRP_KEYS` missing mode-specific parameters (HIGH)

**Root cause:** Seed grouping used only 7 keys: `ticker, experiment, tpsl_mode, entry_minute, prob_thresh, tp_pct, sl_pct`. Mode-specific parameters (`dip_pct`, `tp_mult`, `sl_mult`, `trail_start_mult`, `trail_end_mult`, `grace_minutes`, `max_wait_minutes`, `vol_filter_pct`) were aggregated with `'first'` — picking an arbitrary value.

**Impact:**
- **limit_entry:** Seeds that found different `dip_pct` / `max_wait_minutes` values were merged. Backtest used an arbitrary dip, seed agreement was inflated.
- **ATR/ATR-adaptive:** Seeds with different `tp_mult` / `sl_mult` but same `tp_pct=0, sl_pct=0` were incorrectly grouped. Multipliers used in backtest were arbitrary.
- **trailing_decay:** Different `trail_start_mult` / `trail_end_mult` merged.
- **grace:** Different `grace_minutes` merged.

**Fix:** Expanded `GRP_KEYS` to 15 columns. Mode-specific params are now group keys (unique config identity), not aggregated. Only `vix_threshold`, `tlt_crash`, `spy_ignore` remain as `'first'` aggregates since they're regime filter params, not trade mechanics.

**Also required:** NaN fill for `tp_mult`, `sl_mult`, `trail_start_mult`, `trail_end_mult`, `vol_filter_pct` before groupby (BQ returns NULL for modes that don't use these).

---

## Bug 2: `DEPLOY_EXPERIMENTS` stale (HIGH)

**Root cause:** Most scripts only had `{152_XSECTOR, 160_XSECTOR}` in their experiment filter. This silently excluded all 170/175/180/185 WF data from BQ queries.

**Fix:** Expanded to `{152, 160, 170, 175, 180, 185}` (with `_XSECTOR` suffix).

---

## Bug 3: `WALKFORWARD_FOLDS` keyed wrong or missing (HIGH)

**Root cause:** WALKFORWARD_FOLDS was keyed by the FIRST fold (`170_XSECTOR`) instead of the DEPLOYED fold (`175_XSECTOR`). The `_row_to_config` function uses this dict to set `model_id` and `stats_id` — keying by the first fold caused 175_XSECTOR BQ configs to map to `170_XSECTOR` models, loading wrong model files.

**Fix:** Re-keyed by deployed fold: `175_XSECTOR → [170..175]`, `185_XSECTOR → [180..185]`.

---

## Per-Script Changes

### PRODUCTION (daily scoring pipeline)

**daily_score_combined.py** — 6 fixes:
1. `WF_DEPLOY_EXPERIMENTS`: `{152, 160, 170}` → `{152, 160, 170, 175, 180, 185}`
2. `WF_GRP_KEYS`: 7 → 15 columns (added mode-specific params)
3. `_EXP_MAP`: Removed forward reference to undefined `EXP_170` (was NameError at import). Moved to after all experiment definitions.
4. `_EXP_MAP`: Expanded to map 175→EXP_170, 180→EXP_185, 185→EXP_185
5. `EXPERIMENTS`: Added EXP_185 (was excluded, 185 hardcoded tickers weren't being scored)
6. NaN fill: Added `tp_mult, sl_mult, trail_start_mult, trail_end_mult, vol_filter_pct`
7. `extra_agg`: Removed mode params promoted to group keys

### BACKTESTING

**backtest_tier_comparison.py** — 4 fixes:
1. `GRP_KEYS`: 7 → 15 columns
2. `DEPLOY_EXPERIMENTS`: Added 175, 180
3. `WALKFORWARD_FOLDS`: Re-keyed by deployed fold, added 180-185 chain
4. NaN fill + `extra_agg` cleanup

**backtest_top5_composites.py** — 3 fixes:
1. `GRP_KEYS`: 7 → 15 columns
2. `DEPLOY_EXPERIMENTS`: `{152, 160}` → `{152, 160, 170, 175, 180, 185}`
3. NaN fill + `extra_agg` cleanup

**backtest_composite_compare.py** — 3 fixes:
1. `grp_keys` (local): 7 → 15 columns
2. `DEPLOY_EXPERIMENTS`: `{152, 160}` → `{152, 160, 170, 175, 180, 185}`
3. NaN fill + `extra_agg` cleanup

**backtest_portfolio_v3.py** — 3 fixes:
1. `grp_keys` (local): 7 → 15 columns
2. `DEPLOY_EXPERIMENTS`: `{152, 160}` → `{152, 160, 170, 175, 180, 185}`
3. NaN fill + `extra_agg` cleanup

### ANALYSIS

**analyze_tier1_expanded.py** — 4 fixes:
1. `GRP_KEYS`: 7 → 15 columns
2. `DEPLOY_EXPERIMENTS`: Added 175
3. `WALKFORWARD_FOLDS`: Re-keyed by deployed fold (170→175, 180→185)
4. NaN fill + `extra_agg` cleanup

**analyze_tier1_variants.py** — 4 fixes:
1. `GRP_KEYS`: 7 → 15 columns
2. `DEPLOY_EXPERIMENTS`: `{152, 160, 170}` → `{152, 160, 170, 175, 180, 185}`
3. `WALKFORWARD_FOLDS`: Re-keyed, added 180-185 chain
4. NaN fill + `extra_agg` cleanup

**analyze_selection_methods.py** — 3 fixes:
1. `GRP_KEYS`: 7 → 15 columns
2. `DEPLOY_EXPERIMENTS`: `{152, 160}` → `{152, 160, 170, 175, 180, 185}`
3. NaN fill + `extra_agg` cleanup

**analyze_seed_agreement_sweep.py** — 2 fixes:
1. `GRP_KEYS`: 7 → 15 columns
2. Added NaN fill for mode-specific params before groupby

**deployed_wf_metrics.py** — 2 fixes:
1. `GRP_KEYS`: 5 → 15 columns (was also missing `tp_pct`, `sl_pct`)
2. NaN fill + moved mode params from aggregates to group keys

---

## What's NOT changed

- **WF optimizer scripts** (`optimize_wf_*.py`): These WRITE to BQ, not read+group. No seed grouping logic. Not affected.
- **Analysis scripts without seed grouping** (most `analyze_*.py`): Scripts that just query BQ without grouping by seed are not affected.
- **`audit_wf_v12_completeness.py`**: Has stale `DEPLOY_EXPERIMENTS_CURRENT` but it's intentionally a reference snapshot for comparison. Left as-is.

## Verification

After running any affected script in Colab, compare:
- Seed agreement numbers should be **lower or equal** (correct grouping splits previously-merged groups)
- Config counts may increase (finer groups that were previously merged)
- Limit entry `dip_pct` and `max_wait_minutes` should be consistent per config (not arbitrary `'first'`)

---

# Changelog: Round 2 Fixes (2026-06-19)

## Bug 4: `daily_score_combined.py` — Strategy H → Strategy B (Tier 1+)

**What changed:** Production scoring switched from Strategy H (full exploit, best per ticker × family, all modes) to Strategy B (Tier 1+: limit_entry only, both 152+160 gate, best per ticker across all families).

**Changes:**
1. Selection logic: gate on individual seed rows qualifying in both 152 AND 160, filter to limit_entry, pick best per ticker
2. Hardcoded configs cleared (ALAB, BE, OKLO, RKLB, PL, RDW, SMCI) — Tier 1+ is fully BQ-driven
3. Docstring and labels updated

**Bug fix within:** Initial gate implementation used seed GROUP means (only 2 tickers passed). Fixed to use individual seed rows (10 tickers pass), matching backtest `find_both_exp_tickers` logic.

---

## Bug 5: `backtest_portfolio_v2.py` — 3 fixes (HIGH)

1. `DEPLOY_EXPERIMENTS`: `{152, 160}` → `{152, 160, 170, 175, 180, 185}`
2. `MODE_SUFFIX`: Added `regime_adaptive`, `atr_sl`, `grace`, `limit_entry` (was missing all v12 modes)
3. Suffix stripping: Replaced brittle `rsplit` with proper `_strip_mode()` using MODE_SUFFIX (was missing ATRSL, GRACE, LIMIT, REGADP suffixes)

---

## Bug 6: `backtest_signal_filters.py` — 1 fix (HIGH)

1. `MODE_SUFFIX`: Added `regime_adaptive`, `atr_sl`, `grace`, `limit_entry`

---

## Bug 7: LUNR removed from deployed baselines (MEDIUM)

LUNR removed from hardcoded "A: Deployed" configs in:
- `backtest_tier_comparison.py` — `select_A_deployed()` (EXP_152 3→2 tickers)
- `analyze_selection_methods.py` — `select_A_deployed()` (EXP_152 3→2 tickers)
- `analyze_tier1_variants.py` — `select_A_deployed()`
- `deployed_wf_metrics.py` — `DEPLOYED_HARDCODED` list

---

## Bug 8: `deployed_wf_metrics.py` — stale BQ_DEPLOY_EXPERIMENTS (MEDIUM)

`BQ_DEPLOY_EXPERIMENTS`: `{152, 160}` → `{152, 160, 170, 175, 180, 185}`

---

## Bug 9: `analyze_composite_score.py` — stale DEPLOY_EXPERIMENTS (MEDIUM)

`DEPLOY_EXPERIMENTS`: `{152, 160, 170}` → `{152, 160, 170, 175, 180, 185}`

---

## Bug 10: `analyze_seed_agreement_sweep.py` — missing regime cols (MEDIUM)

Added `extra_agg` for `vix_threshold`, `tlt_crash`, `spy_ignore` as `'first'` aggregates in groupby. Previously these regime filter params were silently dropped.

---

## Stale comments updated

- `daily_micro_all.py` — "Strategy H traded tickers" → "Tier 1+ scored tickers"
- `daily_stitch_master.py` — same comment update
