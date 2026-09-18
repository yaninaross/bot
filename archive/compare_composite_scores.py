"""
compare_composite_scores.py — Compare 10 Composite Scores via Live Backtest
============================================================================
Queries BQ for WF optimization results, applies 10 different composite scores
to select best config per ticker, then runs LIVE TF model inference backtest
for each portfolio. Compares: avg PnL, total PnL, WR, drawdown, degradation.

Efficient: deduplicates scoring — each unique (ticker, model, seeds) combo
is scored ONCE, then reused across all 10 portfolios.

Run in Colab: paste as cell. Requires: TF, pandas_ta, numba, GCS, BQ.
"""

import os, io, gc, sys, json, subprocess, warnings, ctypes
from datetime import datetime, timedelta

os.environ['TF_USE_LEGACY_KERAS'] = '1'

import numpy as np, pandas as pd, pytz

warnings.filterwarnings('ignore')

def _release_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

try:
    import tf_keras
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tf_keras", "--quiet"])

try:
    import tensorflow as tf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "tensorflow[and-cuda]", "tf_keras", "pandas_ta",
                           "google-cloud-storage", "pyarrow", "gcsfs",
                           "numba", "--quiet"])
    import tensorflow as tf

try:
    import pandas_ta as ta
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "pandas_ta", "--no-deps", "--quiet"])
    import pandas_ta as ta

from numba import jit
from google.cloud import storage as gcs_storage, bigquery

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

import keras
print(f"  TF={tf.__version__}, Keras={keras.__version__}")
print(f"  Legacy Keras: {os.environ.get('TF_USE_LEGACY_KERAS', 'NOT SET')}")

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling1D,
    LayerNormalization, MultiHeadAttention, Add,
    Concatenate, Embedding, Flatten,
)
from tensorflow.keras.regularizers import l2

# ==============================================================================
# CONFIG
# ==============================================================================
VERSIONS = ['robust_wf_v11']
WINNING_SCORE = 'S8_mix'
TEST_START = '2026-04-01'
TEST_END   = '2026-06-09'

BUCKET_NAME    = os.environ.get("GCS_BUCKET", "")
BQ_TABLE       = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"
INFERENCE_DIR  = "Data/Inference_v2"
GCS_MODEL_PATH = "Algos/to_deploy/Saved_models_v2/UNIVERSAL/Direction"
STATS_DIR      = "Data/Inference_v2"

TIME_STEPS = 240; EMBED_DIM = 16; EXIT_TIME_MIN = 780
PST = pytz.timezone('America/Los_Angeles')
SLIPPAGE_PCT = 0.0015; VOL_LOOKBACK_DAYS = 30
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']
ROC_HORIZONS = [5, 30, 60, 120]
ASSET_COLS = (['Gap_Pct', 'RSI_14', 'RSI_60', 'RVOL', 'Compression',
               'ATR_14', 'Dist_to_PM_High', 'Bar_Is_Real'] +
              [f'ROC_{h}m' for h in ROC_HORIZONS] + MICRO_COLS)
CONTEXT_ASSETS = ['SPY', 'VIX', 'TLT', 'IWM']
CONTEXT_COLS = []
for _t in CONTEXT_ASSETS:
    CONTEXT_COLS.append(f'{_t}_Anchor_ROC')
    for _h in ROC_HORIZONS:
        CONTEXT_COLS.append(f'{_t}_ROC_{_h}m')
CONTEXT_COLS += ['Time_Pos']
ALL_COLS = ASSET_COLS + CONTEXT_COLS

TEACHERS_13 = ['BBAI','RCAT','LUNR','ASTS','ACHR','MVIS','WULF','CIFR','IREN','KEEL','HUT','CLSK','APLD']
TEACHERS_31 = ['CIFR','IREN','KEEL','HUT','APLD','WULF','RIOT','CLSK','BTBT','ALAB','MARA',
    'RCAT','ASTS','LUNR','BBAI','RKLB','SPIR','BKSY','MNTS','PLUG','BE','FCEL','FLNC',
    'OKLO','SMR','MVIS','BGDE','RDW','PL','VOYG','SMLR']

FAMILY_TEACHERS = {'152_XSECTOR': TEACHERS_13, '160_XSECTOR': TEACHERS_13,
    '170_XSECTOR': TEACHERS_13, '175_XSECTOR': TEACHERS_13, '185_XSECTOR': TEACHERS_31}

MODE_SUFFIX = {'static': '', 'atr': '_ATR', 'atr_adaptive': '_ATRADP',
               'regime_scaled': '_REGIME', 'trailing_decay': '_TRAIL'}
REGIME_TP_SCALE = np.array([0.70, 0.85, 1.00, 1.30, 1.60])
REGIME_SL_SCALE = np.array([0.60, 0.80, 1.00, 1.30, 1.80])

gcs_client = gcs_storage.Client()
bq_client  = bigquery.Client()
W = 120

def get_asset_index(model_id, ticker):
    teachers = FAMILY_TEACHERS.get(model_id, TEACHERS_13)
    return {t: i for i, t in enumerate(teachers)}.get(ticker, len(teachers))

def get_n_assets(model_id):
    return len(FAMILY_TEACHERS.get(model_id, TEACHERS_13)) + 1

# ==============================================================================
# 10 COMPOSITE SCORES — ALL use both VAL + TEST data
# ==============================================================================
def compute_all_scores(df):
    va = df['val_a_avg_pnl'].values
    va_wr = df['val_a_win_rate'].values
    va_n = df['val_a_count'].values.astype(float)
    vb = df['val_b_avg_pnl'].values
    vb_wr = df['val_b_win_rate'].values
    vb_n = df['val_b_count'].values.astype(float)

    combined_n = va_n + vb_n
    wavg_pnl = np.where(combined_n > 0, (va * va_n + vb * vb_n) / combined_n, 0)
    combined_wr = np.where(combined_n > 0, (va_wr * va_n + vb_wr * vb_n) / combined_n, 0)
    min_pnl = np.minimum(va, vb)
    min_wr = np.minimum(va_wr, vb_wr)
    harm_pnl = np.where((va > 0) & (vb > 0), 2 * va * vb / (va + vb), 0)
    geom_pnl = np.where((va > 0) & (vb > 0), np.sqrt(va * vb), 0)
    degradation = np.where(va > 0, np.abs(vb - va) / va, 1.0)
    consistency = np.clip(1 - degradation, 0.2, 1.0)

    va_n_mo = va_n / np.maximum(df['val_mo'].values, 0.5)
    vb_n_mo = vb_n / np.maximum(df['test_mo'].values, 0.5)
    combined_n_mo = (va_n_mo + vb_n_mo) / 2
    nq3 = np.exp(-0.5 * ((combined_n_mo - 3.0) / 2.5)**2)

    wr_both_60 = (va_wr >= 0.60) & (vb_wr >= 0.60)
    wr_both_50 = (va_wr >= 0.50) & (vb_wr >= 0.50)

    def both_above(thresh):
        return (va >= thresh) & (vb >= thresh)

    # --- Effective SL/TP for risk-adjusted scores ---
    sl_pct = df['sl_pct'].values.astype(float)
    tp_pct = df['tp_pct'].values.astype(float)
    sl_mult = df['sl_mult'].values.astype(float)
    tp_mult = df['tp_mult'].values.astype(float)
    sl_eff = np.where(sl_pct > 0, sl_pct, np.where(sl_mult > 0, sl_mult * 0.05, 0.08))
    tp_eff = np.where(tp_pct > 0, tp_pct, np.where(tp_mult > 0, tp_mult * 0.05, 0.16))

    scores = {}

    # --- Filters ---
    asym_filter = ((va >= 0.015) & (vb >= 0.010)) | ((va >= 0.010) & (vb >= 0.015))

    # --- Baselines (kept for reference) ---
    scores['S8_12'] = np.where(both_above(0.012) & wr_both_50,
        wavg_pnl**2 * combined_wr**2 * nq3 * consistency * np.log2(1 + combined_n), 0)
    scores['S8_mix'] = np.where(asym_filter & wr_both_50,
        wavg_pnl**2 * combined_wr**2 * nq3 * consistency * np.log2(1 + combined_n), 0)

    wr_both_65 = (va_wr >= 0.65) & (vb_wr >= 0.65)
    scores['S8_wr'] = np.where(both_above(0.010) & wr_both_65,
        wavg_pnl**2 * min_wr**2 * nq3 * consistency * np.log2(1 + combined_n), 0)

    # ===================================================================
    # R1: SORTINO — Risk-adjusted return, penalizes large SL vs PnL
    # ===================================================================
    def sortino_approx(avg_pnl, wr, sl):
        downside_var = (1.0 - wr) * sl**2
        downside_dev = np.sqrt(downside_var) + 1e-6
        return avg_pnl / downside_dev

    va_sortino = sortino_approx(va, va_wr, sl_eff)
    vb_sortino = sortino_approx(vb, vb_wr, sl_eff)
    blended_sortino = np.minimum(va_sortino, vb_sortino) * 0.6 + \
                      np.where(combined_n > 0, (va_sortino * va_n + vb_sortino * vb_n) / combined_n, 0) * 0.4

    r1_filter = (va >= 0.008) & (vb >= 0.008) & wr_both_50
    scores['R1'] = np.where(r1_filter,
        blended_sortino * nq3 * np.log2(1 + combined_n), 0)

    # ===================================================================
    # R2: KELLY — Kelly criterion sizing signal
    # ===================================================================
    def kelly_fraction(avg_pnl, wr, sl):
        avg_loss = np.maximum(sl, 0.001)
        avg_win = np.where(wr > 0.01, (avg_pnl + (1 - wr) * avg_loss) / wr, 0)
        avg_win = np.clip(avg_win, 0, 0.60)
        gain_loss = avg_win / avg_loss
        kelly = wr - (1 - wr) / np.maximum(gain_loss, 0.01)
        return np.clip(kelly, 0, 1)

    va_kelly = kelly_fraction(va, va_wr, sl_eff)
    vb_kelly = kelly_fraction(vb, vb_wr, sl_eff)
    combined_kelly = np.where((va_kelly > 0) & (vb_kelly > 0),
        np.sqrt(va_kelly * vb_kelly), 0)

    r2_filter = (va >= 0.008) & (vb >= 0.008) & (combined_kelly > 0.05)
    scores['R2'] = np.where(r2_filter,
        combined_kelly * wavg_pnl * nq3 * np.log2(1 + combined_n), 0)

    # ===================================================================
    # R3: ROBUST — Neighborhood density (anti-overfit)
    # Computed per ticker: fraction of configs with positive test PnL
    # ===================================================================
    robustness = pd.Series(0.3, index=df.index)
    for ticker, group in df.groupby('ticker'):
        n_total = len(group)
        if n_total < 3:
            continue
        n_good = ((group['val_a_avg_pnl'] >= 0.008) &
                  (group['val_b_avg_pnl'] >= 0.008) &
                  (group['val_a_win_rate'] >= 0.45) &
                  (group['val_b_win_rate'] >= 0.45)).sum()
        frac_good = n_good / n_total
        r_mult = 0.3 + 1.2 * frac_good
        robustness.loc[group.index] = r_mult
    robustness = robustness.values

    r3_filter = (va >= 0.010) & (vb >= 0.008) & wr_both_50
    scores['R3'] = np.where(r3_filter,
        wavg_pnl**2 * combined_wr**2 * robustness * nq3 * np.log2(1 + combined_n), 0)

    # ===================================================================
    # R4: DECAY — Asymmetric degradation with exponential penalty
    # ===================================================================
    ratio = np.where(va > 0, vb / va, 0.0)
    decay_score = np.where(
        ratio >= 1.0,
        np.minimum(1.0 + 0.2 * (ratio - 1.0), 1.2),
        np.exp(-2.0 * (1.0 - ratio))
    )
    wr_ratio = np.where(va_wr > 0, np.minimum(vb_wr / va_wr, 1.1), 0.5)
    both_stable = decay_score * wr_ratio

    r4_filter = (va >= 0.010) & (vb >= 0.005) & (va_wr >= 0.50)
    scores['R4'] = np.where(r4_filter,
        wavg_pnl * combined_wr * both_stable**2 * nq3 * np.log2(1 + combined_n), 0)

    # ===================================================================
    # R5: MINIMAX — Bayesian worst-case period with shrinkage
    # ===================================================================
    min_period_n = np.minimum(va_n, vb_n)
    shrinkage = np.clip(min_period_n / 10.0, 0.3, 1.0)
    raw_min_pnl = np.minimum(va, vb)
    raw_mean_pnl = wavg_pnl
    shrunk_min_pnl = shrinkage * raw_min_pnl + (1 - shrinkage) * raw_mean_pnl
    shrunk_min_wr = shrinkage * min_wr + (1 - shrinkage) * combined_wr
    min_n = np.minimum(va_n, vb_n)

    r5_filter = (shrunk_min_pnl >= 0.008) & (shrunk_min_wr >= 0.50)
    scores['R5'] = np.where(r5_filter,
        shrunk_min_pnl**2 * shrunk_min_wr**2 * nq3 * np.log2(1 + min_n + 1), 0)

    # ===================================================================
    # HYBRIDS: S8_mix filter + R-score ranking components
    # ===================================================================
    # H1: S8_mix filter + robustness multiplier
    scores['H1'] = np.where(asym_filter & wr_both_50,
        wavg_pnl**2 * combined_wr**2 * robustness * nq3 * consistency * np.log2(1 + combined_n), 0)

    # H2: S8_mix filter + Sortino ranking
    scores['H2'] = np.where(asym_filter & wr_both_50,
        blended_sortino * robustness * nq3 * np.log2(1 + combined_n), 0)

    # H3: S8_mix filter + Kelly ranking
    scores['H3'] = np.where(asym_filter & wr_both_50,
        combined_kelly * wavg_pnl * robustness * nq3 * np.log2(1 + combined_n), 0)

    # H4: S8_mix filter + asymmetric decay + robustness
    scores['H4'] = np.where(asym_filter & wr_both_50,
        wavg_pnl * combined_wr * both_stable**2 * robustness * nq3 * np.log2(1 + combined_n), 0)

    # H5: S8_mix filter + minimax ranking
    scores['H5'] = np.where(asym_filter & wr_both_50,
        shrunk_min_pnl**2 * shrunk_min_wr**2 * robustness * nq3 * np.log2(1 + min_n + 1), 0)

    # ===================================================================
    # HYBRIDS: S8_wr filter + R-score ranking components
    # ===================================================================
    wr_filter = both_above(0.010) & wr_both_65

    # W1: S8_wr filter + robustness
    scores['W1'] = np.where(wr_filter,
        wavg_pnl**2 * min_wr**2 * robustness * nq3 * consistency * np.log2(1 + combined_n), 0)

    # W2: S8_wr filter + Sortino
    scores['W2'] = np.where(wr_filter,
        blended_sortino * robustness * nq3 * np.log2(1 + combined_n), 0)

    # W3: S8_wr filter + Kelly
    scores['W3'] = np.where(wr_filter,
        combined_kelly * wavg_pnl * robustness * nq3 * np.log2(1 + combined_n), 0)

    # W4: S8_wr filter + decay + robustness
    scores['W4'] = np.where(wr_filter,
        wavg_pnl * min_wr * both_stable**2 * robustness * nq3 * np.log2(1 + combined_n), 0)

    # W5: S8_wr filter + minimax
    scores['W5'] = np.where(wr_filter,
        shrunk_min_pnl**2 * shrunk_min_wr**2 * robustness * nq3 * np.log2(1 + min_n + 1), 0)

    # ===================================================================
    # WIDER HYBRIDS: Relaxed filters + strong ranking
    # ===================================================================
    # X1: Wider asymmetric (0.8%/1.2%) + WR≥50% + robustness + Sortino
    wide_asym = ((va >= 0.012) & (vb >= 0.008)) | ((va >= 0.008) & (vb >= 0.012))
    scores['X1'] = np.where(wide_asym & wr_both_50,
        blended_sortino * robustness * nq3 * consistency * np.log2(1 + combined_n), 0)

    # X2: Wider asymmetric + soft WR (relax to 40% when PnL>3% and N<8)
    soft_wr = wr_both_50 | ((va >= 0.030) & (vb >= 0.020) & (combined_n < 8))
    scores['X2'] = np.where(asym_filter & soft_wr,
        wavg_pnl**2 * combined_wr**2 * robustness * nq3 * consistency * np.log2(1 + combined_n), 0)

    # X3: S8_12 filter + robustness + Sortino (middle ground)
    scores['X3'] = np.where(both_above(0.012) & wr_both_50,
        blended_sortino * robustness * nq3 * consistency * np.log2(1 + combined_n), 0)

    for name, vals in scores.items():
        df[name] = vals
    return df

SCORE_LABELS = {
    'S8_mix':'S8_mix',
    'S8_wr': 'S8_wr',
    'S8_12': 'S8_12',
    'H1':    'H1',
    'H2':    'H2',
    'H3':    'H3',
    'H4':    'H4',
    'H5':    'H5',
    'W1':    'W1',
    'W2':    'W2',
    'W3':    'W3',
    'W4':    'W4',
    'W5':    'W5',
    'X1':    'X1',
    'X2':    'X2',
    'X3':    'X3',
}

SCORE_SHORT = {
    'S8_mix': 'Asym filter, base',
    'S8_wr': 'WR>=65% filter, base',
    'S8_12': 'Both>1.2%+WR50, base',
    'H1': 'Asym + robust',
    'H2': 'Asym + Sortino + robust',
    'H3': 'Asym + Kelly + robust',
    'H4': 'Asym + decay + robust',
    'H5': 'Asym + minimax + robust',
    'W1': 'WR65 + robust',
    'W2': 'WR65 + Sortino + robust',
    'W3': 'WR65 + Kelly + robust',
    'W4': 'WR65 + decay + robust',
    'W5': 'WR65 + minimax + robust',
    'X1': 'Wide asym + Sortino+rob',
    'X2': 'Asym + soft WR + robust',
    'X3': 'S8_12 + Sortino + robust',
}

# ==============================================================================
# STEP 1: LOAD BQ DATA FOR ALL VERSIONS + SELECT PORTFOLIOS
# ==============================================================================
ver_str = ', '.join(VERSIONS)
print(f"{'='*W}")
print(f"  VERSION COMPARISON — {ver_str}")
print(f"  Composite: {WINNING_SCORE}  |  TEST: {TEST_START} → {TEST_END}")
print(f"{'='*W}")

ver_sql = ','.join(f"'{v}'" for v in VERSIONS)
q = f"""
SELECT ticker, experiment, seed, tpsl_mode, prob_thresh, model_version,
       opt_avg_pnl, opt_win_rate, opt_count,
       val_a_avg_pnl, val_a_win_rate, val_a_count,
       val_b_avg_pnl, val_b_win_rate, val_b_count,
       tp_pct, sl_pct, tp_mult, sl_mult, trail_start_mult, trail_end_mult,
       vix_threshold, tlt_crash, spy_ignore, vol_filter_pct,
       entry_minute,
       val_a_start, val_a_end, val_b_start, val_b_end,
       survived_a, survived_b, is_ensemble
FROM `{BQ_TABLE}`
WHERE model_version IN ({ver_sql})
  AND survived_a = TRUE
  AND val_b_start = '{TEST_START}'
"""
df_all = bq_client.query(q).to_dataframe()
df_all = df_all.drop_duplicates(subset=['model_version','ticker','experiment','seed','tpsl_mode','prob_thresh'], keep='first')

def mo(s, e):
    try: return max(0.5, (pd.Timestamp(e) - pd.Timestamp(s)).days / 30.44)
    except: return 1.0

ALL_SCORE_NAMES = list(SCORE_LABELS.keys())
portfolios = {}
score_portfolios = {}
df_scored = {}

for ver in VERSIONS:
    df_ver = df_all[df_all['model_version'] == ver].copy()
    df_ver['val_mo'] = df_ver.apply(lambda r: mo(r['val_a_start'], r['val_a_end']), axis=1)
    df_ver['test_mo'] = df_ver.apply(lambda r: mo(r['val_b_start'], r['val_b_end']), axis=1)
    df_ver = compute_all_scores(df_ver)
    df_scored[ver] = df_ver

    valid = df_ver[df_ver[WINNING_SCORE] > 0]
    if valid.empty:
        portfolios[ver] = pd.DataFrame()
    else:
        portfolios[ver] = valid.loc[valid.groupby('ticker')[WINNING_SCORE].idxmax()]

    n_by_score = []
    for score_name in ALL_SCORE_NAMES:
        sv = df_ver[df_ver[score_name] > 0]
        if sv.empty:
            score_portfolios[(ver, score_name)] = pd.DataFrame()
        else:
            score_portfolios[(ver, score_name)] = sv.loc[sv.groupby('ticker')[score_name].idxmax()]
        n_by_score.append(len(score_portfolios[(ver, score_name)]))

    print(f"  {ver}: {len(df_ver)} configs, {df_ver['ticker'].nunique()} tickers → "
          f"{len(portfolios[ver])} qualified ({WINNING_SCORE}), "
          f"max {max(n_by_score)} across all scores")

# BQ stats comparison — overlapping tickers
all_ver_tickers = [set(portfolios[v]['ticker']) for v in VERSIONS if not portfolios[v].empty]
common_tickers = sorted(set.intersection(*all_ver_tickers)) if all_ver_tickers else []

print(f"\n  {'='*W}")
print(f"  BQ STATS — {WINNING_SCORE} selected configs ({len(common_tickers)} common tickers)")
print(f"  {'='*W}")

if common_tickers:
    header_parts = [f"{'Ticker':<7}"]
    for v in VERSIONS:
        tag = v.split('_')[-1]
        header_parts.append(f"  {tag+' PnL':>8} {tag+' WR':>7} {tag+' N':>5}")
    header_parts.append(f"  {'Delta':>7}")
    print(f"\n  " + ''.join(header_parts))
    print(f"  " + "-" * (len(''.join(header_parts)) + 4))

    for ticker in common_tickers:
        parts = [f"  {ticker:<7}"]
        pnls = {}
        for v in VERSIONS:
            tag = v.split('_')[-1]
            row = portfolios[v][portfolios[v]['ticker'] == ticker]
            if len(row) > 0:
                pnl = float(row['val_b_avg_pnl'].iloc[0])
                wr = float(row['val_b_win_rate'].iloc[0])
                n = int(row['val_b_count'].iloc[0])
                pnls[v] = pnl
                parts.append(f"  {pnl*100:>+7.2f}% {wr*100:>6.0f}% {n:>5}")
            else:
                parts.append(f"  {'—':>8} {'—':>7} {'—':>5}")
        if len(pnls) == len(VERSIONS):
            delta = pnls[VERSIONS[-1]] - pnls[VERSIONS[0]]
            parts.append(f"  {delta*100:>+6.2f}%")
        print(''.join(parts))

    for v in VERSIONS:
        tag = v.split('_')[-1]
        p = portfolios[v]
        p_common = p[p['ticker'].isin(common_tickers)]
        if not p_common.empty:
            print(f"\n  {tag} avg (common): PnL={p_common['val_b_avg_pnl'].mean()*100:+.2f}%  "
                  f"WR={p_common['val_b_win_rate'].mean()*100:.0f}%  "
                  f"N={p_common['val_b_count'].mean():.1f}")

# Show per-version qualified assets
for ver in VERSIONS:
    tag = ver.split('_')[-1]
    wp = portfolios[ver]
    if wp.empty: continue
    print(f"\n  {'='*90}")
    print(f"  QUALIFIED ASSETS — {tag} + {WINNING_SCORE} ({len(wp)} tickers)")
    print(f"  {'='*90}")
    print(f"\n  {'Ticker':<7} {'Experiment':<20} {'Seed':>6} {'Mode':<16} {'Thresh':>7} "
          f"{'ValPnL':>8} {'TestPnL':>8} {'WR':>5} {'N':>4} {'TP':>6} {'SL':>6}")
    print(f"  " + "-" * 88)
    for _, r in wp.sort_values('ticker').iterrows():
        tp_str = f"{r.get('tp_pct',0)*100:.1f}%" if pd.notna(r.get('tp_pct')) and r.get('tp_pct',0) > 0 else f"{r.get('tp_mult',0):.1f}x"
        sl_str = f"{r.get('sl_pct',0)*100:.1f}%" if pd.notna(r.get('sl_pct')) and r.get('sl_pct',0) > 0 else f"{r.get('sl_mult',0):.1f}x"
        print(f"  {r['ticker']:<7} {r['experiment']:<20} {int(r['seed']):>6} {r['tpsl_mode']:<16} "
              f"{r['prob_thresh']:>6.3f} {r['val_a_avg_pnl']*100:>+7.2f}% {r['val_b_avg_pnl']*100:>+7.2f}% "
              f"{r['val_b_win_rate']*100:>4.0f}% {int(r['val_b_count']):>4} {tp_str:>6} {sl_str:>6}")
    print(f"\n  DEPLOY_TICKERS_{tag} = {sorted(wp['ticker'].tolist())}")

# ==============================================================================
# STEP 2: IDENTIFY ALL UNIQUE SCORING COMBOS NEEDED (from ALL versions + ALL scores)
# ==============================================================================
all_combos = set()

def _extract_combos(portfolio_df):
    for _, r in portfolio_df.iterrows():
        exp = r['experiment']
        mode_sfx = MODE_SUFFIX.get(r['tpsl_mode'], '')
        base_exp = exp[:-len(mode_sfx)] if mode_sfx and exp.endswith(mode_sfx) else exp
        seed = int(r['seed'])
        is_ens = bool(r['is_ensemble'])
        all_combos.add((r['ticker'], base_exp, seed, is_ens))

for ver in VERSIONS:
    for score_name in ALL_SCORE_NAMES:
        p = score_portfolios.get((ver, score_name), pd.DataFrame())
        if not p.empty:
            _extract_combos(p)

print(f"\n  {len(all_combos)} unique (ticker, model, seed) combos to score (across all scores)")

# ==============================================================================
# STEP 3: CORE FUNCTIONS (from backtest_portfolio_v3.py)
# ==============================================================================
_master_cache = {}
_model_cache = {}
_stats_cache = {}
_market_context_cache = None

def load_master(ticker, start_date=None):
    cache_key = (ticker, start_date)
    if cache_key in _master_cache:
        return _master_cache[cache_key]
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    uri = f"gs://{BUCKET_NAME}/{INFERENCE_DIR}/{clean}_master_inference.parquet"
    try:
        df = pd.read_parquet(uri)
        df = df.reset_index()
        dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        if dt_cols:
            df['Datetime'] = pd.to_datetime(df[dt_cols[0]], utc=True)
        else:
            df['Datetime'] = pd.to_datetime(df[df.columns[0]], unit='ms', utc=True)
        df = df.set_index('Datetime')
        df = df[~df.index.duplicated(keep='first')]
        std_map = {'Total_vol':'Total_Vol','Trade_count':'Trade_Count','Volume_delta':'Volume_Delta',
                    'Delta_pct':'Delta_Pct','Obi':'OBI','Bsize':'bsize','Asize':'asize',
                    'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'}
        lower_map = {k.lower():v for k,v in std_map.items()}
        df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip()) for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()
        for c in MICRO_COLS:
            if c not in df.columns: df[c] = 0.0
        df = df[df.index.notna()].sort_index()
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date, tz='UTC') - pd.Timedelta(days=30)]
        _master_cache[cache_key] = df
        return df
    except Exception as e:
        return None

def heal_data(df):
    df = df.resample('1min').asfreq()
    for c in ['Open','High','Low','Close']:
        if c in df.columns: df[c] = df[c].ffill()
    for c in ['Volume','Total_Vol','Volume_Delta','Trade_Count','Delta_Pct','Spread','OBI','bsize','asize']:
        if c in df.columns: df[c] = df[c].fillna(0)
    return df

def build_market_context(start_date=None):
    global _market_context_cache
    if _market_context_cache is not None: return _market_context_cache
    print("  Building market context...")
    trim_from = pd.Timestamp(start_date, tz='UTC') - pd.Timedelta(days=7) if start_date else None
    frames = []
    for ticker in CONTEXT_ASSETS:
        clean = TICKER_FILE_MAP.get(ticker, ticker)
        uri = f"gs://{BUCKET_NAME}/{INFERENCE_DIR}/{clean}_master_inference.parquet"
        try:
            d = pd.read_parquet(uri)
        except Exception:
            continue
        d = d.reset_index()
        dt_cols = [c for c in d.columns if pd.api.types.is_datetime64_any_dtype(d[c])]
        if dt_cols:
            d['Datetime'] = pd.to_datetime(d[dt_cols[0]], utc=True)
        else:
            d['Datetime'] = pd.to_datetime(d[d.columns[0]], unit='ms', utc=True)
        d = d.set_index('Datetime').sort_index()
        d = d[~d.index.duplicated(keep='first')]
        std_map = {'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'}
        lower_map = {k.lower():v for k,v in std_map.items()}
        d.columns = [lower_map.get(str(c).strip().lower(), str(c).strip()) for c in d.columns]
        if trim_from is not None:
            d = d[d.index >= trim_from]
        d = d[['Close']].copy()
        d = d.resample('1min').asfreq()
        d['Close'] = d['Close'].ffill()
        daily_c = d['Close'].resample('D').last().ffill().shift(1).reindex(d.index, method='ffill')
        cols = {f'{ticker}_Anchor_ROC': (d['Close'] - daily_c) / daily_c}
        for h in ROC_HORIZONS:
            cols[f'{ticker}_ROC_{h}m'] = d['Close'].pct_change(h).fillna(0)
        frames.append(pd.DataFrame(cols, index=d.index))
        del d, daily_c; _release_memory()
        print(f"    {ticker} loaded")
    _market_context_cache = pd.concat(frames, axis=1, sort=True).ffill().fillna(0)
    del frames; _release_memory()
    return _market_context_cache

def engineer_features(df, market_df):
    daily_c = df['Close'].resample('D').last().ffill().shift(1).reindex(df.index, method='ffill')
    df['Gap_Pct'] = (df['Open'] - daily_c) / daily_c
    df['PM_High'] = df.groupby(df.index.date)['High'].cummax()
    df['Dist_to_PM_High'] = (df['Close'] - df['PM_High']) / (df['PM_High'] + 1e-9)
    df['RSI_14'] = ta.rsi(df['Close'], length=14)
    df['RSI_60'] = ta.rsi(df['Close'], length=60)
    vol_real = df['Volume'].where(df['Volume'] > 0)
    df['RVOL'] = df['Volume'] / vol_real.rolling(50, min_periods=5).mean().ffill().clip(lower=1e-9)
    real_mask = df['Volume'] > 0
    atr_raw = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['ATR_14'] = atr_raw.where(real_mask).ffill().fillna(0)
    df['Compression'] = (df['High'] - df['Low']) / (df['ATR_14'] + 1e-9)
    for h in ROC_HORIZONS:
        df[f'ROC_{h}m'] = df['Close'].pct_change(h)
    df = df.join(market_df, how='left').ffill().fillna(0)
    df_pst = df.index.tz_convert(PST)
    mins = df_pst.hour * 60 + df_pst.minute
    df['Time_Pos'] = np.clip((mins - 390) / 390, 0, 1)
    df['Bar_Is_Real'] = (df['Volume'] > 0).astype(float)
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)

def load_scaling_stats(model_id):
    if model_id in _stats_cache: return _stats_cache[model_id]
    blob = gcs_client.bucket(BUCKET_NAME).blob(f"{STATS_DIR}/scaling_stats_experiment_{model_id}.json")
    raw = json.loads(blob.download_as_bytes())
    _stats_cache[model_id] = raw
    return raw

def apply_scaling(df, stats):
    df_s = df.copy()
    for col in ALL_COLS:
        if col in ('Time_Pos','Bar_Is_Real'): continue
        if col in stats:
            df_s[col] = ((df_s[col]-stats[col]['med'])/stats[col]['iqr']).clip(-10,10).astype('float32')
    return df_s

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.4):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    res = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return Add()([x, res])

def build_model(n_context_cols, n_assets, n_asset_cols=None):
    n_ac = n_asset_cols or len(ASSET_COLS)
    in_a = Input(shape=(TIME_STEPS, n_ac), name='asset_in')
    in_c = Input(shape=(n_context_cols,), name='context_in')
    in_id = Input(shape=(1,), dtype=tf.int32, name='asset_id')
    x = transformer_encoder(Dense(128)(in_a), 128, 4, 128)
    x = GlobalAveragePooling1D()(transformer_encoder(x, 128, 4, 128))
    y = Dense(128, activation='relu')(LayerNormalization()(in_c))
    e = Flatten()(Embedding(n_assets, EMBED_DIM, name='asset_embed')(in_id))
    e = Dense(32, activation='relu')(e)
    z = Concatenate()([x, Dropout(0.5)(y), e])
    z = Dense(128, activation='relu', kernel_regularizer=l2(5e-3))(z)
    out = Dense(1, activation='sigmoid', dtype='float32')(Dropout(0.5)(z))
    return Model(inputs=[in_a, in_c, in_id], outputs=out)

def get_model(model_id, seed):
    key = (model_id, seed)
    if key in _model_cache: return _model_cache[key]
    blob_path = f"{GCS_MODEL_PATH}/UNIVERSAL_S{seed}_M{model_id}.h5"
    local_path = f"/tmp/UNIVERSAL_S{seed}_M{model_id}.h5"
    if not os.path.exists(local_path):
        gcs_client.bucket(BUCKET_NAME).blob(blob_path).download_to_filename(local_path)
    model = build_model(len(CONTEXT_COLS), get_n_assets(model_id), n_asset_cols=len(ASSET_COLS))
    model.load_weights(local_path)
    _model_cache[key] = model
    return model

# ==============================================================================
# TRADE SIMULATION (Numba)
# ==============================================================================
@jit(nopython=True)
def run_single_trade(prices, highs, lows, date_ints, time_mins,
                     entry_idx, tp, sl, exit_time):
    n = len(prices)
    if entry_idx >= n - 1: return 0.0, date_ints[entry_idx], 'EOD'
    ep = prices[entry_idx]; ed = date_ints[entry_idx]
    tp_p = ep * (1 + tp); sl_p = ep * (1 - sl)
    for k in range(entry_idx + 1, n):
        if date_ints[k] != ed: return (prices[k-1] - ep) / ep, ed, 'EOD'
        if lows[k] <= sl_p and highs[k] >= tp_p:
            return (-sl if np.random.random() < 0.5 else tp), ed, 'BOTH'
        if lows[k] <= sl_p: return -sl, ed, 'SL'
        if highs[k] >= tp_p: return tp, ed, 'TP'
        if time_mins[k] >= exit_time: return (prices[k] - ep) / ep, ed, 'EOD'
    return (prices[-1] - ep) / ep, ed, 'EOD'

def compute_atr(df_ohlcv, period=14):
    daily = df_ohlcv.resample('D').agg({'High':'max','Low':'min','Close':'last'}).dropna()
    pc = daily['Close'].shift(1)
    tr = pd.concat([daily['High']-daily['Low'],(daily['High']-pc).abs(),(daily['Low']-pc).abs()],axis=1).max(axis=1)
    return (tr.rolling(period).mean() / daily['Close']).reindex(df_ohlcv.index, method='ffill')

def compute_adaptive_atr(df_ohlcv, short=5, long=22, base=14):
    daily = df_ohlcv.resample('D').agg({'High':'max','Low':'min','Close':'last'}).dropna()
    pc = daily['Close'].shift(1)
    tr = pd.concat([daily['High']-daily['Low'],(daily['High']-pc).abs(),(daily['Low']-pc).abs()],axis=1).max(axis=1)
    atr_b = tr.rolling(base).mean()
    accel = (tr.rolling(short).mean() / tr.rolling(long).mean()).clip(0.5, 3.0)
    return ((atr_b * accel) / daily['Close']).reindex(df_ohlcv.index, method='ffill')

# ==============================================================================
# STEP 4: SCORE ALL UNIQUE COMBOS + LOAD OHLCV
# ==============================================================================
print(f"\n  Scoring {len(all_combos)} unique combos + loading OHLCV...")
market_df = build_market_context(start_date=TEST_START)

combo_signals = {}
ticker_ohlcv = {}
ticker_atr = {}
ticker_atr_adp = {}

all_tickers_needed = set(t for t, _, _, _ in all_combos)

ohlcv_trim = pd.Timestamp(TEST_START, tz='UTC') - pd.Timedelta(days=30)
for ticker in sorted(all_tickers_needed):
    df_raw = load_master(ticker, start_date=TEST_START)
    if df_raw is None: continue
    df_raw = df_raw[df_raw.index >= ohlcv_trim]
    df_raw = heal_data(df_raw.sort_index())
    df_pst = df_raw.copy()
    df_pst.index = df_pst.index.tz_convert(PST)
    test_s = pd.Timestamp(TEST_START, tz=PST)
    test_e = pd.Timestamp(TEST_END, tz=PST) + timedelta(days=1)
    ohlcv_test = df_pst[(df_pst.index >= test_s) & (df_pst.index < test_e)]
    if not ohlcv_test.empty:
        ticker_ohlcv[ticker] = {
            'prices': ohlcv_test['Close'].values, 'highs': ohlcv_test['High'].values,
            'lows': ohlcv_test['Low'].values, 'volumes': ohlcv_test['Volume'].values,
            'dates': (ohlcv_test.index.year*10000+ohlcv_test.index.month*100+ohlcv_test.index.day).values.astype(np.int64),
            'times': (ohlcv_test.index.hour*60+ohlcv_test.index.minute).values.astype(np.int64),
            'index': ohlcv_test.index,
        }
        ticker_atr[ticker] = compute_atr(df_pst)
        ticker_atr_adp[ticker] = compute_adaptive_atr(df_pst)
    del df_raw, df_pst; _release_memory()

_master_cache.clear(); _release_memory()
print(f"  {len(ticker_ohlcv)} tickers with OHLCV loaded")

test_start_utc = pd.Timestamp(TEST_START, tz='UTC')
test_end_utc = pd.Timestamp(TEST_END, tz='UTC') + timedelta(days=1)
trim_start = test_start_utc - pd.Timedelta(days=5)

for ci, (ticker, base_exp, seed, is_ens) in enumerate(sorted(all_combos)):
    print(f"  [{ci+1}/{len(all_combos)}] {ticker}/{base_exp}/S{'ENS' if is_ens else seed}...", end=' ')
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    uri = f"gs://{BUCKET_NAME}/{INFERENCE_DIR}/{clean}_master_inference.parquet"
    try:
        df_raw = pd.read_parquet(uri)
    except Exception:
        print("SKIP"); continue
    df_raw = df_raw.reset_index()
    dt_cols = [c for c in df_raw.columns if pd.api.types.is_datetime64_any_dtype(df_raw[c])]
    if dt_cols:
        df_raw['Datetime'] = pd.to_datetime(df_raw[dt_cols[0]], utc=True)
    else:
        df_raw['Datetime'] = pd.to_datetime(df_raw[df_raw.columns[0]], unit='ms', utc=True)
    df_raw = df_raw.set_index('Datetime').sort_index()
    df_raw = df_raw[~df_raw.index.duplicated(keep='first')]
    std_map = {'Total_vol':'Total_Vol','Trade_count':'Trade_Count','Volume_delta':'Volume_Delta',
                'Delta_pct':'Delta_Pct','Obi':'OBI','Bsize':'bsize','Asize':'asize',
                'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'}
    lower_map = {k.lower():v for k,v in std_map.items()}
    df_raw.columns = [lower_map.get(str(c).strip().lower(), str(c).strip()) for c in df_raw.columns]
    df_raw = df_raw.loc[:, ~df_raw.columns.duplicated(keep='first')].copy()
    for c in MICRO_COLS:
        if c not in df_raw.columns: df_raw[c] = 0.0
    df_raw = df_raw[df_raw.index >= trim_start]
    df_raw = heal_data(df_raw)
    df_feat = engineer_features(df_raw.copy(), market_df)
    stats = load_scaling_stats(base_exp)
    df_scaled = apply_scaling(df_feat, stats)
    del df_raw, df_feat; _release_memory()

    asset_id = get_asset_index(base_exp, ticker)
    df_test = df_scaled[(df_scaled.index >= test_start_utc) & (df_scaled.index < test_end_utc)]
    if len(df_test) == 0: del df_scaled; _release_memory(); print("SKIP"); continue

    for em in [360, 365, 370, 375, 380, 385]:
        pst_idx = df_test.index.tz_convert(PST)
        mins = pst_idx.hour * 60 + pst_idx.minute
        positions = [df_scaled.index.get_loc(df_test.index[i])
                     for i in range(len(df_test))
                     if mins[i] == em and df_scaled.index.get_loc(df_test.index[i]) >= TIME_STEPS
                     and ('Volume' not in df_scaled.columns or float(df_scaled['Volume'].iloc[df_scaled.index.get_loc(df_test.index[i])]) > 0)]
        if not positions: continue
        XA = np.stack([df_scaled[ASSET_COLS].values[idx-TIME_STEPS:idx] for idx in positions]).astype(np.float32)
        XC = np.stack([df_scaled[CONTEXT_COLS].values[idx-1] for idx in positions]).astype(np.float32)
        XID = np.full((len(positions), 1), asset_id, dtype=np.int32)
        if is_ens:
            seed_list = list(range(int(base_exp.split('_')[0])*100+1, int(base_exp.split('_')[0])*100+16))
        else:
            seed_list = [seed]
        all_probs = []
        for s in seed_list:
            try:
                m = get_model(base_exp, s)
                all_probs.append(m.predict({'asset_in':XA,'context_in':XC,'asset_id':XID},batch_size=64,verbose=0).flatten())
            except Exception as e:
                if ci < 2:
                    import traceback
                    print(f"\n    MODEL FAIL S{s}: {type(e).__name__}: {e}")
                    traceback.print_exc()
                continue
        if not all_probs: continue
        probs = np.mean(all_probs, axis=0)
        sigs = {}
        for i, idx in enumerate(positions):
            ts = df_scaled.index[idx].tz_convert(PST)
            d = ts.year*10000+ts.month*100+ts.day
            sigs[d] = float(probs[i])
        combo_signals[(ticker, base_exp, seed, is_ens, em)] = sigs
        del XA, XC
    del df_scaled, df_test
    tf.keras.backend.clear_session(); _model_cache.clear(); _release_memory()
    print("done")

print(f"\n  {len(combo_signals)} signal sets computed")

# ==============================================================================
# STEP 5: SIMULATE REAL TRADES FOR EACH PORTFOLIO
# ==============================================================================
STARTING_CAPITAL = 200_000

def simulate_portfolio_real(portfolio_df, combo_signals, ticker_ohlcv, ticker_atr, ticker_atr_adp):
    trades = []
    for _, cfg in portfolio_df.iterrows():
        ticker = cfg['ticker']
        if ticker not in ticker_ohlcv: continue
        ohlcv = ticker_ohlcv[ticker]
        exp = cfg['experiment']
        mode_sfx = MODE_SUFFIX.get(cfg['tpsl_mode'], '')
        base_exp = exp[:-len(mode_sfx)] if mode_sfx and exp.endswith(mode_sfx) else exp
        seed = int(cfg['seed']); is_ens = bool(cfg['is_ensemble'])
        em = int(cfg.get('entry_minute', 380) if pd.notna(cfg.get('entry_minute')) else 380)
        prob_thresh = float(cfg['prob_thresh'])
        mode = cfg['tpsl_mode']

        combo_key = (ticker, base_exp, seed, is_ens, em)
        if combo_key not in combo_signals: continue
        signals = combo_signals[combo_key]
        last_traded = -1

        for d_int, prob in sorted(signals.items()):
            if prob < prob_thresh: continue
            if d_int == last_traded: continue
            entry_idx_arr = np.where((ohlcv['dates'] == d_int) & (ohlcv['times'] == em))[0]
            if len(entry_idx_arr) == 0: continue
            entry_idx = entry_idx_arr[0]
            if ohlcv['volumes'][entry_idx] == 0: continue

            if mode in ('atr', 'trailing_decay'):
                atr_s = ticker_atr.get(ticker)
            elif mode == 'atr_adaptive':
                atr_s = ticker_atr_adp.get(ticker)
            else:
                atr_s = None

            if mode in ('atr', 'atr_adaptive'):
                atr_val = 0.05
                if atr_s is not None:
                    ts = ohlcv['index'][entry_idx]
                    prior = atr_s.index[atr_s.index <= ts]
                    if len(prior) > 0:
                        v = float(atr_s.iloc[atr_s.index.get_loc(prior[-1])])
                        if not np.isnan(v) and v > 0: atr_val = v
                tp = min(max(atr_val * float(cfg.get('tp_mult') or 3), 0.02), 0.60)
                sl = min(max(atr_val * float(cfg.get('sl_mult') or 1.5), 0.01), 0.25)
            elif mode == 'regime_scaled':
                tp = float(cfg.get('tp_pct') or 0.16)
                sl = float(cfg.get('sl_pct') or 0.08)
            else:
                tp = float(cfg.get('tp_pct') or 0.16)
                sl = float(cfg.get('sl_pct') or 0.08)

            pnl, trade_date, exit_type = run_single_trade(
                ohlcv['prices'], ohlcv['highs'], ohlcv['lows'],
                ohlcv['dates'], ohlcv['times'], entry_idx, tp, sl, EXIT_TIME_MIN)
            pnl = float(pnl) - 2 * SLIPPAGE_PCT
            trades.append({'ticker': ticker, 'date_int': int(trade_date), 'prob': prob,
                           'pnl': pnl, 'mode': mode, 'exit': exit_type,
                           'tp': tp, 'sl': sl, 'entry_price': float(ohlcv['prices'][entry_idx])})
            last_traded = d_int
    return pd.DataFrame(trades)

score_trade_results = {}

print(f"\n  Simulating REAL trades for {len(VERSIONS)} version portfolios ({WINNING_SCORE})...")

results = {}
portfolio_trades = {}

for ver in VERSIONS:
    tag = ver.split('_')[-1]
    p = portfolios[ver]
    if p.empty:
        results[ver] = {'trades':0,'avg_pnl':0,'wr':0,'total_pnl':0,'tickers':0}
        portfolio_trades[ver] = pd.DataFrame()
        continue
    tdf = simulate_portfolio_real(p, combo_signals, ticker_ohlcv, ticker_atr, ticker_atr_adp)
    portfolio_trades[ver] = tdf
    if tdf.empty:
        results[ver] = {'trades':0,'avg_pnl':0,'wr':0,'total_pnl':0,'tickers':0}
        continue
    n = len(tdf)
    results[ver] = {
        'trades': n, 'avg_pnl': tdf['pnl'].mean(), 'wr': (tdf['pnl']>0).mean(),
        'total_pnl': tdf['pnl'].sum(), 'tickers': tdf['ticker'].nunique(),
        'trades_per_ticker': n / max(tdf['ticker'].nunique(), 1),
        'n_sl': (tdf['exit']=='SL').sum(), 'n_tp': (tdf['exit']=='TP').sum(),
    }
    print(f"  {tag:<8}: {n:>4} trades, avg {tdf['pnl'].mean()*100:>+5.2f}%, "
          f"WR {(tdf['pnl']>0).mean()*100:.0f}%, {tdf['ticker'].nunique()} tickers")

# ==============================================================================
# STEP 5c: MULTI-SCORE BACKTEST — compare all composite scores
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  MULTI-SCORE COMPARISON — Backtest each composite score's portfolio selection")
print(f"  Signal reuse: each (ticker, model, seed) scored once, portfolios differ by config selection")
print(f"{'='*W}")

for ver in VERSIONS:
    tag = ver.split('_')[-1]
    print(f"\n  --- {tag} ---")
    print(f"  {'Score':<10} {'Description':<30} {'Tickers':>7} {'Trades':>6} "
          f"{'AvgPnL':>8} {'WR':>5} {'TotPnL':>9} {'TP':>4} {'SL':>4} {'EOD':>4}")
    print(f"  " + "-" * 100)

    score_results_ver = []
    for score_name in ALL_SCORE_NAMES:
        p = score_portfolios.get((ver, score_name), pd.DataFrame())
        if p.empty:
            score_results_ver.append({'score': score_name, 'trades': 0, 'avg_pnl': 0,
                                      'wr': 0, 'total_pnl': 0, 'tickers': 0})
            continue
        tdf_s = simulate_portfolio_real(p, combo_signals, ticker_ohlcv, ticker_atr, ticker_atr_adp)
        if tdf_s.empty:
            score_results_ver.append({'score': score_name, 'trades': 0, 'avg_pnl': 0,
                                      'wr': 0, 'total_pnl': 0, 'tickers': 0})
            continue
        n = len(tdf_s)
        sr = {
            'score': score_name,
            'trades': n,
            'avg_pnl': tdf_s['pnl'].mean(),
            'wr': (tdf_s['pnl'] > 0).mean(),
            'total_pnl': tdf_s['pnl'].sum(),
            'tickers': tdf_s['ticker'].nunique(),
            'n_tp': (tdf_s['exit'] == 'TP').sum(),
            'n_sl': (tdf_s['exit'] == 'SL').sum(),
            'n_eod': (tdf_s['exit'] == 'EOD').sum(),
        }
        score_results_ver.append(sr)
        score_trade_results[(ver, score_name)] = tdf_s

    score_results_ver.sort(key=lambda x: -x['total_pnl'])
    for sr in score_results_ver:
        desc = SCORE_SHORT.get(sr['score'], '')[:28]
        if sr['trades'] == 0:
            print(f"  {sr['score']:<10} {desc:<30} {'—':>7}")
            continue
        print(f"  {sr['score']:<10} {desc:<30} {sr['tickers']:>7} {sr['trades']:>6} "
              f"{sr['avg_pnl']*100:>+7.2f}% {sr['wr']*100:>4.0f}% "
              f"{sr['total_pnl']*100:>+8.1f}% "
              f"{sr.get('n_tp',0):>4} {sr.get('n_sl',0):>4} {sr.get('n_eod',0):>4}")

    if score_results_ver:
        best = score_results_ver[0]
        worst_pos = [s for s in score_results_ver if s['total_pnl'] > 0]
        print(f"\n  BEST SCORE: {best['score']} — {best['trades']} trades, "
              f"avg {best['avg_pnl']*100:+.2f}%, total {best['total_pnl']*100:+.1f}%, "
              f"WR {best['wr']*100:.0f}%")
        if len(worst_pos) > 1:
            gap = (best['total_pnl'] - score_results_ver[1]['total_pnl']) * 100
            print(f"  Gap to 2nd ({score_results_ver[1]['score']}): {gap:+.1f}%")
        pos = len([s for s in score_results_ver if s['total_pnl'] > 0])
        neg = len([s for s in score_results_ver if s['total_pnl'] < 0 and s['trades'] > 0])
        print(f"  {pos} scores positive, {neg} scores negative")

    # Cap 2/day per score
    print(f"\n  --- {tag} Cap 2/day per score ---")
    print(f"  {'Score':<10} {'Trades':>6} {'Return':>9} {'Final$':>10} {'WR':>5} {'MaxDD':>7}")
    print(f"  " + "-" * 55)

    cap2_results = []
    for score_name in ALL_SCORE_NAMES:
        tdf_s = score_trade_results.get((ver, score_name), pd.DataFrame())
        if tdf_s.empty:
            continue
        tdf_s2 = tdf_s.copy()
        tdf_s2['date'] = pd.to_datetime(tdf_s2['date_int'].astype(str), format='%Y%m%d')
        capped = tdf_s2.groupby('date').apply(
            lambda g: g.nlargest(2, 'prob') if len(g) > 2 else g
        ).reset_index(drop=True)

        equity = STARTING_CAPITAL
        eq_curve = [equity]
        for day, group in sorted(capped.groupby('date'), key=lambda x: x[0]):
            n_taken = len(group)
            alloc = equity / n_taken
            day_pnl = sum(alloc * t['pnl'] for _, t in group.iterrows())
            equity += day_pnl
            eq_curve.append(equity)

        final = equity
        ret = (final - STARTING_CAPITAL) / STARTING_CAPITAL
        eq_arr = np.array(eq_curve)
        dd = ((eq_arr - np.maximum.accumulate(eq_arr)) / np.maximum.accumulate(eq_arr)).min()
        wr = (capped['pnl'] > 0).mean()
        cap2_results.append({'score': score_name, 'trades': len(capped), 'ret': ret,
                             'final': final, 'wr': wr, 'dd': dd})

    cap2_results.sort(key=lambda x: -x['ret'])
    for cr in cap2_results:
        print(f"  {cr['score']:<10} {cr['trades']:>6} {cr['ret']*100:>+8.1f}% "
              f"${cr['final']/1000:>8.1f}K {cr['wr']*100:>4.0f}% {cr['dd']*100:>+6.1f}%")

    if cap2_results:
        best_cap2 = cap2_results[0]
        print(f"\n  BEST (Cap2): {best_cap2['score']} — {best_cap2['ret']*100:+.1f}% return, "
              f"${best_cap2['final']/1000:.1f}K final, DD {best_cap2['dd']*100:.1f}%")

# ==============================================================================
# STEP 6: CAPITAL ALLOCATION STRATEGIES
# ==============================================================================

ALLOC_STRATEGIES = [
    'equal', 'conviction', 'cap2', 'cap3',
    'hist_pnl', 'kelly', 'risk_parity', 'regime', 'montecarlo', 'logistic',
]

ALLOC_SHORT = {
    'equal': '1-Equal weight',
    'conviction': '2-Conviction (prob)',
    'cap2': '3a-Cap 2/day',
    'cap3': '3b-Cap 3/day',
    'hist_pnl': '4-Hist PnL weight',
    'kelly': '5-Kelly sizing',
    'risk_parity': '6-Risk parity (ATR)',
    'regime': '7-Regime (VIX scale)',
    'montecarlo': '8-Monte Carlo opt',
    'logistic': '9-Logistic ML',
}

def build_ticker_meta(portfolio_df):
    meta = {}
    for _, cfg in portfolio_df.iterrows():
        va_n = float(cfg.get('val_a_count', 1) or 1)
        vb_n = float(cfg.get('val_b_count', 1) or 1)
        total_n = va_n + vb_n
        meta[cfg['ticker']] = {
            'avg_pnl': (cfg['val_a_avg_pnl'] * va_n + cfg['val_b_avg_pnl'] * vb_n) / max(total_n, 1),
            'win_rate': (cfg['val_a_win_rate'] * va_n + cfg['val_b_win_rate'] * vb_n) / max(total_n, 1),
            'sl': float(cfg['sl_pct']) if float(cfg['sl_pct']) > 0 else float(cfg.get('sl_mult', 0)) * 0.05 or 0.08,
            'tp': float(cfg['tp_pct']) if float(cfg['tp_pct']) > 0 else float(cfg.get('tp_mult', 0)) * 0.05 or 0.16,
        }
    return meta

def build_vix_daily(regime_data=None):
    if regime_data is not None:
        return regime_data
    vix_daily = {}
    clean = TICKER_FILE_MAP.get('VIX', 'VIX')
    uri = f"gs://{BUCKET_NAME}/{INFERENCE_DIR}/{clean}_master_inference.parquet"
    try:
        d = pd.read_parquet(uri)
        d = d.reset_index()
        dt_cols = [c for c in d.columns if pd.api.types.is_datetime64_any_dtype(d[c])]
        if dt_cols:
            d['Datetime'] = pd.to_datetime(d[dt_cols[0]], utc=True)
        else:
            d['Datetime'] = pd.to_datetime(d[d.columns[0]], unit='ms', utc=True)
        d = d.set_index('Datetime').sort_index()
        std_map = {'close': 'Close'}
        d.columns = [std_map.get(str(c).strip().lower(), str(c).strip()) for c in d.columns]
        d_pst = d.index.tz_convert(PST)
        mins = d_pst.hour * 60 + d_pst.minute
        at_entry = d[(mins >= 375) & (mins <= 390)]
        daily_vix = at_entry['Close'].resample('D').median().dropna()
        for dt, v in daily_vix.items():
            d_int = int(dt.strftime('%Y%m%d'))
            vix_daily[d_int] = float(v)
        del d, at_entry, daily_vix; _release_memory()
    except Exception:
        pass
    return vix_daily

def get_atr_for_ticker_date(ticker, date_int, ticker_atr_dict):
    atr_s = ticker_atr_dict.get(ticker)
    if atr_s is None:
        return 0.05
    date_str = str(date_int)
    ts = pd.Timestamp(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}", tz=PST)
    prior = atr_s.index[atr_s.index <= ts]
    if len(prior) > 0:
        v = float(atr_s.iloc[-1])
        if not np.isnan(v) and v > 0:
            return v
    return 0.05


def simulate_capital_alloc(trades_df, starting_capital, strategy='equal',
                           ticker_meta=None, ticker_atr_dict=None, vix_daily=None):
    if trades_df.empty:
        return pd.DataFrame(), starting_capital

    tdf = trades_df.copy()
    tdf['date'] = pd.to_datetime(tdf['date_int'].astype(str), format='%Y%m%d')
    daily_groups = tdf.groupby('date')

    equity = starting_capital
    curve = []

    for date, group in sorted(daily_groups, key=lambda x: x[0]):
        trades = group.to_dict('records')
        n = len(trades)
        d_int = trades[0]['date_int']

        if strategy == 'equal':
            weights = [1.0 / n] * n

        elif strategy == 'conviction':
            probs = [t['prob'] for t in trades]
            total_p = sum(probs)
            weights = [p / total_p for p in probs] if total_p > 0 else [1.0/n]*n

        elif strategy == 'cap2':
            if n > 2:
                ranked_idx = sorted(range(n), key=lambda i: trades[i]['prob'], reverse=True)
                weights = [0.0] * n
                for i in ranked_idx[:2]:
                    weights[i] = 0.5
            else:
                weights = [1.0 / n] * n

        elif strategy == 'cap3':
            if n > 3:
                ranked_idx = sorted(range(n), key=lambda i: trades[i]['prob'], reverse=True)
                weights = [0.0] * n
                for i in ranked_idx[:3]:
                    weights[i] = 1.0 / 3
            else:
                weights = [1.0 / n] * n

        elif strategy == 'hist_pnl':
            if ticker_meta:
                raw = [max(ticker_meta.get(t['ticker'], {}).get('avg_pnl', 0.01), 0.001) for t in trades]
                total = sum(raw)
                weights = [r / total for r in raw]
            else:
                weights = [1.0 / n] * n

        elif strategy == 'kelly':
            if ticker_meta:
                kellys = []
                for t in trades:
                    m = ticker_meta.get(t['ticker'], {})
                    wr = m.get('win_rate', 0.5)
                    sl = max(m.get('sl', 0.08), 0.001)
                    tp = max(m.get('tp', 0.16), 0.001)
                    avg_win = (m.get('avg_pnl', 0.02) + (1-wr)*sl) / max(wr, 0.01)
                    gl = max(avg_win / sl, 0.01)
                    k = max(wr - (1-wr) / gl, 0.0)
                    kellys.append(min(k, 0.25))
                total_k = sum(kellys)
                if total_k > 0:
                    weights = [k / total_k for k in kellys]
                else:
                    weights = [1.0 / n] * n
            else:
                weights = [1.0 / n] * n

        elif strategy == 'risk_parity':
            if ticker_atr_dict:
                inv_atrs = [1.0 / max(get_atr_for_ticker_date(t['ticker'], d_int, ticker_atr_dict), 0.01)
                            for t in trades]
                total_inv = sum(inv_atrs)
                weights = [ia / total_inv for ia in inv_atrs]
            else:
                weights = [1.0 / n] * n

        elif strategy == 'regime':
            vix_val = vix_daily.get(d_int, 20.0) if vix_daily else 20.0
            if vix_val < 25:
                scale = 1.0
            elif vix_val < 35:
                scale = 0.5
            else:
                scale = 0.25
            weights = [scale / n] * n

        else:
            weights = [1.0 / n] * n

        day_pnl = sum(equity * weights[i] * trades[i]['pnl'] for i in range(n))
        day_ret = day_pnl / equity if equity > 0 else 0
        equity += day_pnl
        n_active = sum(1 for w in weights if w > 0)
        curve.append({'date': date, 'equity': equity, 'day_ret': day_ret,
                      'n_trades': n_active, 'day_pnl': day_pnl})

    return pd.DataFrame(curve), equity


def run_montecarlo(trades_df, starting_capital, n_sims=5000):
    if trades_df.empty:
        return starting_capital, starting_capital, {}
    tdf = trades_df.copy()
    tdf['date'] = pd.to_datetime(tdf['date_int'].astype(str), format='%Y%m%d')
    tickers = sorted(tdf['ticker'].unique())
    tk_idx = {t: i for i, t in enumerate(tickers)}
    n_tk = len(tickers)

    daily_groups = sorted(tdf.groupby('date'), key=lambda x: x[0])
    daily_trades = []
    for date, group in daily_groups:
        day = []
        for _, t in group.iterrows():
            day.append((tk_idx[t['ticker']], float(t['pnl'])))
        daily_trades.append(day)

    best_final = -1e18
    best_weights = np.ones(n_tk) / n_tk
    rng = np.random.default_rng(42)

    for _ in range(n_sims):
        w = rng.dirichlet(np.ones(n_tk))
        equity = starting_capital
        for day in daily_trades:
            day_w = np.array([w[idx] for idx, _ in day])
            day_w = day_w / day_w.sum() if day_w.sum() > 0 else day_w
            day_pnl = sum(equity * day_w[i] * pnl for i, (_, pnl) in enumerate(day))
            equity += day_pnl
        if equity > best_final:
            best_final = equity
            best_weights = w.copy()

    weight_map = {tickers[i]: float(best_weights[i]) for i in range(n_tk)}
    return best_final, (best_final - starting_capital) / starting_capital, weight_map


def run_logistic_sizer(trades_df, starting_capital, ticker_meta, ticker_atr_dict, vix_daily):
    if trades_df.empty or len(trades_df) < 10:
        return pd.DataFrame(), starting_capital

    tdf = trades_df.copy()
    tdf['date'] = pd.to_datetime(tdf['date_int'].astype(str), format='%Y%m%d')
    tdf['atr'] = tdf.apply(lambda r: get_atr_for_ticker_date(r['ticker'], r['date_int'], ticker_atr_dict), axis=1)
    tdf['vix'] = tdf['date_int'].map(lambda d: vix_daily.get(d, 20.0) if vix_daily else 20.0)
    day_counts = tdf.groupby('date_int')['ticker'].transform('count')
    tdf['n_signals'] = day_counts.values
    tdf['win'] = (tdf['pnl'] > 0).astype(int)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    features = tdf[['prob', 'atr', 'vix', 'n_signals']].values
    target = tdf['win'].values

    split = int(len(tdf) * 0.6)
    if split < 5 or len(tdf) - split < 5:
        return simulate_capital_alloc(trades_df, starting_capital, 'equal')

    scaler = StandardScaler()
    X_train = scaler.fit_transform(features[:split])
    X_test = scaler.transform(features[split:])
    y_train = target[:split]

    model = LogisticRegression(max_iter=500, C=0.1)
    model.fit(X_train, y_train)

    p_win_all = np.zeros(len(tdf))
    p_win_all[:split] = 0.5
    p_win_all[split:] = model.predict_proba(X_test)[:, 1]

    tdf['p_win'] = p_win_all
    daily_groups = tdf.groupby('date')
    equity = starting_capital
    curve = []

    for date, group in sorted(daily_groups, key=lambda x: x[0]):
        trades = group.to_dict('records')
        n = len(trades)
        p_wins = [max(t['p_win'], 0.01) for t in trades]
        total_pw = sum(p_wins)
        weights = [pw / total_pw for pw in p_wins]
        day_pnl = sum(equity * weights[i] * trades[i]['pnl'] for i in range(n))
        equity += day_pnl
        curve.append({'date': date, 'equity': equity, 'day_ret': day_pnl / max(equity - day_pnl, 1),
                      'n_trades': n, 'day_pnl': day_pnl})

    return pd.DataFrame(curve), equity


# ==============================================================================
# STEP 7: CAP 2/DAY SIMULATION — per version
# ==============================================================================
print("  Loading VIX daily data...")
vix_daily = build_vix_daily()

test_s_dt = pd.Timestamp(TEST_START)
test_e_dt = pd.Timestamp(TEST_END)
all_biz_days = pd.bdate_range(test_s_dt, test_e_dt)
total_biz_days = len(all_biz_days)

ver_curves = {}
ver_calendars = {}

for ver in VERSIONS:
    tag = ver.split('_')[-1]
    tdf = portfolio_trades[ver]
    if tdf.empty:
        ver_curves[ver] = pd.DataFrame()
        ver_calendars[ver] = pd.DataFrame()
        continue

    tdf_cal = tdf.copy()
    tdf_cal['date'] = pd.to_datetime(tdf_cal['date_int'].astype(str), format='%Y%m%d')
    daily = tdf_cal.groupby('date').apply(
        lambda g: g.nlargest(2, 'prob') if len(g) > 2 else g
    ).reset_index(drop=True)

    equity = STARTING_CAPITAL
    calendar = []
    for day in all_biz_days:
        day_trades = daily[daily['date'] == day]
        n_signals = len(tdf_cal[tdf_cal['date'] == day])
        n_taken = len(day_trades)
        if n_taken == 0:
            calendar.append({'date': day, 'signals': n_signals, 'taken': 0,
                             'tickers': '', 'day_pnl': 0, 'equity': equity})
            continue
        alloc = equity / n_taken
        day_pnl = sum(alloc * t['pnl'] for _, t in day_trades.iterrows())
        equity += day_pnl
        tickers_str = ', '.join(f"{t['ticker']}({t['pnl']*100:+.1f}%)" for _, t in day_trades.iterrows())
        calendar.append({'date': day, 'signals': n_signals, 'taken': n_taken,
                         'tickers': tickers_str, 'day_pnl': day_pnl, 'equity': equity})

    ver_calendars[ver] = pd.DataFrame(calendar)
    ver_curves[ver] = ver_calendars[ver][['date', 'equity', 'day_pnl']].copy()

# ==============================================================================
# STEP 8: SIDE-BY-SIDE DAY-BY-DAY CALENDAR
# ==============================================================================
print(f"\n\n{'='*140}")
print(f"  SIDE-BY-SIDE TRADING CALENDAR — {WINNING_SCORE} + Cap 2/day")
print(f"  ${STARTING_CAPITAL/1000:.0f}K starting capital each  |  {TEST_START} → {TEST_END}")
print(f"{'='*140}")

v_tags = [v.split('_')[-1] for v in VERSIONS]
hdr = f"  {'Date':<12}"
for tag in v_tags:
    hdr += f"  {tag+' Tickers':<35} {tag+' PnL':>10} {tag+' Equity':>10}"
hdr += f"  {'Better':>7}"
print(f"\n{hdr}")
print(f"  " + "-" * 136)

for day in all_biz_days:
    d_str = day.strftime('%Y-%m-%d')
    line = f"  {d_str:<12}"
    pnls = {}
    for vi, ver in enumerate(VERSIONS):
        tag = v_tags[vi]
        cal = ver_calendars[ver]
        if cal.empty or 'date' not in cal.columns:
            line += f"  {'—':<35} {'$0':>10} ${STARTING_CAPITAL/1000:>8.1f}K"
            continue
        row = cal[cal['date'] == day]
        if row.empty or int(row.iloc[0]['taken']) == 0:
            line += f"  {'—':<35} {'$0':>10} ${STARTING_CAPITAL/1000:>8.1f}K"
        else:
            r = row.iloc[0]
            tickers = r['tickers'][:33]
            pnls[tag] = float(r['day_pnl'])
            line += f"  {tickers:<35} ${r['day_pnl']:>+9.0f} ${r['equity']/1000:>8.1f}K"

    if len(pnls) >= 2:
        better = max(pnls, key=pnls.get)
        if all(v == 0 for v in pnls.values()):
            line += f"  {'—':>7}"
        else:
            line += f"  {better:>7}"
    elif len(pnls) == 1:
        line += f"  {list(pnls.keys())[0]:>7}"
    else:
        line += f"  {'—':>7}"
    print(line)

# ==============================================================================
# STEP 9: VERSION COMPARISON SUMMARY
# ==============================================================================
print(f"\n\n{'='*120}")
print(f"  VERSION COMPARISON SUMMARY — {WINNING_SCORE} + Cap 2/day")
print(f"  ${STARTING_CAPITAL/1000:.0f}K starting capital  |  {TEST_START} → {TEST_END}")
print(f"{'='*120}")

print(f"\n  {'Version':<18} {'Assets':>7} {'Trades':>7} {'AvgPnL':>8} {'WR':>5} "
      f"{'Return':>9} {'MaxDD':>7} {'Sharpe':>7} {'Final$':>10} {'WinDays':>9}")
print(f"  " + "-" * 100)

for ver in VERSIONS:
    tag = ver.split('_')[-1]
    r = results[ver]
    cal = ver_calendars.get(ver, pd.DataFrame())
    if cal.empty or r['trades'] == 0:
        print(f"  {tag:<18} {'—':>7}")
        continue

    curve = ver_curves[ver]
    final = float(cal.iloc[-1]['equity'])
    ret = (final - STARTING_CAPITAL) / STARTING_CAPITAL
    dd = ((cal['equity'] - cal['equity'].cummax()) / cal['equity'].cummax()).min()
    active = cal[cal['taken'] > 0]
    daily_rets = (active['day_pnl'] / (active['equity'] - active['day_pnl'])).values
    sharpe = float(np.mean(daily_rets) / max(np.std(daily_rets), 1e-6) * np.sqrt(252)) if len(daily_rets) > 1 else 0
    win_days = (active['day_pnl'] > 0).sum()
    total_active = len(active)

    print(f"  {tag:<18} {r['tickers']:>7} {r['trades']:>7} {r['avg_pnl']*100:>+7.2f}% {r['wr']*100:>4.0f}% "
          f"{ret*100:>+8.1f}% {dd*100:>+6.1f}% {sharpe:>6.1f} ${final/1000:>8.1f}K "
          f"{win_days}/{total_active}")

# Improvement delta
if len(VERSIONS) == 2 and all(not ver_calendars[v].empty and 'date' in ver_calendars[v].columns for v in VERSIONS):
    cal_a = ver_calendars[VERSIONS[0]]
    cal_b = ver_calendars[VERSIONS[1]]
    final_a = float(cal_a.iloc[-1]['equity'])
    final_b = float(cal_b.iloc[-1]['equity'])
    ret_a = (final_a - STARTING_CAPITAL) / STARTING_CAPITAL
    ret_b = (final_b - STARTING_CAPITAL) / STARTING_CAPITAL
    tag_a = VERSIONS[0].split('_')[-1]
    tag_b = VERSIONS[1].split('_')[-1]
    print(f"\n  DELTA: {tag_b} vs {tag_a} = {(ret_b - ret_a)*100:+.1f}% return "
          f"(${(final_b - final_a):+,.0f})")

    # Days where versions diverged
    diverged_days = 0
    v_better = {tag_a: 0, tag_b: 0, 'tie': 0}
    for day in all_biz_days:
        r_a = cal_a[cal_a['date'] == day]
        r_b = cal_b[cal_b['date'] == day]
        pnl_a = float(r_a.iloc[0]['day_pnl']) if not r_a.empty else 0
        pnl_b = float(r_b.iloc[0]['day_pnl']) if not r_b.empty else 0
        tickers_a = str(r_a.iloc[0]['tickers']) if not r_a.empty else ''
        tickers_b = str(r_b.iloc[0]['tickers']) if not r_b.empty else ''
        if tickers_a != tickers_b:
            diverged_days += 1
            if pnl_a > pnl_b: v_better[tag_a] += 1
            elif pnl_b > pnl_a: v_better[tag_b] += 1
            else: v_better['tie'] += 1
    print(f"  Diverged on {diverged_days}/{total_biz_days} days  "
          f"({tag_a} better: {v_better[tag_a]}, {tag_b} better: {v_better[tag_b]}, tie: {v_better['tie']})")

# ==============================================================================
# STEP 10: MODEL DECAY ANALYSIS — performance by 2-week window
# ==============================================================================
print(f"\n\n{'='*120}")
print(f"  MODEL DECAY ANALYSIS — Is performance degrading over time?")
print(f"{'='*120}")

for ver in VERSIONS:
    tag = ver.split('_')[-1]
    tdf = portfolio_trades.get(ver, pd.DataFrame())
    if tdf.empty:
        continue

    tdf_decay = tdf.copy()
    tdf_decay['date'] = pd.to_datetime(tdf_decay['date_int'].astype(str), format='%Y%m%d')

    tdf_decay['biweek'] = ((tdf_decay['date'] - tdf_decay['date'].min()).dt.days // 14)
    tdf_decay['week'] = ((tdf_decay['date'] - tdf_decay['date'].min()).dt.days // 7)
    tdf_decay['days_since_start'] = (tdf_decay['date'] - tdf_decay['date'].min()).dt.days

    print(f"\n  --- {tag} — Trade-level decay ({len(tdf_decay)} trades) ---")

    # Weekly breakdown
    weekly = tdf_decay.groupby('week').agg(
        start=('date', 'min'),
        end=('date', 'max'),
        trades=('pnl', 'count'),
        avg_pnl=('pnl', 'mean'),
        win_rate=('pnl', lambda x: (x > 0).mean()),
        total_pnl=('pnl', 'sum'),
        avg_prob=('prob', 'mean'),
    )

    print(f"\n  {'Week':>4} {'Period':<23} {'Trades':>6} {'AvgPnL':>8} {'WR':>5} {'TotPnL':>9} {'AvgProb':>8}")
    print(f"  " + "-" * 75)
    for wk, r in weekly.iterrows():
        period = f"{r['start'].strftime('%b %d')} - {r['end'].strftime('%b %d')}"
        print(f"  {wk:>4} {period:<23} {int(r['trades']):>6} {r['avg_pnl']*100:>+7.2f}% "
              f"{r['win_rate']*100:>4.0f}% {r['total_pnl']*100:>+8.1f}% {r['avg_prob']:>7.3f}")

    # First half vs second half
    mid = len(tdf_decay) // 2
    first_half = tdf_decay.iloc[:mid]
    second_half = tdf_decay.iloc[mid:]

    f_pnl = first_half['pnl'].mean()
    s_pnl = second_half['pnl'].mean()
    f_wr = (first_half['pnl'] > 0).mean()
    s_wr = (second_half['pnl'] > 0).mean()
    f_mid_date = first_half['date'].median().strftime('%b %d')
    s_mid_date = second_half['date'].median().strftime('%b %d')

    print(f"\n  FIRST HALF ({len(first_half)} trades, ~{f_mid_date}):")
    print(f"    Avg PnL: {f_pnl*100:+.2f}%   WR: {f_wr*100:.0f}%")
    print(f"  SECOND HALF ({len(second_half)} trades, ~{s_mid_date}):")
    print(f"    Avg PnL: {s_pnl*100:+.2f}%   WR: {s_wr*100:.0f}%")
    print(f"  DECAY: {(s_pnl - f_pnl)*100:+.2f}% per trade  ({(s_pnl/max(f_pnl,0.001)-1)*100:+.0f}%)")

    # Last 5 trading days detail
    last_dates = sorted(tdf_decay['date'].unique())[-5:]
    last_trades = tdf_decay[tdf_decay['date'].isin(last_dates)]

    print(f"\n  LAST 5 TRADING DAYS:")
    for d in last_dates:
        day_trades = last_trades[last_trades['date'] == d]
        tickers = ', '.join(f"{t['ticker']}({t['pnl']*100:+.1f}%)" for _, t in day_trades.iterrows())
        day_pnl = day_trades['pnl'].sum()
        print(f"    {d.strftime('%Y-%m-%d')}  {len(day_trades)} trades  {day_pnl*100:>+6.1f}%  {tickers}")

    # Linear regression on PnL vs time
    from numpy.polynomial import polynomial as P
    x = tdf_decay['days_since_start'].values.astype(float)
    y = tdf_decay['pnl'].values
    if len(x) >= 5:
        coeffs = np.polyfit(x, y, 1)
        slope_per_day = coeffs[0]
        slope_per_month = slope_per_day * 30
        r_squared = 1 - np.sum((y - np.polyval(coeffs, x))**2) / np.sum((y - y.mean())**2)

        print(f"\n  LINEAR TREND:")
        print(f"    Slope: {slope_per_day*100:+.4f}% per day ({slope_per_month*100:+.2f}% per month)")
        print(f"    R²: {r_squared:.3f}  ({'significant' if abs(r_squared) > 0.1 else 'weak — noise dominates'})")

        if slope_per_month < -0.005:
            print(f"    VERDICT: DECAYING — losing {abs(slope_per_month)*100:.2f}% per month")
            print(f"    Recommendation: RETRAIN models (36-day half-life → 2+ months past cutoff)")
        elif slope_per_month > 0.005:
            print(f"    VERDICT: IMPROVING — models still fresh")
        else:
            print(f"    VERDICT: STABLE — no significant trend")

    # Per-ticker decay: which tickers are getting worse?
    print(f"\n  PER-TICKER TREND (last 2 weeks vs first 2 weeks):")
    cutoff = tdf_decay['date'].min() + pd.Timedelta(days=14)
    late_start = tdf_decay['date'].max() - pd.Timedelta(days=14)

    early = tdf_decay[tdf_decay['date'] <= cutoff]
    late = tdf_decay[tdf_decay['date'] >= late_start]

    ticker_decay = []
    all_tickers_in_trades = sorted(tdf_decay['ticker'].unique())
    for t in all_tickers_in_trades:
        e = early[early['ticker'] == t]
        l = late[late['ticker'] == t]
        if len(e) >= 1 and len(l) >= 1:
            delta = l['pnl'].mean() - e['pnl'].mean()
            ticker_decay.append({
                'ticker': t,
                'early_pnl': e['pnl'].mean(),
                'early_n': len(e),
                'late_pnl': l['pnl'].mean(),
                'late_n': len(l),
                'delta': delta,
            })

    if ticker_decay:
        ticker_decay.sort(key=lambda x: x['delta'])
        print(f"  {'Ticker':<7} {'Early PnL':>10} {'(N)':>4} {'Late PnL':>10} {'(N)':>4} {'Delta':>8} {'Status':>10}")
        print(f"  " + "-" * 60)
        for td in ticker_decay:
            status = 'DECAYING' if td['delta'] < -0.01 else ('IMPROVING' if td['delta'] > 0.01 else 'STABLE')
            print(f"  {td['ticker']:<7} {td['early_pnl']*100:>+9.2f}% {td['early_n']:>3} "
                  f"{td['late_pnl']*100:>+9.2f}% {td['late_n']:>3} "
                  f"{td['delta']*100:>+7.2f}% {status:>10}")

# ==============================================================================
# STEP 11: LOSS CATEGORIZATION + WHAT-IF SCENARIOS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  LOSS CATEGORIZATION + WHAT-IF SCENARIOS")
print(f"{'='*W}")

for ver in VERSIONS:
    tag = ver.split('_')[-1]
    tdf = portfolio_trades.get(ver, pd.DataFrame())
    if tdf.empty or len(tdf) < 5:
        continue

    tdf_e = tdf.copy()
    tdf_e['date'] = pd.to_datetime(tdf_e['date_int'].astype(str), format='%Y%m%d')

    # Enrich trades with max_up, max_down, eod_return from OHLCV
    max_ups, max_downs, eod_rets = [], [], []
    for _, tr in tdf_e.iterrows():
        ticker = tr['ticker']
        ohlcv = ticker_ohlcv.get(ticker)
        if ohlcv is None:
            max_ups.append(0); max_downs.append(0); eod_rets.append(0)
            continue
        d_int = int(tr['date_int'])
        em_val = None
        for em_check in [360, 365, 370, 375, 380, 385]:
            idx_arr = np.where((ohlcv['dates'] == d_int) & (ohlcv['times'] == em_check))[0]
            if len(idx_arr) > 0 and abs(float(ohlcv['prices'][idx_arr[0]]) - tr['entry_price']) < 0.01:
                em_val = em_check
                break
        if em_val is None:
            idx_arr = np.where((ohlcv['dates'] == d_int) & (ohlcv['times'] >= 360) & (ohlcv['times'] <= 385))[0]
            if len(idx_arr) > 0:
                diffs = np.abs(ohlcv['prices'][idx_arr] - tr['entry_price'])
                em_val = int(ohlcv['times'][idx_arr[np.argmin(diffs)]])
        if em_val is None:
            max_ups.append(0); max_downs.append(0); eod_rets.append(0)
            continue
        entry_idx_arr = np.where((ohlcv['dates'] == d_int) & (ohlcv['times'] == em_val))[0]
        if len(entry_idx_arr) == 0:
            max_ups.append(0); max_downs.append(0); eod_rets.append(0)
            continue
        eidx = entry_idx_arr[0]
        day_mask = ohlcv['dates'] == d_int
        after_mask = np.arange(len(ohlcv['prices'])) > eidx
        after = np.where(day_mask & after_mask)[0]
        if len(after) == 0:
            max_ups.append(0); max_downs.append(0); eod_rets.append(0)
            continue
        ep = tr['entry_price']
        mh = float(ohlcv['highs'][after].max())
        ml = float(ohlcv['lows'][after].min())
        eod = float(ohlcv['prices'][after[-1]])
        max_ups.append((mh - ep) / ep)
        max_downs.append((ep - ml) / ep)
        eod_rets.append((eod - ep) / ep)

    tdf_e['max_up'] = max_ups
    tdf_e['max_down'] = max_downs
    tdf_e['eod_return'] = eod_rets

    losses_e = tdf_e[tdf_e['pnl'] < 0].copy()
    wins_e = tdf_e[tdf_e['pnl'] > 0]

    # Categorize
    losses_e['sl_too_tight'] = (losses_e['exit'] == 'SL') & (losses_e['eod_return'] > 0)
    losses_e['wrong_direction'] = (losses_e['max_up'] < 0.01) & (losses_e['eod_return'] < -0.02)
    losses_e['whipsaw'] = (losses_e['exit'] == 'SL') & (losses_e['max_up'] > losses_e['tp'] * 0.5)
    losses_e['penny_stock'] = losses_e['entry_price'] < 2.0
    losses_e['low_conviction'] = losses_e['prob'] < 0.55

    print(f"\n  --- {tag}: {len(tdf_e)} trades, {len(wins_e)} wins ({len(wins_e)/len(tdf_e)*100:.0f}%), "
          f"{len(losses_e)} losses ---")

    cats = [
        ('SL Too Tight (recovered by EOD)', 'sl_too_tight'),
        ('Wrong Direction (<1% upside)', 'wrong_direction'),
        ('Whipsaw (SL hit, bounced 50%+ TP)', 'whipsaw'),
        ('Penny Stock (<$2)', 'penny_stock'),
        ('Low Conviction (prob < 0.55)', 'low_conviction'),
    ]
    print(f"\n  {'Category':<40} {'Count':>6} {'%':>5} {'AvgLoss':>8} {'Total':>8}")
    print(f"  " + "-" * 70)
    for label, col in cats:
        mask = losses_e[col]
        n = mask.sum()
        pct = n / max(len(losses_e), 1) * 100
        avg = losses_e[mask]['pnl'].mean() * 100 if n > 0 else 0
        total = losses_e[mask]['pnl'].sum() * 100 if n > 0 else 0
        print(f"  {label:<40} {n:>6} {pct:>4.0f}% {avg:>+7.2f}% {total:>+7.1f}%")

    # All losing trades detail
    print(f"\n  {'Date':<12} {'Ticker':<7} {'Prob':>5} {'Entry$':>7} {'PnL':>7} {'Exit':>5} "
          f"{'MaxUp':>6} {'MaxDn':>6} {'EOD':>7} {'SL':>5} {'TP':>5} {'Tags'}")
    print(f"  " + "-" * 100)
    for _, r in losses_e.sort_values('date').iterrows():
        tags = []
        if r.get('sl_too_tight', False): tags.append('TIGHT')
        if r.get('wrong_direction', False): tags.append('WRONG')
        if r.get('whipsaw', False): tags.append('WHIP')
        if r.get('penny_stock', False): tags.append('PENNY')
        if r.get('low_conviction', False): tags.append('LOWC')
        if not tags: tags.append('NORMAL')
        print(f"  {r['date'].strftime('%Y-%m-%d'):<12} {r['ticker']:<7} {r['prob']:>5.2f} "
              f"${r['entry_price']:>6.2f} {r['pnl']*100:>+6.2f}% {r['exit']:>5} "
              f"{r['max_up']*100:>+5.1f}% {r['max_down']*100:>+5.1f}% {r['eod_return']*100:>+6.2f}% "
              f"{r['sl']*100:>4.0f}% {r['tp']*100:>4.0f}% {','.join(tags)}")

    # WHAT-IF SCENARIOS
    WHIPSAW_TICKERS = ['ASTS', 'FCEL', 'RCAT', 'CLSK', 'LUNR', 'SPIR']

    def run_whatif(t, name, prob_min=0.0, min_price=0.0, sl_mult=1.0, whip_mult=1.0):
        df = t.copy()
        if min_price > 0:
            df = df[df['entry_price'] >= min_price]
        if prob_min > 0:
            df = df[df['prob'] >= prob_min]
        if sl_mult != 1.0 or whip_mult != 1.0:
            new_pnl = []
            for _, r in df.iterrows():
                mult = whip_mult if (r['ticker'] in WHIPSAW_TICKERS and whip_mult != 1.0) else sl_mult
                if mult == 1.0:
                    new_pnl.append(r['pnl']); continue
                new_sl = r['sl'] * mult
                if r['exit'] == 'SL' and r['max_down'] < new_sl:
                    new_pnl.append(r['eod_return'] - 0.003)
                elif r['exit'] == 'SL':
                    new_pnl.append(-new_sl - 0.003)
                else:
                    new_pnl.append(r['pnl'])
            df['pnl'] = new_pnl
        if df.empty:
            return {'name': name, 'n': 0, 'rm': len(t), 'w': 0, 'wr': 0, 'avg': 0, 'tot': 0}
        n = len(df); w = (df['pnl'] > 0).sum()
        return {'name': name, 'n': n, 'rm': len(t)-n, 'w': w,
                'wr': w/n, 'avg': df['pnl'].mean(), 'tot': df['pnl'].sum()}

    scenarios = [
        run_whatif(tdf_e, 'A: Baseline'),
        run_whatif(tdf_e, 'B: Prob >= 0.55', prob_min=0.55),
        run_whatif(tdf_e, 'C: Prob >= 0.60', prob_min=0.60),
        run_whatif(tdf_e, 'D: Prob >= 0.65', prob_min=0.65),
        run_whatif(tdf_e, 'E: Min $2', min_price=2.0),
        run_whatif(tdf_e, 'F: SL +50% all', sl_mult=1.5),
        run_whatif(tdf_e, 'G: SL +50% whipsaw', whip_mult=1.5),
        run_whatif(tdf_e, 'H: SL +100% whipsaw', whip_mult=2.0),
        run_whatif(tdf_e, 'I: P>=0.55 + $2', prob_min=0.55, min_price=2.0),
        run_whatif(tdf_e, 'J: P>=0.55 + $2 + SL50%', prob_min=0.55, min_price=2.0, sl_mult=1.5),
        run_whatif(tdf_e, 'K: P>=0.60 + $2', prob_min=0.60, min_price=2.0),
        run_whatif(tdf_e, 'L: P>=0.60 + $2 + whipSL', prob_min=0.60, min_price=2.0, whip_mult=1.5),
        run_whatif(tdf_e, 'M: P>=0.60 + $2 + SL50%', prob_min=0.60, min_price=2.0, sl_mult=1.5),
        run_whatif(tdf_e, 'N: P>=0.65 + $2', prob_min=0.65, min_price=2.0),
        run_whatif(tdf_e, 'O: P>=0.65 + $2 + SL50%', prob_min=0.65, min_price=2.0, sl_mult=1.5),
    ]

    base_tot = scenarios[0]['tot']
    best_s = max(scenarios[1:], key=lambda x: x['tot'])

    print(f"\n  WHAT-IF SCENARIOS:")
    print(f"  {'Scenario':<30} {'Trades':>6} {'Rm':>4} {'Wins':>5} {'WR':>5} "
          f"{'AvgPnL':>8} {'TotPnL':>9} {'vsBas':>7}")
    print(f"  " + "-" * 80)
    for s in scenarios:
        delta = (s['tot'] - base_tot) * 100
        best = ' <--' if s is best_s else ''
        print(f"  {s['name']:<30} {s['n']:>6} {s['rm']:>4} {s['w']:>5} "
              f"{s['wr']*100:>4.0f}% {s['avg']*100:>+7.2f}% {s['tot']*100:>+8.1f}% "
              f"{delta:>+6.1f}%{best}")

    top3 = sorted(scenarios[1:], key=lambda x: x['tot'], reverse=True)[:3]
    print(f"\n  TOP 3:")
    for i, s in enumerate(top3, 1):
        imp = (s['tot'] - base_tot) * 100
        print(f"    {i}. {s['name']} — {s['n']} trades, WR={s['wr']*100:.0f}%, "
              f"total={s['tot']*100:+.1f}% ({imp:+.1f}% vs baseline)")

# ==============================================================================
# STEP 12: EQUITY CURVE + DECAY CHARTS
# ==============================================================================
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

print(f"\n  Generating charts...")

fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 2]})
fig.suptitle(f'Version Comparison + Model Decay — {WINNING_SCORE} + Cap 2/day\n'
             f'${STARTING_CAPITAL/1000:.0f}K, {TEST_START} → {TEST_END}',
             fontsize=13, fontweight='bold')

colors = ['#F44336', '#2196F3', '#4CAF50', '#FF9800']

# Top: equity curves
ax = axes[0]
for vi, ver in enumerate(VERSIONS):
    tag = ver.split('_')[-1]
    cal = ver_calendars.get(ver, pd.DataFrame())
    if cal.empty: continue
    final = float(cal.iloc[-1]['equity'])
    ret = (final - STARTING_CAPITAL) / STARTING_CAPITAL
    n_tickers = results[ver].get('tickers', 0)
    ax.plot(cal['date'], cal['equity']/1000, color=colors[vi%4], linewidth=2,
            label=f"{tag} ({n_tickers} assets, {ret*100:+.1f}%)")
ax.axhline(y=STARTING_CAPITAL/1000, color='gray', linewidth=0.5, linestyle='--')
ax.set_ylabel('Equity ($K)')
ax.legend(fontsize=10, loc='upper left')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.grid(True, alpha=0.3)
ax.set_title('Portfolio Equity')

# Bottom: rolling avg PnL per trade (decay visualization)
ax2 = axes[1]
for vi, ver in enumerate(VERSIONS):
    tag = ver.split('_')[-1]
    tdf = portfolio_trades.get(ver, pd.DataFrame())
    if tdf.empty: continue
    tdf_sorted = tdf.copy()
    tdf_sorted['date'] = pd.to_datetime(tdf_sorted['date_int'].astype(str), format='%Y%m%d')
    tdf_sorted = tdf_sorted.sort_values('date')
    window = max(5, len(tdf_sorted) // 8)
    rolling_pnl = tdf_sorted['pnl'].rolling(window, min_periods=3).mean()
    rolling_wr = tdf_sorted['pnl'].rolling(window, min_periods=3).apply(lambda x: (x>0).mean())
    ax2.plot(tdf_sorted['date'].values, rolling_pnl.values * 100, color=colors[vi%4],
             linewidth=2, label=f"{tag} rolling avg PnL ({window}-trade)")
ax2.axhline(y=0, color='red', linewidth=0.5, linestyle='--')
ax2.axhline(y=1.8, color='green', linewidth=0.5, linestyle='--', alpha=0.5)
ax2.set_ylabel('Avg PnL per trade (%)')
ax2.set_xlabel('Date')
ax2.legend(fontsize=9, loc='upper right')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax2.grid(True, alpha=0.3)
ax2.set_title('Model Decay — Rolling Avg PnL (green = 1.8% deploy threshold)')

plt.tight_layout()
chart_path = '/tmp/version_comparison_chart.png'
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"  Charts saved: {chart_path}")
try:
    from IPython.display import display, Image
    display(Image(filename=chart_path))
except ImportError:
    plt.show()

print(f"\n{'='*W}")
print(f"  To change versions: edit VERSIONS list at top")
print(f"{'='*W}")
