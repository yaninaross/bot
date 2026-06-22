"""
backtest_magnitude_comparison.py — Backtest 3 TP Strategies Head-to-Head
========================================================================
For each ticker, simulates trades using the SAME binary model signals and
SAME SL (from WF optimizer), but 3 different TP strategies:
  1. Fixed TP (from WF optimizer v8)
  2. Classification TP (from magnitude classifier → predicted class sets TP)
  3. Regression TP (from magnitude regressor → predicted magnitude sets TP)

Same entries, same SL, only TP differs → isolates the TP impact.
Portfolio-level PnL comparison with $200K capital, cap-2 allocation.
"""

import os, io, gc, sys, json, subprocess, warnings, ctypes
from datetime import timedelta
import numpy as np, pandas as pd, pytz

warnings.filterwarnings('ignore')

def _release_memory():
    gc.collect()
    try: ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception: pass

from numba import jit
from google.cloud import storage as gcs_storage, bigquery

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass

# ==============================================================================
# CONFIG
# ==============================================================================
TEST_START = '2026-04-01'
TEST_END = '2026-05-29'
STARTING_CAPITAL = 200_000

BUCKET_NAME = "time-series-bucket-ys"
BQ_TABLE = "ys-time-series.scoring_logs.wf_optimization_results"
INFERENCE_DIR = "Data/Inference_v2"
MAG_CLS_PATH = "Algos/Results_all_v2/Universal_Exp185_MAG_CLS"
MAG_REG_PATH = "Algos/Results_all_v2/Universal_Exp185_MAG_REG"

EXIT_TIME_MIN = 780
PST = pytz.timezone('America/Los_Angeles')
SLIPPAGE_PCT = 0.0015
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

# Classification TP grids — multiple mappings to search
CLS_TP_CONFIGS = [
    {'name': 'tight',        0: 0.015, 1: 0.025, 2: 0.04,  3: 0.06,  4: 0.08},
    {'name': 'conservative', 0: 0.02,  1: 0.03,  2: 0.05,  3: 0.07,  4: 0.10},
    {'name': 'moderate',     0: 0.02,  1: 0.03,  2: 0.055, 3: 0.085, 4: 0.12},
    {'name': 'aggressive',   0: 0.03,  1: 0.04,  2: 0.06,  3: 0.09,  4: 0.15},
    {'name': 'wide',         0: 0.03,  1: 0.05,  2: 0.08,  3: 0.12,  4: 0.20},
    {'name': 'midpoint',     0: 0.01,  1: 0.03,  2: 0.055, 3: 0.085, 4: 0.10},
]

# Regression TP grids — search ratio/floor/cap combos
REG_TP_CONFIGS = [
    {'name': 'r0.6_f2_c15', 'ratio': 0.6, 'floor': 0.02, 'cap': 0.15},
    {'name': 'r0.7_f2_c20', 'ratio': 0.7, 'floor': 0.02, 'cap': 0.20},
    {'name': 'r0.8_f3_c25', 'ratio': 0.8, 'floor': 0.03, 'cap': 0.25},
    {'name': 'r0.9_f3_c25', 'ratio': 0.9, 'floor': 0.03, 'cap': 0.25},
    {'name': 'r1.0_f3_c30', 'ratio': 1.0, 'floor': 0.03, 'cap': 0.30},
    {'name': 'r0.7_f3_c15', 'ratio': 0.7, 'floor': 0.03, 'cap': 0.15},
    {'name': 'r0.5_f2_c20', 'ratio': 0.5, 'floor': 0.02, 'cap': 0.20},
    {'name': 'r0.8_f2_c15', 'ratio': 0.8, 'floor': 0.02, 'cap': 0.15},
]

MODE_SUFFIX = {'static': '', 'atr': '_ATR', 'atr_adaptive': '_ATRADP',
               'regime_scaled': '_REGIME', 'trailing_decay': '_TRAIL'}

gcs_client = gcs_storage.Client()
bq_client = bigquery.Client()

# ==============================================================================
# LOAD DEPLOYED CONFIGS (S8_12 best per ticker from v8)
# ==============================================================================
print("Loading S8_12 deployed configs from BQ...")
q = f"""
WITH base AS (
  SELECT *,
    CASE WHEN (val_a_count + val_b_count) > 0
      THEN (val_a_avg_pnl * val_a_count + val_b_avg_pnl * val_b_count) / (val_a_count + val_b_count)
      ELSE 0 END AS wavg_pnl,
    CASE WHEN val_a_avg_pnl > 0
      THEN GREATEST(1.0 - ABS(val_b_avg_pnl - val_a_avg_pnl) / val_a_avg_pnl, 0.2)
      ELSE 0.2 END AS consistency
  FROM `{BQ_TABLE}`
  WHERE model_version = 'robust_wf_v8'
    AND survived_a = TRUE
    AND val_a_avg_pnl >= 0.012 AND val_b_avg_pnl >= 0.012
    AND val_a_win_rate >= 0.50 AND val_b_win_rate >= 0.50
    AND val_b_start = '{TEST_START}'
),
scored AS (
  SELECT *, POW(wavg_pnl, 2) * POW(
    (val_a_win_rate * val_a_count + val_b_win_rate * val_b_count) / GREATEST(val_a_count + val_b_count, 1), 2
  ) * consistency * LOG(1 + val_a_count + val_b_count, 2) AS composite
  FROM base
),
ranked AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY composite DESC) AS rn FROM scored
)
SELECT ticker, experiment, seed, tpsl_mode, prob_thresh, entry_minute,
       tp_pct, sl_pct, tp_mult, sl_mult, trail_start_mult, trail_end_mult,
       is_ensemble
FROM ranked WHERE rn = 1
"""
configs_df = bq_client.query(q).to_dataframe()
print(f"  {len(configs_df)} tickers\n")

# ==============================================================================
# LOAD MAGNITUDE SCORING CSVs
# ==============================================================================
def load_mag_scores(ticker, mag_path):
    prefix = f"{mag_path}/{ticker}/Seed_"
    blobs = list(gcs_client.bucket(BUCKET_NAME).list_blobs(prefix=prefix))
    if not blobs:
        return None
    best_df = None
    for blob in blobs:
        try:
            df = pd.read_csv(io.BytesIO(blob.download_as_bytes()), index_col=0, parse_dates=True)
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            df.index = df.index.tz_convert(PST)
            if best_df is None or len(df) > len(best_df):
                best_df = df
        except Exception:
            continue
    return best_df

# ==============================================================================
# TRADE SIMULATION
# ==============================================================================
@jit(nopython=True)
def sim_trade(prices, highs, lows, date_ints, time_mins, entry_idx, tp, sl, exit_time):
    n = len(prices)
    if entry_idx >= n - 1: return 0.0, 'EOD'
    ep = prices[entry_idx]; ed = date_ints[entry_idx]
    tp_p = ep * (1 + tp); sl_p = ep * (1 - sl)
    for k in range(entry_idx + 1, n):
        if date_ints[k] != ed: return (prices[k-1] - ep) / ep, 'EOD'
        if lows[k] <= sl_p and highs[k] >= tp_p:
            return (-sl if np.random.random() < 0.5 else tp), 'BOTH'
        if lows[k] <= sl_p: return -sl, 'SL'
        if highs[k] >= tp_p: return tp, 'TP'
        if time_mins[k] >= exit_time: return (prices[k] - ep) / ep, 'EOD'
    return (prices[-1] - ep) / ep, 'EOD'

# ==============================================================================
# MAIN
# ==============================================================================
print(f"Loading OHLCV + magnitude predictions...")

all_trades = {'fixed': [], 'cls': [], 'reg': []}
ticker_summary = []

for _, cfg in configs_df.iterrows():
    ticker = cfg['ticker']
    clean = TICKER_FILE_MAP.get(ticker, ticker)

    # Load OHLCV
    try:
        df_ohlcv = pd.read_parquet(f"gs://{BUCKET_NAME}/{INFERENCE_DIR}/{clean}_master_inference.parquet")
        df_ohlcv = df_ohlcv.reset_index()
        dt_cols = [c for c in df_ohlcv.columns if pd.api.types.is_datetime64_any_dtype(df_ohlcv[c])]
        if dt_cols:
            df_ohlcv['Datetime'] = pd.to_datetime(df_ohlcv[dt_cols[0]], utc=True)
        else:
            df_ohlcv['Datetime'] = pd.to_datetime(df_ohlcv[df_ohlcv.columns[0]], unit='ms', utc=True)
        df_ohlcv = df_ohlcv.set_index('Datetime').sort_index()
        std_map = {'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'}
        lower_map = {k.lower():v for k,v in std_map.items()}
        df_ohlcv.columns = [lower_map.get(str(c).strip().lower(), str(c).strip()) for c in df_ohlcv.columns]
        df_ohlcv.index = df_ohlcv.index.tz_convert(PST)
    except Exception:
        continue

    # Load binary scores
    exp = cfg['experiment']
    mode_sfx = MODE_SUFFIX.get(cfg['tpsl_mode'], '')
    base_exp = exp[:-len(mode_sfx)] if mode_sfx and exp.endswith(mode_sfx) else exp
    seed = int(cfg['seed'])
    is_ens = bool(cfg.get('is_ensemble', False))

    prefix = f"Algos/Results_all_v2/Universal_Exp{base_exp}/{clean}/Seed_"
    blobs = list(gcs_client.bucket(BUCKET_NAME).list_blobs(prefix=prefix))
    binary_df = None
    if is_ens:
        all_probs = []
        for b in blobs:
            try:
                df_s = pd.read_csv(io.BytesIO(b.download_as_bytes()), index_col=0, parse_dates=True)
                if df_s.index.tz is None: df_s.index = df_s.index.tz_localize('UTC')
                df_s.index = df_s.index.tz_convert(PST)
                all_probs.append(df_s['Prob_Spike'])
            except: continue
        if all_probs:
            binary_df = pd.DataFrame({'Prob_Spike': pd.concat(all_probs, axis=1).mean(axis=1)})
    else:
        for b in blobs:
            if f"Seed_{seed}_" in b.name or b.name.endswith(f"Seed_{seed}_Results.csv"):
                try:
                    binary_df = pd.read_csv(io.BytesIO(b.download_as_bytes()), index_col=0, parse_dates=True)
                    if binary_df.index.tz is None: binary_df.index = binary_df.index.tz_localize('UTC')
                    binary_df.index = binary_df.index.tz_convert(PST)
                except: pass
                break

    if binary_df is None or 'Prob_Spike' not in binary_df.columns:
        continue

    cls_df = load_mag_scores(ticker, MAG_CLS_PATH)
    reg_df = load_mag_scores(ticker, MAG_REG_PATH)

    test_s = pd.Timestamp(TEST_START, tz=PST)
    test_e = pd.Timestamp(TEST_END, tz=PST) + timedelta(days=1)
    ohlcv = df_ohlcv[(df_ohlcv.index >= test_s) & (df_ohlcv.index < test_e)]
    if ohlcv.empty: continue

    prices = ohlcv['Close'].values
    highs = ohlcv['High'].values
    lows = ohlcv['Low'].values
    dates = (ohlcv.index.year*10000 + ohlcv.index.month*100 + ohlcv.index.day).values.astype(np.int64)
    times = (ohlcv.index.hour*60 + ohlcv.index.minute).values

    em = int(cfg['entry_minute'])
    prob_thresh = float(cfg['prob_thresh'])
    fixed_tp = float(cfg['tp_pct']) if cfg['tp_pct'] > 0 else 0.16
    fixed_sl = float(cfg['sl_pct']) if cfg['sl_pct'] > 0 else 0.08

    # Collect entry signals first
    entries = []
    last_traded = -1
    for bar_i in range(len(ohlcv)):
        if times[bar_i] != em: continue
        if ohlcv['Volume'].values[bar_i] == 0: continue
        d_int = int(dates[bar_i])
        if d_int == last_traded: continue

        bar_ts = ohlcv.index[bar_i]
        if bar_ts in binary_df.index:
            prob = float(binary_df.loc[bar_ts, 'Prob_Spike'])
        else:
            common = binary_df.index[binary_df.index <= bar_ts]
            if len(common) == 0: continue
            nearest = common[-1]
            if abs((bar_ts - nearest).total_seconds()) > 120: continue
            prob = float(binary_df.loc[nearest, 'Prob_Spike'])

        if prob < prob_thresh: continue
        last_traded = d_int

        # Get magnitude predictions for this bar
        cls_class = None
        if cls_df is not None and bar_ts in cls_df.index:
            if 'Pred_Class' in cls_df.columns:
                cls_class = int(cls_df.loc[bar_ts, 'Pred_Class'])
            else:
                cls_class = int(cls_df.loc[bar_ts, ['Prob_C0','Prob_C1','Prob_C2','Prob_C3','Prob_C4']].values.argmax())

        reg_mag = None
        if reg_df is not None and bar_ts in reg_df.index:
            reg_mag = float(reg_df.loc[bar_ts, 'Pred_Mag'])

        entries.append({'bar_i': bar_i, 'd_int': d_int, 'prob': prob,
                        'cls_class': cls_class, 'reg_mag': reg_mag})

    if not entries:
        del df_ohlcv; _release_memory()
        continue

    # Fixed TP trades
    for e in entries:
        pnl, exit_t = sim_trade(prices, highs, lows, dates, times, e['bar_i'], fixed_tp, fixed_sl, EXIT_TIME_MIN)
        pnl = float(pnl) - 2 * SLIPPAGE_PCT
        all_trades['fixed'].append({'ticker': ticker, 'date_int': e['d_int'], 'prob': e['prob'],
                                     'pnl': pnl, 'exit': exit_t, 'tp': fixed_tp, 'sl': fixed_sl})

    # Classification TP — search all configs
    for cls_cfg in CLS_TP_CONFIGS:
        key = f"cls_{cls_cfg['name']}"
        if key not in all_trades:
            all_trades[key] = []
        for e in entries:
            tp = cls_cfg.get(e['cls_class'], fixed_tp) if e['cls_class'] is not None else fixed_tp
            pnl, exit_t = sim_trade(prices, highs, lows, dates, times, e['bar_i'], tp, fixed_sl, EXIT_TIME_MIN)
            pnl = float(pnl) - 2 * SLIPPAGE_PCT
            all_trades[key].append({'ticker': ticker, 'date_int': e['d_int'], 'prob': e['prob'],
                                     'pnl': pnl, 'exit': exit_t, 'tp': tp, 'sl': fixed_sl})

    # Regression TP — search all configs
    for reg_cfg in REG_TP_CONFIGS:
        key = f"reg_{reg_cfg['name']}"
        if key not in all_trades:
            all_trades[key] = []
        for e in entries:
            if e['reg_mag'] is not None:
                tp = max(min(e['reg_mag'] * reg_cfg['ratio'], reg_cfg['cap']), reg_cfg['floor'])
            else:
                tp = fixed_tp
            pnl, exit_t = sim_trade(prices, highs, lows, dates, times, e['bar_i'], tp, fixed_sl, EXIT_TIME_MIN)
            pnl = float(pnl) - 2 * SLIPPAGE_PCT
            all_trades[key].append({'ticker': ticker, 'date_int': e['d_int'], 'prob': e['prob'],
                                     'pnl': pnl, 'exit': exit_t, 'tp': tp, 'sl': fixed_sl})

    print(f"  {ticker}: {len(entries)} entries scored")
    del df_ohlcv; _release_memory()

# ==============================================================================
# PORTFOLIO SIMULATION (cap-2)
# ==============================================================================
def simulate_portfolio(trades_list, starting_capital):
    if not trades_list: return pd.DataFrame(), starting_capital
    tdf = pd.DataFrame(trades_list)
    tdf['date'] = pd.to_datetime(tdf['date_int'].astype(str), format='%Y%m%d')
    equity = starting_capital
    curve = []
    for date, group in sorted(tdf.groupby('date'), key=lambda x: x[0]):
        day = group.sort_values('prob', ascending=False).head(2)
        n = len(day)
        day_pnl = sum(equity / n * row['pnl'] for _, row in day.iterrows())
        equity += day_pnl
        curve.append({'date': date, 'equity': equity, 'day_pnl': day_pnl, 'n': n})
    return pd.DataFrame(curve), equity

# ==============================================================================
# RESULTS — ALL CONFIGS RANKED
# ==============================================================================
W = 110
print(f"\n{'='*W}")
print(f"  TP STRATEGY GRID SEARCH — Fixed vs {len(CLS_TP_CONFIGS)} Classification vs {len(REG_TP_CONFIGS)} Regression")
print(f"  SL is FIXED across all strategies  |  {TEST_START} to {TEST_END}  |  Cap 2/day")
print(f"{'='*W}")

# Build results for every strategy
all_strategy_names = ['fixed'] + [f"cls_{c['name']}" for c in CLS_TP_CONFIGS] + [f"reg_{r['name']}" for r in REG_TP_CONFIGS]
strategy_results = []

for key in all_strategy_names:
    trades = all_trades.get(key, [])
    if not trades:
        continue
    curve, final = simulate_portfolio(trades, STARTING_CAPITAL)
    ret = (final - STARTING_CAPITAL) / STARTING_CAPITAL
    n = len(trades)
    avg_pnl = np.mean([t['pnl'] for t in trades])
    wr = np.mean([t['pnl'] > 0 for t in trades])
    tp_hits = sum(1 for t in trades if t['exit'] == 'TP')
    sl_hits = sum(1 for t in trades if t['exit'] == 'SL')
    eod_exits = n - tp_hits - sl_hits
    avg_tp = np.mean([t['tp'] for t in trades])
    dd = 0
    if not curve.empty:
        peak = curve['equity'].cummax()
        dd = ((curve['equity'] - peak) / peak).min()

    strategy_results.append({
        'key': key, 'n': n, 'avg_pnl': avg_pnl, 'wr': wr,
        'tp_hits': tp_hits, 'sl_hits': sl_hits, 'eod': eod_exits,
        'avg_tp': avg_tp, 'ret': ret, 'dd': dd, 'final': final,
    })

strategy_results.sort(key=lambda x: x['ret'], reverse=True)

print(f"\n  {'#':>2}  {'Strategy':<22} {'Trades':>6} {'AvgPnL':>7} {'WR':>5} {'AvgTP':>6} "
      f"{'TP#':>4} {'SL#':>4} {'EOD':>4} {'Return':>8} {'MaxDD':>7} {'Final':>8}")
print(f"  " + "-" * (W - 4))

for rank, sr in enumerate(strategy_results, 1):
    marker = " <--" if rank == 1 else ""
    is_fixed = sr['key'] == 'fixed'
    label = "FIXED (baseline)" if is_fixed else sr['key']
    print(f"  {rank:>2}  {label:<22} {sr['n']:>6} {sr['avg_pnl']*100:>+6.2f}% {sr['wr']*100:>4.0f}% "
          f"{sr['avg_tp']*100:>5.1f}% "
          f"{sr['tp_hits']:>4} {sr['sl_hits']:>4} {sr['eod']:>4} "
          f"{sr['ret']*100:>+7.1f}% {sr['dd']*100:>+6.1f}% ${sr['final']/1000:>6.0f}K{marker}")

# Summary by category
fixed_ret = next((s['ret'] for s in strategy_results if s['key'] == 'fixed'), 0)
best_cls = max([s for s in strategy_results if s['key'].startswith('cls_')],
               key=lambda x: x['ret'], default=None)
best_reg = max([s for s in strategy_results if s['key'].startswith('reg_')],
               key=lambda x: x['ret'], default=None)

print(f"\n  {'='*W}")
print(f"  BEST PER CATEGORY")
print(f"  {'='*W}")
print(f"  Fixed TP:              Return {fixed_ret*100:>+7.1f}%")
if best_cls:
    delta = best_cls['ret'] - fixed_ret
    print(f"  Best Classification:   Return {best_cls['ret']*100:>+7.1f}%  ({best_cls['key']})  "
          f"vs Fixed: {delta*100:>+.1f}%")
if best_reg:
    delta = best_reg['ret'] - fixed_ret
    print(f"  Best Regression:       Return {best_reg['ret']*100:>+7.1f}%  ({best_reg['key']})  "
          f"vs Fixed: {delta*100:>+.1f}%")

# Per-ticker detail for top 3 strategies
print(f"\n  {'='*W}")
print(f"  PER-TICKER BREAKDOWN — Top 3 strategies")
print(f"  {'='*W}")

top3 = strategy_results[:3]
for sr in top3:
    trades = all_trades[sr['key']]
    tdf = pd.DataFrame(trades)
    by_tk = tdf.groupby('ticker').agg(
        n=('pnl','count'), avg=('pnl','mean'), total=('pnl','sum'),
        wr=('pnl', lambda x: (x>0).mean()*100),
        avg_tp=('tp','mean'),
        tp_hits=('exit', lambda x: (x=='TP').sum()),
    ).sort_values('total', ascending=False)

    print(f"\n  {sr['key']} — Return {sr['ret']*100:+.1f}%, {sr['n']} trades")
    print(f"  {'Ticker':<7} {'N':>3} {'AvgPnL':>8} {'TotPnL':>8} {'WR':>5} {'AvgTP':>6} {'TP#':>4}")
    print(f"  " + "-" * 45)
    for t, r in by_tk.iterrows():
        print(f"  {t:<7} {int(r['n']):>3} {r['avg']*100:>+7.2f}% {r['total']*100:>+7.1f}% "
              f"{r['wr']:>4.0f}% {r['avg_tp']*100:>5.1f}% {int(r['tp_hits']):>4}")

print(f"\n{'='*W}")
print(f"  KEY: If magnitude TP beats fixed, the model adds value by setting")
print(f"  realistic TP targets that get hit more often than overly ambitious fixed TPs.")
print(f"{'='*W}")
