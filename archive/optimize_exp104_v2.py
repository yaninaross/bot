"""
optimize_exp104_v2.py — Ensemble + simplified optimization for Exp-104
=======================================================================
Key changes from v1:
  - Ensemble: averages Prob_Spike across ALL 15 seeds (no seed selection)
  - 4 optimized params: entry_time, prob_thresh, tp_pct, sl_pct
  - No regime filters (VIX/TLT/SPY removed)
  - MIN_TRADES_PER_MONTH raised to 2.0
  - Stage 1: coarse sweep (4 params: entry_time, prob, TP, SL)
  - Stage 2: fine-tune all 4 jointly around Stage 1 winner
"""

import gc
import pandas as pd
import numpy as np
import io
import itertools
from google.cloud import storage
from numba import jit
try:
    from google.colab import auth
except ImportError:
    auth = None
from datetime import timedelta
import pytz

# ==============================================================================
# CONFIG — Exp-104 (Defense)
# ==============================================================================
MODEL_ID    = "104_DEFENSE"
SEEDS       = list(range(10401, 10416))

TRAIN_TEACHERS = [
    'AVAV', 'KTOS', 'PLTR', 'BBAI',
    'RKLB', 'ASTS', 'LUNR', 'SPIR',
    'JOBY', 'NNDM', 'MVIS',
]
SCORING_POOL = TRAIN_TEACHERS + [
    'GD', 'LMT', 'NOC', 'RTX', 'LHX', 'GE',
    'RCAT', 'ACHR',
]

TICKER_FILE_MAP = {'BITF': 'KEEL'}

ENTRY_TIMES = ["06:00", "06:05", "06:10", "06:15", "06:20", "06:25", "06:30"]

OPT_START_DATE  = "2025-07-01"
OPT_END_DATE    = "2025-12-31"
TEST_START_DATE = "2026-01-01"
TEST_END_DATE   = "2026-04-18"

RESULTS_DIR = f"Algos/Results_all_v2/Universal_Exp{MODEL_ID}"
BUCKET_NAME = os.environ.get("GCS_BUCKET", "")

# ==============================================================================
# GRID — only 3 parameters
# ==============================================================================
GRID_COARSE = {
    'entry_time':  ENTRY_TIMES,
    'prob_thresh': np.arange(0.05, 0.95, 0.05).round(2).tolist(),
    'tp_pct':      np.arange(0.03, 0.26, 0.01).round(2).tolist(),
    'sl_pct':      np.arange(0.02, 0.16, 0.01).round(2).tolist(),
}

MIN_TRADES_PER_MONTH = 2.0
EXIT_TIME_MIN        = 780   # 1:00 PM PST

# ==============================================================================
# SIMULATION ENGINE (Numba)
# ==============================================================================
@jit(nopython=True)
def run_trade_sim(entry_indices, prices, highs, lows, date_ints, time_mins, tp, sl, exit_time):
    n_entries = len(entry_indices)
    n_prices  = len(prices)
    trades      = np.empty(n_entries, dtype=np.float64)
    entry_dates = np.empty(n_entries, dtype=np.int64)

    trade_count       = 0
    current_entry_ptr = 0
    last_traded_date  = -1

    while current_entry_ptr < n_entries:
        idx = entry_indices[current_entry_ptr]
        if idx >= n_prices - 1: break

        entry_date = date_ints[idx]
        if entry_date == last_traded_date:
            current_entry_ptr += 1
            continue

        entry_price  = prices[idx]
        target_price = entry_price * (1 + tp)
        stop_price   = entry_price * (1 - sl)

        outcome    = 0.0
        trade_done = False
        exit_idx   = idx

        for k in range(idx + 1, n_prices):
            if date_ints[k] != entry_date:
                outcome = (prices[k-1] - entry_price) / entry_price
                exit_idx = k; trade_done = True; break
            if lows[k] <= stop_price:
                outcome = -sl; exit_idx = k; trade_done = True; break
            if highs[k] >= target_price:
                outcome = tp;  exit_idx = k; trade_done = True; break
            if time_mins[k] >= exit_time:
                outcome = (prices[k] - entry_price) / entry_price
                exit_idx = k; trade_done = True; break

        if not trade_done:
            outcome  = (prices[n_prices-1] - entry_price) / entry_price
            exit_idx = n_prices

        trades[trade_count]      = outcome
        entry_dates[trade_count] = entry_date
        trade_count += 1
        last_traded_date = entry_date

        next_ptr = current_entry_ptr + 1
        while next_ptr < n_entries:
            if entry_indices[next_ptr] > exit_idx: break
            next_ptr += 1
        current_entry_ptr = next_ptr

    return trades[:trade_count], entry_dates[:trade_count]

# ==============================================================================
# DATA LOADING
# ==============================================================================
def _load_parquet_with_dt_index(gcs_uri, tz_out='America/Los_Angeles'):
    df = pd.read_parquet(gcs_uri)
    df = df.reset_index()
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if dt_cols:
        df['Datetime'] = pd.to_datetime(df[dt_cols[0]], utc=True)
    else:
        num_col = next((c for c in df.columns if c in ('Timestamp', 'Date', 'Time')), df.columns[0])
        df['Datetime'] = pd.to_datetime(df[num_col], unit='ms', utc=True)
    df = df.set_index('Datetime')
    df = df[df.index.notna()].sort_index()
    df = df[~df.index.duplicated(keep='first')]
    df.index = df.index.tz_convert(tz_out)
    return df

def load_ticker_ohlcv(ticker):
    clean = TICKER_FILE_MAP.get(ticker.replace("I:", "").replace("X:", ""),
                                 ticker.replace("I:", "").replace("X:", ""))
    for suffix in ['_master_inference', '_1m_full_historical_data']:
        uri = f"gs://{BUCKET_NAME}/Data/Inference_v2/{clean}{suffix}.parquet"
        try:
            return _load_parquet_with_dt_index(uri)
        except Exception:
            continue
    raise FileNotFoundError(f"No OHLCV data found for {ticker}")

def load_ensemble_probs(ticker):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    all_probs = []

    for seed in SEEDS:
        path = f"{RESULTS_DIR}/{ticker}/Seed_{seed}_Results.csv"
        blob = bucket.blob(path)
        if not blob.exists():
            continue
        df = pd.read_csv(io.BytesIO(blob.download_as_bytes()), index_col=0, parse_dates=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('America/Los_Angeles')
        all_probs.append(df['Prob_Spike'].rename(seed))

    if not all_probs:
        return None, 0

    combined = pd.concat(all_probs, axis=1)
    ensemble_mean = combined.mean(axis=1).sort_index()
    n_seeds = len(all_probs)
    return ensemble_mean, n_seeds

# ==============================================================================
# OHLCV ARRAYS
# ==============================================================================
def precompute_ohlcv_arrays(df_ohlcv, start_date, end_date):
    pst = pytz.timezone('America/Los_Angeles')
    eval_start = pd.to_datetime(start_date).tz_localize(pst)
    eval_end   = pd.to_datetime(end_date).tz_localize(pst) + timedelta(days=1)
    df = df_ohlcv[(df_ohlcv.index >= eval_start) & (df_ohlcv.index < eval_end)]
    if len(df) == 0:
        return None
    return {
        'index':  df.index,
        'prices': df['Close'].values.copy(),
        'highs':  df['High'].values.copy(),
        'lows':   df['Low'].values.copy(),
        'dates':  (df.index.year * 10000 + df.index.month * 100 + df.index.day).values.astype(np.int64),
        'times':  (df.index.hour * 60 + df.index.minute).values,
        'months': max(1.0, (df.index[-1] - df.index[0]).days / 30.44),
    }

# ==============================================================================
# OPTIMIZATION — 2 stages, 3 params, ensemble probs
# ==============================================================================
def optimize_ticker(ticker, ohlcv_arrays, ensemble_prob):
    if ohlcv_arrays is None:
        return None

    idx = ohlcv_arrays['index']
    common = idx.intersection(ensemble_prob.index)
    if len(common) == 0:
        return None

    probs = np.zeros(len(idx), dtype=np.float64)
    common_locs = idx.get_indexer(common)
    probs[common_locs] = ensemble_prob.loc[common].values

    prices = ohlcv_arrays['prices']
    highs  = ohlcv_arrays['highs']
    lows   = ohlcv_arrays['lows']
    dates  = ohlcv_arrays['dates']
    times  = ohlcv_arrays['times']
    months = ohlcv_arrays['months']

    # -- STAGE 1: coarse sweep (4 params: entry_time, prob, TP, SL) --
    best       = None
    best_score = -999.0

    coarse_combos = list(itertools.product(*GRID_COARSE.values()))
    n_combos = len(coarse_combos)

    for entry_t, prob_t, tp, sl in coarse_combos:
        h, m_ = map(int, entry_t.split(':'))
        t_min = h * 60 + m_
        mask = (probs >= prob_t) & (times == t_min)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue

        pnl, raw_dates = run_trade_sim(
            indices, prices, highs, lows, dates,
            times.astype(np.int64), tp, sl, EXIT_TIME_MIN
        )
        if len(pnl) == 0:
            continue

        freq = len(pnl) / months
        if freq < MIN_TRADES_PER_MONTH:
            continue

        avg = float(np.mean(pnl))
        if avg > best_score:
            best_score = avg
            best = {
                'Ticker': ticker,
                'entry_time': entry_t,
                'prob_thresh': prob_t,
                'tp_pct': tp,
                'sl_pct': sl,
                'Avg_PnL':   avg,
                'Total_PnL': float(np.sum(pnl)),
                'Win_Rate':  float(np.mean(pnl > 0)),
                'Freq':      freq,
                'Count':     len(pnl),
                'Stage':     1,
            }

    if best is None:
        return None

    print(f"    Stage 1: Entry={best['entry_time']} prob={best['prob_thresh']:.2f} "
          f"TP={best['tp_pct']:.0%} SL={best['sl_pct']:.0%} "
          f"AvgPnL={best['Avg_PnL']:.2%} N={best['Count']} ({n_combos:,} combos)")

    # -- STAGE 2: fine-tune all 4 jointly around Stage 1 winner --
    b_entry_h, b_entry_m = map(int, best['entry_time'].split(':'))
    b_entry_min = b_entry_h * 60 + b_entry_m
    fine_entry = list(range(max(360, b_entry_min - 2), min(391, b_entry_min + 3)))

    b_prob = best['prob_thresh']
    b_tp   = best['tp_pct']
    b_sl   = best['sl_pct']

    fine_prob = np.arange(
        max(0.01, b_prob - 0.05), min(0.99, b_prob + 0.051), 0.005
    ).round(3).tolist()
    fine_tp = np.arange(
        max(0.02, b_tp - 0.03), min(0.30, b_tp + 0.031), 0.005
    ).round(3)
    fine_sl = np.arange(
        max(0.01, b_sl - 0.03), min(0.20, b_sl + 0.031), 0.005
    ).round(3)

    fine_combos = list(itertools.product(fine_prob, fine_tp, fine_sl))

    for t_min2 in fine_entry:
        for fp, ft, fs in fine_combos:
            mask2 = (probs >= fp) & (times == t_min2)
            indices2 = np.where(mask2)[0]
            if len(indices2) == 0:
                continue

            pnl2, _ = run_trade_sim(
                indices2, prices, highs, lows, dates,
                times.astype(np.int64), float(ft), float(fs), EXIT_TIME_MIN
            )
            if len(pnl2) == 0:
                continue

            freq2 = len(pnl2) / months
            if freq2 < MIN_TRADES_PER_MONTH:
                continue

            avg2 = float(np.mean(pnl2))
            if avg2 > best_score:
                best_score = avg2
                best = {**best,
                        'entry_time': f"{t_min2//60}:{t_min2%60:02d}",
                        'prob_thresh': fp, 'tp_pct': float(ft), 'sl_pct': float(fs),
                        'Avg_PnL': avg2,
                        'Total_PnL': float(np.sum(pnl2)),
                        'Win_Rate': float(np.mean(pnl2 > 0)),
                        'Freq': freq2, 'Count': len(pnl2), 'Stage': 2}

    if best['Stage'] == 2:
        print(f"    Stage 2: Entry={best['entry_time']} prob={best['prob_thresh']:.3f} "
              f"TP={best['tp_pct']:.1%} SL={best['sl_pct']:.1%} "
              f"AvgPnL={best['Avg_PnL']:.2%} N={best['Count']}")
    else:
        print(f"    Stage 2: no improvement")

    return best

# ==============================================================================
# BACKTEST — test set with optimized params
# ==============================================================================
def backtest_on_test(ticker, df_ohlcv, ensemble_prob, opt_result):
    ohlcv_test = precompute_ohlcv_arrays(df_ohlcv, TEST_START_DATE, TEST_END_DATE)
    if ohlcv_test is None:
        return None

    idx = ohlcv_test['index']
    common = idx.intersection(ensemble_prob.index)
    if len(common) == 0:
        return None

    probs = np.zeros(len(idx), dtype=np.float64)
    common_locs = idx.get_indexer(common)
    probs[common_locs] = ensemble_prob.loc[common].values

    h, m_ = map(int, opt_result['entry_time'].split(':'))
    entry_min = h * 60 + m_
    mask = (probs >= opt_result['prob_thresh']) & (ohlcv_test['times'] == entry_min)
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return None

    pnl, _ = run_trade_sim(
        indices, ohlcv_test['prices'], ohlcv_test['highs'], ohlcv_test['lows'],
        ohlcv_test['dates'], ohlcv_test['times'].astype(np.int64),
        opt_result['tp_pct'], opt_result['sl_pct'], EXIT_TIME_MIN
    )
    if len(pnl) == 0:
        return None

    months = ohlcv_test['months']
    return {
        'Ticker':    ticker,
        'Avg_PnL':   float(np.mean(pnl)),
        'Total_PnL': float(np.sum(pnl)),
        'Win_Rate':  float(np.mean(pnl > 0)),
        'Freq':      len(pnl) / months,
        'Count':     len(pnl),
    }

# ==============================================================================
# MAIN
# ==============================================================================
def run_optimization():
    if auth is not None:
        try: auth.authenticate_user()
        except Exception: pass

    n_coarse = len(list(itertools.product(*GRID_COARSE.values())))
    print(f"\n{'='*80}")
    print(f"  EXP-104 v2 — ENSEMBLE OPTIMIZATION (Defense)")
    print(f"  Model: {MODEL_ID}  |  Seeds: {SEEDS[0]}-{SEEDS[-1]} (ensemble of {len(SEEDS)})")
    print(f"  Entry: {ENTRY_TIMES[0]}-{ENTRY_TIMES[-1]} PST (optimized)  |  No regime filters")
    print(f"  Opt: {OPT_START_DATE} -> {OPT_END_DATE}")
    print(f"  Test: {TEST_START_DATE} -> {TEST_END_DATE}")
    print(f"  Min trades/month: {MIN_TRADES_PER_MONTH}")
    print(f"  Stage 1: {n_coarse:,} combos (entry x prob x TP x SL)")
    print(f"  Stage 2: fine-tune all 4 around Stage 1 winner")
    print(f"{'='*80}\n")

    all_opt = []
    all_test = []

    for ticker in SCORING_POOL:
        print(f"\n{'~'*60}")
        print(f"  {ticker}")
        print(f"{'~'*60}")

        ensemble_prob, n_seeds = load_ensemble_probs(ticker)
        if ensemble_prob is None:
            print(f"  No seed data found.")
            continue
        print(f"  Ensemble: {n_seeds} seeds averaged")

        try:
            df_ohlcv = load_ticker_ohlcv(ticker)
        except Exception as e:
            print(f"  Could not load OHLCV: {e}")
            continue

        # --- Ensemble prob distribution ---
        pst = pytz.timezone('America/Los_Angeles')
        opt_start = pd.to_datetime(OPT_START_DATE).tz_localize(pst)
        opt_end = pd.to_datetime(OPT_END_DATE).tz_localize(pst) + timedelta(days=1)
        opt_probs = ensemble_prob[(ensemble_prob.index >= opt_start) & (ensemble_prob.index < opt_end)]
        if len(opt_probs) > 0:
            p50 = opt_probs.median()
            p90 = opt_probs.quantile(0.90)
            p95 = opt_probs.quantile(0.95)
            p99 = opt_probs.quantile(0.99)
            print(f"  Prob distribution (opt): median={p50:.3f} p90={p90:.3f} "
                  f"p95={p95:.3f} p99={p99:.3f}")

        # --- Optimize ---
        ohlcv_opt = precompute_ohlcv_arrays(df_ohlcv, OPT_START_DATE, OPT_END_DATE)
        if ohlcv_opt is None:
            print(f"  No OHLCV data in opt window.")
            del df_ohlcv; gc.collect()
            continue

        result = optimize_ticker(ticker, ohlcv_opt, ensemble_prob)
        if result is None:
            print(f"  No valid configuration found.")
            del df_ohlcv; gc.collect()
            continue

        all_opt.append(result)
        is_teacher = "T" if ticker in TRAIN_TEACHERS else " "
        print(f"  OPT:{is_teacher} AvgPnL={result['Avg_PnL']:.2%}  WR={result['Win_Rate']:.0%}  "
              f"N={result['Count']}  Freq={result['Freq']:.1f}/mo  "
              f"Entry={result['entry_time']}  Prob>{result['prob_thresh']:.3f}  "
              f"TP={result['tp_pct']:.1%}  SL={result['sl_pct']:.1%}")

        # --- Test ---
        bt = backtest_on_test(ticker, df_ohlcv, ensemble_prob, result)
        if bt is not None:
            all_test.append({**bt, **{k: result[k] for k in
                ['entry_time', 'prob_thresh', 'tp_pct', 'sl_pct']}})
            print(f"  TEST: AvgPnL={bt['Avg_PnL']:.2%}  WR={bt['Win_Rate']:.0%}  "
                  f"N={bt['Count']}  Freq={bt['Freq']:.1f}/mo")
        else:
            print(f"  TEST: no trades")

        del df_ohlcv, ohlcv_opt
        gc.collect()

    # -- SUMMARY --
    if not all_opt:
        print("\nNo results.")
        return

    df_opt = pd.DataFrame(all_opt).sort_values('Avg_PnL', ascending=False)

    print(f"\n\n{'='*100}")
    print(f"  OPTIMIZATION RESULTS ({OPT_START_DATE} -> {OPT_END_DATE})")
    print(f"  Ensemble of {len(SEEDS)} seeds | Entry optimized | No regime filters")
    print(f"{'='*100}")
    print(f"{'Ticker':<7} {'AvgPnL':>8} {'TotPnL':>8} {'WR':>6} {'Freq':>6} "
          f"{'Entry':>6} {'Prob':>7} {'TP':>6} {'SL':>6} {'N':>5} {'St':>3}")
    print("~" * 100)
    for _, r in df_opt.iterrows():
        is_teacher = "T" if r['Ticker'] in TRAIN_TEACHERS else " "
        print(f"{r['Ticker']:<6}{is_teacher} {r['Avg_PnL']:>7.2%} {r['Total_PnL']:>7.2%} "
              f"{r['Win_Rate']:>5.0%} {r['Freq']:>5.1f}/mo "
              f"{r['entry_time']:>6} {r['prob_thresh']:>7.3f} {r['tp_pct']:>5.1%} {r['sl_pct']:>5.1%} "
              f"{int(r['Count']):>5} {int(r.get('Stage',1)):>3}")
    print("~" * 100)
    print(f"{'MEAN':<7} {df_opt['Avg_PnL'].mean():>7.2%} "
          f"{df_opt['Total_PnL'].mean():>7.2%} {df_opt['Win_Rate'].mean():>5.0%}")

    # -- TEST RESULTS --
    if all_test:
        df_test = pd.DataFrame(all_test).sort_values('Avg_PnL', ascending=False)

        print(f"\n\n{'='*100}")
        print(f"  TEST RESULTS ({TEST_START_DATE} -> {TEST_END_DATE})  [HELD-OUT]")
        print(f"{'='*100}")
        print(f"{'Ticker':<7} {'AvgPnL':>8} {'TotPnL':>8} {'WR':>6} {'Freq':>6} "
              f"{'Entry':>6} {'Prob':>7} {'TP':>6} {'SL':>6} {'N':>5}")
        print("~" * 100)
        for _, r in df_test.iterrows():
            is_teacher = "T" if r['Ticker'] in TRAIN_TEACHERS else " "
            print(f"{r['Ticker']:<6}{is_teacher} {r['Avg_PnL']:>7.2%} {r['Total_PnL']:>7.2%} "
                  f"{r['Win_Rate']:>5.0%} {r['Freq']:>5.1f}/mo "
                  f"{r['entry_time']:>6} {r['prob_thresh']:>7.3f} {r['tp_pct']:>5.1%} {r['sl_pct']:>5.1%} "
                  f"{int(r['Count']):>5}")
        print("~" * 100)
        print(f"{'MEAN':<7} {df_test['Avg_PnL'].mean():>7.2%} "
              f"{df_test['Total_PnL'].mean():>7.2%} {df_test['Win_Rate'].mean():>5.0%}")

        # -- OPT vs TEST DECAY --
        print(f"\n  OPT -> TEST:")
        print(f"  {'Ticker':<7} {'OPT':>8} {'TEST':>8} {'Decay':>7} {'OPT_N':>6} {'TEST_N':>6}")
        print(f"  {'~'*50}")
        for _, t_row in df_test.iterrows():
            tk = t_row['Ticker']
            opt_row = df_opt[df_opt['Ticker'] == tk]
            if not opt_row.empty:
                o = opt_row.iloc[0]
                decay = 1 - (t_row['Avg_PnL'] / o['Avg_PnL']) if o['Avg_PnL'] != 0 else 0
                print(f"  {tk:<7} {o['Avg_PnL']:>7.2%} {t_row['Avg_PnL']:>7.2%} "
                      f"{decay:>6.0%} {int(o['Count']):>6} {int(t_row['Count']):>6}")

        deploy = df_test[df_test['Avg_PnL'] > 0.018]
        if not deploy.empty:
            print(f"\n  Deploy candidates (test AvgPnL > 1.8%):")
            for _, r in deploy.iterrows():
                tag = "TEACHER" if r['Ticker'] in TRAIN_TEACHERS else "OOS"
                print(f"    {r['Ticker']:<7} AvgPnL={r['Avg_PnL']:.2%}  "
                      f"WR={r['Win_Rate']:.0%}  N={int(r['Count'])}  "
                      f"Freq={r['Freq']:.1f}/mo  [{tag}]")
        else:
            print(f"\n  No tickers with test AvgPnL > 1.8%")

        print(f"\n  v1 (per-seed, 7 params) vs v2 (ensemble, 4 params):")
        print(f"  Compare TEST columns — lower OPT but higher TEST = less overfit")
    else:
        print(f"\n  No test results generated.")

    print(f"\n{'='*100}\n")


if __name__ == "__main__":
    run_optimization()
