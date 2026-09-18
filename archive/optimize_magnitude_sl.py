"""
optimize_magnitude_tp.py — WF Optimizer for Magnitude-Based TP
==============================================================
Uses BOTH binary spike model (Prob_Spike) AND magnitude model predictions
to set dynamic per-trade take-profit targets.

Binary model → decides WHEN to trade (prob > threshold)
Magnitude model → decides WHERE to set TP (predicted spike magnitude)

Two modes:
  MODEL_TYPE = 'classification'
    Uses predicted magnitude class to look up SL from a fixed table.
    Optimizes: prob_thresh, class_sl_mapping

  MODEL_TYPE = 'regression'
    Uses predicted magnitude × sl_ratio as SL.
    Optimizes: prob_thresh, sl_ratio, sl_floor

Walk-forward: same R1/R2/TEST structure as optimize_wf_robust.py.
Results saved to BQ with model_version = 'mag_sl_v1'.
"""

import gc, io, uuid, warnings
import pandas as pd, numpy as np, itertools
from google.cloud import storage, bigquery as bq
from numba import jit
try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass
import pytz
from datetime import timedelta

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIG
# ==============================================================================
MODEL_TYPE = 'classification'  # 'classification' or 'regression'

BUCKET_NAME = os.environ.get("GCS_BUCKET", "")
BQ_TABLE_ID = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"

BINARY_EXP = '185_XSECTOR'
MAG_EXP_CLS = '185_MAG_CLS'
MAG_EXP_REG = '185_MAG_REG'
MAG_EXP = MAG_EXP_CLS if MODEL_TYPE == 'classification' else MAG_EXP_REG

MODEL_VERSION_TAG = 'mag_sl_v1'
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

TEST_START = '2026-04-01'
TEST_END = '2026-05-21'
WF_ROUNDS = [
    {'name': 'R1', 'opt': ('2026-01-02', '2026-02-28'),
                   'val': ('2026-03-01', '2026-03-31'), 'min_trades': 2},
]

EXIT_TIME_MIN = 780
MIN_TRADES_PER_MONTH = 2.0
SLIPPAGE_PCT = 0.0015

# --- Classification TP grid (class → TP target) ---
CLS_TP_MAPS = [
    {'name': 'conservative', 0: 0.02, 1: 0.03, 2: 0.05, 3: 0.07, 4: 0.10},
    {'name': 'moderate',     0: 0.02, 1: 0.03, 2: 0.055, 3: 0.085, 4: 0.12},
    {'name': 'aggressive',   0: 0.03, 1: 0.04, 2: 0.06, 3: 0.09, 4: 0.15},
    {'name': 'wide',         0: 0.03, 1: 0.05, 2: 0.08, 3: 0.12, 4: 0.20},
]

# --- Regression TP grid (TP = pred_mag × ratio, with floor/cap) ---
REG_TP_RATIOS = [0.6, 0.7, 0.8, 0.9, 1.0]
REG_TP_FLOORS = [0.02, 0.03, 0.05]
REG_TP_CAPS = [0.15, 0.20, 0.25]

# --- Shared grid ---
PROB_THRESH_GRID = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80]
ENTRY_TIMES = ["06:15", "06:20", "06:25"]
SL_PCT_GRID = [0.03, 0.05, 0.08, 0.10, 0.14]

FALLBACK_TICKERS = [
    'LUNR', 'RCAT', 'ASTS', 'BBAI', 'ACHR', 'MVIS',
    'CIFR', 'IREN', 'HUT', 'KEEL', 'CLSK', 'APLD',
    'WULF', 'MARA', 'RIOT', 'MSTR', 'RKLB', 'JOBY',
    'ALAB', 'BGDE', 'PLUG', 'FCEL', 'BTBT', 'BE',
    'OKLO', 'SMR', 'FLNC', 'COIN', 'HOOD', 'SMCI',
    'KTOS', 'POWL', 'MOD', 'NNDM', 'CORZ', 'NBIS',
    'AVAV', 'MP', 'PL', 'RDW',
]
EXCLUDE_TICKERS = {'TLT', 'VIX', 'SMLR', 'SPIR'}
TICKERS = [t for t in FALLBACK_TICKERS if t not in EXCLUDE_TICKERS]

_bq_client = bq.Client()
gcs_client = storage.Client()

print(f"\n{'='*80}")
print(f"  MAGNITUDE SL OPTIMIZER — {MODEL_TYPE.upper()}")
print(f"  Binary model: {BINARY_EXP}  |  Magnitude model: {MAG_EXP}")
print(f"  Version: {MODEL_VERSION_TAG}")
print(f"{'='*80}\n")

# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_parquet_pst(gcs_uri):
    df = pd.read_parquet(gcs_uri)
    df = df.reset_index()
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if dt_cols:
        df['Datetime'] = pd.to_datetime(df[dt_cols[0]], utc=True)
    else:
        df['Datetime'] = pd.to_datetime(df[df.columns[0]], unit='ms', utc=True)
    df = df.set_index('Datetime').sort_index()
    df = df[~df.index.duplicated(keep='first')]
    df.index = df.index.tz_convert('America/Los_Angeles')
    std_map = {'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'}
    lower_map = {k.lower():v for k,v in std_map.items()}
    df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip()) for c in df.columns]
    return df


def load_seed_scores(ticker, experiment):
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    prefix = f"Algos/Results_all_v2/Universal_Exp{experiment}/{clean}/Seed_"
    blobs = list(gcs_client.bucket(BUCKET_NAME).list_blobs(prefix=prefix))
    seeds = {}
    for blob in blobs:
        name = blob.name.split('/')[-1]
        if not name.startswith('Seed_') or not name.endswith('_Results.csv'):
            continue
        seed_id = int(name.replace('Seed_', '').replace('_Results.csv', ''))
        df = pd.read_csv(io.BytesIO(blob.download_as_bytes()), index_col=0, parse_dates=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df.index = df.index.tz_convert('America/Los_Angeles')
        seeds[seed_id] = df
    return seeds


def load_ticker_ohlcv(ticker):
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    for suffix in ['_master_inference', '_1m_full_historical_data']:
        uri = f"gs://{BUCKET_NAME}/Data/Inference_v2/{clean}{suffix}.parquet"
        try:
            return load_parquet_pst(uri)
        except Exception:
            continue
    return None


# ==============================================================================
# TRADE SIMULATION (Numba)
# ==============================================================================
@jit(nopython=True)
def run_trade_mag(entry_indices, prices, highs, lows, date_ints, time_mins,
                  tp_arr, sl_arr, exit_time):
    n_entries = len(entry_indices)
    n_prices = len(prices)
    trades = np.empty(n_entries, dtype=np.float64)
    entry_dates = np.empty(n_entries, dtype=np.int64)
    trade_count = 0
    last_traded_date = -1

    for ptr in range(n_entries):
        idx = entry_indices[ptr]
        if idx >= n_prices - 1:
            break
        entry_date = date_ints[idx]
        if entry_date == last_traded_date:
            continue

        ep = prices[idx]
        tp = tp_arr[ptr]
        sl = sl_arr[ptr]
        tp_price = ep * (1 + tp)
        sl_price = ep * (1 - sl)

        outcome = 0.0
        for k in range(idx + 1, n_prices):
            if date_ints[k] != entry_date:
                outcome = (prices[k-1] - ep) / ep
                break
            if lows[k] <= sl_price and highs[k] >= tp_price:
                outcome = -sl if np.random.random() < 0.5 else tp
                break
            if lows[k] <= sl_price:
                outcome = -sl
                break
            if highs[k] >= tp_price:
                outcome = tp
                break
            if time_mins[k] >= exit_time:
                outcome = (prices[k] - ep) / ep
                break
        else:
            outcome = (prices[n_prices-1] - ep) / ep

        trades[trade_count] = outcome
        entry_dates[trade_count] = entry_date
        trade_count += 1
        last_traded_date = entry_date

    return trades[:trade_count], entry_dates[:trade_count]


# ==============================================================================
# MAGNITUDE TP COMPUTATION
# ==============================================================================
def get_magnitude_tp(mag_data, indices, tp_params):
    n = len(indices)
    tp_arr = np.full(n, 0.16, dtype=np.float64)

    if MODEL_TYPE == 'classification':
        tp_map = tp_params
        for i, idx in enumerate(indices):
            if idx < len(mag_data):
                pred_class = int(mag_data['pred_class'].iloc[idx])
                tp_arr[i] = tp_map.get(pred_class, 0.16)

    else:
        ratio = tp_params['ratio']
        floor = tp_params['floor']
        cap = tp_params['cap']
        for i, idx in enumerate(indices):
            if idx < len(mag_data):
                pred_mag = float(mag_data['pred_mag'].iloc[idx])
                tp = max(pred_mag * ratio, floor)
                tp = min(tp, cap)
                tp_arr[i] = tp

    return tp_arr


# ==============================================================================
# GRID SEARCH
# ==============================================================================
def run_grid_search(ohlcv_df, binary_probs, mag_data, start_date, end_date,
                    tp_params_list, min_trades):
    pst = pytz.timezone('America/Los_Angeles')
    eval_start = pd.Timestamp(start_date, tz=pst)
    eval_end = pd.Timestamp(end_date, tz=pst) + timedelta(days=1)

    df = ohlcv_df[(ohlcv_df.index >= eval_start) & (ohlcv_df.index < eval_end)]
    if len(df) == 0:
        return []

    prices = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    dates = (df.index.year*10000 + df.index.month*100 + df.index.day).values.astype(np.int64)
    times = (df.index.hour*60 + df.index.minute).values
    months = max(1.0, (df.index[-1] - df.index[0]).days / 30.44)

    common = df.index.intersection(binary_probs.index)
    probs = np.zeros(len(df), dtype=np.float64)
    if len(common) > 0:
        locs = df.index.get_indexer(common)
        probs[locs] = binary_probs.loc[common].values

    mag_aligned = mag_data.reindex(df.index)

    results = []
    for entry_time in ENTRY_TIMES:
        h, m = map(int, entry_time.split(':'))
        t_min = h * 60 + m

        for prob_thresh in PROB_THRESH_GRID:
            mask = (probs >= prob_thresh) & (times == t_min)
            entry_idx = np.where(mask)[0]
            if len(entry_idx) == 0:
                continue

            for sl_pct in SL_PCT_GRID:
                sl_arr = np.full(len(entry_idx), sl_pct, dtype=np.float64)

                for tp_params in tp_params_list:
                    tp_arr = get_magnitude_tp(mag_aligned, entry_idx, tp_params)

                    pnl, trade_dates = run_trade_mag(
                        entry_idx, prices, highs, lows, dates, times,
                        tp_arr, sl_arr, EXIT_TIME_MIN)

                    if len(pnl) > 0:
                        pnl = pnl - 2 * SLIPPAGE_PCT

                    if len(pnl) < min_trades:
                        continue
                    freq = len(pnl) / months
                    if freq < MIN_TRADES_PER_MONTH:
                        continue

                    tp_name = tp_params.get('name', f"r{tp_params.get('ratio','')}")
                    results.append({
                        'entry_time': entry_time,
                        'prob_thresh': prob_thresh,
                        'sl_pct': sl_pct,
                        'tp_config': tp_name,
                        'Avg_PnL': float(np.mean(pnl)),
                        'Win_Rate': float(np.mean(pnl > 0)),
                        'Count': len(pnl),
                        'Freq': freq,
                        'avg_tp': float(np.mean(tp_arr)),
                    })

    return results


# ==============================================================================
# BQ SAVE
# ==============================================================================
def save_to_bq(row, run_id, ticker, seed):
    try:
        h, m = map(int, row['entry_time'].split(':'))
        bq_row = {
            'run_id': run_id,
            'run_timestamp': pd.Timestamp.utcnow().isoformat(),
            'experiment': MAG_EXP,
            'ticker': ticker, 'seed': seed,
            'entry_time': row['entry_time'], 'entry_minute': h*60+m,
            'prob_thresh': row['prob_thresh'],
            'tp_pct': row.get('avg_tp', 0.16), 'sl_pct': row.get('sl_pct', 0.08),
            'vix_threshold': 999.0, 'tlt_crash': -999.0, 'spy_ignore': 999.0,
            'min_vol_bars': 0, 'vol_filter_pct': 0,
            'opt_avg_pnl': row['Avg_PnL'], 'opt_win_rate': row['Win_Rate'],
            'opt_freq': row['Freq'], 'opt_count': row['Count'],
            'val_a_avg_pnl': row.get('val_a_avg_pnl', 0),
            'val_a_win_rate': row.get('val_a_win_rate', 0),
            'val_a_freq': row.get('val_a_freq', 0),
            'val_a_count': row.get('val_a_count', 0),
            'survived_a': row.get('survived_a', False),
            'val_b_avg_pnl': row.get('val_b_avg_pnl', 0),
            'val_b_win_rate': row.get('val_b_win_rate', 0),
            'val_b_freq': row.get('val_b_freq', 0),
            'val_b_count': row.get('val_b_count', 0),
            'survived_b': row.get('survived_b', False),
            'opt_start': WF_ROUNDS[0]['opt'][0],
            'opt_end': WF_ROUNDS[0]['val'][1],
            'val_a_start': WF_ROUNDS[0]['val'][0],
            'val_a_end': WF_ROUNDS[0]['val'][1],
            'val_b_start': TEST_START, 'val_b_end': TEST_END,
            'tpsl_mode': f'mag_{MODEL_TYPE[:3]}',
            'model_version': MODEL_VERSION_TAG,
            'v0_fixed': True,
            'is_ensemble': False, 'n_seeds_avg': 0,
        }
        errors = _bq_client.insert_rows_json(BQ_TABLE_ID, [bq_row])
        if errors:
            print(f"      BQ error: {errors[0]}")
        else:
            print(f"      BQ: saved {ticker}/S{seed}")
    except Exception as e:
        print(f"      BQ save failed: {e}")


# ==============================================================================
# MAIN
# ==============================================================================
if MODEL_TYPE == 'classification':
    tp_params_list = CLS_TP_MAPS
else:
    tp_params_list = [
        {'name': f'r{r}_f{f}_c{c}', 'ratio': r, 'floor': f, 'cap': c}
        for r in REG_TP_RATIOS for f in REG_TP_FLOORS for c in REG_TP_CAPS
    ]

print(f"  TP configurations: {len(tp_params_list)}")
print(f"  Prob thresholds: {PROB_THRESH_GRID}")
print(f"  SL grid: {SL_PCT_GRID}")
print(f"  Entry times: {ENTRY_TIMES}")
total_combos = len(ENTRY_TIMES) * len(PROB_THRESH_GRID) * len(SL_PCT_GRID) * len(tp_params_list)
print(f"  Total combos per seed: {total_combos}")

RUN_ID = str(uuid.uuid4())
print(f"  BQ run_id: {RUN_ID[:8]}...\n")

for ticker in TICKERS:
    print(f"\n{'='*60}")
    print(f"  {ticker}")
    print(f"{'='*60}")

    df_ohlcv = load_ticker_ohlcv(ticker)
    if df_ohlcv is None:
        print(f"  No OHLCV data — skipping")
        continue

    binary_seeds = load_seed_scores(ticker, BINARY_EXP)
    mag_seeds = load_seed_scores(ticker, MAG_EXP)

    if not binary_seeds:
        print(f"  No binary seed scores — skipping")
        continue
    if not mag_seeds:
        print(f"  No magnitude seed scores — skipping")
        continue

    common_seeds = sorted(set(binary_seeds.keys()) & set(mag_seeds.keys()))
    if not common_seeds:
        print(f"  No common seeds between binary and magnitude — skipping")
        continue
    print(f"  {len(common_seeds)} common seeds: {common_seeds[0]}-{common_seeds[-1]}")

    for seed in common_seeds:
        print(f"\n    Seed {seed}:")

        binary_prob = binary_seeds[seed]['Prob_Spike']

        mag_df = mag_seeds[seed]
        if MODEL_TYPE == 'classification':
            if 'Pred_Class' in mag_df.columns:
                mag_data = pd.DataFrame({'pred_class': mag_df['Pred_Class']}, index=mag_df.index)
            else:
                pred_class = mag_df[['Prob_C0','Prob_C1','Prob_C2','Prob_C3','Prob_C4']].values.argmax(axis=1)
                mag_data = pd.DataFrame({'pred_class': pred_class}, index=mag_df.index)
        else:
            mag_data = pd.DataFrame({'pred_mag': mag_df['Pred_Mag']}, index=mag_df.index)

        # --- R1: Optimize on OPT period, validate on VAL ---
        r1 = WF_ROUNDS[0]
        opt_results = run_grid_search(
            df_ohlcv, binary_prob, mag_data,
            r1['opt'][0], r1['opt'][1], tp_params_list, r1['min_trades'])

        if not opt_results:
            print(f"      R1: no valid configs")
            continue

        opt_results.sort(key=lambda x: x['Avg_PnL'], reverse=True)
        survivors = [r for r in opt_results if r['Avg_PnL'] > 0][:20]

        if not survivors:
            print(f"      R1: no positive PnL configs")
            continue

        best = None
        best_val_pnl = -999
        for cfg in survivors:
            val_results = run_grid_search(
                df_ohlcv, binary_prob, mag_data,
                r1['val'][0], r1['val'][1],
                [next(s for s in tp_params_list if s.get('name') == cfg['tp_config'])],
                r1['min_trades'])
            val_match = [v for v in val_results
                         if v['entry_time'] == cfg['entry_time']
                         and v['prob_thresh'] == cfg['prob_thresh']
                         and v['sl_pct'] == cfg['sl_pct']]
            if val_match and val_match[0]['Avg_PnL'] > best_val_pnl:
                best_val_pnl = val_match[0]['Avg_PnL']
                best = cfg
                best['val_a_avg_pnl'] = val_match[0]['Avg_PnL']
                best['val_a_win_rate'] = val_match[0]['Win_Rate']
                best['val_a_count'] = val_match[0]['Count']
                best['val_a_freq'] = val_match[0]['Freq']
                best['survived_a'] = val_match[0]['Avg_PnL'] > 0

        if best is None or best_val_pnl <= 0:
            print(f"      No config survived validation")
            continue

        print(f"      OPT: AvgPnL={best['Avg_PnL']:.2%}  WR={best['Win_Rate']:.1%}  "
              f"N={best['Count']}  TP={best['tp_config']}  SL={best.get('sl_pct',0)*100:.0f}%  prob>{best['prob_thresh']}")
        print(f"      VAL: AvgPnL={best['val_a_avg_pnl']:.2%}")

        # --- TEST ---
        test_results = run_grid_search(
            df_ohlcv, binary_prob, mag_data,
            TEST_START, TEST_END,
            [next(s for s in tp_params_list if s.get('name') == best['tp_config'])],
            1)
        test_match = [v for v in test_results
                      if v['entry_time'] == best['entry_time']
                      and v['prob_thresh'] == best['prob_thresh']
                      and v['sl_pct'] == best['sl_pct']]

        if test_match:
            t = test_match[0]
            best['val_b_avg_pnl'] = t['Avg_PnL']
            best['val_b_win_rate'] = t['Win_Rate']
            best['val_b_count'] = t['Count']
            best['val_b_freq'] = t['Freq']
            best['survived_b'] = t['Avg_PnL'] > 0
            tag = "PASS" if t['Avg_PnL'] > 0 else "FAIL"
            print(f"      TEST: AvgPnL={t['Avg_PnL']:.2%}  WR={t['Win_Rate']:.1%}  N={t['Count']}  [{tag}]")
        else:
            best['val_b_avg_pnl'] = 0
            best['val_b_win_rate'] = 0
            best['val_b_count'] = 0
            best['val_b_freq'] = 0
            best['survived_b'] = False
            print(f"      TEST: no trades")

        save_to_bq(best, RUN_ID, ticker, seed)

    del df_ohlcv; gc.collect()

print(f"\n{'='*60}")
print(f"  OPTIMIZATION COMPLETE — {MODEL_TYPE.upper()}")
print(f"  Results in BQ: model_version='{MODEL_VERSION_TAG}'")
print(f"{'='*60}")
