# %% ── Post-Trade SHORT — 4-Strategy EQ-WT Comparison ────────────────────────
#
# Side-by-side equal-weight PnL comparison (capital split equally per entry):
#   1) BQ LOGGED   — deployed M_CONSENSUS GO signals (fire-count ranked)
#   2) 9-MODEL     — all BQ candidates that fired, re-ranked by prob, top 3
#   3) 5-MODEL     — same but only 5 newest model versions
#   4) ORACLE      — hindsight best 3 by actual PnL
#
# PnL = avg(trade PnLs) per entry point. Combined = avg of both windows.
# ALL deploy tickers simulated with complete master_inference data.
# Colab-ready. GPU not required.

import os, sys, subprocess, warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import pytz

warnings.filterwarnings('ignore')

try:
    from google.cloud import bigquery
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "google-cloud-bigquery", "--quiet"])
    from google.cloud import bigquery

from google.cloud import storage as gcs_storage

try:
    from google.colab import auth
    _IN_COLAB = True
    try:
        auth.authenticate_user()
    except Exception:
        pass
except ImportError:
    _IN_COLAB = False

import matplotlib
if not _IN_COLAB:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ==============================================================================
# CONFIG
# ==============================================================================
BUCKET_NAME      = os.environ.get("GCS_BUCKET", "")
WF_BQ_TABLE      = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"
BQ_PREDICTIONS   = f"{GCP_PROJECT}.scoring_logs.live_predictions"
INFERENCE_DIR    = "Data/Inference_v2"
PST              = pytz.timezone('America/Los_Angeles')

ANALYSIS_DATE    = '2026-08-24'

SCORE_WINDOWS = [
    {'label': '6:15→6:30', 'score_min': 375, 'entry_min': 390},
    {'label': '6:45→7:00', 'score_min': 405, 'entry_min': 420},
]

EXIT_TIME_MIN    = 780
SLIPPAGE_PCT     = 0.0015

BASE_PNL_THRESH  = 0.018
F7_VAL_B_THRESH  = 0.030

MAX_TRADES       = 3

TICKER_FILE_MAP  = {'BITF': 'KEEL', 'MIGI': 'BGDE'}
EXCLUDE_TICKERS  = {'SPY', 'VIX', 'TLT', 'IWM', 'XLE', 'ITA', 'BITF', 'MIGI',
                     'SMLR', 'BTCUSD', 'ETHUSD'}

FAMILIES = {
    'L5-1': {
        'bq_versions': {
            5: 'spike_transformer_short_L5_1_fold5',
            6: 'spike_transformer_short_L5_1_fold6',
            7: 'spike_transformer_short_L5_1_fold7',
        },
    },
    'L5-2': {
        'bq_versions': {
            5: 'spike_transformer_short_L5_2_fold5',
            6: 'spike_transformer_short_L5_2_fold6',
            7: 'spike_transformer_short_L5_2_fold7',
        },
    },
    'MR-v2': {
        'bq_versions': {
            6: 'spike_transformer_short_L5_2_multires_v2_fold6',
            7: 'spike_transformer_short_L5_2_multires_v2_fold7',
            8: 'spike_transformer_short_L5_2_multires_v2_fold8',
        },
    },
}

DEPLOYED_STRATEGY = 'ST_SHORT_HYB_MCONS5'

NEWEST_5_VERSIONS = {
    'spike_transformer_short_L5_1_fold7',
    'spike_transformer_short_L5_2_fold7',
    'spike_transformer_short_L5_2_multires_v2_fold6',
    'spike_transformer_short_L5_2_multires_v2_fold7',
    'spike_transformer_short_L5_2_multires_v2_fold8',
}

ALL_9_VERSIONS = set()
for fam in FAMILIES.values():
    for mv in fam['bq_versions'].values():
        ALL_9_VERSIONS.add(mv)

W = 120

bq_client  = bigquery.Client()
gcs_client = gcs_storage.Client()

_NEEDED_COLS = {'Open', 'High', 'Low', 'Close', 'Volume',
                'open', 'high', 'low', 'close', 'volume',
                'Datetime', 'Date', 'Time', 'Timestamp', 'Level_0'}


# ==============================================================================
# BQ CONFIG LOADING
# ==============================================================================
def load_deploy_configs():
    family_fold_cfgs = {}
    for family_name, fam in FAMILIES.items():
        for fold_id, model_version in fam['bq_versions'].items():
            q = f"""
            SELECT ticker, experiment, tpsl_mode, prob_thresh, tp_pct, sl_pct,
                   grace_minutes, dip_pct, max_wait_minutes,
                   val_b_avg_pnl
            FROM `{WF_BQ_TABLE}`
            WHERE model_version = '{model_version}'
              AND survived_a = TRUE
              AND val_b_avg_pnl >= {BASE_PNL_THRESH}
            """
            rows = bq_client.query(q).to_dataframe()
            if len(rows) == 0:
                family_fold_cfgs[(family_name, fold_id)] = {}
                continue
            best = rows.loc[rows.groupby('ticker')['val_b_avg_pnl'].idxmax()]
            cfgs = {}
            for _, row in best.iterrows():
                tk = row['ticker']
                if tk in EXCLUDE_TICKERS:
                    continue
                mode = row['tpsl_mode']
                if pd.isna(mode) or mode == '':
                    exp = row['experiment']
                    if '_GRACE' in exp:
                        mode = 'grace'
                    elif '_LIMIT' in exp:
                        mode = 'limit_entry'
                    else:
                        mode = 'static'
                cfgs[tk] = {
                    'family': family_name, 'fold': fold_id,
                    'prob_thresh': float(row['prob_thresh']),
                    'tp_pct': float(row['tp_pct']),
                    'sl_pct': float(row['sl_pct']),
                    'mode': mode,
                    'grace_minutes': int(row['grace_minutes']) if pd.notna(row['grace_minutes']) else 0,
                    'dip_pct': float(row['dip_pct']) if pd.notna(row['dip_pct']) else 0,
                    'max_wait': int(row['max_wait_minutes']) if pd.notna(row['max_wait_minutes']) else 0,
                    'val_b_avg_pnl': float(row['val_b_avg_pnl']),
                }
            family_fold_cfgs[(family_name, fold_id)] = cfgs

    deploy = {}
    all_tickers = set()
    for cfgs in family_fold_cfgs.values():
        all_tickers |= set(cfgs.keys())
    for tk in sorted(all_tickers):
        best_val, best_cfg = -999, None
        for (fam, fold), cfgs in family_fold_cfgs.items():
            if tk not in cfgs:
                continue
            cfg = cfgs[tk]
            v = cfg['val_b_avg_pnl']
            if fold >= 7 and v < F7_VAL_B_THRESH:
                continue
            if v > best_val:
                best_val = v
                best_cfg = cfg
        if best_cfg is not None:
            deploy[tk] = best_cfg
    return deploy


# ==============================================================================
# BQ LIVE PREDICTIONS
# ==============================================================================
def load_bq_predictions(date_str):
    q = f"""
    SELECT ticker, prob, threshold, signal, tp_pct, sl_pct, tpsl_mode,
           close as bq_price, model_version, strategy, entry_minute,
           run_timestamp
    FROM `{BQ_PREDICTIONS}`
    WHERE score_date = '{date_str}'
      AND strategy LIKE '%SHORT%'
    ORDER BY entry_minute, run_timestamp DESC, ticker
    """
    return bq_client.query(q).to_dataframe()


# ==============================================================================
# MASTER INFERENCE LOADER
# ==============================================================================
def load_master_day(ticker, date_str):
    clean = ticker.replace("I:", "").replace("X:", "")
    file_ticker = TICKER_FILE_MAP.get(clean, clean)
    blob_path = f"{INFERENCE_DIR}/{file_ticker}_master_inference.parquet"
    local_path = f"/tmp/_pt_{file_ticker}_master.parquet"
    try:
        gcs_client.bucket(BUCKET_NAME).blob(blob_path).download_to_filename(local_path)
        import pyarrow.parquet as pq
        schema = pq.read_schema(local_path)
        all_cols = [f.name for f in schema]
        use_cols = [c for c in all_cols if c in _NEEDED_COLS]
        df = pd.read_parquet(local_path, columns=use_cols if use_cols else None)
        os.remove(local_path)
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        if dt_cols:
            df['Datetime'] = pd.to_datetime(df[dt_cols[0]], utc=True)
        else:
            num_col = next(
                (c for c in df.columns
                 if c in ('Datetime', 'Date', 'Time', 'Timestamp', 'Level_0')),
                df.columns[0])
            df['Datetime'] = pd.to_datetime(df[num_col], unit='ms', utc=True)
        df = df.set_index('Datetime').sort_index()
        df = df[~df.index.duplicated(keep='first')]
        col_map = {'open': 'Open', 'high': 'High', 'low': 'Low',
                    'close': 'Close', 'volume': 'Volume'}
        df.columns = [col_map.get(c.lower().strip(), c) for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()
        for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if c not in df.columns:
                df[c] = 0.0
        target_date = pd.Timestamp(date_str, tz='UTC')
        return df[(df.index >= target_date) &
                  (df.index < target_date + pd.Timedelta(days=1))].copy()
    except Exception:
        if os.path.exists(local_path):
            os.remove(local_path)
        return None


# ==============================================================================
# TRADE SIM — SHORT
# ==============================================================================
def simulate_trade(day_df, entry_minute, cfg):
    day_pst = day_df.index.tz_convert(PST)
    day_mins = day_pst.hour * 60 + day_pst.minute

    entry_mask = day_mins == entry_minute
    if not entry_mask.any():
        return None
    entry_idx = np.where(entry_mask)[0][0]

    prices = day_df['Close'].values.astype(np.float64)
    highs = day_df['High'].values.astype(np.float64)
    lows = day_df['Low'].values.astype(np.float64)
    times = (day_pst.hour * 60 + day_pst.minute).values.astype(np.int64)

    tp, sl, mode = cfg['tp_pct'], cfg['sl_pct'], cfg['mode']
    entry_price = prices[entry_idx]

    if mode == 'limit_entry':
        dip_pct = cfg.get('dip_pct', 0) or 0
        max_wait = cfg.get('max_wait', 30) or 30
        limit_price = entry_price * (1 + dip_pct)
        entry_time = times[entry_idx]
        filled, fill_idx = False, -1
        for k in range(entry_idx + 1, len(prices)):
            if times[k] - entry_time > max_wait:
                break
            if highs[k] >= limit_price:
                entry_price, fill_idx, filled = limit_price, k, True
                break
        if not filled:
            return {'pnl': 0.0, 'entry': prices[entry_idx], 'exit': prices[entry_idx],
                    'outcome': 'NO_FILL', 'filled': False, 'mode': mode}
        start_k = fill_idx + 1
    else:
        filled = True
        start_k = entry_idx + 1

    tgt = entry_price * (1 - tp)
    stp = entry_price * (1 + sl)
    grace_min = cfg.get('grace_minutes', 0) or 0
    entry_time = times[entry_idx]

    for k in range(start_k, len(prices)):
        if times[k] >= EXIT_TIME_MIN:
            pnl = (entry_price - prices[k]) / entry_price
            return {'pnl': pnl - 2 * SLIPPAGE_PCT, 'entry': entry_price,
                    'exit': prices[k], 'outcome': 'EOD', 'filled': True, 'mode': mode}
        tp_hit = lows[k] <= tgt
        sl_hit = (highs[k] >= stp) if mode != 'grace' else \
                 ((times[k] - entry_time >= grace_min) and (highs[k] >= stp))
        if tp_hit and sl_hit:
            if np.random.random() < 0.5:
                return {'pnl': tp - 2 * SLIPPAGE_PCT, 'entry': entry_price,
                        'exit': tgt, 'outcome': 'TP', 'filled': True, 'mode': mode}
            else:
                return {'pnl': -sl - 2 * SLIPPAGE_PCT, 'entry': entry_price,
                        'exit': stp, 'outcome': 'SL', 'filled': True, 'mode': mode}
        if tp_hit:
            return {'pnl': tp - 2 * SLIPPAGE_PCT, 'entry': entry_price,
                    'exit': tgt, 'outcome': 'TP', 'filled': True, 'mode': mode}
        if sl_hit:
            return {'pnl': -sl - 2 * SLIPPAGE_PCT, 'entry': entry_price,
                    'exit': stp, 'outcome': 'SL', 'filled': True, 'mode': mode}

    last_price = prices[-1]
    pnl = (entry_price - last_price) / entry_price
    return {'pnl': pnl - 2 * SLIPPAGE_PCT, 'entry': entry_price,
            'exit': last_price, 'outcome': 'EOD', 'filled': True, 'mode': mode}


# ==============================================================================
# EQUAL-WEIGHT PORTFOLIO PnL
# ==============================================================================
def eq_wt_pnl(pnl_list):
    if not pnl_list:
        return 0.0
    return np.mean(pnl_list)


def strat_stats(pnl_list):
    n = len(pnl_list)
    if n == 0:
        return {'n': 0, 'avg': 0, 'w': 0, 'l': 0}
    avg = eq_wt_pnl(pnl_list)
    w = sum(1 for p in pnl_list if p > 0)
    l = n - w
    return {'n': n, 'avg': avg, 'w': w, 'l': l}


# ==============================================================================
# MAIN
# ==============================================================================
print(f"\n{'='*W}")
print(f"  POST-TRADE SHORT — 4-STRATEGY EQ-WT COMPARISON")
print(f"  Date: {ANALYSIS_DATE}")
for sw in SCORE_WINDOWS:
    print(f"  Window: {sw['label']} PT")
print(f"  Exit: {EXIT_TIME_MIN//60}:{EXIT_TIME_MIN%60:02d} PT | Slippage: {2*SLIPPAGE_PCT:.2%} RT")
print(f"  PnL = equal-weight average (capital split equally per entry point)")
print(f"{'='*W}")

# --- Step 1: Load deploy configs ---
print(f"\n  Step 1: Loading deploy configs from BQ...")
DEPLOY_CONFIGS = load_deploy_configs()
deploy_tickers = sorted(DEPLOY_CONFIGS.keys())
print(f"    {len(deploy_tickers)} deploy tickers")

# --- Step 2: Load BQ predictions ---
print(f"\n  Step 2: Loading BQ live predictions for {ANALYSIS_DATE}...")
bq_preds = load_bq_predictions(ANALYSIS_DATE)
print(f"    {len(bq_preds)} total rows")

strategies_found = bq_preds['strategy'].value_counts()
print(f"    Strategies in BQ:")
for strat, cnt in strategies_found.items():
    n_go = len(bq_preds[(bq_preds['strategy'] == strat) & (bq_preds['signal'] == 'GO')])
    print(f"      {strat}: {cnt} rows, {n_go} GO")

bq_entry_mins = sorted(bq_preds['entry_minute'].dropna().unique())
print(f"    BQ entry_minute values: {[int(m) for m in bq_entry_mins]}")

# --- Step 2b: Deduplicate — latest run per (strategy, entry_minute) ---
for sw in SCORE_WINDOWS:
    matching = [m for m in bq_entry_mins if abs(m - sw['entry_min']) <= 10]
    sw['bq_min'] = int(matching[0]) if matching else sw['entry_min']

    window_rows = bq_preds[bq_preds['entry_minute'] == sw['bq_min']].copy()
    mcons_rows = window_rows[window_rows['strategy'] == DEPLOYED_STRATEGY].copy()

    if len(mcons_rows) > 0 and 'run_timestamp' in mcons_rows.columns:
        mcons_rows['run_timestamp'] = pd.to_datetime(mcons_rows['run_timestamp'], utc=True)
        latest_ts = mcons_rows['run_timestamp'].max()
        latest_rows = mcons_rows[mcons_rows['run_timestamp'] == latest_ts].copy()
    else:
        latest_rows = mcons_rows

    sw['bq_rows'] = latest_rows
    sw['bq_all_rows'] = window_rows

    go_rows = latest_rows[latest_rows['signal'] == 'GO']
    sw['go_tickers'] = sorted(go_rows['ticker'].unique())

    all_fired = latest_rows[latest_rows['prob'] >= latest_rows['threshold']].copy()
    all_fired = all_fired.sort_values('prob', ascending=False)
    sw['9model_candidates'] = sorted(all_fired['ticker'].unique())
    sw['9model_picks'] = list(all_fired.head(MAX_TRADES)['ticker'].values)

    five_fired = all_fired[all_fired['model_version'].isin(NEWEST_5_VERSIONS)].copy()
    sw['5model_candidates'] = sorted(five_fired['ticker'].unique())
    sw['5model_picks'] = list(five_fired.head(MAX_TRADES)['ticker'].values)

    n_total = len(latest_rows)
    n_fired = len(all_fired)
    n_5fire = len(five_fired)
    print(f"    {sw['label']} (BQ min={sw['bq_min']}): "
          f"{n_total} scored, {n_fired} fired (prob≥thresh), "
          f"{n_5fire} from 5NEWEST")
    print(f"      BQ LOGGED GO: {', '.join(sw['go_tickers']) or 'NONE'}")
    print(f"      9-MODEL top3: {', '.join(sw['9model_picks']) or 'NONE'}")
    print(f"      5-MODEL top3: {', '.join(sw['5model_picks']) or 'NONE'}")

# --- Step 3: Load MI data for ALL deploy tickers ---
print(f"\n  Step 3: Loading master_inference for ALL {len(deploy_tickers)} deploy tickers...")
price_data = {}
loaded, failed = 0, 0
for ticker in deploy_tickers:
    day_df = load_master_day(ticker, ANALYSIS_DATE)
    if day_df is not None and len(day_df) > 0:
        price_data[ticker] = day_df
        loaded += 1
    else:
        failed += 1
print(f"    Loaded: {loaded}, Failed: {failed}")

sample_tk = next(iter(price_data), None)
if sample_tk:
    sample_df = price_data[sample_tk]
    day_pst = sample_df.index.tz_convert(PST)
    day_mins = day_pst.hour * 60 + day_pst.minute
    first_min, last_min = day_mins[0], day_mins[-1]
    has_eod = (day_mins >= EXIT_TIME_MIN).any()
    print(f"    Sample ({sample_tk}): {len(sample_df)} bars, "
          f"{first_min//60}:{first_min%60:02d}-{last_min//60}:{last_min%60:02d} PT, "
          f"EOD={'YES' if has_eod else 'NO'}")


# ==============================================================================
# Step 4: SIMULATE ALL TICKERS AT BOTH ENTRY TIMES
# ==============================================================================
print(f"\n  Step 4: Simulating ALL deploy tickers at both entry times...")

sims = {}
for ticker in deploy_tickers:
    cfg = DEPLOY_CONFIGS[ticker]
    sims[ticker] = {}
    for sw in SCORE_WINDOWS:
        entry_min = sw['entry_min']
        if ticker in price_data:
            result = simulate_trade(price_data[ticker], entry_min, cfg)
            sims[ticker][entry_min] = result
        else:
            sims[ticker][entry_min] = None


# ==============================================================================
# HELPER: get PnL for a ticker at an entry time
# ==============================================================================
def get_pnl(ticker, entry_min):
    sim = sims.get(ticker, {}).get(entry_min)
    if sim is None:
        return None
    return sim.get('pnl')


def get_sim(ticker, entry_min):
    return sims.get(ticker, {}).get(entry_min)


# ==============================================================================
# Step 5: 4-STRATEGY COMPARISON PER WINDOW
# ==============================================================================
all_window_results = {}

for sw in SCORE_WINDOWS:
    lbl = sw['label']
    entry_min = sw['entry_min']
    go_tickers = sw['go_tickers']

    print(f"\n\n{'='*W}")
    print(f"  {lbl} — 4-STRATEGY COMPARISON (EQ-WT)")
    print(f"{'='*W}")

    # --- Build all sims for this entry time, sorted by PnL ---
    all_sims = []
    for ticker in deploy_tickers:
        cfg = DEPLOY_CONFIGS[ticker]
        sim = get_sim(ticker, entry_min)
        pnl = sim['pnl'] if sim and sim.get('pnl') is not None else None
        filled = sim['filled'] if sim else False
        outcome = sim['outcome'] if sim else ('NO_DATA' if ticker not in price_data else 'NO_BAR')
        all_sims.append({
            'ticker': ticker, 'pnl': pnl, 'filled': filled, 'outcome': outcome,
            'entry': sim['entry'] if sim else None,
            'exit': sim['exit'] if sim else None,
            'mode': cfg['mode'], 'tp': cfg['tp_pct'], 'sl': cfg['sl_pct'],
            'family': cfg['family'], 'fold': cfg['fold'],
        })

    # Oracle: best MAX_TRADES by PnL (only filled trades)
    filled_sims = [s for s in all_sims if s['filled'] and s['pnl'] is not None and s['pnl'] != 0]
    filled_sims.sort(key=lambda x: -x['pnl'])
    oracle_picks = [s['ticker'] for s in filled_sims[:MAX_TRADES]]

    # --- Strategy picks ---
    strategies = {
        'BQ LOGGED':  go_tickers,
        '9-MODEL':    sw['9model_picks'],
        '5-MODEL':    sw['5model_picks'],
        'ORACLE':     oracle_picks,
    }

    # --- Compute PnL per strategy ---
    strat_results = {}
    for name, picks in strategies.items():
        pnl_list = []
        for tk in picks:
            p = get_pnl(tk, entry_min)
            pnl_list.append(p if p is not None else 0.0)
        strat_results[name] = {
            'picks': picks,
            'pnls': pnl_list,
            'stats': strat_stats(pnl_list),
        }

    # --- Per-ticker detail table ---
    # Collect all tickers that appear in any strategy
    strat_tickers = set()
    for picks in strategies.values():
        strat_tickers |= set(picks)

    # Also show all fired candidates for context
    all_9cands = set(sw['9model_candidates'])
    show_tickers = strat_tickers | all_9cands

    show_sims = {s['ticker']: s for s in all_sims if s['ticker'] in show_tickers}

    # Sort by PnL descending
    show_list = sorted(show_sims.values(), key=lambda x: -(x['pnl'] if x['pnl'] is not None else -999))

    print(f"\n  {'Ticker':<7} {'BQ':>3} {'9M':>3} {'5M':>3} {'OR':>3}  "
          f"{'Entry':>9} {'Exit':>9} {'PnL':>8} {'Out':<8} {'Mode':<6} "
          f"{'TP%':>5} {'SL%':>5} {'Config':<10}")
    print(f"  {'─'*95}")

    for s in show_list:
        tk = s['ticker']
        marks = {
            'BQ': '✓' if tk in strategies['BQ LOGGED'] else '',
            '9M': '✓' if tk in strategies['9-MODEL'] else '',
            '5M': '✓' if tk in strategies['5-MODEL'] else '',
            'OR': '✓' if tk in strategies['ORACLE'] else '',
        }
        entry_str = f"${s['entry']:.2f}" if s['entry'] else 'N/A'
        exit_str = f"${s['exit']:.2f}" if s['exit'] else 'N/A'
        pnl_str = f"{s['pnl']:+.2%}" if s['pnl'] is not None else 'N/A'
        cfg_str = f"{s['family']} F{s['fold']}"

        print(f"  {tk:<7} {marks['BQ']:>3} {marks['9M']:>3} {marks['5M']:>3} {marks['OR']:>3}  "
              f"{entry_str:>9} {exit_str:>9} {pnl_str:>8} {s['outcome']:<8} {s['mode']:<6} "
              f"{s['tp']:>5.1%} {s['sl']:>5.1%} {cfg_str:<10}")

    print(f"  {'─'*95}")

    # --- Strategy comparison table ---
    print(f"\n  {'Strategy':<14} {'Picks':<25} {'N':>2} {'EQ-WT PnL':>10} {'W/L':>5}")
    print(f"  {'─'*62}")

    for name in ['BQ LOGGED', '9-MODEL', '5-MODEL', 'ORACLE']:
        r = strat_results[name]
        st = r['stats']
        tk_str = ', '.join(r['picks'][:5])
        print(f"  {name:<14} {tk_str:<25} {st['n']:>2} {st['avg']:>+10.2%} "
              f"{st['w']}W/{st['l']}L")

    # --- Universe stats ---
    universe_pnls = [s['pnl'] for s in all_sims
                     if s['filled'] and s['pnl'] is not None]
    n_pos = sum(1 for p in universe_pnls if p > 0)
    n_neg = sum(1 for p in universe_pnls if p <= 0)
    avg_univ = eq_wt_pnl(universe_pnls)
    print(f"\n  UNIVERSE: {len(universe_pnls)} filled, "
          f"{n_pos} positive / {n_neg} negative, avg {avg_univ:+.2%}")

    bq_avg = strat_results['BQ LOGGED']['stats']['avg']
    or_avg = strat_results['ORACLE']['stats']['avg']
    print(f"  GAP: Oracle {or_avg:+.2%} - BQ Logged {bq_avg:+.2%} = {or_avg - bq_avg:+.2%}")

    # Store for combined
    all_window_results[lbl] = {
        'strat_results': strat_results,
        'all_sims': all_sims,
        'universe_pnls': universe_pnls,
    }
    sw['strat_results'] = strat_results


# ==============================================================================
# COMBINED SUMMARY ACROSS WINDOWS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  COMBINED SUMMARY — {ANALYSIS_DATE} (EQ-WT)")
print(f"{'='*W}")
print(f"  Capital split equally per entry point. Combined = avg of both windows.\n")

header = f"  {'Strategy':<14} │"
for sw in SCORE_WINDOWS:
    h = sw['label'].split('→')[1]
    header += f" {h:>10} │"
header += f" {'COMBINED':>10}"
print(header)
print(f"  {'─'*14}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*10}")

for name in ['BQ LOGGED', '9-MODEL', '5-MODEL', 'ORACLE']:
    line = f"  {name:<14} │"
    window_avgs = []
    for sw in SCORE_WINDOWS:
        avg = sw['strat_results'][name]['stats']['avg']
        window_avgs.append(avg)
        line += f" {avg:>+10.2%} │"
    combined = np.mean(window_avgs)
    line += f" {combined:>+10.2%}"
    print(line)

print(f"  {'─'*14}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*10}")

# Universe avg per window
line = f"  {'UNIVERSE':<14} │"
window_univs = []
for sw in SCORE_WINDOWS:
    lbl = sw['label']
    r = all_window_results[lbl]
    avg = eq_wt_pnl(r['universe_pnls'])
    window_univs.append(avg)
    line += f" {avg:>+10.2%} │"
line += f" {np.mean(window_univs):>+10.2%}"
print(line)


# ==============================================================================
# CROSS-WINDOW: GO TICKERS AT BOTH ENTRY TIMES
# ==============================================================================
if len(SCORE_WINDOWS) == 2:
    sw0, sw1 = SCORE_WINDOWS[0], SCORE_WINDOWS[1]

    # Union of all strategy picks across both windows
    all_picks = set()
    for sw in SCORE_WINDOWS:
        for name in ['BQ LOGGED', '9-MODEL', '5-MODEL', 'ORACLE']:
            all_picks |= set(sw['strat_results'][name]['picks'])

    if all_picks:
        print(f"\n\n{'='*W}")
        print(f"  ALL STRATEGY PICKS — BOTH ENTRY TIMES")
        print(f"{'='*W}\n")

        h0 = sw0['label'].split('→')[1]
        h1 = sw1['label'].split('→')[1]

        print(f"  {'Ticker':<7} │ {'@'+h0:>6} {'PnL':>8} {'Out':<7} │ "
              f"{'@'+h1:>6} {'PnL':>8} {'Out':<7} │ Strategies")
        print(f"  {'─'*7}─┼─{'─'*6}─{'─'*8}─{'─'*7}─┼─"
              f"{'─'*6}─{'─'*8}─{'─'*7}─┼─{'─'*20}")

        for tk in sorted(all_picks):
            def fmt_cell(sw_ref, tk):
                sim = get_sim(tk, sw_ref['entry_min'])
                if sim is None:
                    return ('  N/A', '   N/A', '')
                e = f"${sim['entry']:.0f}" if sim.get('entry') and sim['entry'] > 100 else \
                    (f"${sim['entry']:.2f}" if sim.get('entry') else '  N/A')
                p = f"{sim['pnl']:+.2%}" if sim.get('pnl') is not None else '   N/A'
                o = sim['outcome'][:7] if sim.get('outcome') else ''
                return (e, p, o)

            e0, p0, o0 = fmt_cell(sw0, tk)
            e1, p1, o1 = fmt_cell(sw1, tk)

            # Which strategies picked this ticker in each window?
            strat_marks = []
            for sw in SCORE_WINDOWS:
                h = sw['label'].split('→')[1].strip()
                for sname, abbr in [('BQ LOGGED','BQ'), ('9-MODEL','9M'),
                                    ('5-MODEL','5M'), ('ORACLE','OR')]:
                    if tk in sw['strat_results'][sname]['picks']:
                        strat_marks.append(f"{abbr}@{h}")
            strat_str = ' '.join(strat_marks)

            print(f"  {tk:<7} │ {e0:>6} {p0:>8} {o0:<7} │ "
                  f"{e1:>6} {p1:>8} {o1:<7} │ {strat_str}")


# ==============================================================================
# FULL UNIVERSE TABLE (all 48 tickers, sorted by PnL)
# ==============================================================================
for sw in SCORE_WINDOWS:
    lbl = sw['label']
    entry_min = sw['entry_min']
    r = all_window_results[lbl]

    sim_list = sorted(r['all_sims'], key=lambda x: -(x['pnl'] if x['pnl'] is not None else -999))

    print(f"\n\n{'='*W}")
    print(f"  FULL UNIVERSE — {lbl} (sorted by PnL)")
    print(f"{'='*W}\n")

    print(f"  {'#':>3} {'Ticker':<7} {'Entry':>9} {'Exit':>9} {'PnL':>8} "
          f"{'Out':<8} {'Mode':<6} {'Config':<10}")
    print(f"  {'─'*75}")

    for i, s in enumerate(sim_list):
        entry_str = f"${s['entry']:.2f}" if s['entry'] else 'N/A'
        exit_str = f"${s['exit']:.2f}" if s['exit'] else 'N/A'
        pnl_str = f"{s['pnl']:+.2%}" if s['pnl'] is not None else 'N/A'
        cfg_str = f"{s['family']} F{s['fold']}"

        print(f"  {i+1:>3} {s['ticker']:<7} {entry_str:>9} {exit_str:>9} {pnl_str:>8} "
              f"{s['outcome']:<8} {s['mode']:<6} {cfg_str:<10}")


# ==============================================================================
# CHARTS
# ==============================================================================

# --- Chart 1: Intraday price for all strategy picks ---
chart_tickers_set = set()
for sw in SCORE_WINDOWS:
    for name in ['BQ LOGGED', '9-MODEL', '5-MODEL']:
        chart_tickers_set |= set(sw['strat_results'][name]['picks'])

chart_tickers = sorted(chart_tickers_set & set(price_data.keys()))
n_charts = min(len(chart_tickers), 12)

if n_charts > 0:
    ncols = min(4, n_charts)
    nrows = (n_charts + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    fig.suptitle(f'SHORT Post-Trade — {ANALYSIS_DATE}\n'
                 f'Blue▼=6:30  Red▼=7:00  Green---=TP  Red---=SL',
                 fontsize=11, fontweight='bold')

    for i, ticker in enumerate(chart_tickers[:n_charts]):
        ax = axes[i]
        day_df = price_data[ticker]
        day_pst = day_df.index.tz_convert(PST)
        day_mins = day_pst.hour * 60 + day_pst.minute
        prices = day_df['Close'].values

        ax.plot(day_pst, prices, color='#333', linewidth=1)

        cfg = DEPLOY_CONFIGS.get(ticker)
        colors_entry = ['#1565C0', '#D32F2F']
        for wi, sw in enumerate(SCORE_WINDOWS):
            entry_min = sw['entry_min']
            mask = day_mins == entry_min
            if mask.any():
                idx = np.where(mask)[0][0]
                ep = prices[idx]
                ax.plot(day_pst[idx], ep, 'v', color=colors_entry[wi], markersize=8)
                ax.axhline(ep, color=colors_entry[wi], linewidth=0.7,
                           linestyle='--', alpha=0.4)
                if cfg and wi == 0:
                    ax.axhline(ep * (1 - cfg['tp_pct']), color='green',
                               linewidth=0.8, linestyle=':', alpha=0.6)
                    ax.axhline(ep * (1 + cfg['sl_pct']), color='red',
                               linewidth=0.8, linestyle=':', alpha=0.6)

        sim0 = get_sim(ticker, SCORE_WINDOWS[0]['entry_min'])
        pnl_str = f" {sim0['pnl']:+.1%}" if sim0 and sim0.get('pnl') is not None else ""
        mode_str = f" [{cfg['mode'][:4]}]" if cfg else ""
        ax.set_title(f'{ticker}{mode_str}{pnl_str}', fontsize=9, fontweight='bold')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    for j in range(n_charts, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    chart_path = f'/tmp/post_trade_short_{ANALYSIS_DATE}.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"\n  Chart saved: {chart_path}")
    if _IN_COLAB:
        plt.show()
    plt.close()

# --- Chart 2: 4-strategy comparison bar chart ---
fig, axes = plt.subplots(1, len(SCORE_WINDOWS), figsize=(7 * len(SCORE_WINDOWS), 5))
if len(SCORE_WINDOWS) == 1:
    axes = [axes]

strat_names = ['BQ LOGGED', '9-MODEL', '5-MODEL', 'ORACLE']
strat_colors = ['#D32F2F', '#1565C0', '#FF9800', '#4CAF50']

for wi, sw in enumerate(SCORE_WINDOWS):
    ax = axes[wi]
    avgs = [sw['strat_results'][name]['stats']['avg'] * 100 for name in strat_names]
    bars = ax.bar(strat_names, avgs, color=strat_colors, width=0.6)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel('EQ-WT PnL (%)')
    ax.set_title(f'{sw["label"]}', fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:+.2f}%', ha='center', va='bottom' if val >= 0 else 'top',
                fontsize=9, fontweight='bold')

plt.tight_layout()
bar_path = f'/tmp/post_trade_short_4strat_{ANALYSIS_DATE}.png'
plt.savefig(bar_path, dpi=150, bbox_inches='tight')
print(f"  Strategy chart saved: {bar_path}")
if _IN_COLAB:
    plt.show()
plt.close()

# --- Chart 3: PnL distribution bar chart (all deploy tickers) ---
fig, axes = plt.subplots(1, len(SCORE_WINDOWS), figsize=(8 * len(SCORE_WINDOWS), 8))
if len(SCORE_WINDOWS) == 1:
    axes = [axes]

for wi, sw in enumerate(SCORE_WINDOWS):
    ax = axes[wi]
    lbl = sw['label']
    r = all_window_results[lbl]
    sim_list = sorted(r['all_sims'],
                      key=lambda x: (x['pnl'] if x['pnl'] is not None else -999))
    sim_list = [s for s in sim_list if s['pnl'] is not None]

    tks = [s['ticker'] for s in sim_list]
    pnls = [s['pnl'] * 100 for s in sim_list]

    go_set = set(sw['go_tickers'])
    nine_set = set(sw['9model_picks'])
    five_set = set(sw['5model_picks'])

    colors = []
    for s in sim_list:
        tk = s['ticker']
        if tk in go_set:
            colors.append('#D32F2F')
        elif tk in nine_set:
            colors.append('#1565C0')
        elif tk in five_set:
            colors.append('#FF9800')
        elif s['pnl'] > 0:
            colors.append('#4CAF50')
        else:
            colors.append('#999')

    ax.barh(range(len(tks)), pnls, color=colors)
    ax.set_yticks(range(len(tks)))
    ax.set_yticklabels(tks, fontsize=7)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('PnL (%)')
    ax.set_title(f'{lbl} — All Deploy Tickers\n'
                 f'Red=BQ  Blue=9M  Orange=5M  Green=Win  Gray=Loss',
                 fontsize=9)
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
univ_path = f'/tmp/post_trade_short_universe_{ANALYSIS_DATE}.png'
plt.savefig(univ_path, dpi=150, bbox_inches='tight')
print(f"  Universe chart saved: {univ_path}")
if _IN_COLAB:
    plt.show()
plt.close()


# ==============================================================================
# DIAGNOSIS
# ==============================================================================
print(f"\n\n{'='*W}")
print(f"  DIAGNOSIS — {ANALYSIS_DATE}")
print(f"{'='*W}\n")

bq_combined = np.mean([sw['strat_results']['BQ LOGGED']['stats']['avg']
                        for sw in SCORE_WINDOWS])
or_combined = np.mean([sw['strat_results']['ORACLE']['stats']['avg']
                        for sw in SCORE_WINDOWS])
nine_combined = np.mean([sw['strat_results']['9-MODEL']['stats']['avg']
                          for sw in SCORE_WINDOWS])
five_combined = np.mean([sw['strat_results']['5-MODEL']['stats']['avg']
                          for sw in SCORE_WINDOWS])

print(f"  1. DID THE STRATEGY LOSE MONEY?")
print(f"     BQ LOGGED EQ-WT: {bq_combined:+.2%}")
print(f"     {'YES — genuine losing day' if bq_combined < 0 else 'NO — profitable day'}")

print(f"\n  2. WOULD DIFFERENT MODEL SELECTION HELP?")
print(f"     9-MODEL (prob-ranked): {nine_combined:+.2%}  "
      f"({'better' if nine_combined > bq_combined else 'worse'} than BQ by "
      f"{nine_combined - bq_combined:+.2%})")
print(f"     5-MODEL (5 newest):    {five_combined:+.2%}  "
      f"({'better' if five_combined > bq_combined else 'worse'} than BQ by "
      f"{five_combined - bq_combined:+.2%})")

print(f"\n  3. WAS THE UNIVERSE ALSO NEGATIVE?")
for sw in SCORE_WINDOWS:
    lbl = sw['label']
    r = all_window_results[lbl]
    avg = eq_wt_pnl(r['universe_pnls'])
    n_pos = sum(1 for p in r['universe_pnls'] if p > 0)
    n_neg = sum(1 for p in r['universe_pnls'] if p <= 0)
    print(f"     {lbl}: avg {avg:+.2%}, {n_pos} positive / {n_neg} negative")

print(f"\n  4. COULD BETTER PICKS HAVE SAVED THE DAY?")
print(f"     Oracle EQ-WT: {or_combined:+.2%}")
print(f"     Gap vs BQ:    {or_combined - bq_combined:+.2%}")
if or_combined > 0 and bq_combined < 0:
    print(f"     YES — profitable picks existed, model selected wrong tickers")
elif or_combined < 0:
    print(f"     NO — even the best picks lost money")
else:
    gap = or_combined - bq_combined
    print(f"     PARTIAL — oracle profitable, gap is {'small' if abs(gap) < 0.01 else 'large'}")

print(f"\n  NOTE: 9-MODEL and 5-MODEL rank by hybrid prob (single model).")
print(f"  BQ LOGGED uses M_CONSENSUS fire count (all 9 models), which BQ does not store.")
print(f"  For true 9-model/5-model comparison, add n_fires to BQ logging schema.")

print(f"\n{'='*W}")
print(f"  POST-TRADE ANALYSIS COMPLETE — {ANALYSIS_DATE}")
print(f"{'='*W}")
