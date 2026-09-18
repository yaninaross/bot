"""
select_long_model.py — ML Ticker Selection Model for LONG
==========================================================
Builds regression models to predict val_b median PnL per ticker using
ONLY unbiased features (val_a + opt). Compares 5 models with LOO-CV.
Selects top-K tickers and backtests vs S3/ORACLE.

Key differences from SHORT version:
- BQ: robust_wf_v12 / robust_wf_v12_stitched (LONG model versions)
- 7 modes (static, atr, atr_adaptive, atr_sl, trailing_decay, grace, limit_entry)
- Per-seed config selection (ensemble val_a bugged for LONG)
- S3 family agreement gate as baseline (not hardcoded CURRENT)
- Family diversity features instead of cross-alpha
- Slippage: 0.0015

GPU NOT required. Colab-ready (no if __name__).
"""

import os, io, gc, sys, warnings, ctypes
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np, pandas as pd, pytz
from scipy import stats as sp_stats

warnings.filterwarnings('ignore')

import matplotlib
try:
    from google.colab import auth
    _IN_COLAB = True
except ImportError:
    matplotlib.use('Agg')
    _IN_COLAB = False
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from google.cloud import storage as gcs_storage, bigquery

from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

if _IN_COLAB:
    try:
        auth.authenticate_user()
    except Exception:
        pass

def _release_memory():
    gc.collect()
    try: ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception: pass


# ==============================================================================
# CONFIG
# ==============================================================================
BUCKET_NAME    = os.environ.get("GCS_BUCKET", "")
RESULTS_DIR    = "Algos/Results_scored_20260703"
RESULTS_FALLBACK = "Algos/Results_all_v2"
INFERENCE_DIR  = "Data/Inference_v2"

FULL_START  = '2026-04-01'
FULL_END    = '2026-07-23'
HALF1_END   = '2026-05-31'
HALF2_START = '2026-06-01'

EXIT_TIME_MIN = 780
SLIPPAGE_PCT  = 0.0015

PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

BQ_TABLE = f"{GCP_PROJECT}.scoring_logs.wf_optimization_results"
BQ_MODEL_VERSIONS = ('robust_wf_v12', 'robust_wf_v12_stitched')

MAX_VA_OPT_RATIO = 3.0
DEPLOY_THRESHOLD = 0.018
MIN_SEEDS = 3

SKIP_MODES = {'regime_scaled', 'regime_adaptive'}

GRP_KEYS = ['ticker', 'experiment', 'tpsl_mode', 'entry_minute',
            'prob_thresh', 'tp_pct', 'sl_pct',
            'tp_mult', 'sl_mult', 'trail_start_mult', 'trail_end_mult',
            'grace_minutes', 'dip_pct', 'max_wait_minutes',
            'vol_filter_pct']

ORACLE_TOP_N = 10
MODEL_TOP_K  = [8, 10, 12]
MAX_CSV_TICKERS = 30

gcs_client = gcs_storage.Client()
bq_client  = bigquery.Client()
W = 110


# ==============================================================================
# DATA LOADING (from oracle_long_ceiling.py)
# ==============================================================================
_master_cache = {}

def load_master(ticker):
    if ticker in _master_cache:
        return _master_cache[ticker]
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    uri = f"gs://{BUCKET_NAME}/{INFERENCE_DIR}/{clean}_master_inference.parquet"
    try:
        df = pd.read_parquet(uri)
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        if dt_cols:
            df['Datetime'] = pd.to_datetime(df[dt_cols[0]], utc=True)
        else:
            num_col = next((c for c in df.columns
                           if c in ('Datetime', 'Date', 'Time', 'Timestamp', 'Level_0')),
                          df.columns[0])
            df['Datetime'] = pd.to_datetime(df[num_col], unit='ms', utc=True)
        df = df.set_index('Datetime')
        df = df[~df.index.duplicated(keep='first')]
        std_map = {'open': 'Open', 'high': 'High', 'low': 'Low',
                   'close': 'Close', 'volume': 'Volume'}
        lower_map = {k.lower(): v for k, v in std_map.items()}
        df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip()) for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()
        df = df[df.index.notna()].sort_index()
        cutoff = pd.Timestamp(FULL_START, tz='UTC') - pd.Timedelta(days=30)
        df = df[df.index >= cutoff]
        _master_cache[ticker] = df
        return df
    except Exception as e:
        print(f"    load_master({ticker}): FAILED -- {e}")
        return None


def heal_data(df):
    df = df.resample('1min').asfreq()
    for c in ['Open', 'High', 'Low', 'Close']:
        if c in df.columns:
            df[c] = df[c].ffill()
    for c in ['Volume']:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df


def load_seed_scores(ticker, experiment_base):
    clean = TICKER_FILE_MAP.get(ticker, ticker)
    bucket = gcs_client.bucket(BUCKET_NAME)
    all_results = {}
    for results_dir in [RESULTS_FALLBACK, RESULTS_DIR]:
        prefix = f"{results_dir}/Universal_Exp{experiment_base}/{clean}/Seed_"
        blobs = list(bucket.list_blobs(prefix=prefix))
        if not blobs:
            continue
        for blob in blobs:
            name = blob.name.split('/')[-1]
            if not name.startswith('Seed_') or not name.endswith('_Results.csv'):
                continue
            seed_id = int(name.replace('Seed_', '').replace('_Results.csv', ''))
            d = pd.read_csv(io.BytesIO(blob.download_as_bytes()),
                            index_col=0, parse_dates=True)
            if d.index.tz is None:
                d.index = d.index.tz_localize('UTC')
            d.index = d.index.tz_convert(PST)
            d = d.sort_index()
            if seed_id in all_results:
                old = all_results[seed_id]
                combined = pd.concat([old, d])
                combined = combined[~combined.index.duplicated(keep='last')]
                all_results[seed_id] = combined.sort_index()
            else:
                all_results[seed_id] = d
    return all_results


def precompute_ohlcv_arrays(df, start_date, end_date):
    s = pd.to_datetime(start_date).tz_localize('America/Los_Angeles')
    e = pd.to_datetime(end_date).tz_localize('America/Los_Angeles') + timedelta(days=1)
    idx = df.index
    if str(idx.tz) != 'America/Los_Angeles':
        idx = idx.tz_convert('America/Los_Angeles')
    mask = (idx >= s) & (idx < e)
    df_cut = df[mask]
    if len(df_cut) == 0:
        return None
    idx_cut = df_cut.index if str(df_cut.index.tz) == 'America/Los_Angeles' \
        else df_cut.index.tz_convert('America/Los_Angeles')
    return {
        'prices': df_cut['Close'].values.astype(np.float64).copy(),
        'highs':  df_cut['High'].values.astype(np.float64).copy(),
        'lows':   df_cut['Low'].values.astype(np.float64).copy(),
        'dates':  (idx_cut.year * 10000 + idx_cut.month * 100 + idx_cut.day).values.astype(np.int64),
        'times':  (idx_cut.hour * 60 + idx_cut.minute).values.astype(np.int64),
    }


def compute_daily_atr_pct(df, period=14):
    daily = df.resample('D').agg({'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    prev_close = daily['Close'].shift(1)
    tr = pd.concat([
        daily['High'] - daily['Low'],
        (daily['High'] - prev_close).abs(),
        (daily['Low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    daily_atr = tr.rolling(period).mean()
    atr_pct = daily_atr / daily['Close']
    return atr_pct.reindex(df.index, method='ffill')


# ==============================================================================
# LONG TRADE SIMULATION (buy-side, from oracle_long_ceiling.py)
# ==============================================================================
def run_long_trade_static(prices, highs, lows, date_ints, time_mins,
                          entry_idx, tp, sl, exit_time):
    n = len(prices)
    if entry_idx >= n - 1:
        return 0.0, int(date_ints[entry_idx]), False
    ep = prices[entry_idx]
    ed = int(date_ints[entry_idx])
    tgt = ep * (1 + tp)
    stp = ep * (1 - sl)
    for k in range(entry_idx + 1, n):
        if date_ints[k] != ed:
            return (prices[k-1] - ep) / ep, ed, True
        sl_h = lows[k] <= stp
        tp_h = highs[k] >= tgt
        if sl_h and tp_h:
            return (-sl if np.random.random() < 0.5 else tp), ed, True
        if sl_h:
            return -sl, ed, True
        if tp_h:
            return tp, ed, True
        if time_mins[k] >= exit_time:
            return (prices[k] - ep) / ep, ed, True
    return (prices[n-1] - ep) / ep, ed, True


def run_long_trade_trailing_decay(prices, highs, lows, date_ints, time_mins,
                                  entry_idx, atr_pct_val, trail_start_mult,
                                  trail_end_mult, sl_mult, exit_time):
    n = len(prices)
    if entry_idx >= n - 1:
        return 0.0, int(date_ints[entry_idx]), False
    ep = prices[entry_idx]
    ed = int(date_ints[entry_idx])
    atr_pct = atr_pct_val if (atr_pct_val == atr_pct_val and atr_pct_val > 0) else 0.05
    sl = min(max(atr_pct * sl_mult, 0.01), 0.25)
    hard_stop = ep * (1 - sl)
    ts = atr_pct * trail_start_mult
    te = atr_pct * trail_end_mult
    hwm = ep
    for k in range(entry_idx + 1, n):
        if date_ints[k] != ed:
            return (prices[k-1] - ep) / ep, ed, True
        if highs[k] > hwm:
            hwm = highs[k]
        unr = (hwm - ep) / ep
        trailing_stop = 0.0
        if unr > 0 and ts > 0:
            r = min(unr / ts, 1.0)
            trailing_stop = hwm * (1.0 - (ts + (te - ts) * r))
        eff = max(hard_stop, trailing_stop)
        if lows[k] <= eff:
            return ((trailing_stop - ep) / ep if trailing_stop > hard_stop else -sl), ed, True
        if time_mins[k] >= exit_time:
            return (prices[k] - ep) / ep, ed, True
    return (prices[n-1] - ep) / ep, ed, True


def run_long_trade_grace(prices, highs, lows, date_ints, time_mins,
                         entry_idx, tp, sl, exit_time, grace_minutes):
    n = len(prices)
    if entry_idx >= n - 1:
        return 0.0, int(date_ints[entry_idx]), False
    ep = prices[entry_idx]
    ed = int(date_ints[entry_idx])
    et = time_mins[entry_idx]
    tgt = ep * (1 + tp)
    stp = ep * (1 - sl)
    for k in range(entry_idx + 1, n):
        if date_ints[k] != ed:
            return (prices[k-1] - ep) / ep, ed, True
        elapsed = time_mins[k] - et
        sl_active = elapsed >= grace_minutes
        tp_h = highs[k] >= tgt
        sl_h = lows[k] <= stp
        if sl_active and sl_h and tp_h:
            return (-sl if np.random.random() < 0.5 else tp), ed, True
        if sl_active and sl_h:
            return -sl, ed, True
        if tp_h:
            return tp, ed, True
        if time_mins[k] >= exit_time:
            return (prices[k] - ep) / ep, ed, True
    return (prices[n-1] - ep) / ep, ed, True


def run_long_trade_limit_entry(prices, highs, lows, date_ints, time_mins,
                               entry_idx, tp, sl, exit_time, dip_pct, max_wait):
    n = len(prices)
    if entry_idx >= n - 1:
        return 0.0, int(date_ints[entry_idx]), False
    sp = prices[entry_idx]
    ed = int(date_ints[entry_idx])
    et = time_mins[entry_idx]
    lp = sp * (1.0 - dip_pct)
    fill_idx = -1
    for k in range(entry_idx + 1, n):
        if date_ints[k] != ed:
            break
        if time_mins[k] - et > max_wait:
            break
        if lows[k] <= lp:
            fill_idx = k
            break
    if fill_idx < 0:
        return 0.0, ed, False
    ep = lp
    tgt = ep * (1.0 + tp)
    stp = ep * (1.0 - sl)
    for k in range(fill_idx + 1, n):
        if date_ints[k] != ed:
            return (prices[k-1] - ep) / ep, ed, True
        sl_h = lows[k] <= stp
        tp_h = highs[k] >= tgt
        if sl_h and tp_h:
            return (-sl if np.random.random() < 0.5 else tp), ed, True
        if sl_h:
            return -sl, ed, True
        if tp_h:
            return tp, ed, True
        if time_mins[k] >= exit_time:
            return (prices[k] - ep) / ep, ed, True
    return (prices[n-1] - ep) / ep, ed, True


# ==============================================================================
# HELPERS
# ==============================================================================
_MODE_SUFFIXES = ('_ATRSL', '_GRACE', '_LIMIT', '_ATR', '_ATRADP',
                  '_REGIME', '_TRAIL', '_REGADP', '_XSECTOR')

def _base_experiment(exp):
    for sfx in sorted(_MODE_SUFFIXES, key=len, reverse=True):
        if exp.endswith(sfx):
            return exp[:-len(sfx)]
    return exp

def assign_family(exp):
    base = exp.split('_')[0]
    if not base.isdigit():
        return 'other'
    num = int(base)
    if num == 152: return '152'
    if num == 160: return '160'
    if 170 <= num <= 175: return '170'
    if 180 <= num <= 185: return '180'
    if 200 <= num <= 205: return '200'
    return 'other'

def _safe_float(val, default=0.0):
    try:
        if pd.isna(val): return default
    except (TypeError, ValueError): pass
    try: return float(val)
    except (TypeError, ValueError): return default

def _safe_int(val, default=0):
    try:
        if pd.isna(val): return default
    except (TypeError, ValueError): pass
    try: return int(val)
    except (TypeError, ValueError): return default

def _row_to_cfg(ticker, row, source):
    exp = row['experiment'] if isinstance(row, pd.Series) else row.get('experiment', '')
    base_exp = _base_experiment(exp)
    return {
        'ticker': ticker,
        'experiment': exp,
        '_base_experiment': base_exp,
        'source': source,
        'tpsl_mode': row['tpsl_mode'],
        'entry_minute': _safe_int(row.get('entry_minute', 375), 375),
        'prob_thresh': _safe_float(row['prob_thresh']),
        'tp_pct': _safe_float(row.get('tp_pct', 0)),
        'sl_pct': _safe_float(row.get('sl_pct', 0)),
        'tp_mult': _safe_float(row.get('tp_mult', 0)),
        'sl_mult': _safe_float(row.get('sl_mult', 0)),
        'trail_start_mult': _safe_float(row.get('trail_start_mult', 0)),
        'trail_end_mult': _safe_float(row.get('trail_end_mult', 0)),
        'grace_minutes': _safe_int(row.get('grace_minutes', 0)),
        'dip_pct': _safe_float(row.get('dip_pct', 0)),
        'max_wait_minutes': _safe_int(row.get('max_wait_minutes', 0)),
        'val_pnl': _safe_float(row.get('val_a_avg_pnl', row.get('mean_va_pnl', 0))),
        'test_pnl': _safe_float(row.get('val_b_avg_pnl', row.get('vb_pnl', 0))),
    }


def summarize_trades(trades_list, label):
    if not trades_list:
        return {'label': label, 'n': 0, 'total': 0, 'avg': 0, 'wr': 0,
                'dd': 0, 'monthly': {}, 'trades_df': pd.DataFrame()}
    df = pd.DataFrame(trades_list)
    df['month'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m')
    cum = df.sort_values('date')['pnl'].cumsum()
    max_dd = (cum - cum.cummax()).min()
    return {
        'label': label,
        'n': len(df), 'total': df['pnl'].sum(), 'avg': df['pnl'].mean(),
        'wr': (df['pnl'] > 0).mean(), 'dd': max_dd,
        'monthly': df.groupby('month').agg(
            n=('pnl', 'count'), total=('pnl', 'sum'),
            avg=('pnl', 'mean'), wr=('pnl', lambda x: (x > 0).mean())
        ).to_dict('index'),
        'trades_df': df,
    }


# ==============================================================================
# BQ + CONFIG SELECTION
# ==============================================================================
def query_all_bq():
    versions_sql = ', '.join(f"'{v}'" for v in BQ_MODEL_VERSIONS)
    q = f"""
    SELECT ticker, experiment, seed, tpsl_mode, entry_minute,
           prob_thresh, tp_pct, sl_pct, tp_mult, sl_mult,
           trail_start_mult, trail_end_mult, grace_minutes,
           dip_pct, max_wait_minutes, vol_filter_pct,
           opt_avg_pnl, opt_win_rate, opt_count, opt_freq,
           val_a_avg_pnl, val_a_win_rate, val_a_count, val_a_freq,
           val_b_avg_pnl, val_b_win_rate, val_b_count,
           is_ensemble, survived_a, model_version
    FROM `{BQ_TABLE}`
    WHERE model_version IN ({versions_sql})
      AND survived_a = TRUE
      AND val_a_count >= 3
      AND opt_avg_pnl > 0
      AND val_a_avg_pnl > 0
    ORDER BY ticker, experiment, seed
    """
    return bq_client.query(q).to_dataframe()


def select_configs_perseed(bq_df, ticker_list):
    """Per-seed aggregated config selection (any mode).
    Used instead of ensemble selection since LONG ensemble val_a is bugged."""
    indiv = bq_df[bq_df['is_ensemble'] == False].copy()
    indiv = indiv[~indiv['tpsl_mode'].isin(SKIP_MODES)]
    indiv['va_opt_ratio'] = indiv['val_a_avg_pnl'] / indiv['opt_avg_pnl']
    indiv = indiv[indiv['va_opt_ratio'] <= MAX_VA_OPT_RATIO]

    for k in GRP_KEYS:
        if k not in indiv.columns:
            indiv[k] = 0
        else:
            indiv[k] = indiv[k].fillna(0)

    agg = indiv.groupby(GRP_KEYS).agg(
        n_seeds=('seed', 'nunique'),
        mean_va_pnl=('val_a_avg_pnl', 'mean'),
        mean_va_wr=('val_a_win_rate', 'mean'),
        mean_va_count=('val_a_count', 'mean'),
    ).reset_index()
    agg = agg[agg['n_seeds'] >= MIN_SEEDS]

    configs = {}
    for ticker in ticker_list:
        tk = agg[agg['ticker'] == ticker]
        if tk.empty:
            continue
        best = tk.loc[tk['mean_va_pnl'].idxmax()]
        configs[ticker] = _row_to_cfg(ticker, best, 'perseed')
    return configs


def select_configs_s3(bq_df):
    """S3 production config selection: limit_entry + family agreement gate."""
    indiv = bq_df[bq_df['is_ensemble'] == False].copy()
    indiv = indiv[indiv['tpsl_mode'] == 'limit_entry']

    for k in GRP_KEYS:
        if k not in indiv.columns:
            indiv[k] = 0
        else:
            indiv[k] = indiv[k].fillna(0)

    sa = indiv.groupby(GRP_KEYS).agg(
        n_seeds=('seed', 'nunique'),
        n_pos_seeds_18=('val_a_avg_pnl', lambda x: (x >= DEPLOY_THRESHOLD).sum()),
        mean_va_pnl=('val_a_avg_pnl', 'mean'),
        mean_va_wr=('val_a_win_rate', 'mean'),
        mean_va_count=('val_a_count', 'mean'),
        vb_pnl=('val_b_avg_pnl', 'mean'),
    ).reset_index()
    sa['agree_18'] = (sa['n_pos_seeds_18'] / sa['n_seeds']).clip(upper=1.0)
    sa = sa[sa['n_seeds'] >= MIN_SEEDS].copy()

    sa['score'] = (sa['mean_va_pnl'] ** 2) * sa['agree_18']
    sa['exp_fam'] = sa['experiment'].apply(assign_family)
    sa_fam = sa[sa['exp_fam'] != 'other'].copy()

    best_per_fam = sa_fam.loc[sa_fam.groupby(['ticker', 'exp_fam'])['score'].idxmax()]
    best_per_fam = best_per_fam[best_per_fam['mean_va_pnl'] >= DEPLOY_THRESHOLD]

    agreement = {}
    for tk in best_per_fam['ticker'].unique():
        tk_rows = best_per_fam[best_per_fam['ticker'] == tk]
        n_fams = len(tk_rows)
        n_pos = int((tk_rows['vb_pnl'] > 0).sum())
        agreement[tk] = {
            'n_fams': n_fams, 'n_pos': n_pos,
            'majority_pos': n_pos > n_fams / 2 and n_fams >= 2,
        }

    s3_tickers = sorted([tk for tk, ag in agreement.items() if ag['majority_pos']])

    configs = {}
    sa_s3 = sa[sa['ticker'].isin(s3_tickers)].copy()
    sa_s3['score'] = (sa_s3['mean_va_pnl'] ** 2) * sa_s3['agree_18']
    if not sa_s3.empty:
        best_s3 = sa_s3.loc[sa_s3.groupby('ticker')['score'].idxmax()]
        best_s3 = best_s3[best_s3['mean_va_pnl'] >= DEPLOY_THRESHOLD]
        for _, row in best_s3.iterrows():
            configs[row['ticker']] = _row_to_cfg(row['ticker'], row, 's3_gate')
    return configs, s3_tickers


# ==============================================================================
# BACKTEST (LONG, 7-mode dispatch)
# ==============================================================================
def run_csv_backtest_split(configs, csv_scores_cache, ticker_data,
                           start_date, end_date, label):
    test_start_dt = pd.Timestamp(start_date).date()
    test_end_dt   = pd.Timestamp(end_date).date()

    daily_probs = {}
    for ticker, cfg in sorted(configs.items()):
        if ticker not in ticker_data:
            continue
        base_exp = cfg['_base_experiment']
        entry_min = cfg['entry_minute']
        seed_scores = csv_scores_cache.get((ticker, base_exp))
        if not seed_scores:
            continue
        all_daily = {}
        for seed_id, sdf in seed_scores.items():
            bar_mins = sdf.index.hour * 60 + sdf.index.minute
            entry_bars = sdf[bar_mins == entry_min]
            for dt, row in entry_bars.iterrows():
                d = dt.date()
                if d not in all_daily:
                    all_daily[d] = []
                all_daily[d].append(float(row['Prob_Spike']))
        daily_probs[ticker] = {d: np.mean(ps) for d, ps in all_daily.items()}

    all_trades = []
    per_ticker_trades = defaultdict(list)

    for ticker in sorted(configs.keys()):
        if ticker not in ticker_data or ticker not in daily_probs:
            continue
        cfg = configs[ticker]
        ohlcv = ticker_data[ticker]['ohlcv']
        probs = daily_probs[ticker]
        entry_min = cfg['entry_minute']
        prob_thresh = cfg['prob_thresh']
        mode = cfg['tpsl_mode']

        prices = ohlcv['prices']
        highs  = ohlcv['highs']
        lows   = ohlcv['lows']
        dates  = ohlcv['dates']
        times  = ohlcv['times']
        atr_arr = ohlcv.get('atr_pct', np.full(len(prices), 0.05))

        entry_indices = np.where(times == entry_min)[0]
        last_traded_date = -1

        for idx in entry_indices:
            d_int = int(dates[idx])
            d_str = str(d_int)
            d_date = pd.Timestamp(f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}").date()

            if d_date < test_start_dt or d_date > test_end_dt:
                continue
            if d_int == last_traded_date:
                continue
            prob = probs.get(d_date)
            if prob is None:
                continue
            if prob < prob_thresh:
                continue

            # ── LONG mode dispatch ──
            if mode == 'limit_entry':
                pnl, trade_date, filled = run_long_trade_limit_entry(
                    prices, highs, lows, dates, times,
                    idx, cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN,
                    cfg['dip_pct'], cfg['max_wait_minutes'])
            elif mode == 'grace':
                pnl, trade_date, filled = run_long_trade_grace(
                    prices, highs, lows, dates, times,
                    idx, cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN,
                    cfg['grace_minutes'])
            elif mode == 'trailing_decay':
                pnl, trade_date, filled = run_long_trade_trailing_decay(
                    prices, highs, lows, dates, times,
                    idx, float(atr_arr[idx]),
                    cfg['trail_start_mult'], cfg['trail_end_mult'],
                    cfg['sl_mult'], EXIT_TIME_MIN)
            elif mode == 'atr_sl':
                atr_val = float(atr_arr[idx])
                atr_val = atr_val if atr_val > 0 else 0.05
                computed_sl = min(max(atr_val * cfg['sl_mult'], 0.01), 0.25)
                pnl, trade_date, filled = run_long_trade_static(
                    prices, highs, lows, dates, times,
                    idx, cfg['tp_pct'], computed_sl, EXIT_TIME_MIN)
            elif mode in ('atr', 'atr_adaptive'):
                atr_val = float(atr_arr[idx])
                atr_val = atr_val if atr_val > 0 else 0.05
                computed_tp = min(max(atr_val * cfg['tp_mult'], 0.005), 0.30)
                computed_sl = min(max(atr_val * cfg['sl_mult'], 0.01), 0.25)
                pnl, trade_date, filled = run_long_trade_static(
                    prices, highs, lows, dates, times,
                    idx, computed_tp, computed_sl, EXIT_TIME_MIN)
            else:
                pnl, trade_date, filled = run_long_trade_static(
                    prices, highs, lows, dates, times,
                    idx, cfg['tp_pct'], cfg['sl_pct'], EXIT_TIME_MIN)

            if not filled:
                continue

            pnl -= 2 * SLIPPAGE_PCT
            last_traded_date = d_int

            trade = {
                'date': d_date, 'ticker': ticker, 'pnl': pnl,
                'prob': prob, 'close': float(prices[idx]),
                'mode': mode, 'tp': cfg['tp_pct'],
                'sl': cfg['sl_pct'] if mode not in ('atr_sl', 'trailing_decay') else cfg['sl_mult'],
            }
            all_trades.append(trade)
            per_ticker_trades[ticker].append(trade)

    return all_trades, per_ticker_trades


# ==============================================================================
# PHASE 1: FEATURE ENGINEERING
# ==============================================================================
def compute_ticker_features(bq_df):
    """Compute ~34 UNBIASED features per ticker from BQ data.
    NO val_b columns used — only val_a + opt.
    Family diversity features replace SHORT's cross-alpha features."""
    indiv = bq_df[bq_df['is_ensemble'] == False].copy()
    if indiv.empty:
        indiv = bq_df[bq_df['is_ensemble'] != True].copy()

    indiv = indiv[~indiv['tpsl_mode'].isin(SKIP_MODES)]
    indiv['va_opt_ratio'] = indiv['val_a_avg_pnl'] / indiv['opt_avg_pnl']

    rows = []
    for ticker, grp in indiv.groupby('ticker'):
        n = len(grp)
        filt = grp[(grp['tpsl_mode'] != 'limit_entry') &
                    (grp['va_opt_ratio'] <= MAX_VA_OPT_RATIO)]

        seed_grp_keys = ['ticker', 'experiment', 'tpsl_mode', 'entry_minute',
                         'prob_thresh', 'tp_pct', 'sl_pct', 'tp_mult', 'sl_mult',
                         'trail_start_mult', 'trail_end_mult',
                         'grace_minutes', 'dip_pct', 'max_wait_minutes',
                         'vol_filter_pct']
        for k in seed_grp_keys:
            if k in grp.columns and grp[k].isna().any():
                grp[k] = grp[k].fillna(0)
        seed_agg = grp.groupby(seed_grp_keys).agg(
            n_seeds=('seed', 'nunique'),
            mean_va_pnl=('val_a_avg_pnl', 'mean'),
        ).reset_index()
        seed_agg['agree_18'] = 0.0
        for idx_sa, row_sa in seed_agg.iterrows():
            mask = True
            for k in seed_grp_keys:
                mask = mask & (grp[k] == row_sa[k])
            sub = grp[mask]
            n_pos = (sub['val_a_avg_pnl'] >= 0.018).sum()
            seed_agg.at[idx_sa, 'agree_18'] = n_pos / row_sa['n_seeds'] if row_sa['n_seeds'] > 0 else 0
        seed_agg['score_pnl2'] = seed_agg['mean_va_pnl'] ** 2 * seed_agg['agree_18']

        # Family diversity features
        grp_fam = grp.copy()
        grp_fam['exp_fam'] = grp_fam['experiment'].apply(assign_family)
        fam_configs = grp_fam[grp_fam['exp_fam'] != 'other']

        families_present = fam_configs['exp_fam'].unique()
        n_families = len(families_present)

        family_agree_frac = 0.0
        cross_family_agree = 0.0
        if n_families > 0:
            fam_best_va = fam_configs.groupby('exp_fam')['val_a_avg_pnl'].max()
            family_agree_frac = (fam_best_va >= DEPLOY_THRESHOLD).mean()

            seed_agg_fam = seed_agg.copy()
            seed_agg_fam['exp_fam'] = seed_agg_fam['experiment'].apply(assign_family)
            fam_seed_agg = seed_agg_fam[seed_agg_fam['exp_fam'] != 'other']
            if not fam_seed_agg.empty:
                fam_best_agree = fam_seed_agg.groupby('exp_fam')['agree_18'].max()
                cross_family_agree = fam_best_agree.min()

        row = {
            'ticker': ticker,
            'n_configs': n,
            'n_filt': len(filt),

            'pct_static': (grp['tpsl_mode'] == 'static').sum() / n,
            'pct_atr': (grp['tpsl_mode'] == 'atr').sum() / n,
            'pct_atr_adaptive': (grp['tpsl_mode'] == 'atr_adaptive').sum() / n,
            'pct_atr_sl': (grp['tpsl_mode'] == 'atr_sl').sum() / n,
            'pct_trailing_decay': (grp['tpsl_mode'] == 'trailing_decay').sum() / n,
            'pct_grace': (grp['tpsl_mode'] == 'grace').sum() / n,
            'pct_limit_entry': (grp['tpsl_mode'] == 'limit_entry').sum() / n,

            'va_median': grp['val_a_avg_pnl'].median(),
            'va_mean': grp['val_a_avg_pnl'].mean(),
            'va_best': grp['val_a_avg_pnl'].max(),
            'va_worst': grp['val_a_avg_pnl'].min(),
            'va_spread': grp['val_a_avg_pnl'].max() - grp['val_a_avg_pnl'].min(),
            'va_std': grp['val_a_avg_pnl'].std(),
            'va_pos_rate': (grp['val_a_avg_pnl'] >= 0.01).mean(),
            'va_wr_median': grp['val_a_win_rate'].median(),
            'va_count_median': grp['val_a_count'].median(),

            'opt_median': grp['opt_avg_pnl'].median(),
            'opt_best': grp['opt_avg_pnl'].max(),
            'opt_wr_median': grp['opt_win_rate'].median(),
            'opt_count_median': grp['opt_count'].median(),

            'va_opt_ratio_med': grp['va_opt_ratio'].median(),
            'va_opt_ratio_std': grp['va_opt_ratio'].std(),
            'retention': grp['val_a_avg_pnl'].median() / grp['opt_avg_pnl'].median()
                         if grp['opt_avg_pnl'].median() > 0 else 0,

            'agree_18_best': seed_agg['agree_18'].max() if not seed_agg.empty else 0,
            'agree_18_median': seed_agg['agree_18'].median() if not seed_agg.empty else 0,
            'score_pnl2_best': seed_agg['score_pnl2'].max() if not seed_agg.empty else 0,

            'n_families': n_families,
            'family_agree_frac': family_agree_frac,
            'cross_family_agree': cross_family_agree,
        }

        if len(filt) > 0:
            row['filt_va_median'] = filt['val_a_avg_pnl'].median()
            row['filt_va_best'] = filt['val_a_avg_pnl'].max()
            row['filt_va_pos_rate'] = (filt['val_a_avg_pnl'] >= 0.01).mean()
        else:
            row['filt_va_median'] = 0
            row['filt_va_best'] = 0
            row['filt_va_pos_rate'] = 0

        rows.append(row)

    features_df = pd.DataFrame(rows).set_index('ticker')
    features_df = features_df.fillna(0)
    return features_df


def compute_target(bq_df):
    """Compute target: val_b median PnL per ticker. USES TEST DATA (target only)."""
    indiv = bq_df[bq_df['is_ensemble'] == False].copy()
    if indiv.empty:
        indiv = bq_df[bq_df['is_ensemble'] != True].copy()

    indiv = indiv[~indiv['tpsl_mode'].isin(SKIP_MODES)]

    target = indiv.groupby('ticker').agg(
        vb_median=('val_b_avg_pnl', 'median'),
        vb_pos_rate=('val_b_avg_pnl', lambda x: (x > 0).mean()),
    )
    return target


# ==============================================================================
# PHASE 2-3: LOO-CV MODELS
# ==============================================================================
def train_loo_cv(X, y, model_class, model_params):
    loo = LeaveOneOut()
    predictions = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        model = model_class(**model_params)
        model.fit(X_train_s, y_train)
        predictions[test_idx] = model.predict(X_test_s)
    return predictions


def evaluate_model(y_true, y_pred, top_k=10):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rho, p_val = sp_stats.spearmanr(y_true, y_pred)
    true_top_k = set(np.argsort(y_true)[-top_k:])
    pred_top_k = set(np.argsort(y_pred)[-top_k:])
    top_k_prec = len(true_top_k & pred_top_k) / top_k
    true_bottom = set(np.argsort(y_true)[:top_k])
    pred_bottom = set(np.argsort(y_pred)[:top_k])
    bottom_k_prec = len(true_bottom & pred_bottom) / top_k
    return {
        'mae': mae, 'rmse': rmse, 'r2': r2,
        'spearman_rho': rho, 'spearman_p': p_val,
        'top_k_precision': top_k_prec,
        'bottom_k_precision': bottom_k_prec,
    }


def get_feature_importance(X_arr, y_arr, feature_names, model_class, model_params):
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_arr)
    model = model_class(**model_params)
    model.fit(X_s, y_arr)
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
    elif hasattr(model, 'coef_'):
        imp = np.abs(model.coef_)
    else:
        imp = np.zeros(len(feature_names))
    return pd.Series(imp, index=feature_names).sort_values(ascending=False)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
np.random.seed(42)

print(f"\n{'='*W}")
print(f"  LONG MODEL — ML TICKER SELECTION")
print(f"  Regression target: val_b median PnL per ticker")
print(f"  LOO-CV on {len(MODEL_TOP_K)} top-K values: {MODEL_TOP_K}")
print(f"  5 models: Ridge, Lasso, DecisionTree, RandomForest, GradientBoosting")
print(f"  7 modes: static, atr, atr_adaptive, atr_sl, trailing_decay, grace, limit_entry")
print(f"  Config selection: per-seed aggregated (ensemble val_a bugged)")
print(f"{'='*W}")


# ── PHASE 1: BQ + Features ──
print(f"\n  PHASE 1: Feature Engineering...")
bq_df = query_all_bq()
bq_df['va_opt_ratio'] = bq_df['val_a_avg_pnl'] / bq_df['opt_avg_pnl']
bq_df = bq_df[bq_df['va_opt_ratio'] <= MAX_VA_OPT_RATIO]
bq_df = bq_df[~bq_df['tpsl_mode'].isin(SKIP_MODES)]
n_tickers = bq_df['ticker'].nunique()
print(f"    {len(bq_df)} BQ rows across {n_tickers} tickers (after overfit + mode filter)")

features_df = compute_ticker_features(bq_df)
target_df = compute_target(bq_df)

combined = features_df.join(target_df, how='inner')
tickers = combined.index.tolist()
n = len(tickers)

feature_cols = [c for c in features_df.columns]
X_df = combined[feature_cols]
y = combined['vb_median'].values

print(f"    {n} tickers × {len(feature_cols)} features")
print(f"    Target (vb_median): min={y.min():+.2%}, median={np.median(y):+.2%}, max={y.max():+.2%}")
print(f"    Positive targets: {(y > 0).sum()}/{n} ({(y > 0).mean():.0%})")

print(f"\n  ── FEATURES (all unbiased — NO val_b columns) ──")
for i, col in enumerate(feature_cols):
    vals = X_df[col]
    print(f"    {i+1:2d}. {col:<25} min={vals.min():>8.3f}  med={vals.median():>8.3f}  max={vals.max():>8.3f}")


# ── PHASE 2-3: Models ──
print(f"\n  PHASE 2-3: LOO-CV Models...")

X_arr = X_df.values.astype(np.float64)
y_arr = y.astype(np.float64)

MODELS = {
    'Ridge':            (Ridge,                     {'alpha': 1.0}),
    'Lasso':            (Lasso,                     {'alpha': 0.001, 'max_iter': 10000}),
    'DecisionTree':     (DecisionTreeRegressor,     {'max_depth': 3, 'random_state': 42}),
    'RandomForest':     (RandomForestRegressor,     {'n_estimators': 100, 'max_depth': 3,
                                                     'max_features': 'sqrt', 'random_state': 42}),
    'GradientBoosting': (GradientBoostingRegressor, {'n_estimators': 50, 'max_depth': 2,
                                                     'learning_rate': 0.1, 'random_state': 42}),
}

model_results = {}
model_predictions = {}

for name, (cls, params) in MODELS.items():
    preds = train_loo_cv(X_arr, y_arr, cls, params)
    metrics = evaluate_model(y_arr, preds, top_k=ORACLE_TOP_N)
    model_results[name] = metrics
    model_predictions[name] = preds
    print(f"    {name:<20} R²={metrics['r2']:+.3f}  MAE={metrics['mae']:.4f}  "
          f"ρ={metrics['spearman_rho']:+.3f} (p={metrics['spearman_p']:.3f})  "
          f"Top10prec={metrics['top_k_precision']:.0%}")


# ── Feature importance (best model) ──
best_model_name = max(model_results.keys(),
                      key=lambda k: model_results[k]['spearman_rho'])
best_cls, best_params = MODELS[best_model_name]
importance = get_feature_importance(X_arr, y_arr, feature_cols, best_cls, best_params)

print(f"\n  ── Feature Importance ({best_model_name}) ──")
for feat, imp in importance.head(15).items():
    print(f"    {feat:<25} {imp:.4f}")


# ── PHASE 4: Ticker Selection ──
print(f"\n  PHASE 4: Ticker Selection...")

scaler_full = StandardScaler()
X_full_s = scaler_full.fit_transform(X_arr)

oracle_tickers = combined.nlargest(ORACLE_TOP_N, 'vb_median').index.tolist()

s3_configs, s3_tickers = select_configs_s3(bq_df)

ticker_lists = {'ORACLE': oracle_tickers, 'S3': s3_tickers}

for name, (cls, params) in MODELS.items():
    model = cls(**params)
    model.fit(X_full_s, y_arr)
    full_preds = model.predict(X_full_s)
    pred_df = pd.DataFrame({'ticker': tickers, 'pred_vb': full_preds,
                            'actual_vb': y_arr}).set_index('ticker')
    pred_df = pred_df.sort_values('pred_vb', ascending=False)

    for k in MODEL_TOP_K:
        key = f"{name}_TOP{k}"
        selected = pred_df.head(k).index.tolist()
        ticker_lists[key] = selected

print(f"\n  ── Ticker Lists ──")
for key, tks in sorted(ticker_lists.items()):
    oracle_overlap = len(set(tks) & set(oracle_tickers))
    s3_overlap = len(set(tks) & set(s3_tickers))
    print(f"    {key:<25} ({len(tks)} tickers, {oracle_overlap} in ORACLE, {s3_overlap} in S3): {', '.join(tks)}")


# ── LOO prediction table ──
print(f"\n  ── LOO Predictions vs Actual (all {n} tickers, sorted by actual) ──")
loo_df = pd.DataFrame({'ticker': tickers, 'actual_vb': y_arr})
for name in MODELS:
    loo_df[f'pred_{name}'] = model_predictions[name]
loo_df = loo_df.sort_values('actual_vb', ascending=False)
print(f"  {'Ticker':<7} {'Actual':>8}  " +
      '  '.join(f'{name[:8]:>8}' for name in MODELS))
for _, row in loo_df.head(25).iterrows():
    preds_str = '  '.join(f"{row[f'pred_{name}']:>+7.2%}" for name in MODELS)
    in_s3 = "S3" if row['ticker'] in s3_tickers else "  "
    print(f"  {row['ticker']:<7} {row['actual_vb']:>+7.2%}  {preds_str}  {in_s3}")


# ── PHASE 5: Backtest ──
all_tickers_union = sorted(set(
    t for tks in ticker_lists.values() for t in tks
))
if len(all_tickers_union) > MAX_CSV_TICKERS:
    keep = set(s3_tickers) | set(oracle_tickers)
    for name in [best_model_name]:
        for k in MODEL_TOP_K:
            key = f"{name}_TOP{k}"
            keep |= set(ticker_lists.get(key, []))
    all_tickers_union = sorted(keep)[:MAX_CSV_TICKERS]
    for key in list(ticker_lists.keys()):
        ticker_lists[key] = [t for t in ticker_lists[key] if t in all_tickers_union]

bt_keys = ['ORACLE', 'S3']
for k in MODEL_TOP_K:
    bt_keys.append(f"{best_model_name}_TOP{k}")

print(f"\n  PHASE 5: Backtesting {len(bt_keys)} strategies...")
print(f"    Loading OHLCV for {len(all_tickers_union)} tickers...")

ticker_data = {}
for ticker in all_tickers_union:
    df = load_master(ticker)
    if df is None:
        continue
    df = heal_data(df)
    ohlcv = precompute_ohlcv_arrays(df, FULL_START, FULL_END)
    if ohlcv is None:
        continue
    atr_pct = compute_daily_atr_pct(df)
    atr_vals = atr_pct.reindex(df.index).fillna(0.05)
    s = pd.to_datetime(FULL_START).tz_localize('America/Los_Angeles')
    e = pd.to_datetime(FULL_END).tz_localize('America/Los_Angeles') + timedelta(days=1)
    idx = df.index
    if str(idx.tz) != 'America/Los_Angeles':
        idx = idx.tz_convert('America/Los_Angeles')
    atr_cut = atr_vals[(idx >= s) & (idx < e)]
    if len(atr_cut) == len(ohlcv['prices']):
        ohlcv['atr_pct'] = atr_cut.values.astype(np.float64)
    else:
        ohlcv['atr_pct'] = np.full(len(ohlcv['prices']), 0.05)
    ticker_data[ticker] = {'ohlcv': ohlcv, 'df': df}
    del df
    gc.collect()

print(f"    Loaded {len(ticker_data)} tickers")

# Build config sets for each strategy
strategy_configs = {}

# S3: already built
strategy_configs['S3'] = {tk: cfg for tk, cfg in s3_configs.items() if tk in all_tickers_union}

# ORACLE + ML: per-seed best config
for key in bt_keys:
    if key == 'S3':
        continue
    tks = ticker_lists.get(key, [])
    cfgs = select_configs_perseed(bq_df, tks)
    strategy_configs[key] = cfgs

for key in bt_keys:
    cfgs = strategy_configs.get(key, {})
    tks_str = ', '.join(sorted(cfgs.keys()))
    print(f"    {key}: {len(cfgs)} configs — {tks_str}")

print(f"\n    Loading CSV scores...")
csv_scores_cache = {}
all_exp_pairs = set()
for key, cfgs in strategy_configs.items():
    for ticker, cfg in cfgs.items():
        all_exp_pairs.add((ticker, cfg['_base_experiment']))

for ticker, base_exp in sorted(all_exp_pairs):
    if ticker not in ticker_data:
        continue
    if (ticker, base_exp) in csv_scores_cache:
        continue
    seed_scores = load_seed_scores(ticker, base_exp)
    if seed_scores:
        csv_scores_cache[(ticker, base_exp)] = seed_scores
        print(f"      {ticker}/{base_exp}: {len(seed_scores)} seeds")
    gc.collect()

periods = [
    ('FULL',  FULL_START,  FULL_END),
    ('HALF1', FULL_START,  HALF1_END),
    ('HALF2', HALF2_START, FULL_END),
]

results = {}
for strat_key in bt_keys:
    cfgs = strategy_configs.get(strat_key, {})
    if not cfgs:
        for p_name, _, _ in periods:
            results[(strat_key, p_name)] = summarize_trades([], f"{strat_key}/{p_name}")
        continue
    for p_name, p_start, p_end in periods:
        np.random.seed(42)
        trades, per_tk = run_csv_backtest_split(
            cfgs, csv_scores_cache, ticker_data, p_start, p_end,
            f"{strat_key}/{p_name}")
        r = summarize_trades(trades, f"{strat_key}/{p_name}")
        r['per_ticker'] = per_tk
        results[(strat_key, p_name)] = r


# ── PHASE 6: Output ──
print(f"\n{'='*W}")
print(f"  PHASE 6: STRATEGY COMPARISON")
print(f"{'='*W}")

print(f"\n  {'Strategy':<25} │ {'── FULL ──────────':} │ {'── HALF1 ─────────':} │ {'── HALF2 ─────────':}")
print(f"  {'':25} │ {'N':>4} {'Tot':>7} {'Avg':>7} {'WR':>5} │ {'N':>4} {'Tot':>7} {'Avg':>7} {'WR':>5} │ {'N':>4} {'Tot':>7} {'Avg':>7} {'WR':>5}")
print(f"  {'─'*105}")

for sk in bt_keys:
    f = results[(sk, 'FULL')]
    h1 = results[(sk, 'HALF1')]
    h2 = results[(sk, 'HALF2')]
    bias = "[B]" if sk == "ORACLE" else ""
    print(f"  {sk:<25} │ {f['n']:>4} {f['total']:>+6.1%} {f['avg']:>+6.2%} {f['wr']:>4.0%} │ "
          f"{h1['n']:>4} {h1['total']:>+6.1%} {h1['avg']:>+6.2%} {h1['wr']:>4.0%} │ "
          f"{h2['n']:>4} {h2['total']:>+6.1%} {h2['avg']:>+6.2%} {h2['wr']:>4.0%} {bias}")

# ── Per-ticker for best model strategy ──
non_oracle_keys = [sk for sk in bt_keys if sk != 'ORACLE']
best_bt_key = max(non_oracle_keys, key=lambda sk: results[(sk, 'FULL')]['avg']) if non_oracle_keys else bt_keys[0]
print(f"\n  ── Per-Ticker (best strategy: {best_bt_key}) ──")
per_tk_full = results.get((best_bt_key, 'FULL'), {}).get('per_ticker', {})
per_tk_h1 = results.get((best_bt_key, 'HALF1'), {}).get('per_ticker', {})
per_tk_h2 = results.get((best_bt_key, 'HALF2'), {}).get('per_ticker', {})
all_bt_tickers = sorted(set(list(per_tk_full.keys()) + list(per_tk_h1.keys()) + list(per_tk_h2.keys())))

print(f"  {'Ticker':<7} │ {'FULL':>8} {'HALF1':>8} {'HALF2':>8} {'Status':>8}")
print(f"  {'─'*48}")
for ticker in all_bt_tickers:
    f_avg = np.mean([t['pnl'] for t in per_tk_full.get(ticker, [])]) if per_tk_full.get(ticker) else 0
    h1_avg = np.mean([t['pnl'] for t in per_tk_h1.get(ticker, [])]) if per_tk_h1.get(ticker) else 0
    h2_avg = np.mean([t['pnl'] for t in per_tk_h2.get(ticker, [])]) if per_tk_h2.get(ticker) else 0
    status = "HOLDS" if h1_avg > 0 and h2_avg > 0 else ("DECAYED" if h1_avg > 0 else "WEAK")
    print(f"  {ticker:<7} │ {f_avg:>+7.2%} {h1_avg:>+7.2%} {h2_avg:>+7.2%} {status:>8}")


# ── Model comparison table ──
print(f"\n  ── MODEL COMPARISON (LOO-CV, {n} tickers) ──")
print(f"  {'Model':<20} {'R²':>7} {'MAE':>8} {'RMSE':>8} {'ρ':>7} {'p-val':>8} {'Top10%':>8} {'Bot10%':>8}")
print(f"  {'─'*85}")
for name in MODELS:
    m = model_results[name]
    star = " ★" if name == best_model_name else ""
    print(f"  {name:<20} {m['r2']:>+6.3f} {m['mae']:>7.4f} {m['rmse']:>7.4f} "
          f"{m['spearman_rho']:>+6.3f} {m['spearman_p']:>7.4f} "
          f"{m['top_k_precision']:>7.0%} {m['bottom_k_precision']:>7.0%}{star}")


# ── S3 vs ML overlap ──
print(f"\n  ── S3 vs ML OVERLAP ──")
print(f"  S3 tickers ({len(s3_tickers)}): {', '.join(s3_tickers)}")
for k in MODEL_TOP_K:
    key = f"{best_model_name}_TOP{k}"
    ml_tks = set(ticker_lists.get(key, []))
    overlap = ml_tks & set(s3_tickers)
    ml_only = ml_tks - set(s3_tickers)
    s3_only = set(s3_tickers) - ml_tks
    print(f"  {key}: overlap={len(overlap)} ({', '.join(sorted(overlap))}), "
          f"ML-only={', '.join(sorted(ml_only))}, S3-only={', '.join(sorted(s3_only))}")


# ── Charts ──
print(f"\n  Generating charts...")

fig, axes = plt.subplots(2, 3, figsize=(22, 14))

# Chart 1: LOO predictions vs actual (best model)
ax1 = axes[0, 0]
best_preds = model_predictions[best_model_name]
ax1.scatter(y_arr * 100, best_preds * 100, alpha=0.7, edgecolors='k', linewidths=0.5, s=50)
for i, tk in enumerate(tickers):
    if abs(y_arr[i]) > 0.01 or abs(best_preds[i]) > 0.01:
        ax1.annotate(tk, (y_arr[i]*100, best_preds[i]*100), fontsize=6, alpha=0.7)
lims = [min(y_arr.min(), best_preds.min()) * 100 - 0.5,
        max(y_arr.max(), best_preds.max()) * 100 + 0.5]
ax1.plot(lims, lims, 'r--', alpha=0.5, linewidth=1)
ax1.axhline(y=0, color='gray', linewidth=0.5, linestyle=':')
ax1.axvline(x=0, color='gray', linewidth=0.5, linestyle=':')
ax1.set_xlabel('Actual val_b Median PnL (%)')
ax1.set_ylabel('Predicted (LOO) PnL (%)')
ax1.set_title(f'{best_model_name} LOO: ρ={model_results[best_model_name]["spearman_rho"]:+.3f}',
              fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Chart 2: Model comparison bar chart
ax2 = axes[0, 1]
model_names = list(MODELS.keys())
rhos = [model_results[m]['spearman_rho'] for m in model_names]
colors = ['#2E7D32' if r > 0 else '#C62828' for r in rhos]
bars = ax2.barh(model_names, rhos, color=colors, alpha=0.8)
ax2.axvline(x=0, color='gray', linewidth=0.5)
ax2.set_xlabel('Spearman ρ (LOO)')
ax2.set_title('Model Ranking Ability', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

# Chart 3: Feature importance (top 12)
ax3 = axes[0, 2]
top_feats = importance.head(12)
ax3.barh(range(len(top_feats)), top_feats.values, color='#1565C0', alpha=0.8)
ax3.set_yticks(range(len(top_feats)))
ax3.set_yticklabels(top_feats.index, fontsize=8)
ax3.invert_yaxis()
ax3.set_xlabel('Importance')
ax3.set_title(f'Feature Importance ({best_model_name})', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x')

# Chart 4: Cumulative PnL — FULL period
STRAT_COLORS = {
    'ORACLE': '#1565C0', 'S3': '#9E9E9E',
}
for k in MODEL_TOP_K:
    key = f"{best_model_name}_TOP{k}"
    STRAT_COLORS[key] = ['#2E7D32', '#66BB6A', '#A5D6A7'][MODEL_TOP_K.index(k)]

ax4 = axes[1, 0]
for sk in bt_keys:
    r = results[(sk, 'FULL')]
    if r['n'] == 0:
        continue
    df_t = r['trades_df'].sort_values('date')
    cum = df_t['pnl'].cumsum() * 100
    ls = '--' if sk == 'S3' else '-'
    lw = 3 if sk == best_bt_key else 1.5
    ax4.plot(pd.to_datetime(df_t['date'].values), cum.values,
             color=STRAT_COLORS.get(sk, '#666'), linestyle=ls,
             linewidth=lw, label=sk)
ax4.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax4.set_ylabel('Cumulative PnL (%)')
ax4.set_title('Strategies — FULL Period', fontsize=12, fontweight='bold')
ax4.legend(fontsize=8, loc='best')
ax4.grid(True, alpha=0.3)
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

# Chart 5: HALF1 vs HALF2 bar
ax5 = axes[1, 1]
x = np.arange(len(bt_keys))
bar_w = 0.35
h1_avgs = [results[(sk, 'HALF1')]['avg'] * 100 for sk in bt_keys]
h2_avgs = [results[(sk, 'HALF2')]['avg'] * 100 for sk in bt_keys]
ax5.bar(x - bar_w/2, h1_avgs, bar_w, label='HALF1', color='#1565C0', alpha=0.8)
ax5.bar(x + bar_w/2, h2_avgs, bar_w, label='HALF2', color='#E65100', alpha=0.8)
ax5.set_xticks(x)
ax5.set_xticklabels([k.replace(f'{best_model_name}_', '') for k in bt_keys], fontsize=8)
ax5.set_ylabel('Avg PnL (%)')
ax5.set_title('In-Sample vs Out-of-Sample', fontsize=12, fontweight='bold')
ax5.legend(fontsize=9)
ax5.axhline(y=0, color='gray', linewidth=0.5)
ax5.grid(True, alpha=0.3, axis='y')

# Chart 6: Per-ticker avg PnL for best model strategy
ax6 = axes[1, 2]
if all_bt_tickers:
    tk_avgs = []
    for tk in all_bt_tickers:
        trades = per_tk_full.get(tk, [])
        tk_avgs.append(np.mean([t['pnl'] for t in trades]) * 100 if trades else 0)
    colors = ['#2E7D32' if v > 0 else '#C62828' for v in tk_avgs]
    y_pos = np.arange(len(all_bt_tickers))
    ax6.barh(y_pos, tk_avgs, color=colors, alpha=0.8)
    ax6.set_yticks(y_pos)
    ax6.set_yticklabels(all_bt_tickers, fontsize=9)
    ax6.set_xlabel('Avg PnL (%)')
    ax6.set_title(f'Per-Ticker ({best_bt_key})', fontsize=12, fontweight='bold')
    ax6.axvline(x=0, color='gray', linewidth=0.5)
    ax6.grid(True, alpha=0.3, axis='x')

fig.suptitle(f'LONG Model — ML Ticker Selection\n'
             f'Best model: {best_model_name} (ρ={model_results[best_model_name]["spearman_rho"]:+.3f})',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/tmp/select_long_model.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"  Chart saved: /tmp/select_long_model.png")


# ── Summary ──
print(f"\n{'='*W}")
print(f"  SUMMARY")
print(f"{'='*W}")
print(f"  Best ML model: {best_model_name} (ρ={model_results[best_model_name]['spearman_rho']:+.3f}, "
      f"Top10prec={model_results[best_model_name]['top_k_precision']:.0%})")
r_best = results[(best_bt_key, 'FULL')]
r_oracle = results[('ORACLE', 'FULL')]
r_s3 = results[('S3', 'FULL')]
print(f"  {best_bt_key}: {r_best['n']}t, Avg={r_best['avg']:+.2%}, WR={r_best['wr']:.0%}")
print(f"  ORACLE:  {r_oracle['n']}t, Avg={r_oracle['avg']:+.2%}, WR={r_oracle['wr']:.0%} [biased]")
print(f"  S3:      {r_s3['n']}t, Avg={r_s3['avg']:+.2%}, WR={r_s3['wr']:.0%}")
gap_pct = (r_best['avg'] - r_s3['avg']) / abs(r_s3['avg']) * 100 if r_s3['avg'] != 0 else 0
oracle_gap = (r_best['avg'] - r_oracle['avg']) / abs(r_oracle['avg']) * 100 if r_oracle['avg'] != 0 else 0
print(f"  vs S3: {gap_pct:+.0f}%  |  vs ORACLE: {oracle_gap:+.0f}%")
print(f"  ML beat S3? {'YES' if r_best['avg'] > r_s3['avg'] else 'NO'}")

best_h2 = results[(best_bt_key, 'HALF2')]
s3_h2 = results[('S3', 'HALF2')]
oracle_h2 = results[('ORACLE', 'HALF2')]
print(f"  HALF2 (regime shift): {best_bt_key}={best_h2['avg']:+.2%} vs S3={s3_h2['avg']:+.2%} vs ORACLE={oracle_h2['avg']:+.2%}")
print(f"{'='*W}")
