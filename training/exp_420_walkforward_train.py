"""
Exp 420-425 Walkforward Training (10AM SHORT Reversal Model)
==============================================================
Targets the 10:00 AM EST morning reversal for short entries:
  - Model scores at 6:45-6:50 AM PST (9:45-9:50 EST, min 405-410)
  - Trade executes by 7:00 AM PST (10:00 EST) — 10-15 min lead time
  - Target: price drops ≥2.5% from scoring bar Close before EOD
  - New features: Morning_Ext (9:30→scoring time spike), Gap×VIX interaction
  - Same Transformer architecture + ASL loss as Exp 400-405

Analysis basis:
  - 10:00 AM EST optimal entry: +1.36% avg return vs +0.91% at 9:30 bell
  - MFE ~2.7% → 2.5% dump threshold
  - Gap traps: overnight gap + high VIX → high short probability
  - Model sees 240-bar lookback including the 9:30-9:50 morning spike

Single fold: 425_10AM (cutoff Jan 2026).
  OPT: Jan-Mar 2026 | VAL: Apr-May 2026 | TEST: Jun-Jul 2026
"""

import sys
import subprocess
import os
import json
import random
import io

import numpy as np
import pandas as pd
import pytz
from datetime import datetime
import gc
import ctypes

def _release_memory():
    gc.collect()
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

# ==============================================================================
# ENVIRONMENT SYNC & AUTH
# ==============================================================================
SCRIPT_VERSION = "420WF.1"
print(f"Script version: {SCRIPT_VERSION}")
print("Syncing environment for Experiment 420-425 Walkforward (10AM SHORT Reversal)...")

try:
    import tensorflow as tf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "tensorflow[and-cuda]", "pandas_ta",
                           "google-cloud-storage", "pyarrow", "gcsfs", "--quiet"])
    import tensorflow as tf

try:
    import pandas_ta as ta
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas_ta", "--no-deps", "--quiet"])
    import pandas_ta as ta

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling1D,
    LayerNormalization, MultiHeadAttention, Add,
    BatchNormalization, Concatenate, Embedding, Flatten,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras import mixed_precision
from google.cloud import storage

try:
    from google.colab import auth
    auth.authenticate_user()
except (ImportError, Exception):
    pass
client = storage.Client()
bucket = client.bucket(os.environ.get("GCS_BUCKET", ""))

# ==============================================================================
# GPU + MULTI-GPU SETUP
# ==============================================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    mixed_precision.set_global_policy('mixed_float16')
    print(f"GPU ACTIVE: {len(gpus)} GPU(s) detected, Mixed Precision Enabled")

if len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy()
    print(f"Multi-GPU: MirroredStrategy with {strategy.num_replicas_in_sync} replicas")
else:
    strategy = tf.distribute.get_strategy()
    print(f"Single-device strategy")

NUM_REPLICAS = strategy.num_replicas_in_sync

# ==============================================================================
# COLUMN DEFINITIONS
# ==============================================================================
MICRO_COLS = ['Spread', 'OBI', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'bsize', 'asize', 'Total_Vol']

ROC_HORIZONS = [5, 30, 60, 120]

ASSET_COLS = (['Gap_Pct', 'RSI_14', 'RSI_60', 'RVOL', 'Compression',
               'ATR_14', 'Dist_to_PM_High', 'Bar_Is_Real',
               'Morning_Ext', 'Gap_VIX'] +
              [f'ROC_{h}m' for h in ROC_HORIZONS] + MICRO_COLS)

TIME_STEPS = 240
EMBED_DIM  = 16
PST = pytz.timezone('America/Los_Angeles')
TICKER_FILE_MAP = {'BITF': 'KEEL', 'MIGI': 'BGDE'}

# ==============================================================================
# CONFIGURATION — SINGLE FOLD (matches deployed 414)
# ==============================================================================
ALL_FOLDS = [
    {'exp_id': '425_10AM', 'cutoff': '2026-01-01', 'seed_base': 42500},
]

FOLDS_TO_TRAIN = ALL_FOLDS
SEEDS_PER_FOLD = 15

# ==============================================================================
# TEACHER SELECTION — Script-selected by dump rate + sector diversity
# ==============================================================================
SECTORS = {
    'crypto': {
        'tickers': ['RIOT', 'CLSK', 'HUT', 'MARA', 'CIFR', 'IREN', 'APLD',
                     'KEEL', 'WULF', 'COIN', 'MSTR', 'CORZ', 'BTBT', 'HOOD'],
        'label': 'Crypto/Blockchain',
    },
    'defense': {
        'tickers': ['LUNR', 'RKLB', 'KTOS', 'AVAV', 'PL', 'BKSY', 'RDW',
                     'LMT', 'NOC', 'RTX', 'LHX', 'GD', 'BWXT', 'SPIR',
                     'MNTS', 'NNDM'],
        'label': 'Defense/Aero',
    },
    'nuclear': {
        'tickers': ['SMR', 'OKLO', 'CCJ', 'CEG', 'NXE', 'NBIS', 'NNE'],
        'label': 'Nuclear',
    },
    'clean_energy': {
        'tickers': ['FCEL', 'PLUG', 'BE', 'FLNC', 'ENPH', 'FSLR', 'BEPC'],
        'label': 'Clean Energy',
    },
    'oil_gas': {
        'tickers': ['XOM', 'CVX', 'COP', 'EQT', 'KMI', 'LNG', 'WMB', 'NRG', 'TPL'],
        'label': 'Oil & Gas',
    },
    'utility': {
        'tickers': ['DUK', 'NEE', 'PEG', 'VST', 'TLN', 'GEV'],
        'label': 'Utility/Power',
    },
    'space_evtol': {
        'tickers': ['ASTS', 'ACHR', 'JOBY', 'RCAT', 'BBAI', 'SOUN', 'SMLR', 'VOYG'],
        'label': 'Space/eVTOL/AI',
    },
    'semiconductor': {
        'tickers': ['NVDA', 'AMD', 'MU', 'SMCI', 'ALAB', 'ARM', 'CRDO',
                     'MXL', 'UCTT', 'WOLF', 'AEHR', 'ACMR'],
        'label': 'Semiconductor',
    },
    'tech_infra': {
        'tickers': ['PLTR', 'VRT', 'POWL', 'MOD', 'NET', 'EQIX', 'MVIS',
                     'ETN', 'GE', 'HUBB', 'ONDS', 'NXT'],
        'label': 'Tech/Infra',
    },
    'ev_auto': {
        'tickers': ['TSLA', 'RIVN', 'LCID', 'NIO', 'LI', 'XPEV', 'F', 'GM', 'VWAGY'],
        'label': 'EV/Auto',
    },
    'biotech': {
        'tickers': ['MRNA', 'BEAM', 'NTLA', 'TWST', 'SLS', 'NEOV'],
        'label': 'Biotech/Genomic',
    },
    'fintech': {
        'tickers': ['SQ', 'UPST', 'ROKU'],
        'label': 'Fintech/Consumer',
    },
    'materials': {
        'tickers': ['MP', 'CLF', 'FCX', 'BHP', 'LIN', 'CAT'],
        'label': 'Materials/Indust',
    },
    'mega_tech': {
        'tickers': ['AMZN', 'BABA'],
        'label': 'Mega-cap Tech',
    },
}

# ── 10AM model: 2.5% dump threshold (MFE ~2.7%), wider acceptance ──
DUMP_THRESHOLD_SELECT = 0.025
TARGET_TEACHERS = 30
MIN_PER_SECTOR = 1
MIN_DUMP_RATE = 0.10
MAX_PER_SECTOR = 8
MAX_CRYPTO = 10

# ── Scoring window: 6:45-6:50 AM PST (9:45-9:50 EST), trade by 7:00 AM PST ──
ENTRY_START_MIN = 405
ENTRY_END_MIN = 410
EXIT_MIN = 780

ticker_to_sector = {}
all_candidate_tickers = []
for sec, info in SECTORS.items():
    for t in info['tickers']:
        ticker_to_sector[t] = sec
        if t not in all_candidate_tickers:
            all_candidate_tickers.append(t)

print(f"\n{'='*70}")
print(f"TEACHER SELECTION: Scanning {len(all_candidate_tickers)} tickers for dump rate at {DUMP_THRESHOLD_SELECT:.1%}")
print(f"  Entry window: min {ENTRY_START_MIN}-{ENTRY_END_MIN} PST (9:45-9:50 AM EST), trade by 10:00 AM EST")
print(f"{'='*70}")

ticker_dump_rates = {}
for ticker in sorted(all_candidate_tickers):
    gcs_tk = TICKER_FILE_MAP.get(ticker, ticker)
    blob_path = f"Data/Inference/{gcs_tk}_1m_full_historical_data.parquet"
    blob = bucket.blob(blob_path)
    if not blob.exists():
        continue

    try:
        df = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        df = df.sort_index()

        for c in ['Open', 'High', 'Low', 'Close']:
            if c in df.columns:
                df[c] = df[c].ffill()
        df['Volume'] = df.get('Volume', pd.Series(0, index=df.index)).fillna(0)

        df_pst = df.index.tz_convert(PST)
        mins = df_pst.hour * 60 + df_pst.minute

        is_before_exit = mins <= EXIT_MIN
        is_real = df['Volume'] > 0
        df['Safe_Low'] = np.where(is_before_exit & is_real, df['Low'], np.nan)
        df['date'] = df_pst.date

        future_mins_list = []
        for dt, grp in df.groupby('date'):
            sl = grp['Safe_Low'].values.copy()
            fm_lo = np.full_like(sl, np.nan)
            running_min = np.nan
            for i in range(len(sl) - 1, 0, -1):
                if not np.isnan(sl[i]):
                    running_min = sl[i] if np.isnan(running_min) else min(running_min, sl[i])
                fm_lo[i - 1] = running_min
            future_mins_list.append(pd.Series(fm_lo, index=grp.index))

        df['Future_Min'] = pd.concat(future_mins_list)

        is_entry = ((mins >= ENTRY_START_MIN) & (mins <= ENTRY_END_MIN) & (df['Volume'] > 0))
        entry_bars = df[is_entry].copy()

        if len(entry_bars) == 0:
            continue

        close_safe = entry_bars['Close'].replace(0, np.nan)
        entry_bars['max_drop'] = (1 - (entry_bars['Future_Min'] / close_safe)).clip(lower=0)

        daily_best = entry_bars.groupby('date')['max_drop'].max()
        n_days = len(daily_best)
        n_dump = int((daily_best >= DUMP_THRESHOLD_SELECT).sum())
        dump_rate = n_dump / n_days if n_days > 0 else 0

        master_tk = TICKER_FILE_MAP.get(ticker, ticker)
        master_blob = bucket.blob(f"Data/Inference_v2/{master_tk}_master_inference.parquet")
        if not master_blob.exists():
            print(f"   {ticker}: OHLCV only (no master_inference) — skipped as teacher")
            del df; _release_memory()
            continue

        ticker_dump_rates[ticker] = {
            'dump_rate': dump_rate,
            'n_dump': n_dump,
            'n_days': n_days,
            'sector': ticker_to_sector.get(ticker, 'other'),
        }
    except Exception as e:
        print(f"   {ticker}: scan failed — {e}")
        continue

    del df
    _release_memory()

print(f"\n   Scanned {len(ticker_dump_rates)} tickers successfully")

pre_filter = len(ticker_dump_rates)
ticker_dump_rates = {t: info for t, info in ticker_dump_rates.items()
                     if info['dump_rate'] >= MIN_DUMP_RATE}
print(f"   After MIN_DUMP_RATE filter ({MIN_DUMP_RATE:.0%}): {len(ticker_dump_rates)}/{pre_filter} tickers remain")

by_sector = {}
for t, info in ticker_dump_rates.items():
    sec = info['sector']
    if sec not in by_sector:
        by_sector[sec] = []
    by_sector[sec].append((t, info['n_dump'], info['dump_rate']))

for sec in by_sector:
    by_sector[sec].sort(key=lambda x: -x[1])

selected = set()
selected_list = []

for sec in sorted(by_sector.keys()):
    candidates = by_sector[sec]
    for t, n_dump, rate in candidates[:MIN_PER_SECTOR]:
        if t not in selected:
            selected.add(t)
            selected_list.append((t, n_dump, rate, sec))

all_ranked = sorted(ticker_dump_rates.items(), key=lambda x: -x[1]['n_dump'])
sector_counts = {}
for _, _, _, sec in selected_list:
    sector_counts[sec] = sector_counts.get(sec, 0) + 1

for t, info in all_ranked:
    if len(selected) >= TARGET_TEACHERS:
        break
    if t not in selected:
        sec = info['sector']
        cap = MAX_CRYPTO if sec == 'crypto' else MAX_PER_SECTOR
        if sector_counts.get(sec, 0) >= cap:
            continue
        selected.add(t)
        selected_list.append((t, info['n_dump'], info['dump_rate'], sec))
        sector_counts[sec] = sector_counts.get(sec, 0) + 1

selected_list.sort(key=lambda x: (-x[1], x[3], x[0]))

# Ensure core 6 SHORT tickers are always included as teachers
CORE_SHORT_TICKERS = ['BE', 'IREN', 'OKLO', 'ONDS', 'RKLB', 'WOLF']
for t in CORE_SHORT_TICKERS:
    if t not in selected and t in ticker_dump_rates:
        sec = ticker_to_sector.get(t, 'other')
        info = ticker_dump_rates[t]
        selected.add(t)
        selected_list.append((t, info['n_dump'], info['dump_rate'], sec))

selected_list.sort(key=lambda x: (-x[1], x[3], x[0]))

TRAIN_TEACHERS = [t for t, _, _, _ in selected_list]

print(f"\n   Selected {len(TRAIN_TEACHERS)} teachers (ranked by #dumps, min_rate={MIN_DUMP_RATE:.0%}):")
print(f"   {'Ticker':<8} {'Sector':<18} {'#Dumps':>6} {'DumpRate':>8} {'#Days':>6}")
print(f"   {'─'*8} {'─'*18} {'─'*6} {'─'*8} {'─'*6}")
for t, n_dump, rate, sec in selected_list:
    info = ticker_dump_rates[t]
    core_flag = " ★" if t in CORE_SHORT_TICKERS else ""
    print(f"   {t:<8} {SECTORS.get(sec, {}).get('label', sec):<18} {n_dump:>6} {rate:>7.1%} {info['n_days']:>6}{core_flag}")

sector_counts_print = {}
for _, _, _, sec in selected_list:
    sector_counts_print[sec] = sector_counts_print.get(sec, 0) + 1
print(f"\n   Sector coverage: {len(sector_counts_print)}/{len(SECTORS)} sectors")
for sec, cnt in sorted(sector_counts_print.items(), key=lambda x: -x[1]):
    cap_label = f" (cap={MAX_CRYPTO})" if sec == 'crypto' else f" (cap={MAX_PER_SECTOR})"
    print(f"      {SECTORS.get(sec, {}).get('label', sec):<20} {cnt} teachers{cap_label}")

del ticker_dump_rates, by_sector, all_ranked, selected, selected_list, sector_counts, sector_counts_print
_release_memory()

# ==============================================================================
# TICKER START DATES & POOLS
# ==============================================================================
TICKER_START_DATES = {}

STATS_POOL   = TRAIN_TEACHERS
SCORING_POOL = list(TRAIN_TEACHERS) + [
    'ACHR', 'ACMR', 'AEHR', 'ALAB', 'AMD', 'APLD', 'ARM', 'ASTS', 'AVAV',
    'BABA', 'BBAI', 'BE', 'BEAM', 'BHP', 'BKSY', 'BTBT',
    'CAT', 'CCJ', 'CEG', 'CIFR', 'CLSK', 'CLF', 'COIN', 'COP', 'CORZ', 'CRDO', 'CVX',
    'DUK', 'ENPH', 'EQIX', 'EQT', 'ETN',
    'F', 'FCEL', 'FCX', 'FLNC', 'FSLR',
    'GD', 'GE', 'GEV', 'GM',
    'HOOD', 'HUT',
    'IREN',
    'JOBY',
    'KEEL', 'KMI', 'KTOS',
    'LHX', 'LIN', 'LMT', 'LNG', 'LUNR',
    'MARA', 'MNT', 'MNTS', 'MOD', 'MP', 'MRNA', 'MSTR', 'MU', 'MVIS', 'MXL',
    'NBIS', 'NEE', 'NEOV', 'NET', 'NIO', 'NNE', 'NNDM', 'NOC', 'NRG', 'NTLA', 'NVDA', 'NXE',
    'OKLO', 'ONDS',
    'PL', 'PLTR', 'PLUG', 'POWL',
    'RCAT', 'RDW', 'RIOT', 'RIVN', 'RKLB', 'ROKU', 'RTX',
    'SLS', 'SMCI', 'SMR', 'SOUN', 'SQ',
    'TSLA', 'TWST',
    'UCTT', 'UPST',
    'VOYG', 'VRT', 'VST',
    'WOLF', 'WULF',
    'XOM', 'XPEV',
]
SCORING_POOL = sorted(set(SCORING_POOL))

CONTEXT_ASSETS = ['SPY', 'VIX', 'TLT', 'IWM']

CONTEXT_COLS = []
for _t in CONTEXT_ASSETS:
    CONTEXT_COLS.append(f'{_t}_Anchor_ROC')
    for _h in ROC_HORIZONS:
        CONTEXT_COLS.append(f'{_t}_ROC_{_h}m')
CONTEXT_COLS += ['Time_Pos', 'VIX_Level']
ALL_COLS = ASSET_COLS + CONTEXT_COLS

DATA_START_DATE   = "2022-01-01"
TRAIN_START_DATE  = "2022-02-01"
DUMP_THRESHOLD    = 0.025
ASL_GAMMA_NEG     = 3.0
ASL_GAMMA_POS     = 0.5
ASL_CLIP          = 0.05
ASL_ALPHA_POS     = 2.0
SAMPLE_WEIGHT_CAP = 5.0
TARGET_POS_RATIO  = 0.25
PREC_TOP_K        = 0.02

BUCKET_NAME   = os.environ.get("GCS_BUCKET", "")
INFERENCE_DIR = "Data/Inference_v2"
OHLCV_DIR     = "Data/Inference"

LOCAL_RESULTS_DIR = "Results"
LOCAL_MODEL_DIR   = "Models"
os.makedirs(LOCAL_RESULTS_DIR, exist_ok=True)
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

GCS_MODEL_PATH   = "Algos/to_deploy/Saved_models_v2/UNIVERSAL/Direction"
STATS_DIR        = "Data/Inference_v2"

PER_REPLICA_BATCH = 256
GLOBAL_BATCH_SIZE = PER_REPLICA_BATCH * NUM_REPLICAS

ASSET_INDEX = {ticker: i for i, ticker in enumerate(TRAIN_TEACHERS)}
N_ASSETS    = len(TRAIN_TEACHERS) + 1
OTHER_IDX   = len(TRAIN_TEACHERS)

print(f"\nConfig: {len(TRAIN_TEACHERS)} teachers, {N_ASSETS} embeddings, "
      f"batch={PER_REPLICA_BATCH}x{NUM_REPLICAS}={GLOBAL_BATCH_SIZE}")
print(f"Dump threshold: {DUMP_THRESHOLD:.1%}  ASL: gamma_neg={ASL_GAMMA_NEG}, alpha_pos={ASL_ALPHA_POS}")
print(f"Scoring window: min {ENTRY_START_MIN}-{ENTRY_END_MIN} PST (9:45-9:50 AM EST)")
print(f"Target pos ratio: {TARGET_POS_RATIO:.0%}")
print(f"New features: Morning_Ext, Gap_VIX, VIX_Level")
print(f"Asset features: {len(ASSET_COLS)}  Context features: {len(CONTEXT_COLS)}")
print(f"Folds to train: {[f['exp_id'] for f in FOLDS_TO_TRAIN]}")
print(f"Seeds per fold: {SEEDS_PER_FOLD}")

# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_master(ticker, tail_days=None, add_micro_zeros=True):
    clean = ticker.replace("I:", "").replace("X:", "")
    file_ticker = TICKER_FILE_MAP.get(clean, clean)

    uri = f"gs://{BUCKET_NAME}/{INFERENCE_DIR}/{file_ticker}_master_inference.parquet"
    fallback_uri = f"gs://{BUCKET_NAME}/{OHLCV_DIR}/{file_ticker}_1m_full_historical_data.parquet"

    df = None
    for attempt_uri in [uri, fallback_uri]:
        try:
            df = pd.read_parquet(attempt_uri)
            if attempt_uri == fallback_uri:
                print(f"      {ticker}: loaded from OHLCV fallback (no master_inference)")
            break
        except Exception:
            continue

    if df is None:
        print(f"   Failed to load {ticker}: no master_inference or OHLCV found")
        return None

    try:
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
        std_map = {
            'Total_vol': 'Total_Vol', 'Trade_count': 'Trade_Count',
            'Volume_delta': 'Volume_Delta', 'Delta_pct': 'Delta_Pct',
            'Obi': 'OBI', 'Bsize': 'bsize', 'Asize': 'asize',
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume',
        }
        lower_map = {k.lower(): v for k, v in std_map.items()}
        df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip())
                      for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()
        if add_micro_zeros:
            for c in MICRO_COLS:
                if c not in df.columns:
                    df[c] = 0.0
        df = df[df.index.notna()].sort_index()
        if tail_days is not None:
            cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=tail_days)
            df = df[df.index >= cutoff]
            if df.empty:
                print(f"      {ticker}: no recent data -- skipping")
                return None
        return df
    except Exception as e:
        print(f"   Failed to load {ticker}: {e}")
        return None


def heal_data(df):
    df = df.resample('1min').asfreq()
    for c in ['Open', 'High', 'Low', 'Close']:
        if c in df.columns:
            df[c] = df[c].ffill()
    for c in ['Volume', 'Total_Vol', 'Volume_Delta', 'Trade_Count', 'Delta_Pct',
              'Spread', 'OBI', 'bsize', 'asize']:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df


# ==============================================================================
# MARKET CONTEXT (with VIX_Level for Gap×VIX interaction)
# ==============================================================================
def build_market_context():
    print("   Building market context...")
    frames = []
    vix_close_series = None
    for ticker in CONTEXT_ASSETS:
        df = load_master(ticker)
        if df is None:
            print(f"      Could not load context asset {ticker}")
            continue
        df = df.sort_index()
        df = heal_data(df)

        if ticker == 'VIX':
            vix_close_series = df['Close'].copy()

        daily_c = df['Close'].resample('D').last().ffill().shift(1).reindex(
            df.index, method='ffill')
        cols = {f'{ticker}_Anchor_ROC': (df['Close'] - daily_c) / daily_c}
        for h in ROC_HORIZONS:
            cols[f'{ticker}_ROC_{h}m'] = df['Close'].pct_change(h).fillna(0)
        frames.append(pd.DataFrame(cols, index=df.index))
        del df, daily_c
        _release_memory()
    if not frames:
        raise RuntimeError("No context assets loaded.")
    result = pd.concat(frames, axis=1).ffill().fillna(0)

    if vix_close_series is not None:
        result['VIX_Level'] = vix_close_series.reindex(result.index).ffill().fillna(0)
    else:
        result['VIX_Level'] = 0.0

    return result


# ==============================================================================
# FEATURE ENGINEERING (with Morning_Ext + Gap×VIX)
# ==============================================================================
def engineer_features(df, market_df):
    df_pst = df.index.tz_convert(PST)
    mins = df_pst.hour * 60 + df_pst.minute
    dates = df_pst.date

    # ── Standard asset features ──
    daily_c = df['Close'].resample('D').last().ffill().shift(1).reindex(
        df.index, method='ffill')
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

    # ── Morning Extension: return from 9:30 AM EST open to current bar ──
    # First real bar at market open (9:30 AM EST = minute 390 PST)
    open_window = (mins >= 388) & (mins <= 400) & (df['Volume'] > 0)
    temp_dates = pd.Series(dates, index=df.index)
    if open_window.any():
        daily_open_930 = df.loc[open_window].groupby(temp_dates[open_window])['Open'].first()
        mapped_open = temp_dates.map(daily_open_930)
    else:
        mapped_open = pd.Series(np.nan, index=df.index)

    after_open = mins >= 390
    df['Morning_Ext'] = np.where(
        after_open & (mapped_open.fillna(0) > 0),
        (df['Close'] / mapped_open.fillna(1).clip(lower=1e-8)) - 1,
        0.0
    )

    # ── Join market context ──
    df = df.join(market_df, how='left').ffill().fillna(0)

    # ── Gap × VIX interaction ──
    vix_level = df['VIX_Level'] if 'VIX_Level' in df.columns else pd.Series(0.0, index=df.index)
    df['Gap_VIX'] = df['Gap_Pct'] * vix_level

    # ── Time position + real bar flag ──
    df_pst2 = df.index.tz_convert(PST)
    mins2 = df_pst2.hour * 60 + df_pst2.minute
    df['Time_Pos'] = np.clip((mins2 - 390) / 390, 0, 1)
    df['Bar_Is_Real'] = (df['Volume'] > 0).astype(float)

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def apply_scaling(df, stats, inplace=False):
    df_s = df if inplace else df.copy()
    for col in ALL_COLS:
        if col in ('Time_Pos', 'Bar_Is_Real'):
            continue
        if col in stats:
            df_s[col] = ((df_s[col] - stats[col]['med'])
                         / stats[col]['iqr']).clip(-10, 10).astype('float32')
    return df_s


# ==============================================================================
# TARGET COMPUTATION — 10AM ENTRY, 2.5% DUMP
# ==============================================================================
def add_targets(df):
    df_pst = df.index.tz_convert(PST)
    mins = df_pst.hour * 60 + df_pst.minute
    is_safe = (df_pst.hour < 13) | ((df_pst.hour == 13) & (df_pst.minute == 0))
    is_real = df['Volume'] > 0

    df['Safe_Low'] = np.where(is_safe & is_real, df['Low'], np.inf)
    df['Future_Min'] = df.groupby(df.index.date)['Safe_Low'].transform(
        lambda x: x.shift(-1).fillna(np.inf).iloc[::-1].cummin().iloc[::-1])
    df['Target'] = (df['Future_Min'] < (df['Close'] * (1 - DUMP_THRESHOLD))).astype(int)

    # Valid_Row: 6:45-6:50 AM PST (9:45-9:50 EST) = minute 405-410
    df['Valid_Row'] = ((mins >= ENTRY_START_MIN) & (mins <= ENTRY_END_MIN)
                       & (df['Volume'] > 0)).astype(int)

    close_safe = df['Close'].replace(0, np.nan)
    df['Dump_Mag'] = (1 - (df['Future_Min'] / close_safe)).clip(lower=0).fillna(0)
    df['Dump_Mag'] = df['Dump_Mag'].replace([np.inf, -np.inf], 0)
    df['Sample_Weight'] = np.where(
        df['Target'] == 0, 1.0,
        np.clip(df['Dump_Mag'] / DUMP_THRESHOLD, 1.0, SAMPLE_WEIGHT_CAP))

    keep_cols = ALL_COLS + ['Target', 'Valid_Row', 'Sample_Weight',
                            'Open', 'High', 'Low', 'Close', 'Volume']
    drop_cols = [c for c in df.columns if c not in keep_cols]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)

    return df


def engineer_asset(asset_name, market_df):
    df = load_master(asset_name, tail_days=None, add_micro_zeros=True)
    if df is None:
        return None
    df = df.sort_index()
    ticker_start = TICKER_START_DATES.get(asset_name, DATA_START_DATE)
    df = df.loc[pd.to_datetime(ticker_start).tz_localize('UTC'):]
    if asset_name in TICKER_START_DATES:
        print(f"   {asset_name}: using custom start date {ticker_start}")
    df = heal_data(df)
    df = engineer_features(df, market_df)
    df = add_targets(df)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


# ==============================================================================
# MODEL ARCHITECTURE (identical to 400-405)
# ==============================================================================
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.4):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    res = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return Add()([x, res])


def build_universal_model():
    in_a  = Input(shape=(TIME_STEPS, len(ASSET_COLS)), name='asset_in')
    in_c  = Input(shape=(len(CONTEXT_COLS),),          name='context_in')
    in_id = Input(shape=(1,), dtype=tf.int32,          name='asset_id')
    x = transformer_encoder(Dense(128)(in_a), 128, 4, 128)
    x = GlobalAveragePooling1D()(transformer_encoder(x, 128, 4, 128))
    y = Dense(128, activation='relu')(LayerNormalization()(in_c))
    e = Embedding(N_ASSETS, EMBED_DIM, name='asset_embed')(in_id)
    e = Flatten()(e)
    e = Dense(32, activation='relu')(e)
    z = Concatenate()([x, Dropout(0.5)(y), e])
    z = Dense(128, activation='relu', kernel_regularizer=l2(5e-3))(z)
    out = Dense(1, activation='sigmoid', dtype='float32')(Dropout(0.5)(z))
    return Model(inputs=[in_a, in_c, in_id], outputs=out)


# ==============================================================================
# ASL LOSS
# ==============================================================================
def asymmetric_loss(gamma_neg=ASL_GAMMA_NEG, gamma_pos=ASL_GAMMA_POS,
                    clip=ASL_CLIP, alpha_pos=ASL_ALPHA_POS):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        loss_pos = alpha_pos * y_true * tf.pow(1.0 - y_pred, gamma_pos) * tf.math.log(y_pred)
        y_pred_neg = tf.clip_by_value(y_pred - clip, 0.0, 1.0)
        loss_neg = (1.0 - y_true) * tf.pow(y_pred_neg, gamma_neg) * tf.math.log(
            tf.clip_by_value(1.0 - y_pred_neg, 1e-7, 1.0))
        return -tf.reduce_mean(loss_pos + loss_neg)
    return loss


# ==============================================================================
# CUSTOM METRICS
# ==============================================================================
class PrecisionAtTopK(tf.keras.metrics.Metric):
    def __init__(self, k=PREC_TOP_K, name='prec_top2pct', **kwargs):
        super().__init__(name=name, **kwargs)
        self.k     = k
        self.tp    = self.add_weight(name='tp',    initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_flat = tf.reshape(y_pred,  [-1])
        y_true_flat = tf.reshape(tf.cast(y_true, tf.float32), [-1])
        n = tf.shape(y_pred_flat)[0]
        k = tf.maximum(tf.cast(tf.round(tf.cast(n, tf.float32) * self.k), tf.int32), 1)
        _, top_idx  = tf.math.top_k(y_pred_flat, k=k)
        top_labels  = tf.gather(y_true_flat, top_idx)
        self.tp.assign_add(tf.reduce_sum(top_labels))
        self.total.assign_add(tf.cast(k, tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.tp, self.total)

    def reset_state(self):
        self.tp.assign(0.)
        self.total.assign(0.)


# ==============================================================================
# STREAMING STATS (per-ticker, no giant concat)
# ==============================================================================
def _compute_stats_streaming(market_df, aware_cutoff):
    print("\nComputing scaling stats (per-ticker streaming)...")
    col_arrays = {col: [] for col in ALL_COLS if col not in ('Time_Pos', 'Bar_Is_Real')}
    diag = {}

    for ticker in STATS_POOL:
        print(f"   Loading {ticker}...")
        df = engineer_asset(ticker, market_df)
        if df is None:
            continue
        df = df.loc[TRAIN_START_DATE:]
        print(f"      rows={len(df):,}  range: {df.index.min()} to {df.index.max()}")

        mask_vr = df['Valid_Row'] == 1
        vr_count = int(mask_vr.sum())
        sr = float(df.loc[mask_vr, 'Target'].mean()) if vr_count > 0 else 0
        n_val = int((df.loc[mask_vr].index > aware_cutoff).sum())

        mask_pos = mask_vr & (df['Target'] == 1) & (df.index <= aware_cutoff)
        sw_info = None
        if mask_pos.sum() > 0:
            sw = df.loc[mask_pos, 'Sample_Weight']
            sw_info = {'n_pos': int(mask_pos.sum()), 'avg': float(sw.mean()),
                       'max': float(sw.max()), 'pct_cap': float((sw >= SAMPLE_WEIGHT_CAP).mean())}

        diag[ticker] = {'valid_rows': vr_count, 'dump_rate': sr,
                         'n_val': n_val, 'sw': sw_info}

        stats_mask = mask_vr & (df.index <= aware_cutoff)
        df_stats = df[stats_mask]
        for col in col_arrays:
            if col in df_stats.columns:
                col_arrays[col].append(df_stats[col].values.astype('float32'))

        del df, df_stats
        _release_memory()

    stats = {}
    for col, arrs in col_arrays.items():
        if not arrs:
            continue
        v = np.concatenate(arrs)
        stats[col] = {'med': float(np.median(v)),
                      'iqr': float(max(1e-6, np.percentile(v, 75) - np.percentile(v, 25)))}
        del v
    del col_arrays
    _release_memory()

    return stats, diag


def _print_diagnostics(diag):
    for t, info in diag.items():
        core_flag = " ★" if t in CORE_SHORT_TICKERS else ""
        print(f"   {t:<6}: valid_rows={info['valid_rows']:,}  dump_rate={info['dump_rate']:.1%}  val_rows={info['n_val']:,}{core_flag}")

    print(f"\nAsset Embedding Index:")
    for t, idx in ASSET_INDEX.items():
        print(f"   [{idx}] {t}")
    print(f"   [{OTHER_IDX}] OTHER (out-of-training tickers)")

    print(f"\nSample weight distribution (train positives only, cap={SAMPLE_WEIGHT_CAP}x):")
    for t, info in diag.items():
        if info['sw']:
            sw = info['sw']
            print(f"   {t:<6}: n_pos={sw['n_pos']:,}  avg_weight={sw['avg']:.2f}  "
                  f"max_weight={sw['max']:.2f}  pct_at_cap={sw['pct_cap']:.1%}")


# ==============================================================================
# PER-TICKER ARRAY BUILDER (with per-ticker equalized weighting)
# ==============================================================================
def _build_ticker_arrays(market_df, stats, tickers, aware_cutoff, steps):
    print("\nBuilding train+val arrays (per-ticker streaming)...")
    tr_arrays = []
    val_arrays = []

    for ticker in tickers:
        df = engineer_asset(ticker, market_df)
        if df is None:
            continue
        df = df.loc[TRAIN_START_DATE:]
        apply_scaling(df, stats, inplace=True)

        asset_id_val = np.int32(ASSET_INDEX.get(ticker, OTHER_IDX))

        for label, df_slice, arr_list in [
            ('TRAIN', df[df.index <= aware_cutoff], tr_arrays),
            ('VAL',   df[df.index > aware_cutoff],  val_arrays),
        ]:
            if len(df_slice) < steps:
                continue
            XA = df_slice[ASSET_COLS].values.astype('float32')
            XC = df_slice[CONTEXT_COLS].values.astype('float32')
            Y  = df_slice['Target'].values
            SW = df_slice['Sample_Weight'].values.astype('float32') if 'Sample_Weight' in df_slice.columns \
                 else np.ones(len(df_slice), dtype=np.float32)
            vi = np.where(df_slice['Valid_Row'].values)[0]
            vi = vi[vi >= steps]
            if len(vi) > 0:
                arr_list.append((XA, XC, Y, SW, vi.copy(), asset_id_val))
                if label == 'TRAIN':
                    dump_rate = Y[vi].mean()
                    print(f"   [{ticker}] valid={len(vi)}  dump={dump_rate:.1%}  id={asset_id_val}")

        del df
        _release_memory()

    total_pos_w = 0.0
    total_neg_w = 0.0
    per_ticker_stats = []
    for XA, XC, Y, SW, vi, aid in tr_arrays:
        pos_mask = Y[vi] == 1
        neg_mask = Y[vi] == 0
        pw = float(SW[vi][pos_mask].sum()) if pos_mask.sum() > 0 else 0.0
        nw = float(SW[vi][neg_mask].sum()) if neg_mask.sum() > 0 else 0.0
        total_pos_w += pw
        total_neg_w += nw
        ticker_name = [t for t, i in ASSET_INDEX.items() if i == aid]
        ticker_name = ticker_name[0] if ticker_name else f"id={aid}"
        per_ticker_stats.append((ticker_name, int(pos_mask.sum()), int(neg_mask.sum()), pw))

    total_w = total_pos_w + total_neg_w
    natural_ratio = total_pos_w / total_w if total_w > 0 else 0.0
    T = TARGET_POS_RATIO

    if natural_ratio > 0 and natural_ratio < T:
        pos_mult = (T * total_neg_w) / ((1.0 - T) * total_pos_w)
        for XA, XC, Y, SW, vi, aid in tr_arrays:
            pos_mask_full = Y == 1
            SW[pos_mask_full] *= np.float32(pos_mult)
    else:
        pos_mult = 1.0

    print(f"\n   Class-ratio rebalancing (train only):")
    print(f"      Natural pos ratio: {natural_ratio:.2%}  ({total_pos_w:.0f} / {total_w:.0f})")
    print(f"      Target  pos ratio: {T:.0%}")
    print(f"      Positive class multiplier: {pos_mult:.3f}")
    print(f"      Per-ticker contribution (natural, no equalization):")
    for tname, npos, nneg, pw in sorted(per_ticker_stats, key=lambda x: -x[3]):
        print(f"         {tname:<6}: pos={npos:5d}  neg={nneg:6d}  pos_weight={pw:8.1f}")

    print(f"   Train: {len(tr_arrays)} tickers   Val: {len(val_arrays)} tickers")
    return tr_arrays, val_arrays


# ==============================================================================
# DATASET FROM PRE-BUILT ARRAYS
# ==============================================================================
def _make_ds_from_arrays(ticker_arrays, steps, batch_size, use_weights=True):
    if not ticker_arrays:
        return None, 0

    combined = np.array(
        [(t, int(i)) for t, (_, _, _, _, idx, _) in enumerate(ticker_arrays) for i in idx],
        dtype=np.int64)

    def gen():
        while True:
            perm = np.random.permutation(len(combined))
            for p in perm:
                t_idx, i = combined[p]
                XA, XC, Y, SW, _, asset_id_val = ticker_arrays[t_idx]
                w = SW[i] if use_weights else np.float32(1.0)
                xa_sample = XA[i - steps:i]

                yield (
                    {'asset_in': xa_sample,
                     'context_in': XC[i - 1],
                     'asset_id': np.array([asset_id_val], dtype=np.int32)},
                    Y[i], w)

    ds = (
        tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                {'asset_in':   tf.TensorSpec((steps, len(ASSET_COLS)), tf.float32),
                 'context_in': tf.TensorSpec((len(CONTEXT_COLS),),     tf.float32),
                 'asset_id':   tf.TensorSpec((1,),                     tf.int32)},
                tf.TensorSpec((), tf.float32),
                tf.TensorSpec((), tf.float32)))
        .repeat()
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE))
    return ds, len(combined) // batch_size


# ==============================================================================
# EXECUTION
# ==============================================================================

FORCE_RETRAIN = True

print(f"\n{'='*70}")
if FORCE_RETRAIN:
    print(f"FORCE_RETRAIN=True — deleting old models and scored CSVs on GCS...")
else:
    print(f"PRE-SCAN: Checking existing models and scored CSVs on GCS...")
print(f"{'='*70}")
any_work = False
for fold_cfg in FOLDS_TO_TRAIN:
    fid = fold_cfg['exp_id']
    seeds = [fold_cfg['seed_base'] + 1 + i for i in range(SEEDS_PER_FOLD)]
    gcs_res = f"Algos/Results_all_v2/Universal_Exp{fid}"

    if FORCE_RETRAIN:
        n_deleted = 0
        for s in seeds:
            model_blob = bucket.blob(f"{GCS_MODEL_PATH}/UNIVERSAL_S{s}_M{fid}.h5")
            if model_blob.exists():
                model_blob.delete()
                n_deleted += 1
            local_model = os.path.join(LOCAL_MODEL_DIR, f"UNIVERSAL_S{s}_M{fid}.h5")
            if os.path.exists(local_model):
                os.remove(local_model)
        blobs_to_del = list(bucket.list_blobs(prefix=gcs_res + "/"))
        for b in blobs_to_del:
            b.delete()
        print(f"  {fid}: deleted {n_deleted} models + {len(blobs_to_del)} scored CSVs")
        any_work = True
    else:
        n_trained = 0
        n_scored = 0
        for s in seeds:
            if bucket.blob(f"{GCS_MODEL_PATH}/UNIVERSAL_S{s}_M{fid}.h5").exists():
                n_trained += 1
                missing = [t for t in SCORING_POOL
                           if not bucket.blob(f"{gcs_res}/{t}/Seed_{s}_Results.csv").exists()]
                if not missing:
                    n_scored += 1
        n_need_train = SEEDS_PER_FOLD - n_trained
        n_need_score = n_trained - n_scored
        status = "SKIP" if (n_need_train == 0 and n_need_score == 0) else "WORK"
        print(f"  {fid}: {n_trained}/{SEEDS_PER_FOLD} trained, {n_scored}/{SEEDS_PER_FOLD} fully scored "
              f"→ {n_need_train} to train, {n_need_score} to score  [{status}]")
        if n_need_train > 0 or n_need_score > 0:
            any_work = True

if not any_work:
    print(f"\nAll folds fully trained and scored. Nothing to do.")
    import sys; sys.exit(0)

print(f"\nProceeding with data load and training/scoring...\n")

market_df = build_market_context()


# ==============================================================================
# WALKFORWARD FOLD LOOP
# ==============================================================================
for fold_idx, fold_cfg in enumerate(FOLDS_TO_TRAIN):
    FOLD_EXP_ID = fold_cfg['exp_id']
    FOLD_CUTOFF = fold_cfg['cutoff']
    FOLD_SEEDS  = [fold_cfg['seed_base'] + 1 + i for i in range(SEEDS_PER_FOLD)]

    aware_cutoff = pd.to_datetime(FOLD_CUTOFF).tz_localize('UTC')
    gcs_results_path = f"Algos/Results_all_v2/Universal_Exp{FOLD_EXP_ID}"

    print(f"\n{'='*70}")
    print(f"FOLD {fold_idx+1}/{len(FOLDS_TO_TRAIN)}: Exp {FOLD_EXP_ID}  cutoff={FOLD_CUTOFF}  "
          f"seeds={FOLD_SEEDS[0]}-{FOLD_SEEDS[-1]}")
    print(f"  Entry: min {ENTRY_START_MIN}-{ENTRY_END_MIN} PST  Dump threshold: {DUMP_THRESHOLD:.1%}")
    print(f"{'='*70}")

    if FORCE_RETRAIN:
        needs_training = list(FOLD_SEEDS)
        needs_scoring = []
        remaining_seeds = list(FOLD_SEEDS)
        print(f"   FORCE_RETRAIN: all {len(FOLD_SEEDS)} seeds will retrain and rescore")
    else:
        trained_seeds = []
        fully_scored_seeds = []
        for seed in FOLD_SEEDS:
            model_blob_path = f"{GCS_MODEL_PATH}/UNIVERSAL_S{seed}_M{FOLD_EXP_ID}.h5"
            if bucket.blob(model_blob_path).exists():
                trained_seeds.append(seed)
                missing = [t for t in SCORING_POOL
                           if not bucket.blob(f"{gcs_results_path}/{t}/Seed_{seed}_Results.csv").exists()]
                if not missing:
                    fully_scored_seeds.append(seed)
                else:
                    print(f"   Seed {seed}: model exists, {len(missing)} tickers unscored")

        needs_training = [s for s in FOLD_SEEDS if s not in trained_seeds]
        needs_scoring  = [s for s in trained_seeds if s not in fully_scored_seeds]
        remaining_seeds = needs_training + needs_scoring

        if not remaining_seeds:
            print(f"   All {len(FOLD_SEEDS)} seeds fully trained and scored. Nothing to do.")
            continue

        print(f"   {len(fully_scored_seeds)} seeds fully done, {len(needs_training)} need training, "
              f"{len(needs_scoring)} need scoring only: {remaining_seeds}")

    try:
        stats, diag = _compute_stats_streaming(market_df, aware_cutoff)

        blob = client.bucket(BUCKET_NAME).blob(
            f"{STATS_DIR}/scaling_stats_experiment_{FOLD_EXP_ID}.json")
        blob.upload_from_string(json.dumps(stats, indent=4), content_type='application/json')
        print(f"SAVED STATS TO: {STATS_DIR}/scaling_stats_experiment_{FOLD_EXP_ID}.json")

        _print_diagnostics(diag)
        del diag
        _release_memory()

        if needs_training:
            tr_arrays, val_arrays = _build_ticker_arrays(
                market_df, stats, TRAIN_TEACHERS, aware_cutoff, TIME_STEPS)

        # ==================================================================
        # PHASE 1: TRAINING
        # ==================================================================
        for seed in needs_training:
            print(f"\nTRAINING SEED {seed} (Fold {FOLD_EXP_ID})")
            tf.keras.backend.clear_session()
            random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

            local_model_path = os.path.join(LOCAL_MODEL_DIR, f"UNIVERSAL_S{seed}_M{FOLD_EXP_ID}.h5")
            gcs_model_blob = f"{GCS_MODEL_PATH}/UNIVERSAL_S{seed}_M{FOLD_EXP_ID}.h5"

            print(f"   Building datasets (no augmentation, global_batch={GLOBAL_BATCH_SIZE})...")
            ds_tr, st_tr = _make_ds_from_arrays(
                tr_arrays, TIME_STEPS, GLOBAL_BATCH_SIZE, use_weights=True)
            ds_v, st_v = _make_ds_from_arrays(
                val_arrays, TIME_STEPS, GLOBAL_BATCH_SIZE, use_weights=False)

            if ds_tr is None or st_tr == 0:
                print(f"   Seed {seed}: no valid training samples found. Skipping.")
                del ds_tr, ds_v; _release_memory()
                continue

            print(f"   Steps/epoch: {st_tr}  Val steps: {st_v if st_v else 'N/A'}")

            with strategy.scope():
                model = build_universal_model()

                try:
                    optimizer = tf.keras.optimizers.AdamW(learning_rate=5e-5, weight_decay=5e-4)
                except AttributeError:
                    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

                model.compile(
                    optimizer=optimizer,
                    loss=asymmetric_loss(),
                    metrics=[
                        'accuracy',
                        tf.keras.metrics.AUC(name='auc'),
                        tf.keras.metrics.AUC(curve='PR', name='pr_auc'),
                        PrecisionAtTopK(k=PREC_TOP_K, name='prec_top2pct'),
                    ])

            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    local_model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=4, restore_best_weights=True, mode='min', verbose=1),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, mode='min', verbose=1),
            ]

            model.fit(
                ds_tr, validation_data=ds_v,
                epochs=50, steps_per_epoch=st_tr, validation_steps=st_v,
                callbacks=callbacks, verbose=1)

            if os.path.exists(local_model_path):
                bucket.blob(gcs_model_blob).upload_from_filename(local_model_path)
                print(f"   UPLOADED MODEL TO: {gcs_model_blob}")
            else:
                print("   Model file not found locally, upload skipped.")

            del model, ds_tr, ds_v
            tf.keras.backend.clear_session()
            _release_memory()
            print(f"   Seed {seed} training complete — memory released.")

        if needs_training:
            del tr_arrays, val_arrays
        _release_memory()
        print(f"\n   Training phase done.")

        # ==================================================================
        # PHASE 2: SCORING
        # ==================================================================
        all_seeds_to_score = list(range(fold_cfg['seed_base'] + 1,
                                        fold_cfg['seed_base'] + 1 + SEEDS_PER_FOLD))
        seeds_needing_any_scoring = []
        for seed in all_seeds_to_score:
            local_model_path = os.path.join(LOCAL_MODEL_DIR, f"UNIVERSAL_S{seed}_M{FOLD_EXP_ID}.h5")
            gcs_model_blob = f"{GCS_MODEL_PATH}/UNIVERSAL_S{seed}_M{FOLD_EXP_ID}.h5"
            if not os.path.exists(local_model_path):
                if bucket.blob(gcs_model_blob).exists():
                    bucket.blob(gcs_model_blob).download_to_filename(local_model_path)
                else:
                    continue
            if FORCE_RETRAIN:
                seeds_needing_any_scoring.append(seed)
            else:
                missing = [t for t in SCORING_POOL
                           if not bucket.blob(f"{gcs_results_path}/{t}/Seed_{seed}_Results.csv").exists()]
                if missing:
                    seeds_needing_any_scoring.append(seed)

        if not seeds_needing_any_scoring:
            print(f"\n   All seeds fully scored. Skipping scoring phase.")
        else:
            print(f"\n   SCORING PHASE: {len(seeds_needing_any_scoring)} seeds need scoring")
            print(f"   Loading scoring ticker data (cached per ticker, not per seed)...")

            for score_ticker in SCORING_POOL:
                if FORCE_RETRAIN:
                    seeds_for_this_ticker = list(seeds_needing_any_scoring)
                else:
                    seeds_for_this_ticker = []
                    for seed in seeds_needing_any_scoring:
                        gcs_csv_check = f"{gcs_results_path}/{score_ticker}/Seed_{seed}_Results.csv"
                        if not bucket.blob(gcs_csv_check).exists():
                            seeds_for_this_ticker.append(seed)

                if not seeds_for_this_ticker:
                    continue

                print(f"\n   Scoring {score_ticker} ({len(seeds_for_this_ticker)} seeds)...")
                df_target = engineer_asset(score_ticker, market_df)
                if df_target is None:
                    continue
                df_target_scaled = apply_scaling(df_target, stats)

                test_mask = df_target_scaled.index > aware_cutoff
                df_test   = df_target_scaled[test_mask]

                idx_v = np.where(df_test['Valid_Row'].values)[0]
                idx_v = idx_v[idx_v >= TIME_STEPS]
                if len(idx_v) == 0:
                    del df_target, df_target_scaled, df_test
                    _release_memory()
                    continue

                XA_v = df_test[ASSET_COLS].values
                XC_v = df_test[CONTEXT_COLS].values
                asset_id_val = np.int32(ASSET_INDEX.get(score_ticker, OTHER_IDX))

                for seed in seeds_for_this_ticker:
                    local_model_path = os.path.join(LOCAL_MODEL_DIR, f"UNIVERSAL_S{seed}_M{FOLD_EXP_ID}.h5")
                    if not os.path.exists(local_model_path):
                        continue

                    tf.keras.backend.clear_session()
                    with strategy.scope():
                        model = build_universal_model()
                        model.compile(optimizer='adam', loss='binary_crossentropy')
                    model.load_weights(local_model_path)

                    probs = []
                    for i in range(0, len(idx_v), 4096):
                        batch_idx = idx_v[i:i+4096]
                        XA_b  = np.array([XA_v[j - TIME_STEPS:j] for j in batch_idx])
                        XC_b  = np.array([XC_v[j - 1] for j in batch_idx])
                        XID_b = np.full((len(batch_idx), 1), asset_id_val, dtype=np.int32)
                        p = model.predict([XA_b, XC_b, XID_b], verbose=0).flatten()
                        probs.extend(p)

                    results_df = pd.DataFrame({
                        'Target':     df_test['Target'].values[idx_v],
                        'Prob_Spike': probs,
                        'Prediction': (np.array(probs) > 0.5).astype(int),
                        'Close':      df_test['Close'].values[idx_v],
                        'Ticker':     score_ticker
                    }, index=df_test.index[idx_v])

                    local_csv_name = f"Seed_{seed}_{score_ticker}_Results.csv"
                    local_csv_path = os.path.join(LOCAL_RESULTS_DIR, local_csv_name)
                    results_df.to_csv(local_csv_path)

                    gcs_csv_path = f"{gcs_results_path}/{score_ticker}/Seed_{seed}_Results.csv"
                    bucket.blob(gcs_csv_path).upload_from_filename(local_csv_path)
                    print(f"      S{seed} → {gcs_csv_path}")

                    del model
                    _release_memory()

                del df_target, df_target_scaled, df_test, XA_v, XC_v
                _release_memory()

        _release_memory()
        tf.keras.backend.clear_session()
        print(f"\n  Fold {FOLD_EXP_ID} complete.")

    except Exception as e:
        print(f"\n  *** FOLD {FOLD_EXP_ID} FAILED: {e}")
        import traceback; traceback.print_exc()
        tf.keras.backend.clear_session()
        _release_memory()
        print(f"  Continuing to next fold...")

print(f"\n{'='*70}")
print(f"ALL FOLDS COMPLETE — {[f['exp_id'] for f in FOLDS_TO_TRAIN]}")
print(f"{'='*70}")
print(f"\nNext steps:")
print(f"  1. Create optimize_wf_v12_425_10am.py:")
print(f"     OPT: 2026-01-01 → 2026-03-31  |  VAL: 2026-04-01 → 2026-05-31  |  TEST: 2026-06-01 → 2026-07-31")
print(f"     ENTRY_MIN=420 (trade at 10AM EST), TP grid 1.5-3.5%")
print(f"  2. Create daily_score_short_10am.py for live scoring at 6:45-6:50 AM PST")
print(f"  3. Run backtest_short_10am.py to compare vs original 9:30 entry model")
