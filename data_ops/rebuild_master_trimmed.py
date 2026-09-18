"""
rebuild_master_trimmed.py — Rebuild master_inference with OHLCV trimmed to 2022+
=================================================================================
Fixes the micro coverage dilution for tickers whose OHLCV goes back to 2003+
but raw T&Q (and therefore micro) only starts from 2022.

The problem: heal_data().resample('1min').asfreq() creates a 1-min bar for every
minute from OHLCV start to end. For BE/MSTR/FCEL the OHLCV starts 2003, producing
~12M bars — but micro only covers 2022+ (~1.2M bars). The left join fills 2003-2021
with zeros, diluting Volume_Delta nonzero% below the 5% scoring threshold.

The fix: clip OHLCV to DATA_START before the join. No data loss — models train
from 2022+ anyway.

Does NOT re-process raw T&Q or rebuild micro — those are already healthy.
Only rebuilds the master_inference join.

Usage (Colab cell):
  RUN_TICKERS = ['BE', 'FCEL', 'MSTR']
  main(RUN_TICKERS)
"""

import sys
import io
import gc
import warnings

import pandas as pd
import pytz
from datetime import datetime
from google.cloud import storage

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BUCKET_NAME = os.environ.get("GCS_BUCKET", "")
GCS_MICRO_DIR  = "Data/Inference/Microstructure"
GCS_OHLCV_DIR  = "Data/Inference"
GCS_MASTER_DIR = "Data/Inference_v2"

DATA_START = pd.Timestamp("2022-01-01", tz='UTC')

TICKER_DATA_START = {
    'FCEL': pd.Timestamp("2025-08-01", tz='UTC'),
    'MU':   pd.Timestamp("2026-01-01", tz='UTC'),
}

client = storage.Client()
bucket = client.bucket(BUCKET_NAME)


# ==============================================================================
# HEAL (from rebuild_micro_from_raw.py)
# ==============================================================================
def heal_data(df):
    df = df.resample('1min').asfreq()
    for c in ['Open', 'High', 'Low', 'Close', 'Spread', 'OBI', 'bsize', 'asize']:
        if c in df.columns:
            df[c] = df[c].ffill()
    for c in ['Volume', 'Total_Vol', 'Volume_Delta', 'Trade_Count', 'Delta_Pct']:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    return df


# ==============================================================================
# MAIN
# ==============================================================================
def main(ticker_list=None):
    print("=" * 70)
    print("  REBUILD MASTER_INFERENCE (OHLCV trimmed to 2022+)")
    print(f"  Run time: {datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  DATA_START: {DATA_START.strftime('%Y-%m-%d')}")
    print("=" * 70)

    if ticker_list:
        tickers = [t.upper() for t in ticker_list]
    else:
        cli_tickers = [a.upper() for a in sys.argv[1:]
                       if not a.startswith('-') and not a.endswith('.json')]
        tickers = cli_tickers if cli_tickers else []

    if not tickers:
        print("  ERROR: No tickers specified. Pass a list to main() or via CLI.")
        return

    print(f"  Tickers: {tickers}\n")

    summary = []

    for ticker in tickers:
        print(f"{'─' * 70}")
        print(f"  {ticker}")
        print(f"{'─' * 70}")

        ohlcv_path = f"{GCS_OHLCV_DIR}/{ticker}_1m_full_historical_data.parquet"
        micro_path = f"{GCS_MICRO_DIR}/{ticker}_1m_microstructure.parquet"
        master_path = f"{GCS_MASTER_DIR}/{ticker}_master_inference.parquet"

        # ── Load OHLCV ──
        ohlcv_blob = bucket.blob(ohlcv_path)
        if not ohlcv_blob.exists():
            print(f"  No OHLCV at {ohlcv_path} — skipping")
            summary.append((ticker, 'NO_OHLCV', '-', '-', '-', '-'))
            continue

        df_ohlcv = pd.read_parquet(io.BytesIO(ohlcv_blob.download_as_bytes()))
        if df_ohlcv.index.tz is None:
            df_ohlcv.index = df_ohlcv.index.tz_localize('UTC')

        before_trim = len(df_ohlcv)
        ohlcv_start = df_ohlcv.index.min()
        print(f"  OHLCV: {before_trim:,} rows ({ohlcv_start.strftime('%Y-%m-%d')} → {df_ohlcv.index.max().strftime('%Y-%m-%d')})")

        # ── Trim to DATA_START (per-ticker override if needed) ──
        trim_start = TICKER_DATA_START.get(ticker, DATA_START)
        df_ohlcv = df_ohlcv[df_ohlcv.index >= trim_start]
        trimmed = before_trim - len(df_ohlcv)
        print(f"  Trimmed: {trimmed:,} rows removed (pre-{trim_start.strftime('%Y-%m-%d')})")
        print(f"  Kept: {len(df_ohlcv):,} rows ({df_ohlcv.index.min().strftime('%Y-%m-%d')} → {df_ohlcv.index.max().strftime('%Y-%m-%d')})")

        # ── Load micro ──
        micro_blob = bucket.blob(micro_path)
        if not micro_blob.exists():
            print(f"  No micro at {micro_path} — master will have zero micro columns")
            summary.append((ticker, 'NO_MICRO', f'{len(df_ohlcv):,}', '-', '-', '-'))
            del df_ohlcv
            gc.collect()
            continue

        df_micro = pd.read_parquet(io.BytesIO(micro_blob.download_as_bytes()))
        if df_micro.index.tz is None:
            df_micro.index = df_micro.index.tz_localize('UTC')
        print(f"  Micro: {len(df_micro):,} rows ({df_micro.index.min().strftime('%Y-%m-%d')} → {df_micro.index.max().strftime('%Y-%m-%d')})")

        # ── Join ──
        df_master = df_ohlcv.join(df_micro, how='left')
        df_master = df_master[~df_master.index.duplicated(keep='last')]
        del df_ohlcv, df_micro
        gc.collect()

        # ── Heal ──
        print(f"  Healing {len(df_master):,} rows...")
        df_master = heal_data(df_master)
        df_master = df_master[df_master.index.notna()].sort_index()

        # ── Stats ──
        vd_nz = (df_master['Volume_Delta'] != 0).sum() if 'Volume_Delta' in df_master.columns else 0
        tc_nz = (df_master['Trade_Count'] > 0).sum() if 'Trade_Count' in df_master.columns else 0
        vd_pct = vd_nz / len(df_master) * 100
        tc_pct = tc_nz / len(df_master) * 100

        threshold_status = "PASS" if vd_pct >= 5.0 else "FAIL"
        print(f"  After heal: {len(df_master):,} rows")
        print(f"  Volume_Delta nonzero: {vd_nz:,} / {len(df_master):,} = {vd_pct:.1f}% — {threshold_status}")
        print(f"  Trade_Count > 0:      {tc_nz:,} / {len(df_master):,} = {tc_pct:.1f}%")

        # ── Save ──
        print(f"  Saving to {master_path}...")
        buf = io.BytesIO()
        df_master.to_parquet(buf, engine='pyarrow')
        buf.seek(0)
        bucket.blob(master_path).upload_from_file(buf)

        size_mb = buf.tell() / (1024 * 1024)
        print(f"  Saved: {len(df_master):,} rows, {size_mb:.1f} MB")

        summary.append((ticker, threshold_status, f'{len(df_master):,}',
                         f'{trimmed:,}', f'{vd_pct:.1f}%', f'{tc_pct:.1f}%'))

        del df_master
        gc.collect()
        print()

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"  REBUILD SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n  {'Ticker':<8} {'5%Gate':<8} {'MasterRows':>12} {'Trimmed':>10} {'VD_nz%':>8} {'TC_nz%':>8}")
    print(f"  {'─'*8} {'─'*8} {'─'*12} {'─'*10} {'─'*8} {'─'*8}")
    for ticker, status, rows, trimmed, vd, tc in summary:
        print(f"  {ticker:<8} {status:<8} {rows:>12} {trimmed:>10} {vd:>8} {tc:>8}")

    print(f"\n  Next steps:")
    print(f"  1. Re-run exp_152_score_only.py and exp_160_score_only.py")
    print(f"     — these tickers will now pass the 5% quality gate")
    print(f"  2. After scoring, re-run optimize_wf_v12.py to get limit_entry configs")
    print(f"\n{'=' * 70}")


# ==============================================================================
# RUN CONFIG — edit before running in Colab
# ==============================================================================
RUN_TICKERS = ['FCEL', 'FCX', 'MOD', 'MU']

main(RUN_TICKERS)
