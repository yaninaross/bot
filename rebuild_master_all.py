"""
rebuild_master_all.py — Full Rebuild of All master_inference Parquets
=====================================================================
Reads existing raw OHLCV + microstructure parquets from GCS and rebuilds
every master_inference.parquet with heal_data applied to the ENTIRE file.

This is the one-time rebuild script. Use after:
  - Bulk OHLCV re-download
  - Bulk T&Q re-download + micro rebuild
  - BITF->KEEL migration
  - Any time you need clean, fully-healed master files

Does NOT fetch from Polygon — uses whatever raw data is already in GCS.

Usage:
  python rebuild_master_all.py              # rebuild all tickers
  python rebuild_master_all.py MARA KEEL    # rebuild specific tickers only
"""

import sys
import pandas as pd
from datetime import datetime
import pytz
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
TARGET_TICKERS = [
    # --- Exp-97 (Crypto Cluster) ---
    'MARA', 'IREN', 'WULF', 'APLD', 'RIOT', 'CLSK', 'MSTR', 'COIN',
    'KEEL', 'HUT', 'CIFR', 'BTBT', 'MIGI', 'HOOD', 'CORZ',
    # 'SMLR' — REMOVED: delisted ~Jan 2026

    # --- Additional Inference Targets ---
    'SMCI', 'TSLA', 'VRT',

    # --- Exp-100 (Defense Sector) ---
    'AVAV', 'KTOS', 'PLTR', 'BBAI', 'RKLB', 'ASTS', 'LUNR', 'SPIR',
    'JOBY', 'NNDM', 'MVIS', 'RCAT', 'ACHR',

    # --- Exp-150 (Energy / AI Power) ---
    'VST', 'CEG', 'GEV', 'BE', 'CCJ', 'NEE', 'DUK',

    # --- Exp-190 (Space / Dual-Use) ---
    'PL', 'RDW', 'BKSY', 'MNTS', 'VOYG',

    # --- Defense Primes ---
    'GD', 'LHX', 'LMT', 'NOC', 'RTX', 'GE',
]

CONTEXT_ASSETS = ['ITA', 'SMH', 'MOD', 'I:VIX', 'TLT', 'IWM', 'X:BTCUSD', 'SPY']

BUCKET_NAME   = "time-series-bucket-ys"
GCS_OHLCV_DIR = f"gs://{BUCKET_NAME}/Data/Inference"
GCS_MICRO_DIR = f"gs://{BUCKET_NAME}/Data/Inference/Microstructure"

# ==============================================================================
# HEAL — resample to 1-min continuous + ffill/zero-fill
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
if __name__ == "__main__":
    print("rebuild_master_all.py — Full Master Inference Rebuild")
    print(f"   Run time: {datetime.now(pytz.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    # Build asset list: (ticker, is_context)
    ALL_ASSETS = (
        [(t, False) for t in list(set(TARGET_TICKERS))] +
        [(t, True)  for t in list(set(CONTEXT_ASSETS))]
    )

    # Optional: filter to specific tickers from CLI args
    # Filter out Jupyter kernel args (e.g. -f kernel.json)
    cli_tickers = [a.upper() for a in sys.argv[1:] if not a.startswith('-') and not a.endswith('.json')]
    if cli_tickers:
        ALL_ASSETS = [(t, ctx) for t, ctx in ALL_ASSETS
                      if t.upper() in cli_tickers or t.replace("I:","").replace("X:","").upper() in cli_tickers]
        print(f"   Filtered to: {[t for t,_ in ALL_ASSETS]}")
    else:
        print(f"   Rebuilding ALL {len(ALL_ASSETS)} assets")

    success, failed, skipped = 0, 0, 0

    for ticker, is_context in ALL_ASSETS:
        print(f"\n{'─'*52}")
        print(f"{'[context]' if is_context else '[target]'} {ticker}")
        clean_ticker = ticker.replace("I:", "").replace("X:", "")

        ohlcv_uri  = f"{GCS_OHLCV_DIR}/{clean_ticker}_1m_full_historical_data.parquet"
        micro_uri  = f"{GCS_MICRO_DIR}/{clean_ticker}_1m_microstructure.parquet"
        master_uri = f"{GCS_OHLCV_DIR}/{clean_ticker}_master_inference.parquet"

        # ── Load raw OHLCV ──────────────────────────────────────────────────
        try:
            df_ohlcv = pd.read_parquet(ohlcv_uri)
            if df_ohlcv.index.tz is None:
                df_ohlcv.index = df_ohlcv.index.tz_localize('UTC')
            print(f"   OHLCV: {len(df_ohlcv):,} rows  "
                  f"({df_ohlcv.index.min().date()} to {df_ohlcv.index.max().date()})")
        except Exception as e:
            print(f"   OHLCV not found — skipping ({e})")
            skipped += 1
            continue

        # ── Load micro (target assets only) ─────────────────────────────────
        df_micro = pd.DataFrame()
        if not is_context:
            try:
                df_micro = pd.read_parquet(micro_uri)
                if df_micro.index.tz is None:
                    df_micro.index = df_micro.index.tz_localize('UTC')
                print(f"   Micro: {len(df_micro):,} rows  "
                      f"({df_micro.index.min().date()} to {df_micro.index.max().date()})")
            except Exception:
                print(f"   Micro not found — master will have NaN micro columns")

        # ── Join ────────────────────────────────────────────────────────────
        if not is_context and not df_micro.empty:
            df_unified = df_ohlcv.join(df_micro, how='left')
        else:
            df_unified = df_ohlcv.copy()

        raw_rows = len(df_unified)

        # ── Heal ENTIRE dataset ─────────────────────────────────────────────
        print(f"   Healing entire dataset ({raw_rows:,} raw rows)...")
        try:
            df_unified = heal_data(df_unified)
            df_unified = df_unified[df_unified.index.notna()].sort_index()
            df_unified.to_parquet(master_uri, engine='pyarrow')
            print(f"   master_inference saved: {len(df_unified):,} rows "
                  f"(expanded from {raw_rows:,})")
            success += 1
        except Exception as e:
            print(f"   FAILED: {e}")
            failed += 1

    print(f"\n{'='*52}")
    print(f"DONE — {success} rebuilt, {skipped} skipped (no OHLCV), {failed} failed")
    print("="*52)
