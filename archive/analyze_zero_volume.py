"""
Analyze Volume=0 frequency at entry bars across all 24 scored tickers.

Loads master_inference parquet from GCS, converts to PST, checks each ticker's
configured entry_minute over the last 60 trading days.

Usage:  python3 analyze_zero_volume.py
"""

import sys
import numpy as np
import pandas as pd
import pytz
from datetime import datetime, timedelta

# ============================================================================
# CONFIG: all 24 scored tickers with their entry_minute and GCS file name
# ============================================================================
BUCKET_NAME = "time-series-bucket-ys"
INFERENCE_DIR = "Data/Inference"
PST = pytz.timezone("America/Los_Angeles")
LOOKBACK_TRADING_DAYS = 60
# Pre-market window: 4:00 AM to 9:30 AM ET = 1:00 AM to 6:30 AM PT
# All entry_minutes are in the 360-390 range (6:00-6:30 AM PST)
# We consider a "trading day" as any day with at least 1 bar between 6:00-6:30 AM PST

# Ticker -> (entry_minute, gcs_file_ticker)
# entry_minute = minutes from midnight PST
# gcs_file_ticker = the filename on GCS (handles BGDE->BGDE, KEEL->KEEL remapping)
TICKER_CONFIG = {
    # EXP-97 CRYPTO
    "BGDE":  (380, "BGDE"),
    "WULF":  (382, "WULF"),
    "HOOD":  (368, "HOOD"),
    # EXP-103 DEFENSE
    "KTOS":  (385, "KTOS"),
    "RKLB":  (364, "RKLB"),
    "PLTR":  (365, "PLTR"),
    # EXP-104 DEFENSE (8% spike)
    "JOBY":  (377, "JOBY"),
    "LUNR":  (380, "LUNR"),
    # EXP-105 DEFENSE (cluster)
    "APLD":  (381, "APLD"),
    "RIOT":  (382, "RIOT"),
    "KEEL":  (386, "KEEL"),
    # EXP-106 DEFENSE (8% cluster)
    "RCAT":  (364, "RCAT"),
    "BBAI":  (385, "BBAI"),
    "ACHR":  (360, "ACHR"),
    "ASTS":  (365, "ASTS"),
    # EXP-112 CRYPTO (8% spike)
    "VRT":   (367, "VRT"),
    "HUT":   (381, "HUT"),
    "BTBT":  (362, "BTBT"),
    "CIFR":  (383, "CIFR"),
    "COIN":  (373, "COIN"),
    # EXP-113 CRYPTO (6% spike)
    "IREN":  (364, "IREN"),
    "CLSK":  (372, "CLSK"),
    "MARA":  (360, "MARA"),
    # EXP-150 ENERGY
    "BE":    (379, "BE"),
}


def load_parquet_from_gcs(file_ticker):
    """Load master_inference parquet from GCS via pyarrow GCS filesystem."""
    uri = f"gs://{BUCKET_NAME}/{INFERENCE_DIR}/{file_ticker}_master_inference.parquet"
    df = pd.read_parquet(uri)

    # Normalize index to UTC datetime
    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if dt_cols:
        df["Datetime"] = pd.to_datetime(df[dt_cols[0]], utc=True)
    else:
        fallback = next(
            (c for c in df.columns if c in ("Datetime", "Date", "Time", "Timestamp")),
            df.columns[0],
        )
        df["Datetime"] = pd.to_datetime(df[fallback], unit="ms", utc=True)

    df = df.set_index("Datetime").sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # Normalize column names
    col_map = {
        "volume": "Volume", "Volume": "Volume",
        "close": "Close", "Close": "Close",
        "open": "Open", "high": "High", "low": "Low",
    }
    lower_map = {k.lower(): v for k, v in col_map.items()}
    df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip()) for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    return df


def minute_of_day_pst(ts_index):
    """Convert UTC index to PST and return minute-of-day."""
    pst_index = ts_index.tz_convert(PST)
    return pst_index.hour * 60 + pst_index.minute


def analyze_ticker(ticker, entry_minute, file_ticker):
    """Analyze V=0 frequency at entry bar for one ticker."""
    try:
        df = load_parquet_from_gcs(file_ticker)
    except Exception as e:
        return {
            "ticker": ticker,
            "entry_minute": entry_minute,
            "error": str(e),
        }

    if "Volume" not in df.columns:
        return {
            "ticker": ticker,
            "entry_minute": entry_minute,
            "error": "No Volume column",
        }

    # Convert to PST for filtering
    df_pst_idx = df.index.tz_convert(PST)
    df["minute_of_day"] = df_pst_idx.hour * 60 + df_pst_idx.minute
    df["date_pst"] = df_pst_idx.date

    # Filter to only the entry_minute bars
    entry_bars = df[df["minute_of_day"] == entry_minute].copy()

    if entry_bars.empty:
        return {
            "ticker": ticker,
            "entry_minute": entry_minute,
            "error": f"No bars found at minute {entry_minute}",
        }

    # Get last N trading days (unique dates with entry bars)
    unique_dates = sorted(entry_bars["date_pst"].unique())
    if len(unique_dates) > LOOKBACK_TRADING_DAYS:
        unique_dates = unique_dates[-LOOKBACK_TRADING_DAYS:]
    entry_bars = entry_bars[entry_bars["date_pst"].isin(unique_dates)]

    days_checked = len(unique_dates)
    v0_days = int((entry_bars["Volume"] == 0).sum())
    v0_pct = (v0_days / days_checked * 100) if days_checked > 0 else 0.0
    avg_vol = entry_bars["Volume"].mean()
    med_vol = entry_bars["Volume"].median()

    # Find most recent V=0 day for surrounding-bar analysis
    v0_entries = entry_bars[entry_bars["Volume"] == 0]
    surrounding = None
    if not v0_entries.empty:
        last_v0_date = max(v0_entries["date_pst"])
        # Get all bars on that date within entry_minute +/- 5
        day_bars = df[df["date_pst"] == last_v0_date].copy()
        surround = day_bars[
            (day_bars["minute_of_day"] >= entry_minute - 5)
            & (day_bars["minute_of_day"] <= entry_minute + 5)
        ].copy()
        if not surround.empty:
            surrounding = {
                "date": last_v0_date,
                "bars": surround[["minute_of_day", "Volume", "Close"]].copy(),
            }

    return {
        "ticker": ticker,
        "entry_minute": entry_minute,
        "days_checked": days_checked,
        "v0_days": v0_days,
        "v0_pct": v0_pct,
        "avg_vol": avg_vol,
        "med_vol": med_vol,
        "surrounding": surrounding,
        "error": None,
    }


def entry_time_str(entry_minute):
    """Convert entry_minute to HH:MM PST string."""
    h = entry_minute // 60
    m = entry_minute % 60
    return f"{h}:{m:02d}"


def main():
    print(f"Volume=0 at Entry Bar Analysis")
    print(f"Date: {datetime.now(PST).strftime('%Y-%m-%d %H:%M PST')}")
    print(f"Lookback: {LOOKBACK_TRADING_DAYS} trading days")
    print(f"Tickers: {len(TICKER_CONFIG)}")
    print()

    results = []
    for ticker, (entry_minute, file_ticker) in sorted(TICKER_CONFIG.items()):
        sys.stdout.write(f"  Loading {ticker:<6} (entry {entry_time_str(entry_minute)}) ... ")
        sys.stdout.flush()
        r = analyze_ticker(ticker, entry_minute, file_ticker)
        if r.get("error"):
            print(f"ERROR: {r['error']}")
        else:
            print(f"done  V0={r['v0_days']}/{r['days_checked']} ({r['v0_pct']:.1f}%)")
        results.append(r)

    # ---- Summary table ----
    print()
    print("=" * 105)
    header = (
        f"{'Ticker':<8} {'Entry_Min':>9} {'Entry_Time':>10} "
        f"{'Days_Chk':>8} {'V0_Days':>7} {'V0_Pct':>7} "
        f"{'Avg_Vol':>12} {'Median_Vol':>12}"
    )
    print(header)
    print("-" * 105)

    ok_results = [r for r in results if r.get("error") is None]
    err_results = [r for r in results if r.get("error") is not None]

    # Sort by V0_Pct descending
    ok_results.sort(key=lambda r: r["v0_pct"], reverse=True)

    for r in ok_results:
        print(
            f"{r['ticker']:<8} {r['entry_minute']:>9} "
            f"{entry_time_str(r['entry_minute']):>10} "
            f"{r['days_checked']:>8} {r['v0_days']:>7} "
            f"{r['v0_pct']:>6.1f}% "
            f"{r['avg_vol']:>12,.0f} {r['med_vol']:>12,.0f}"
        )

    if err_results:
        print()
        print("ERRORS:")
        for r in err_results:
            print(f"  {r['ticker']:<8}  {r['error']}")

    print("=" * 105)

    # ---- Summary stats ----
    if ok_results:
        total_v0 = sum(r["v0_days"] for r in ok_results)
        total_days = sum(r["days_checked"] for r in ok_results)
        tickers_with_v0 = sum(1 for r in ok_results if r["v0_days"] > 0)
        high_v0 = [r for r in ok_results if r["v0_pct"] > 10]

        print()
        print(f"Overall: {total_v0} V=0 bars out of {total_days} ticker-days "
              f"({total_v0/total_days*100:.1f}%)" if total_days > 0 else "")
        print(f"Tickers with any V=0: {tickers_with_v0}/{len(ok_results)}")
        print(f"Tickers with >10% V=0 rate: {len(high_v0)}")

    # ---- Surrounding-bars analysis for high-V0 tickers ----
    high_v0_results = [r for r in ok_results if r["v0_pct"] > 10 and r.get("surrounding")]
    if high_v0_results:
        print()
        print("=" * 105)
        print("SURROUNDING BARS ANALYSIS (tickers with >10% V=0 rate)")
        print("Shows volume pattern +/- 5 minutes around entry on most recent V=0 day")
        print("=" * 105)

        for r in high_v0_results:
            surr = r["surrounding"]
            bars_df = surr["bars"]
            entry_min = r["entry_minute"]
            print()
            print(f"  {r['ticker']}  entry_minute={entry_min} "
                  f"({entry_time_str(entry_min)} PST)  "
                  f"V0 date: {surr['date']}")
            print(f"  {'Minute':>8} {'Time':>8} {'Volume':>12} {'Close':>10} {'Note'}")
            print(f"  {'-'*50}")

            for _, row in bars_df.iterrows():
                minute = int(row["minute_of_day"])
                time_str = entry_time_str(minute)
                vol = row["Volume"]
                close = row["Close"]
                marker = " <-- ENTRY" if minute == entry_min else ""
                v0_flag = " [V=0]" if vol == 0 else ""
                print(
                    f"  {minute:>8} {time_str:>8} {vol:>12,.0f} "
                    f"{close:>10.4f}{marker}{v0_flag}"
                )

            # Also check: how many of the surrounding bars also had V=0?
            surr_v0 = int((bars_df["Volume"] == 0).sum())
            surr_total = len(bars_df)
            print(f"  -> {surr_v0}/{surr_total} bars in window have V=0")

    # Also show surrounding bars for any ticker that had V=0 today (even if <10% rate)
    # Filter: tickers where the most recent V0 date is "recent" (last 5 trading days)
    recent_cutoff = datetime.now(PST).date() - timedelta(days=7)
    recent_v0 = [
        r for r in ok_results
        if r.get("surrounding")
        and r["surrounding"]["date"] >= recent_cutoff
        and r["v0_pct"] <= 10
    ]
    if recent_v0:
        print()
        print("=" * 105)
        print("RECENT V=0 EVENTS (last 7 calendar days, tickers with <=10% V=0 rate)")
        print("=" * 105)
        for r in recent_v0:
            surr = r["surrounding"]
            bars_df = surr["bars"]
            entry_min = r["entry_minute"]
            print()
            print(f"  {r['ticker']}  entry_minute={entry_min} "
                  f"({entry_time_str(entry_min)} PST)  "
                  f"V0 date: {surr['date']}  "
                  f"Overall V0 rate: {r['v0_pct']:.1f}%")
            print(f"  {'Minute':>8} {'Time':>8} {'Volume':>12} {'Close':>10} {'Note'}")
            print(f"  {'-'*50}")
            for _, row in bars_df.iterrows():
                minute = int(row["minute_of_day"])
                time_str = entry_time_str(minute)
                vol = row["Volume"]
                close = row["Close"]
                marker = " <-- ENTRY" if minute == entry_min else ""
                v0_flag = " [V=0]" if vol == 0 else ""
                print(
                    f"  {minute:>8} {time_str:>8} {vol:>12,.0f} "
                    f"{close:>10.4f}{marker}{v0_flag}"
                )
            surr_v0 = int((bars_df["Volume"] == 0).sum())
            surr_total = len(bars_df)
            print(f"  -> {surr_v0}/{surr_total} bars in window have V=0")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
