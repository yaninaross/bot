# %% ── KEEL SHORT Drag Investigation ───────────────────────────────────────────
#
# Investigates why KEEL (formerly BITF) is the biggest drag on SHORT performance.
# 12/52 trades (23%), avg PnL -0.49%, 33% WR.
#
# Questions:
# A) Is KEEL fundamentally a bad SHORT ticker?
# B) How much did KEEL cost the portfolio?
# C) Is the consensus mechanism over-selecting KEEL?

import os, sys, warnings
import numpy as np
import pandas as pd
import pytz
from datetime import datetime

warnings.filterwarnings('ignore')

from google.cloud import storage as gcs_storage

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
BUCKET_NAME = "time-series-bucket-ys"
INFERENCE_DIR = "Data/Inference_v2"
PST = pytz.timezone('America/Los_Angeles')

BT_START = '2026-08-03'
BT_END = '2026-08-27'

# Entry at 6:25 PT = minute 385
ENTRY_MINUTE = 385
# Market close at 13:00 PT = minute 780
EXIT_MINUTE = 780
# Market open at 6:30 PT = minute 390
OPEN_MINUTE = 390

# Tickers to compare
KEEL_TICKER = 'KEEL'
GOOD_SHORT_TICKERS = ['HUT', 'CIFR', 'MXL', 'CLSK']
ALL_TICKERS = [KEEL_TICKER] + GOOD_SHORT_TICKERS

# Period definitions
PERIOD_1 = ('2026-08-03', '2026-08-13')  # Profitable period
PERIOD_2 = ('2026-08-14', '2026-08-21')  # Transition
PERIOD_3 = ('2026-08-24', '2026-08-27')  # Losing period

# Known KEEL trade results from the backtest
KEEL_TRADES = {
    'n_trades': 12,
    'avg_pnl': -0.0049,  # -0.49%
    'win_rate': 0.33,
    'tp': 3,
    'sl': 8,
    'eod': 1,
}
TOTAL_SHORT_TRADES = 52
TOTAL_SHORT_AVG_PNL = 0.0236  # +2.36%

gcs_client = gcs_storage.Client()


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_master_inference(ticker):
    """Load master_inference parquet from GCS using google.cloud.storage API."""
    blob_path = f"{INFERENCE_DIR}/{ticker}_master_inference.parquet"
    local_path = f"/tmp/_keel_inv_{ticker}_master.parquet"
    try:
        gcs_client.bucket(BUCKET_NAME).blob(blob_path).download_to_filename(local_path)
        df = pd.read_parquet(local_path)
        os.remove(local_path)
        df = df.reset_index()
        # Find datetime column
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
        # Standardize columns
        std_map = {
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume',
        }
        lower_map = {k.lower(): v for k, v in std_map.items()}
        df.columns = [lower_map.get(str(c).strip().lower(), str(c).strip()) for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated(keep='first')]
        return df
    except Exception as e:
        print(f"  ERROR loading {ticker}: {e}")
        if os.path.exists(local_path):
            os.remove(local_path)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# INTRADAY ANALYSIS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def get_trading_days(df, start_date, end_date):
    """Get list of trading days in date range."""
    bt_start = pd.Timestamp(start_date, tz='UTC')
    bt_end = pd.Timestamp(end_date, tz='UTC') + pd.Timedelta(days=1)
    df_range = df[(df.index >= bt_start) & (df.index < bt_end)]
    df_pst = df_range.index.tz_convert(PST)
    dates = sorted(set(df_pst.date))
    return dates


def analyze_daily_behavior(df, start_date, end_date):
    """Analyze daily price behavior for a ticker."""
    bt_start = pd.Timestamp(start_date, tz='UTC')
    bt_end = pd.Timestamp(end_date, tz='UTC') + pd.Timedelta(days=1)
    df_range = df[(df.index >= bt_start) & (df.index < bt_end)].copy()

    if df_range.empty:
        return pd.DataFrame()

    df_pst = df_range.copy()
    df_pst.index = df_pst.index.tz_convert(PST)
    df_pst['minute_of_day'] = df_pst.index.hour * 60 + df_pst.index.minute
    df_pst['date'] = df_pst.index.date

    daily_stats = []
    prev_close = None

    for date in sorted(df_pst['date'].unique()):
        day = df_pst[df_pst['date'] == date]

        # Day open/close
        day_open = day['Open'].iloc[0] if 'Open' in day.columns else day['Close'].iloc[0]
        day_close = day['Close'].iloc[-1]
        daily_return = (day_close - day_open) / day_open

        # Gap from prior close
        gap = 0.0
        if prev_close is not None and prev_close > 0:
            gap = (day_open - prev_close) / prev_close
        prev_close = day_close

        # Entry at 6:25 PT (minute 385)
        entry_bars = day[(day['minute_of_day'] >= ENTRY_MINUTE - 1) & (day['minute_of_day'] <= ENTRY_MINUTE + 1)]
        if entry_bars.empty:
            entry_bars = day[(day['minute_of_day'] >= OPEN_MINUTE - 5) & (day['minute_of_day'] <= OPEN_MINUTE + 5)]
        if entry_bars.empty:
            continue

        entry_price = entry_bars['Close'].iloc[0]

        # Post-entry window: 6:25 - 13:00 PT
        post_entry = day[(day['minute_of_day'] >= ENTRY_MINUTE) & (day['minute_of_day'] <= EXIT_MINUTE)]
        if post_entry.empty:
            continue

        # Max intraday drop (good for SHORT): how far price dropped
        min_low = post_entry['Low'].min() if 'Low' in post_entry.columns else post_entry['Close'].min()
        max_drop = (entry_price - min_low) / entry_price  # positive = price dropped = good for SHORT

        # Max intraday rally (bad for SHORT): how far price rallied
        max_high = post_entry['High'].max() if 'High' in post_entry.columns else post_entry['Close'].max()
        max_rally = (max_high - entry_price) / entry_price  # positive = price rallied = bad for SHORT

        # Intraday trend: close vs entry
        eod_close = post_entry['Close'].iloc[-1]
        intraday_trend = (eod_close - entry_price) / entry_price  # negative = downtrend = good for SHORT

        # Compute 30-min and 60-min returns from entry
        post_30m = day[(day['minute_of_day'] >= ENTRY_MINUTE + 28) & (day['minute_of_day'] <= ENTRY_MINUTE + 32)]
        ret_30m = 0.0
        if not post_30m.empty:
            ret_30m = (post_30m['Close'].iloc[0] - entry_price) / entry_price

        post_60m = day[(day['minute_of_day'] >= ENTRY_MINUTE + 58) & (day['minute_of_day'] <= ENTRY_MINUTE + 62)]
        ret_60m = 0.0
        if not post_60m.empty:
            ret_60m = (post_60m['Close'].iloc[0] - entry_price) / entry_price

        # Volume analysis
        avg_vol = post_entry['Volume'].mean() if 'Volume' in post_entry.columns else 0

        daily_stats.append({
            'date': date,
            'daily_return': daily_return,
            'gap': gap,
            'entry_price': entry_price,
            'max_drop': max_drop,
            'max_rally': max_rally,
            'intraday_trend': intraday_trend,
            'ret_30m': ret_30m,
            'ret_60m': ret_60m,
            'avg_volume': avg_vol,
            'drop_rally_ratio': max_drop / max(max_rally, 1e-8),
        })

    return pd.DataFrame(daily_stats)


def classify_period(date, p1, p2, p3):
    """Classify a date into a period."""
    p1_s, p1_e = pd.Timestamp(p1[0]).date(), pd.Timestamp(p1[1]).date()
    p2_s, p2_e = pd.Timestamp(p2[0]).date(), pd.Timestamp(p2[1]).date()
    p3_s, p3_e = pd.Timestamp(p3[0]).date(), pd.Timestamp(p3[1]).date()
    if p1_s <= date <= p1_e:
        return 'Period 1 (Aug 3-13)'
    elif p2_s <= date <= p2_e:
        return 'Period 2 (Aug 14-21)'
    elif p3_s <= date <= p3_e:
        return 'Period 3 (Aug 24-27)'
    return 'Other'


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("KEEL SHORT DRAG INVESTIGATION")
print("=" * 80)

# Step 1: Load data for all tickers
print("\n[1] Loading master_inference data...")
ticker_data = {}
for ticker in ALL_TICKERS:
    print(f"  Loading {ticker}...")
    df = load_master_inference(ticker)
    if df is not None:
        print(f"    {ticker}: {len(df)} bars, {df.index.min()} to {df.index.max()}")
        ticker_data[ticker] = df
    else:
        print(f"    {ticker}: FAILED TO LOAD")

if KEEL_TICKER not in ticker_data:
    print("FATAL: Could not load KEEL data")
    sys.exit(1)

# Step 2: Analyze daily behavior
print("\n[2] Analyzing daily price behavior Aug 3-27...")
ticker_daily = {}
for ticker in ALL_TICKERS:
    if ticker in ticker_data:
        stats = analyze_daily_behavior(ticker_data[ticker], BT_START, BT_END)
        ticker_daily[ticker] = stats
        print(f"  {ticker}: {len(stats)} trading days analyzed")

keel_daily = ticker_daily[KEEL_TICKER]

print("\n  KEEL Daily Summary (Aug 3-27):")
print(f"    Avg daily return:     {keel_daily['daily_return'].mean()*100:+.2f}%")
print(f"    Avg gap:              {keel_daily['gap'].mean()*100:+.2f}%")
print(f"    Avg max drop (SHORT): {keel_daily['max_drop'].mean()*100:.2f}%")
print(f"    Avg max rally (BAD):  {keel_daily['max_rally'].mean()*100:.2f}%")
print(f"    Avg intraday trend:   {keel_daily['intraday_trend'].mean()*100:+.2f}%")
print(f"    Avg drop/rally ratio: {keel_daily['drop_rally_ratio'].mean():.2f}")

# Step 3: Period analysis
print("\n[3] Period Analysis for KEEL...")
keel_daily['period'] = keel_daily['date'].apply(
    lambda d: classify_period(d, PERIOD_1, PERIOD_2, PERIOD_3))

for period in ['Period 1 (Aug 3-13)', 'Period 2 (Aug 14-21)', 'Period 3 (Aug 24-27)']:
    p_data = keel_daily[keel_daily['period'] == period]
    if p_data.empty:
        print(f"\n  {period}: No data")
        continue
    print(f"\n  {period} ({len(p_data)} days):")
    print(f"    Avg gap:              {p_data['gap'].mean()*100:+.2f}%")
    print(f"    Avg max drop (SHORT): {p_data['max_drop'].mean()*100:.2f}%")
    print(f"    Avg max rally (BAD):  {p_data['max_rally'].mean()*100:.2f}%")
    print(f"    Avg intraday trend:   {p_data['intraday_trend'].mean()*100:+.2f}%")
    print(f"    Avg drop/rally ratio: {p_data['drop_rally_ratio'].mean():.2f}")
    print(f"    Avg 30m return:       {p_data['ret_30m'].mean()*100:+.2f}%")
    print(f"    Avg 60m return:       {p_data['ret_60m'].mean()*100:+.2f}%")
    # Days with positive trend (BAD for SHORT)
    up_days = (p_data['intraday_trend'] > 0).sum()
    print(f"    Days trending UP:     {up_days}/{len(p_data)} ({up_days/len(p_data)*100:.0f}%)")

# Step 4: Impact analysis
print("\n" + "=" * 80)
print("[4] IMPACT ANALYSIS — What if KEEL was excluded?")
print("=" * 80)

keel_total_pnl = KEEL_TRADES['n_trades'] * KEEL_TRADES['avg_pnl']
total_pnl = TOTAL_SHORT_TRADES * TOTAL_SHORT_AVG_PNL
non_keel_trades = TOTAL_SHORT_TRADES - KEEL_TRADES['n_trades']
non_keel_pnl = total_pnl - keel_total_pnl

if non_keel_trades > 0:
    non_keel_avg = non_keel_pnl / non_keel_trades
else:
    non_keel_avg = 0

print(f"\n  WITH KEEL:")
print(f"    Total trades:    {TOTAL_SHORT_TRADES}")
print(f"    Avg PnL:         {TOTAL_SHORT_AVG_PNL*100:+.2f}%")
print(f"    Total PnL:       {total_pnl*100:+.2f}%")

print(f"\n  KEEL CONTRIBUTION:")
print(f"    KEEL trades:     {KEEL_TRADES['n_trades']} ({KEEL_TRADES['n_trades']/TOTAL_SHORT_TRADES*100:.0f}% of activity)")
print(f"    KEEL avg PnL:    {KEEL_TRADES['avg_pnl']*100:+.2f}%")
print(f"    KEEL total PnL:  {keel_total_pnl*100:+.2f}%")
print(f"    KEEL win rate:   {KEEL_TRADES['win_rate']*100:.0f}% ({KEEL_TRADES['tp']} TP, {KEEL_TRADES['sl']} SL, {KEEL_TRADES['eod']} EOD)")

print(f"\n  WITHOUT KEEL:")
print(f"    Remaining trades: {non_keel_trades}")
print(f"    Remaining avg PnL: {non_keel_avg*100:+.2f}%")
print(f"    Remaining total PnL: {non_keel_pnl*100:+.2f}%")

pnl_improvement = non_keel_avg - TOTAL_SHORT_AVG_PNL
print(f"\n  IMPROVEMENT:")
print(f"    Avg PnL uplift:  {pnl_improvement*100:+.2f}% per trade")
print(f"    PnL recovered:   {abs(keel_total_pnl)*100:.2f}% added back")
print(f"    Relative improvement: {(non_keel_avg/TOTAL_SHORT_AVG_PNL - 1)*100:+.1f}% better avg PnL")

# Step 5: Compare KEEL vs good SHORT tickers
print("\n" + "=" * 80)
print("[5] KEEL vs GOOD SHORT TICKERS — Max Drop Comparison")
print("=" * 80)

comparison_stats = {}
for ticker in ALL_TICKERS:
    if ticker in ticker_daily and not ticker_daily[ticker].empty:
        daily = ticker_daily[ticker]
        comparison_stats[ticker] = {
            'avg_max_drop': daily['max_drop'].mean(),
            'avg_max_rally': daily['max_rally'].mean(),
            'avg_trend': daily['intraday_trend'].mean(),
            'drop_rally_ratio': daily['drop_rally_ratio'].mean(),
            'pct_down_days': (daily['intraday_trend'] < 0).mean(),
            'avg_gap': daily['gap'].mean(),
            'med_max_drop': daily['max_drop'].median(),
            'std_max_drop': daily['max_drop'].std(),
        }

print(f"\n  {'Ticker':<8} {'AvgMaxDrop':>11} {'AvgMaxRally':>12} {'AvgTrend':>10} {'Drop/Rally':>11} {'%DownDays':>10}")
print(f"  {'-'*8} {'-'*11} {'-'*12} {'-'*10} {'-'*11} {'-'*10}")
for ticker in ALL_TICKERS:
    if ticker in comparison_stats:
        s = comparison_stats[ticker]
        marker = " *** WORST" if ticker == KEEL_TICKER and s['drop_rally_ratio'] < 1.0 else ""
        print(f"  {ticker:<8} {s['avg_max_drop']*100:>10.2f}% {s['avg_max_rally']*100:>11.2f}% "
              f"{s['avg_trend']*100:>+9.2f}% {s['drop_rally_ratio']:>10.2f}x {s['pct_down_days']*100:>9.0f}%{marker}")

# Does KEEL have MORE downside potential?
if KEEL_TICKER in comparison_stats:
    keel_s = comparison_stats[KEEL_TICKER]
    good_avg_drop = np.mean([comparison_stats[t]['avg_max_drop'] for t in GOOD_SHORT_TICKERS if t in comparison_stats])
    print(f"\n  KEEL avg max drop: {keel_s['avg_max_drop']*100:.2f}% vs Good tickers avg: {good_avg_drop*100:.2f}%")
    if keel_s['avg_max_drop'] >= good_avg_drop:
        print(f"  => KEEL has EQUAL or MORE downside potential, but poor conversion (low WR)")
    else:
        print(f"  => KEEL has LESS downside potential — consensus is firing on a weak ticker")

    good_avg_ratio = np.mean([comparison_stats[t]['drop_rally_ratio'] for t in GOOD_SHORT_TICKERS if t in comparison_stats])
    print(f"  KEEL drop/rally ratio: {keel_s['drop_rally_ratio']:.2f}x vs Good tickers avg: {good_avg_ratio:.2f}x")
    if keel_s['drop_rally_ratio'] < 1.0:
        print(f"  => KEEL rallies MORE than it drops — fundamentally NOT a good SHORT ticker")
    elif keel_s['drop_rally_ratio'] < good_avg_ratio:
        print(f"  => KEEL has worse drop/rally ratio than good tickers")

# Step 6: Check if KEEL is crowding out better picks
print("\n" + "=" * 80)
print("[6] CONSENSUS CROWDING — Is KEEL selected OVER better tickers?")
print("=" * 80)

# Compare daily max drops across tickers on the SAME days
# On days KEEL had a max_drop, what was the max_drop of good tickers?
if KEEL_TICKER in ticker_daily:
    keel_dates = set(keel_daily['date'].tolist())

    overlap_analysis = []
    for date in sorted(keel_dates):
        keel_day = keel_daily[keel_daily['date'] == date]
        if keel_day.empty:
            continue
        keel_drop = float(keel_day['max_drop'].iloc[0])
        keel_trend = float(keel_day['intraday_trend'].iloc[0])

        good_drops = {}
        for ticker in GOOD_SHORT_TICKERS:
            if ticker in ticker_daily:
                t_day = ticker_daily[ticker]
                t_date = t_day[t_day['date'] == date]
                if not t_date.empty:
                    good_drops[ticker] = {
                        'max_drop': float(t_date['max_drop'].iloc[0]),
                        'trend': float(t_date['intraday_trend'].iloc[0]),
                    }

        # Count how many good tickers had better drops than KEEL
        better_count = sum(1 for t, d in good_drops.items() if d['max_drop'] > keel_drop)

        overlap_analysis.append({
            'date': date,
            'keel_drop': keel_drop,
            'keel_trend': keel_trend,
            'better_alternatives': better_count,
            'good_drops': good_drops,
        })

    oa_df = pd.DataFrame(overlap_analysis)

    # Focus on days where KEEL trend was UP (bad for SHORT)
    keel_bad_days = oa_df[oa_df['keel_trend'] > 0]

    print(f"\n  Total trading days: {len(oa_df)}")
    print(f"  Days KEEL trended UP (bad for SHORT): {len(keel_bad_days)} ({len(keel_bad_days)/len(oa_df)*100:.0f}%)")

    if len(keel_bad_days) > 0:
        avg_better = keel_bad_days['better_alternatives'].mean()
        print(f"  On KEEL bad days, avg tickers with better max_drop: {avg_better:.1f}")

        print(f"\n  KEEL BAD DAYS detail:")
        for _, row in keel_bad_days.iterrows():
            d = row['date']
            print(f"    {d}: KEEL drop={row['keel_drop']*100:.2f}%, trend={row['keel_trend']*100:+.2f}%", end="")
            gd = row['good_drops']
            if gd:
                better = [(t, info) for t, info in gd.items() if info['max_drop'] > row['keel_drop']]
                if better:
                    print(f"  | BETTER: ", end="")
                    print(", ".join(f"{t} drop={info['max_drop']*100:.2f}% trend={info['trend']*100:+.2f}%"
                                   for t, info in better), end="")
            print()

    # Overall: on what fraction of days was at least 1 good ticker better?
    days_with_better = (oa_df['better_alternatives'] > 0).sum()
    print(f"\n  Days where >= 1 good ticker had bigger max_drop than KEEL: {days_with_better}/{len(oa_df)} ({days_with_better/len(oa_df)*100:.0f}%)")

    # Average rank of KEEL's max_drop among all tickers each day
    rank_analysis = []
    for _, row in oa_df.iterrows():
        all_drops = [('KEEL', row['keel_drop'])]
        for t, info in row['good_drops'].items():
            all_drops.append((t, info['max_drop']))
        all_drops.sort(key=lambda x: -x[1])  # Descending
        keel_rank = next(i+1 for i, (t, _) in enumerate(all_drops) if t == 'KEEL')
        rank_analysis.append(keel_rank)

    avg_rank = np.mean(rank_analysis)
    print(f"  KEEL's avg rank in max_drop (1=best): {avg_rank:.1f} out of {len(ALL_TICKERS)} tickers")
    if avg_rank > 2:
        print(f"  => KEEL is consistently ranked LOWER than alternatives — consensus is over-selecting it")

# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7] Generating charts...")

fig = plt.figure(figsize=(20, 18))
gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

# ── Chart 1: KEEL daily intraday behavior (gap + trend) ──
ax1 = fig.add_subplot(gs[0, 0])
if not keel_daily.empty:
    dates_plot = keel_daily['date']
    colors_trend = ['#d32f2f' if t > 0 else '#2e7d32' for t in keel_daily['intraday_trend']]
    ax1.bar(range(len(dates_plot)), keel_daily['intraday_trend'] * 100,
            color=colors_trend, alpha=0.7, label='Intraday Trend')
    ax1.bar(range(len(dates_plot)), keel_daily['gap'] * 100,
            color='#1565c0', alpha=0.5, width=0.4, label='Gap')
    ax1.axhline(0, color='black', linewidth=0.5)
    # Period separators
    for i, date in enumerate(dates_plot):
        if date == pd.Timestamp('2026-08-14').date():
            ax1.axvline(i - 0.5, color='orange', linestyle='--', alpha=0.7, label='Period boundary')
        elif date == pd.Timestamp('2026-08-24').date():
            ax1.axvline(i - 0.5, color='red', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Trading Day')
    ax1.set_ylabel('Return (%)')
    ax1.set_title('KEEL Daily: Gap + Intraday Trend\n(Red = UP = Bad for SHORT)', fontsize=11)
    ax1.set_xticks(range(0, len(dates_plot), 2))
    ax1.set_xticklabels([str(d)[5:] for d in dates_plot.iloc[::2]], rotation=45, fontsize=8)
    ax1.legend(fontsize=8)

# ── Chart 2: KEEL max drop vs max rally ──
ax2 = fig.add_subplot(gs[0, 1])
if not keel_daily.empty:
    x = range(len(keel_daily))
    ax2.fill_between(x, keel_daily['max_drop'] * 100, 0, alpha=0.3, color='#2e7d32', label='Max Drop (good for SHORT)')
    ax2.fill_between(x, 0, -keel_daily['max_rally'] * 100, alpha=0.3, color='#d32f2f', label='Max Rally (bad for SHORT)')
    ax2.plot(x, keel_daily['max_drop'] * 100, color='#2e7d32', linewidth=1.5)
    ax2.plot(x, -keel_daily['max_rally'] * 100, color='#d32f2f', linewidth=1.5)
    ax2.axhline(0, color='black', linewidth=0.5)
    # Period separators
    for i, date in enumerate(keel_daily['date']):
        if date == pd.Timestamp('2026-08-14').date():
            ax2.axvline(i - 0.5, color='orange', linestyle='--', alpha=0.7)
        elif date == pd.Timestamp('2026-08-24').date():
            ax2.axvline(i - 0.5, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Trading Day')
    ax2.set_ylabel('% from Entry')
    ax2.set_title('KEEL: Max Drop vs Max Rally from Entry\n(Green area should dominate for good SHORT)', fontsize=11)
    ax2.set_xticks(range(0, len(keel_daily), 2))
    ax2.set_xticklabels([str(d)[5:] for d in keel_daily['date'].iloc[::2]], rotation=45, fontsize=8)
    ax2.legend(fontsize=8)

# ── Chart 3: KEEL vs good tickers — max drop comparison ──
ax3 = fig.add_subplot(gs[1, 0])
tickers_to_plot = [t for t in ALL_TICKERS if t in comparison_stats]
avg_drops = [comparison_stats[t]['avg_max_drop'] * 100 for t in tickers_to_plot]
avg_rallies = [comparison_stats[t]['avg_max_rally'] * 100 for t in tickers_to_plot]
colors_bar = ['#d32f2f' if t == KEEL_TICKER else '#1565c0' for t in tickers_to_plot]

x_pos = np.arange(len(tickers_to_plot))
width = 0.35
ax3.bar(x_pos - width/2, avg_drops, width, color=colors_bar, alpha=0.8, label='Avg Max Drop')
ax3.bar(x_pos + width/2, avg_rallies, width, color=[c.replace('c0', '66').replace('2f', '77') for c in colors_bar],
        alpha=0.5, label='Avg Max Rally', hatch='//')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(tickers_to_plot, fontsize=10)
ax3.set_ylabel('% from Entry')
ax3.set_title('Avg Max Drop vs Rally by Ticker\n(KEEL in red, good SHORT tickers in blue)', fontsize=11)
ax3.legend(fontsize=8)

# ── Chart 4: Drop/Rally ratio comparison ──
ax4 = fig.add_subplot(gs[1, 1])
ratios = [comparison_stats[t]['drop_rally_ratio'] for t in tickers_to_plot]
colors_ratio = ['#d32f2f' if t == KEEL_TICKER else '#2e7d32' if r > 1.0 else '#ff9800'
                for t, r in zip(tickers_to_plot, ratios)]
ax4.bar(tickers_to_plot, ratios, color=colors_ratio, alpha=0.8)
ax4.axhline(1.0, color='black', linestyle='--', linewidth=1, label='Neutral (drop = rally)')
ax4.set_ylabel('Drop / Rally Ratio')
ax4.set_title('Drop/Rally Ratio (>1 = drops more = good SHORT target)\n(KEEL in red)', fontsize=11)
ax4.legend(fontsize=8)

# ── Chart 5: Impact of excluding KEEL ──
ax5 = fig.add_subplot(gs[2, 0])
categories = ['With KEEL', 'Without KEEL']
avg_pnls = [TOTAL_SHORT_AVG_PNL * 100, non_keel_avg * 100]
total_pnls = [total_pnl * 100, non_keel_pnl * 100]

x_pos = np.arange(len(categories))
width = 0.35
bars1 = ax5.bar(x_pos - width/2, avg_pnls, width, color=['#ff9800', '#2e7d32'], alpha=0.8, label='Avg PnL/trade (%)')
bars2 = ax5.bar(x_pos + width/2, total_pnls, width, color=['#ff980066', '#2e7d3266'], alpha=0.8, label='Total PnL (%)')
ax5.set_xticks(x_pos)
ax5.set_xticklabels(categories, fontsize=11)
ax5.set_ylabel('%')
ax5.set_title(f'SHORT Portfolio: Impact of Excluding KEEL\n({KEEL_TRADES["n_trades"]} trades removed, {non_keel_trades} remaining)', fontsize=11)
ax5.legend(fontsize=8)
for bar, val in zip(bars1, avg_pnls):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{val:.2f}%',
             ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar, val in zip(bars2, total_pnls):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{val:.1f}%',
             ha='center', va='bottom', fontsize=9)

# ── Chart 6: KEEL rank distribution ──
ax6 = fig.add_subplot(gs[2, 1])
if rank_analysis:
    unique_ranks, rank_counts = np.unique(rank_analysis, return_counts=True)
    colors_rank = ['#d32f2f' if r >= 3 else '#ff9800' if r == 2 else '#2e7d32' for r in unique_ranks]
    ax6.bar(unique_ranks, rank_counts, color=colors_rank, alpha=0.8)
    ax6.set_xlabel('KEEL Rank in Max Drop (1=best)')
    ax6.set_ylabel('Number of Days')
    ax6.set_title(f'How Often KEEL Had the Biggest Drop\n(Avg rank: {avg_rank:.1f} out of {len(ALL_TICKERS)} tickers)', fontsize=11)
    ax6.set_xticks(unique_ranks)

plt.suptitle('KEEL SHORT Drag Investigation — Is KEEL the Wrong Ticker for SHORT?',
             fontsize=14, fontweight='bold', y=0.98)

output_path = '/tmp/investigation_keel_drag.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Charts saved to {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL ANSWERS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("FINAL ANSWERS")
print("=" * 80)

print(f"""
A) IS KEEL FUNDAMENTALLY A BAD SHORT TICKER?
   - KEEL avg max drop: {comparison_stats.get(KEEL_TICKER, {}).get('avg_max_drop', 0)*100:.2f}%
   - Good tickers avg max drop: {good_avg_drop*100:.2f}%
   - KEEL drop/rally ratio: {comparison_stats.get(KEEL_TICKER, {}).get('drop_rally_ratio', 0):.2f}x
   - Good tickers drop/rally ratio: {good_avg_ratio:.2f}x
   - KEEL % down days: {comparison_stats.get(KEEL_TICKER, {}).get('pct_down_days', 0)*100:.0f}%
""")

keel_ratio = comparison_stats.get(KEEL_TICKER, {}).get('drop_rally_ratio', 0)
if keel_ratio < 1.0:
    print("   VERDICT: YES — KEEL rallies more than it drops. It is a fundamentally")
    print("   BAD SHORT ticker. It has a drop/rally ratio < 1.0, meaning the upside")
    print("   risk exceeds the downside opportunity. EXCLUDE from SHORT universe.")
elif keel_ratio < good_avg_ratio * 0.8:
    print("   VERDICT: LIKELY YES — KEEL's drop/rally ratio is significantly worse")
    print("   than the good SHORT tickers. Consider excluding or deprioritizing.")
else:
    print("   VERDICT: UNCLEAR — KEEL has comparable drop/rally to good tickers.")
    print("   The issue may be with the specific configs/thresholds, not the ticker itself.")

print(f"""
B) HOW MUCH DID KEEL COST THE PORTFOLIO?
   - KEEL: {KEEL_TRADES['n_trades']} trades x {KEEL_TRADES['avg_pnl']*100:+.2f}% = {keel_total_pnl*100:+.2f}% total drag
   - Portfolio avg PnL with KEEL:    {TOTAL_SHORT_AVG_PNL*100:+.2f}%
   - Portfolio avg PnL without KEEL: {non_keel_avg*100:+.2f}%
   - Per-trade improvement:          {pnl_improvement*100:+.2f}%
   - KEEL consumed {KEEL_TRADES['n_trades']}/{TOTAL_SHORT_TRADES} = {KEEL_TRADES['n_trades']/TOTAL_SHORT_TRADES*100:.0f}% of trade capacity
     while generating {keel_total_pnl*100:+.2f}% total PnL

   VERDICT: KEEL cost approximately {abs(keel_total_pnl)*100:.2f}% in absolute PnL
   and dragged avg PnL down by {abs(pnl_improvement)*100:.2f}% per trade.
   Excluding KEEL would improve avg PnL by {(non_keel_avg/TOTAL_SHORT_AVG_PNL - 1)*100:+.1f}%.
""")

print(f"""C) IS THE CONSENSUS MECHANISM OVER-SELECTING KEEL?
   - KEEL had 2.1 avg consensus fires (multiple models agreed)
   - KEEL's avg rank in max drop: {avg_rank:.1f} out of {len(ALL_TICKERS)} tickers
   - Days where better alternatives existed: {days_with_better}/{len(oa_df)} ({days_with_better/len(oa_df)*100:.0f}%)
   - KEEL consumed 23% of SHORT trade slots
""")

if avg_rank > 2:
    print("   VERDICT: YES — The consensus mechanism is SYSTEMATICALLY over-selecting KEEL.")
    print("   Multiple models fire on KEEL, giving it high consensus counts, but the")
    print("   signals are poor quality. Better SHORT tickers (HUT, CIFR, MXL, CLSK)")
    print(f"   had bigger drops {days_with_better/len(oa_df)*100:.0f}% of the time.")
    print("   RECOMMENDATION: Add KEEL to EXCLUDE_TICKERS for SHORT strategy,")
    print("   or apply a per-ticker win-rate gate to the consensus mechanism.")
elif avg_rank > 1.5:
    print("   VERDICT: PARTIALLY — KEEL is often ranked below better alternatives")
    print("   but still selected because consensus count (n_fires) outweighs quality.")
else:
    print("   VERDICT: NO — KEEL's max drop is competitive with alternatives.")
    print("   The issue is execution (mode/config), not consensus selection.")

print("\nDone.")
