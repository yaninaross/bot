"""
Ticker-Level Microstructure Analysis: April-May vs June 2026
Investigates why S3 spike strategy stopped being profitable in June 2026.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────
PRIMARY_TICKERS = ['KEEL', 'CIFR', 'CLSK', 'FLNC', 'HUT', 'PL', 'RCAT', 'RIOT']
SECONDARY_TICKERS = ['WULF', 'SMCI']
ALL_TICKERS = PRIMARY_TICKERS + SECONDARY_TICKERS

MONTHS = {
    'Apr': ('2026-04-01', '2026-04-30'),
    'May': ('2026-05-01', '2026-05-31'),
    'Jun': ('2026-06-01', '2026-06-30'),
}

# Trading window in PST
MORNING_START = 6   # 6:00 AM PST
MORNING_END   = 7   # 7:30 AM PST (we'll use 7:30 exactly)
MORNING_END_MIN = 30

# ── Load data ───────────────────────────────────────────────────────────
def load_ticker(ticker):
    uri = f"gs://time-series-bucket-ys/Data/Inference_v2/{ticker}_master_inference.parquet"
    df = pd.read_parquet(uri).reset_index()
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if dt_cols:
        df['Datetime'] = pd.to_datetime(df[dt_cols[0]], utc=True)
    df = df.set_index('Datetime').sort_index()
    df.index = df.index.tz_convert('America/Los_Angeles')
    # Filter to Apr-Jun 2026
    df = df['2026-04-01':'2026-06-30']
    return df

# ── Daily aggregation ──────────────────────────────────────────────────
def compute_daily_stats(df, ticker):
    """Compute daily statistics from 1-min bars."""
    rows = []
    for date, day_df in df.groupby(df.index.date):
        if len(day_df) < 10:
            continue

        # Full day OHLCV
        d_open = day_df['Open'].iloc[0]
        d_high = day_df['High'].max()
        d_low  = day_df['Low'].min()
        d_close = day_df['Close'].iloc[-1]
        d_vol   = day_df['Volume'].sum()

        # Daily range
        daily_range_pct = (d_high - d_low) / d_close * 100 if d_close > 0 else 0

        # Max intraday move from open
        max_up_from_open = (d_high - d_open) / d_open * 100 if d_open > 0 else 0
        max_dn_from_open = (d_open - d_low) / d_open * 100 if d_open > 0 else 0

        # Close vs high ratio (how much is given back)
        close_vs_high = (d_close - d_low) / (d_high - d_low) * 100 if (d_high - d_low) > 0 else 50

        # Morning window (6:00-7:30 PST)
        morning = day_df.between_time('06:00', '07:30')
        if len(morning) > 0:
            m_high = morning['High'].max()
            m_low  = morning['Low'].min()
            m_open = morning['Open'].iloc[0]
            m_close = morning['Close'].iloc[-1]
            m_vol  = morning['Volume'].sum()
            m_range_pct = (m_high - m_low) / m_open * 100 if m_open > 0 else 0
            m_move = (m_close - m_open) / m_open * 100 if m_open > 0 else 0
            m_max_up = (m_high - m_open) / m_open * 100 if m_open > 0 else 0
            m_vol_pct = m_vol / d_vol * 100 if d_vol > 0 else 0

            # Spread and OBI in morning
            m_spread = morning['Spread'].mean() if 'Spread' in morning.columns else np.nan
            m_obi = morning['OBI'].mean() if 'OBI' in morning.columns else np.nan

            # Dip check: did price dip 1-2% from open within first 30 min (6:00-6:30)?
            first30 = day_df.between_time('06:00', '06:30')
            if len(first30) > 0 and m_open > 0:
                min_price_30 = first30['Low'].min()
                dip_pct = (m_open - min_price_30) / m_open * 100
                had_1pct_dip = dip_pct >= 1.0
                had_2pct_dip = dip_pct >= 2.0
            else:
                dip_pct = 0
                had_1pct_dip = False
                had_2pct_dip = False
        else:
            m_range_pct = m_move = m_max_up = m_vol_pct = 0
            m_spread = m_obi = np.nan
            m_vol = 0
            dip_pct = 0
            had_1pct_dip = False
            had_2pct_dip = False

        # Afternoon window (12:00-13:00 PST = last hour)
        afternoon = day_df.between_time('12:00', '13:00')
        if len(afternoon) > 0:
            a_move = (afternoon['Close'].iloc[-1] - afternoon['Open'].iloc[0]) / afternoon['Open'].iloc[0] * 100
        else:
            a_move = 0

        # Time of daily high (hour in PST)
        high_time = day_df['High'].idxmax()
        high_hour = high_time.hour + high_time.minute / 60 if pd.notna(high_time) else np.nan
        high_in_morning = 6.0 <= high_hour <= 7.5 if pd.notna(high_hour) else False

        # Gap from previous close
        # (will be computed after sorting)

        # Volume microstructure
        d_trade_count = day_df['Trade_Count'].sum() if 'Trade_Count' in day_df.columns else np.nan
        m_trade_count = morning['Trade_Count'].sum() if len(morning) > 0 and 'Trade_Count' in morning.columns else np.nan

        rows.append({
            'ticker': ticker,
            'date': pd.Timestamp(date),
            'month': pd.Timestamp(date).strftime('%b'),
            'open': d_open,
            'high': d_high,
            'low': d_low,
            'close': d_close,
            'daily_vol': d_vol,
            'daily_range_pct': daily_range_pct,
            'max_up_from_open': max_up_from_open,
            'max_dn_from_open': max_dn_from_open,
            'close_vs_high_ratio': close_vs_high,
            'morning_range_pct': m_range_pct,
            'morning_move_pct': m_move,
            'morning_max_up_pct': m_max_up,
            'morning_vol': m_vol,
            'morning_vol_pct': m_vol_pct,
            'morning_spread': m_spread,
            'morning_obi': m_obi,
            'afternoon_move_pct': a_move,
            'high_hour_pst': high_hour,
            'high_in_morning': high_in_morning,
            'dip_pct_first30': dip_pct,
            'had_1pct_dip': had_1pct_dip,
            'had_2pct_dip': had_2pct_dip,
            'daily_trade_count': d_trade_count,
            'morning_trade_count': m_trade_count,
        })

    daily = pd.DataFrame(rows)

    # Compute gap %
    if len(daily) > 0:
        daily = daily.sort_values('date')
        daily['prev_close'] = daily['close'].shift(1)
        daily['gap_pct'] = ((daily['open'] - daily['prev_close']) / daily['prev_close'] * 100)
        daily['abs_gap_pct'] = daily['gap_pct'].abs()

    return daily

# ── Main analysis ──────────────────────────────────────────────────────
print("=" * 90)
print("TICKER-LEVEL MICROSTRUCTURE ANALYSIS: Apr-May vs June 2026")
print("=" * 90)

all_daily = []
for ticker in ALL_TICKERS:
    print(f"\nLoading {ticker}...", end=" ", flush=True)
    try:
        df = load_ticker(ticker)
        daily = compute_daily_stats(df, ticker)
        all_daily.append(daily)
        print(f"OK ({len(daily)} trading days)")
    except Exception as e:
        print(f"FAILED: {e}")

all_daily = pd.concat(all_daily, ignore_index=True)
all_daily['month'] = pd.Categorical(all_daily['month'], categories=['Apr', 'May', 'Jun'], ordered=True)

# ── 1. Per-Ticker Monthly Comparison ──────────────────────────────────
print("\n" + "=" * 90)
print("SECTION 1: PER-TICKER MONTHLY COMPARISON")
print("=" * 90)

metrics = [
    ('daily_range_pct',    'Daily Range %',         '{:.2f}'),
    ('max_up_from_open',   'Max Up from Open %',    '{:.2f}'),
    ('max_dn_from_open',   'Max Down from Open %',  '{:.2f}'),
    ('abs_gap_pct',        'Abs Gap %',             '{:.2f}'),
    ('morning_range_pct',  'Morning Range %',       '{:.2f}'),
    ('morning_max_up_pct', 'Morning Max Up %',      '{:.2f}'),
    ('morning_move_pct',   'Morning Move %',        '{:.2f}'),
    ('morning_vol_pct',    'Morning Vol %/Day',     '{:.1f}'),
    ('daily_vol',          'Daily Volume',          '{:.0f}'),
    ('morning_spread',     'Morning Spread',        '{:.4f}'),
    ('morning_obi',        'Morning OBI',           '{:.3f}'),
    ('close_vs_high_ratio','Close/High Ratio',      '{:.1f}'),
    ('high_in_morning',    '% High in Morning',     '{:.1f}'),
    ('had_1pct_dip',       '% Days w/ 1% Dip',     '{:.1f}'),
    ('had_2pct_dip',       '% Days w/ 2% Dip',     '{:.1f}'),
]

for ticker in ALL_TICKERS:
    t_df = all_daily[all_daily['ticker'] == ticker]
    if len(t_df) == 0:
        continue

    is_primary = ticker in PRIMARY_TICKERS
    label = "PRIMARY" if is_primary else "SECONDARY"
    print(f"\n{'─' * 70}")
    print(f"  {ticker} [{label}]")
    print(f"{'─' * 70}")
    print(f"  {'Metric':<25s}  {'Apr':>10s}  {'May':>10s}  {'Jun':>10s}  {'Δ Jun vs AM':>12s}")
    print(f"  {'─'*25}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*12}")

    for col, label, fmt in metrics:
        vals = {}
        for m in ['Apr', 'May', 'Jun']:
            m_df = t_df[t_df['month'] == m]
            if len(m_df) == 0:
                vals[m] = np.nan
            elif col in ['high_in_morning', 'had_1pct_dip', 'had_2pct_dip']:
                vals[m] = m_df[col].mean() * 100
            else:
                vals[m] = m_df[col].mean()

        apr_may_avg = np.nanmean([vals['Apr'], vals['May']])
        if apr_may_avg != 0 and not np.isnan(apr_may_avg) and not np.isnan(vals.get('Jun', np.nan)):
            delta = (vals['Jun'] - apr_may_avg) / abs(apr_may_avg) * 100
            delta_str = f"{delta:+.1f}%"
        else:
            delta_str = "N/A"

        row_strs = []
        for m in ['Apr', 'May', 'Jun']:
            v = vals[m]
            if np.isnan(v):
                row_strs.append(f"{'N/A':>10s}")
            else:
                row_strs.append(f"{fmt.format(v):>10s}")

        print(f"  {label:<25s}  {'  '.join(row_strs)}  {delta_str:>12s}")

# ── 2. Aggregated Statistics ──────────────────────────────────────────
print("\n" + "=" * 90)
print("SECTION 2: AGGREGATED STATISTICS (ALL TICKERS)")
print("=" * 90)

print(f"\n  {'Metric':<25s}  {'Apr':>10s}  {'May':>10s}  {'Jun':>10s}  {'Δ Jun vs AM':>12s}")
print(f"  {'─'*25}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*12}")

for col, label, fmt in metrics:
    vals = {}
    for m in ['Apr', 'May', 'Jun']:
        m_df = all_daily[all_daily['month'] == m]
        if len(m_df) == 0:
            vals[m] = np.nan
        elif col in ['high_in_morning', 'had_1pct_dip', 'had_2pct_dip']:
            vals[m] = m_df[col].mean() * 100
        else:
            vals[m] = m_df[col].mean()

    apr_may_avg = np.nanmean([vals['Apr'], vals['May']])
    if apr_may_avg != 0 and not np.isnan(apr_may_avg) and not np.isnan(vals.get('Jun', np.nan)):
        delta = (vals['Jun'] - apr_may_avg) / abs(apr_may_avg) * 100
        delta_str = f"{delta:+.1f}%"
    else:
        delta_str = "N/A"

    row_strs = []
    for m in ['Apr', 'May', 'Jun']:
        v = vals[m]
        if np.isnan(v):
            row_strs.append(f"{'N/A':>10s}")
        else:
            row_strs.append(f"{fmt.format(v):>10s}")

    print(f"  {label:<25s}  {'  '.join(row_strs)}  {delta_str:>12s}")

# ── 2b. Primary vs Secondary comparison ──────────────────────────────
print(f"\n  --- PRIMARY TICKERS ONLY ---")
print(f"  {'Metric':<25s}  {'Apr':>10s}  {'May':>10s}  {'Jun':>10s}  {'Δ Jun vs AM':>12s}")
print(f"  {'─'*25}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*12}")

prim_df = all_daily[all_daily['ticker'].isin(PRIMARY_TICKERS)]
for col, label, fmt in metrics:
    vals = {}
    for m in ['Apr', 'May', 'Jun']:
        m_df = prim_df[prim_df['month'] == m]
        if len(m_df) == 0:
            vals[m] = np.nan
        elif col in ['high_in_morning', 'had_1pct_dip', 'had_2pct_dip']:
            vals[m] = m_df[col].mean() * 100
        else:
            vals[m] = m_df[col].mean()

    apr_may_avg = np.nanmean([vals['Apr'], vals['May']])
    if apr_may_avg != 0 and not np.isnan(apr_may_avg) and not np.isnan(vals.get('Jun', np.nan)):
        delta = (vals['Jun'] - apr_may_avg) / abs(apr_may_avg) * 100
        delta_str = f"{delta:+.1f}%"
    else:
        delta_str = "N/A"

    row_strs = []
    for m in ['Apr', 'May', 'Jun']:
        v = vals[m]
        if np.isnan(v):
            row_strs.append(f"{'N/A':>10s}")
        else:
            row_strs.append(f"{fmt.format(v):>10s}")

    print(f"  {label:<25s}  {'  '.join(row_strs)}  {delta_str:>12s}")

# ── 3. Spike Size Distribution ───────────────────────────────────────
print("\n" + "=" * 90)
print("SECTION 3: SPIKE SIZE DISTRIBUTION (Morning Max Up from Open)")
print("=" * 90)

for bucket_label, bucket_range in [('<1%', (0, 1)), ('1-2%', (1, 2)), ('2-3%', (2, 3)), ('3-5%', (3, 5)), ('>5%', (5, 100))]:
    counts = {}
    totals = {}
    for m in ['Apr', 'May', 'Jun']:
        m_df = all_daily[all_daily['month'] == m]
        mask = (m_df['morning_max_up_pct'] >= bucket_range[0]) & (m_df['morning_max_up_pct'] < bucket_range[1])
        counts[m] = mask.sum()
        totals[m] = len(m_df)

    pcts = {m: counts[m]/totals[m]*100 if totals[m] > 0 else 0 for m in ['Apr', 'May', 'Jun']}
    print(f"  {bucket_label:<8s}: Apr {pcts['Apr']:5.1f}% ({counts['Apr']:3d}/{totals['Apr']})"
          f"  May {pcts['May']:5.1f}% ({counts['May']:3d}/{totals['May']})"
          f"  Jun {pcts['Jun']:5.1f}% ({counts['Jun']:3d}/{totals['Jun']})")

# ── 4. Fill Rate Proxy ────────────────────────────────────────────────
print("\n" + "=" * 90)
print("SECTION 4: LIMIT ENTRY FILL RATE PROXY")
print("=" * 90)
print("  (Days where price dipped >= 1% from open within first 30 min of trading)")

for ticker in ALL_TICKERS:
    t_df = all_daily[all_daily['ticker'] == ticker]
    if len(t_df) == 0:
        continue

    fill_rates = {}
    for m in ['Apr', 'May', 'Jun']:
        m_df = t_df[t_df['month'] == m]
        if len(m_df) > 0:
            fill_rates[m] = m_df['had_1pct_dip'].mean() * 100
        else:
            fill_rates[m] = np.nan

    apr_may = np.nanmean([fill_rates.get('Apr', np.nan), fill_rates.get('May', np.nan)])
    jun_val = fill_rates.get('Jun', np.nan)
    if not np.isnan(apr_may) and not np.isnan(jun_val) and apr_may > 0:
        delta = jun_val - apr_may
    else:
        delta = np.nan

    delta_str = f"{delta:+.1f}pp" if not np.isnan(delta) else "N/A"

    print(f"  {ticker:<6s}: Apr {fill_rates.get('Apr',0):5.1f}%  May {fill_rates.get('May',0):5.1f}%  Jun {fill_rates.get('Jun',0):5.1f}%  Δ={delta_str}")

# ── 5. Time-of-High Analysis ─────────────────────────────────────────
print("\n" + "=" * 90)
print("SECTION 5: TIME OF DAILY HIGH (PST hours)")
print("=" * 90)

for m in ['Apr', 'May', 'Jun']:
    m_df = all_daily[all_daily['month'] == m]
    if len(m_df) == 0:
        continue
    median_h = m_df['high_hour_pst'].median()
    pct_morning = m_df['high_in_morning'].mean() * 100
    print(f"  {m}: Median high at {median_h:.1f}h PST | {pct_morning:.1f}% of days have high in 6:00-7:30 window")

# Per-ticker
print("\n  Per-ticker % of days with daily high in morning (6:00-7:30 PST):")
print(f"  {'Ticker':<8s}  {'Apr':>8s}  {'May':>8s}  {'Jun':>8s}  {'Δ':>8s}")
for ticker in ALL_TICKERS:
    t_df = all_daily[all_daily['ticker'] == ticker]
    vals = {}
    for m in ['Apr', 'May', 'Jun']:
        m_df = t_df[t_df['month'] == m]
        vals[m] = m_df['high_in_morning'].mean() * 100 if len(m_df) > 0 else np.nan

    am = np.nanmean([vals.get('Apr', np.nan), vals.get('May', np.nan)])
    j = vals.get('Jun', np.nan)
    delta = j - am if not np.isnan(am) and not np.isnan(j) else np.nan
    d_str = f"{delta:+.1f}pp" if not np.isnan(delta) else "N/A"

    print(f"  {ticker:<8s}  {vals.get('Apr',0):7.1f}%  {vals.get('May',0):7.1f}%  {vals.get('Jun',0):7.1f}%  {d_str:>8s}")

# ── 6. Morning vs Afternoon Edge ──────────────────────────────────────
print("\n" + "=" * 90)
print("SECTION 6: MORNING (6:00-7:30) vs AFTERNOON (12:00-13:00) MOVE")
print("=" * 90)

for m in ['Apr', 'May', 'Jun']:
    m_df = all_daily[all_daily['month'] == m]
    morn_avg = m_df['morning_move_pct'].mean()
    aftn_avg = m_df['afternoon_move_pct'].mean()
    morn_rng = m_df['morning_range_pct'].mean()
    print(f"  {m}: Morning move avg={morn_avg:+.3f}%  range={morn_rng:.3f}%  |  Afternoon move avg={aftn_avg:+.3f}%")

# ── 7. Spread / OBI Analysis ─────────────────────────────────────────
print("\n" + "=" * 90)
print("SECTION 7: MICROSTRUCTURE (Morning Spread & OBI)")
print("=" * 90)

for ticker in ALL_TICKERS:
    t_df = all_daily[all_daily['ticker'] == ticker]
    if len(t_df) == 0:
        continue

    vals_s = {}
    vals_o = {}
    for m in ['Apr', 'May', 'Jun']:
        m_df = t_df[t_df['month'] == m]
        vals_s[m] = m_df['morning_spread'].mean() if len(m_df) > 0 else np.nan
        vals_o[m] = m_df['morning_obi'].mean() if len(m_df) > 0 else np.nan

    am_s = np.nanmean([vals_s.get('Apr', np.nan), vals_s.get('May', np.nan)])
    j_s = vals_s.get('Jun', np.nan)
    if am_s > 0 and not np.isnan(j_s):
        d_s = (j_s - am_s) / am_s * 100
    else:
        d_s = np.nan

    d_s_str = f"{d_s:+.1f}%" if not np.isnan(d_s) else "N/A"

    print(f"  {ticker:<6s}: Spread Apr={vals_s.get('Apr',0):.4f}  May={vals_s.get('May',0):.4f}  Jun={vals_s.get('Jun',0):.4f}  Δ={d_s_str}"
          f"  |  OBI Apr={vals_o.get('Apr',0):+.3f}  May={vals_o.get('May',0):+.3f}  Jun={vals_o.get('Jun',0):+.3f}")

# ── 8. Volume Trend ──────────────────────────────────────────────────
print("\n" + "=" * 90)
print("SECTION 8: VOLUME TRENDS")
print("=" * 90)

print(f"\n  {'Ticker':<8s}  {'Apr Vol (M)':>12s}  {'May Vol (M)':>12s}  {'Jun Vol (M)':>12s}  {'Δ Jun/AM':>10s}  {'AM Vol% Apr':>12s}  {'AM Vol% May':>12s}  {'AM Vol% Jun':>12s}")
for ticker in ALL_TICKERS:
    t_df = all_daily[all_daily['ticker'] == ticker]
    vals_v = {}
    vals_mp = {}
    for m in ['Apr', 'May', 'Jun']:
        m_df = t_df[t_df['month'] == m]
        vals_v[m] = m_df['daily_vol'].mean() / 1e6 if len(m_df) > 0 else np.nan
        vals_mp[m] = m_df['morning_vol_pct'].mean() if len(m_df) > 0 else np.nan

    am = np.nanmean([vals_v.get('Apr', np.nan), vals_v.get('May', np.nan)])
    j = vals_v.get('Jun', np.nan)
    if am > 0 and not np.isnan(j):
        delta = (j - am) / am * 100
        d_str = f"{delta:+.1f}%"
    else:
        d_str = "N/A"

    print(f"  {ticker:<8s}  {vals_v.get('Apr',0):11.2f}M  {vals_v.get('May',0):11.2f}M  {vals_v.get('Jun',0):11.2f}M  {d_str:>10s}"
          f"  {vals_mp.get('Apr',0):11.1f}%  {vals_mp.get('May',0):11.1f}%  {vals_mp.get('Jun',0):11.1f}%")

# ── 9. Biggest Changers ──────────────────────────────────────────────
print("\n" + "=" * 90)
print("SECTION 9: RANKED CHANGES — BIGGEST DETERIORATIONS")
print("=" * 90)

change_records = []
key_metrics = [
    ('morning_max_up_pct', 'Morning Max Up %', 'higher_better'),
    ('daily_range_pct', 'Daily Range %', 'higher_better'),
    ('morning_range_pct', 'Morning Range %', 'higher_better'),
    ('morning_vol_pct', 'Morning Vol % of Day', 'higher_better'),
    ('had_1pct_dip', 'Fill Rate (1% dip)', 'higher_better'),
    ('abs_gap_pct', 'Abs Gap %', 'higher_better'),
    ('morning_spread', 'Morning Spread', 'lower_better'),
]

for ticker in ALL_TICKERS:
    t_df = all_daily[all_daily['ticker'] == ticker]
    for col, label, direction in key_metrics:
        vals = {}
        for m in ['Apr', 'May', 'Jun']:
            m_df = t_df[t_df['month'] == m]
            if len(m_df) == 0:
                vals[m] = np.nan
                continue
            if col in ['had_1pct_dip', 'had_2pct_dip']:
                vals[m] = m_df[col].mean() * 100
            else:
                vals[m] = m_df[col].mean()

        am = np.nanmean([vals.get('Apr', np.nan), vals.get('May', np.nan)])
        j = vals.get('Jun', np.nan)

        if np.isnan(am) or np.isnan(j) or am == 0:
            continue

        pct_change = (j - am) / abs(am) * 100

        # For "higher_better" metrics, negative change is bad
        # For "lower_better" metrics, positive change is bad
        if direction == 'higher_better':
            impact = -pct_change  # Negative change = positive impact (bad)
        else:
            impact = pct_change   # Positive change = positive impact (bad)

        change_records.append({
            'ticker': ticker,
            'metric': label,
            'apr_may_avg': am,
            'jun': j,
            'pct_change': pct_change,
            'impact': impact,  # Higher = worse for strategy
        })

changes_df = pd.DataFrame(change_records)
changes_df = changes_df.sort_values('impact', ascending=False)

print("\n  Top 20 worst changes (most harmful to strategy):")
print(f"  {'Rank':<5s}  {'Ticker':<8s}  {'Metric':<25s}  {'Apr-May':>10s}  {'Jun':>10s}  {'Change':>10s}")
print(f"  {'─'*5}  {'─'*8}  {'─'*25}  {'─'*10}  {'─'*10}  {'─'*10}")
for i, row in changes_df.head(20).iterrows():
    print(f"  {changes_df.index.get_loc(i)+1:<5d}  {row['ticker']:<8s}  {row['metric']:<25s}  {row['apr_may_avg']:10.2f}  {row['jun']:10.2f}  {row['pct_change']:+9.1f}%")

# ── 10. Summary ──────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("SECTION 10: EXECUTIVE SUMMARY")
print("=" * 90)

# Aggregate key changes
agg_changes = {}
for col, label, direction in key_metrics:
    vals = {}
    for m in ['Apr', 'May', 'Jun']:
        m_df = all_daily[all_daily['month'] == m]
        if col in ['had_1pct_dip', 'had_2pct_dip']:
            vals[m] = m_df[col].mean() * 100
        else:
            vals[m] = m_df[col].mean()
    am = np.nanmean([vals['Apr'], vals['May']])
    j = vals['Jun']
    if am != 0:
        pct = (j - am) / abs(am) * 100
    else:
        pct = 0
    agg_changes[label] = {'apr_may': am, 'jun': j, 'pct_change': pct}

print("\n  Aggregate changes (all tickers, Jun vs Apr-May avg):")
for label, d in agg_changes.items():
    print(f"    {label:<25s}: {d['apr_may']:.3f} -> {d['jun']:.3f}  ({d['pct_change']:+.1f}%)")

# Count tickers with major deterioration
print("\n  Tickers with >20% deterioration in Morning Max Up:")
for ticker in ALL_TICKERS:
    t_df = all_daily[all_daily['ticker'] == ticker]
    vals = {}
    for m in ['Apr', 'May', 'Jun']:
        m_df = t_df[t_df['month'] == m]
        vals[m] = m_df['morning_max_up_pct'].mean() if len(m_df) > 0 else np.nan
    am = np.nanmean([vals.get('Apr', np.nan), vals.get('May', np.nan)])
    j = vals.get('Jun', np.nan)
    if not np.isnan(am) and not np.isnan(j) and am > 0:
        pct = (j - am) / am * 100
        if pct < -20:
            print(f"    {ticker}: {am:.2f}% -> {j:.2f}% ({pct:+.1f}%)")

print("\n  Tickers with >30% deterioration in Fill Rate (1% dip):")
for ticker in ALL_TICKERS:
    t_df = all_daily[all_daily['ticker'] == ticker]
    vals = {}
    for m in ['Apr', 'May', 'Jun']:
        m_df = t_df[t_df['month'] == m]
        vals[m] = m_df['had_1pct_dip'].mean() * 100 if len(m_df) > 0 else np.nan
    am = np.nanmean([vals.get('Apr', np.nan), vals.get('May', np.nan)])
    j = vals.get('Jun', np.nan)
    if not np.isnan(am) and not np.isnan(j) and am > 0:
        pct = (j - am) / am * 100
        if pct < -30:
            print(f"    {ticker}: {am:.1f}% -> {j:.1f}% ({pct:+.1f}%)")

print("\n" + "=" * 90)
print("END OF ANALYSIS")
print("=" * 90)
