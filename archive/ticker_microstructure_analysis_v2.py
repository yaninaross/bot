"""
Follow-up analysis: WHY did the strategy stop working despite bigger moves?
Key hypothesis: The character of moves shifted from "spike up" to "spike down"
The strategy is LONG spikes with limit entry on dip - what matters is:
  1. Up/Down asymmetry (are up moves > down moves?)
  2. Stop-loss hit rate (are drawdowns deeper before the spike?)
  3. Morning move direction (net long bias vs net short bias)
  4. Speed of recovery after dip (fast V-recovery = strategy works; slow grind = SL hit)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

PRIMARY_TICKERS = ['KEEL', 'CIFR', 'CLSK', 'FLNC', 'HUT', 'PL', 'RCAT', 'RIOT']
SECONDARY_TICKERS = ['WULF', 'SMCI']
ALL_TICKERS = PRIMARY_TICKERS + SECONDARY_TICKERS

def load_ticker(ticker):
    uri = f"gs://time-series-bucket-ys/Data/Inference_v2/{ticker}_master_inference.parquet"
    df = pd.read_parquet(uri).reset_index()
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if dt_cols:
        df['Datetime'] = pd.to_datetime(df[dt_cols[0]], utc=True)
    df = df.set_index('Datetime').sort_index()
    df.index = df.index.tz_convert('America/Los_Angeles')
    df = df['2026-04-01':'2026-06-30']
    return df

def analyze_intraday_character(df, ticker):
    """Deep analysis of intraday move character per day."""
    rows = []
    for date, day_df in df.groupby(df.index.date):
        morning = day_df.between_time('06:00', '07:30')
        if len(morning) < 5:
            continue

        m_open = morning['Open'].iloc[0]
        if m_open <= 0:
            continue

        # --- Up/Down Asymmetry ---
        m_high = morning['High'].max()
        m_low = morning['Low'].min()
        max_up = (m_high - m_open) / m_open * 100
        max_dn = (m_open - m_low) / m_open * 100
        updown_ratio = max_up / max_dn if max_dn > 0.001 else 10.0

        # --- Sequence analysis: did the dip come BEFORE or AFTER the spike? ---
        high_idx = morning['High'].idxmax()
        low_idx = morning['Low'].idxmin()
        dip_before_spike = low_idx < high_idx  # Good for strategy: dip fills limit, then spikes

        # --- Max drawdown from entry (simulating limit buy on 1% dip) ---
        # Find first bar where Low dips 1% below open
        prices = morning[['Open', 'High', 'Low', 'Close']].copy()
        entry_price = None
        entry_idx = None
        for ts, row in prices.iterrows():
            if row['Low'] <= m_open * 0.99:
                entry_price = m_open * 0.99  # Limit order fills at -1%
                entry_idx = ts
                break

        if entry_price is not None and entry_idx is not None:
            # After entry, what's the max drawdown and max gain?
            post_entry = morning.loc[entry_idx:]
            if len(post_entry) > 0:
                post_low = post_entry['Low'].min()
                post_high = post_entry['High'].max()
                max_dd_from_entry = (entry_price - post_low) / entry_price * 100
                max_gain_from_entry = (post_high - entry_price) / entry_price * 100

                # Exit at 7:30 close
                exit_price = morning['Close'].iloc[-1]
                pnl_at_close = (exit_price - entry_price) / entry_price * 100

                # Would a 3% SL get hit?
                sl_3pct_hit = max_dd_from_entry >= 3.0
                # Would a 2% SL get hit?
                sl_2pct_hit = max_dd_from_entry >= 2.0

                # Time from entry to high
                time_to_high = (post_entry['High'].idxmax() - entry_idx).total_seconds() / 60
            else:
                max_dd_from_entry = 0
                max_gain_from_entry = 0
                pnl_at_close = 0
                sl_3pct_hit = False
                sl_2pct_hit = False
                time_to_high = 0
            trade_filled = True
        else:
            max_dd_from_entry = np.nan
            max_gain_from_entry = np.nan
            pnl_at_close = np.nan
            sl_3pct_hit = np.nan
            sl_2pct_hit = np.nan
            time_to_high = np.nan
            trade_filled = False

        # --- Reversal character ---
        # Break morning into 3 segments: first 30min, middle 30min, last 30min
        seg1 = day_df.between_time('06:00', '06:30')
        seg2 = day_df.between_time('06:30', '07:00')
        seg3 = day_df.between_time('07:00', '07:30')

        seg1_move = (seg1['Close'].iloc[-1] - seg1['Open'].iloc[0]) / seg1['Open'].iloc[0] * 100 if len(seg1) > 0 and seg1['Open'].iloc[0] > 0 else 0
        seg2_move = (seg2['Close'].iloc[-1] - seg2['Open'].iloc[0]) / seg2['Open'].iloc[0] * 100 if len(seg2) > 0 and seg2['Open'].iloc[0] > 0 else 0
        seg3_move = (seg3['Close'].iloc[-1] - seg3['Open'].iloc[0]) / seg3['Open'].iloc[0] * 100 if len(seg3) > 0 and seg3['Open'].iloc[0] > 0 else 0

        # V-recovery: dipped in seg1 then recovered in seg2+3
        v_recovery = seg1_move < -0.5 and (seg2_move + seg3_move) > abs(seg1_move) * 0.5

        # Continuation down: kept going down
        continuation_dn = seg1_move < -0.5 and seg2_move < 0

        # Full day context
        d_close = day_df['Close'].iloc[-1]
        d_open = day_df['Open'].iloc[0]
        full_day_return = (d_close - d_open) / d_open * 100 if d_open > 0 else 0

        # Afternoon reversal (did morning gains get wiped?)
        afternoon = day_df.between_time('10:00', '13:00')
        m_close = morning['Close'].iloc[-1]
        if len(afternoon) > 0 and m_close > 0:
            aftn_return = (afternoon['Close'].iloc[-1] - m_close) / m_close * 100
        else:
            aftn_return = 0

        rows.append({
            'ticker': ticker,
            'date': pd.Timestamp(date),
            'month': pd.Timestamp(date).strftime('%b'),
            'max_up': max_up,
            'max_dn': max_dn,
            'updown_ratio': min(updown_ratio, 10),  # cap at 10
            'dip_before_spike': dip_before_spike,
            'trade_filled': trade_filled,
            'max_dd_from_entry': max_dd_from_entry,
            'max_gain_from_entry': max_gain_from_entry,
            'pnl_at_close': pnl_at_close,
            'sl_2pct_hit': sl_2pct_hit,
            'sl_3pct_hit': sl_3pct_hit,
            'time_to_high_min': time_to_high,
            'seg1_move': seg1_move,
            'seg2_move': seg2_move,
            'seg3_move': seg3_move,
            'v_recovery': v_recovery,
            'continuation_dn': continuation_dn,
            'full_day_return': full_day_return,
            'aftn_reversal': aftn_return,
        })

    return pd.DataFrame(rows)

# ── Load and analyze ─────────────────────────────────────────────────
print("=" * 90)
print("DEEP DIVE: WHY STRATEGY STOPPED WORKING DESPITE BIGGER MOVES")
print("=" * 90)

all_analysis = []
for ticker in ALL_TICKERS:
    print(f"Analyzing {ticker}...", end=" ", flush=True)
    try:
        df = load_ticker(ticker)
        analysis = analyze_intraday_character(df, ticker)
        all_analysis.append(analysis)
        print(f"OK ({len(analysis)} days)")
    except Exception as e:
        print(f"FAILED: {e}")

all_df = pd.concat(all_analysis, ignore_index=True)
all_df['month'] = pd.Categorical(all_df['month'], categories=['Apr', 'May', 'Jun'], ordered=True)

# ── 1. Up/Down Asymmetry Shift ───────────────────────────────────────
print("\n" + "=" * 90)
print("1. UP/DOWN ASYMMETRY (Morning 6:00-7:30 PST)")
print("   Ratio > 1 = up moves bigger than down moves (good for long strategy)")
print("=" * 90)

print(f"\n  {'Ticker':<8s}  {'Apr U/D':>8s}  {'May U/D':>8s}  {'Jun U/D':>8s}  {'Apr Up%':>8s}  {'Apr Dn%':>8s}  {'Jun Up%':>8s}  {'Jun Dn%':>8s}")
print(f"  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

for ticker in ALL_TICKERS:
    t_df = all_df[all_df['ticker'] == ticker]
    vals = {}
    ups = {}
    dns = {}
    for m in ['Apr', 'May', 'Jun']:
        m_df = t_df[t_df['month'] == m]
        vals[m] = m_df['updown_ratio'].median() if len(m_df) > 0 else np.nan
        ups[m] = m_df['max_up'].mean() if len(m_df) > 0 else np.nan
        dns[m] = m_df['max_dn'].mean() if len(m_df) > 0 else np.nan

    print(f"  {ticker:<8s}  {vals.get('Apr',0):8.2f}  {vals.get('May',0):8.2f}  {vals.get('Jun',0):8.2f}"
          f"  {ups.get('Apr',0):7.2f}%  {dns.get('Apr',0):7.2f}%  {ups.get('Jun',0):7.2f}%  {dns.get('Jun',0):7.2f}%")

# Aggregate
print(f"\n  AGGREGATE:")
for m in ['Apr', 'May', 'Jun']:
    m_df = all_df[all_df['month'] == m]
    ratio = m_df['updown_ratio'].median()
    up = m_df['max_up'].mean()
    dn = m_df['max_dn'].mean()
    print(f"    {m}: Median U/D ratio = {ratio:.2f}  |  Avg Up = {up:.2f}%  |  Avg Down = {dn:.2f}%")

# ── 2. Simulated Trade Performance ───────────────────────────────────
print("\n" + "=" * 90)
print("2. SIMULATED LIMIT ENTRY TRADE (buy on -1% dip, hold to 7:30)")
print("=" * 90)

for ticker in ALL_TICKERS:
    t_df = all_df[all_df['ticker'] == ticker]
    print(f"\n  {ticker}:")
    for m in ['Apr', 'May', 'Jun']:
        m_df = t_df[t_df['month'] == m]
        filled = m_df[m_df['trade_filled']]
        n_days = len(m_df)
        n_filled = len(filled)
        fill_rate = n_filled / n_days * 100 if n_days > 0 else 0

        if len(filled) > 0:
            avg_pnl = filled['pnl_at_close'].mean()
            win_rate = (filled['pnl_at_close'] > 0).mean() * 100
            avg_gain = filled[filled['pnl_at_close'] > 0]['pnl_at_close'].mean() if (filled['pnl_at_close'] > 0).any() else 0
            avg_loss = filled[filled['pnl_at_close'] <= 0]['pnl_at_close'].mean() if (filled['pnl_at_close'] <= 0).any() else 0
            sl2_rate = filled['sl_2pct_hit'].mean() * 100 if filled['sl_2pct_hit'].notna().any() else 0
            sl3_rate = filled['sl_3pct_hit'].mean() * 100 if filled['sl_3pct_hit'].notna().any() else 0
            avg_dd = filled['max_dd_from_entry'].mean()
            avg_maxgain = filled['max_gain_from_entry'].mean()
        else:
            avg_pnl = win_rate = avg_gain = avg_loss = sl2_rate = sl3_rate = avg_dd = avg_maxgain = 0

        print(f"    {m}: Fill {fill_rate:4.1f}% ({n_filled}/{n_days}) | PnL {avg_pnl:+.2f}% | WR {win_rate:.0f}% | "
              f"AvgW {avg_gain:+.2f}% AvgL {avg_loss:+.2f}% | MaxDD {avg_dd:.2f}% | MaxGain {avg_maxgain:.2f}% | "
              f"SL2% {sl2_rate:.0f}% SL3% {sl3_rate:.0f}%")

# Aggregate
print(f"\n  AGGREGATE (all tickers):")
for m in ['Apr', 'May', 'Jun']:
    m_df = all_df[all_df['month'] == m]
    filled = m_df[m_df['trade_filled']]
    n_days = len(m_df)
    n_filled = len(filled)
    fill_rate = n_filled / n_days * 100 if n_days > 0 else 0
    if len(filled) > 0:
        avg_pnl = filled['pnl_at_close'].mean()
        win_rate = (filled['pnl_at_close'] > 0).mean() * 100
        avg_dd = filled['max_dd_from_entry'].mean()
        avg_maxgain = filled['max_gain_from_entry'].mean()
        sl2_rate = filled['sl_2pct_hit'].mean() * 100
        sl3_rate = filled['sl_3pct_hit'].mean() * 100
    else:
        avg_pnl = win_rate = avg_dd = avg_maxgain = sl2_rate = sl3_rate = 0
    print(f"    {m}: Fill {fill_rate:4.1f}% ({n_filled}/{n_days}) | PnL {avg_pnl:+.2f}% | WR {win_rate:.0f}% | "
          f"MaxDD {avg_dd:.2f}% | MaxGain {avg_maxgain:.2f}% | SL2% {sl2_rate:.0f}% SL3% {sl3_rate:.0f}%")

# ── 3. Intraday Sequence Analysis ────────────────────────────────────
print("\n" + "=" * 90)
print("3. INTRADAY SEQUENCE (Dip-before-spike vs Spike-then-crash)")
print("   'Dip before spike' is favorable for limit entry strategy")
print("=" * 90)

for m in ['Apr', 'May', 'Jun']:
    m_df = all_df[all_df['month'] == m]
    dbs = m_df['dip_before_spike'].mean() * 100
    vrec = m_df['v_recovery'].mean() * 100
    cont = m_df['continuation_dn'].mean() * 100
    print(f"  {m}: Dip-before-spike {dbs:.1f}% | V-recovery {vrec:.1f}% | Continuation-down {cont:.1f}%")

print(f"\n  Per-ticker dip-before-spike % (favorable pattern):")
print(f"  {'Ticker':<8s}  {'Apr':>8s}  {'May':>8s}  {'Jun':>8s}  {'Δ Jun-AM':>10s}")
for ticker in ALL_TICKERS:
    t_df = all_df[all_df['ticker'] == ticker]
    vals = {}
    for m in ['Apr', 'May', 'Jun']:
        m_df = t_df[t_df['month'] == m]
        vals[m] = m_df['dip_before_spike'].mean() * 100 if len(m_df) > 0 else np.nan
    am = np.nanmean([vals.get('Apr', np.nan), vals.get('May', np.nan)])
    j = vals.get('Jun', np.nan)
    delta = j - am if not np.isnan(am) and not np.isnan(j) else np.nan
    d_str = f"{delta:+.1f}pp" if not np.isnan(delta) else "N/A"
    print(f"  {ticker:<8s}  {vals.get('Apr',0):7.1f}%  {vals.get('May',0):7.1f}%  {vals.get('Jun',0):7.1f}%  {d_str:>10s}")

# ── 4. Segment-by-Segment Morning Breakdown ──────────────────────────
print("\n" + "=" * 90)
print("4. MORNING SEGMENT BREAKDOWN")
print("   Seg1: 6:00-6:30  |  Seg2: 6:30-7:00  |  Seg3: 7:00-7:30")
print("=" * 90)

for m in ['Apr', 'May', 'Jun']:
    m_df = all_df[all_df['month'] == m]
    s1 = m_df['seg1_move'].mean()
    s2 = m_df['seg2_move'].mean()
    s3 = m_df['seg3_move'].mean()
    print(f"  {m}: Seg1 {s1:+.3f}%  |  Seg2 {s2:+.3f}%  |  Seg3 {s3:+.3f}%  |  Total {s1+s2+s3:+.3f}%")

# Per ticker segment analysis
print(f"\n  Per-ticker Seg1 (6:00-6:30) move — this is where limit orders fill:")
print(f"  {'Ticker':<8s}  {'Apr':>8s}  {'May':>8s}  {'Jun':>8s}")
for ticker in ALL_TICKERS:
    t_df = all_df[all_df['ticker'] == ticker]
    vals = {}
    for m in ['Apr', 'May', 'Jun']:
        m_df = t_df[t_df['month'] == m]
        vals[m] = m_df['seg1_move'].mean() if len(m_df) > 0 else np.nan
    print(f"  {ticker:<8s}  {vals.get('Apr',0):+7.3f}%  {vals.get('May',0):+7.3f}%  {vals.get('Jun',0):+7.3f}%")

# ── 5. Close/High Deterioration (Giving Back Gains) ──────────────────
print("\n" + "=" * 90)
print("5. AFTERNOON REVERSAL (% return from morning close to EOD)")
print("   Negative = morning gains given back")
print("=" * 90)

print(f"  {'Ticker':<8s}  {'Apr':>8s}  {'May':>8s}  {'Jun':>8s}")
for ticker in ALL_TICKERS:
    t_df = all_df[all_df['ticker'] == ticker]
    vals = {}
    for m in ['Apr', 'May', 'Jun']:
        m_df = t_df[t_df['month'] == m]
        vals[m] = m_df['aftn_reversal'].mean() if len(m_df) > 0 else np.nan
    print(f"  {ticker:<8s}  {vals.get('Apr',0):+7.3f}%  {vals.get('May',0):+7.3f}%  {vals.get('Jun',0):+7.3f}%")

agg = {}
for m in ['Apr', 'May', 'Jun']:
    agg[m] = all_df[all_df['month'] == m]['aftn_reversal'].mean()
print(f"\n  AGGREGATE: Apr {agg['Apr']:+.3f}%  May {agg['May']:+.3f}%  Jun {agg['Jun']:+.3f}%")

# ── 6. Full Day Return Context ───────────────────────────────────────
print("\n" + "=" * 90)
print("6. FULL DAY RETURN (Open to Close)")
print("=" * 90)

print(f"  {'Ticker':<8s}  {'Apr':>8s}  {'May':>8s}  {'Jun':>8s}")
for ticker in ALL_TICKERS:
    t_df = all_df[all_df['ticker'] == ticker]
    vals = {}
    for m in ['Apr', 'May', 'Jun']:
        m_df = t_df[t_df['month'] == m]
        vals[m] = m_df['full_day_return'].mean() if len(m_df) > 0 else np.nan
    print(f"  {ticker:<8s}  {vals.get('Apr',0):+7.3f}%  {vals.get('May',0):+7.3f}%  {vals.get('Jun',0):+7.3f}%")

agg = {}
for m in ['Apr', 'May', 'Jun']:
    agg[m] = all_df[all_df['month'] == m]['full_day_return'].mean()
print(f"\n  AGGREGATE: Apr {agg['Apr']:+.3f}%  May {agg['May']:+.3f}%  Jun {agg['Jun']:+.3f}%")

# ── 7. Time to Peak After Entry ──────────────────────────────────────
print("\n" + "=" * 90)
print("7. TIME TO PEAK AFTER LIMIT ENTRY (minutes)")
print("   Longer time = slower recovery, higher chance of SL hit")
print("=" * 90)

for m in ['Apr', 'May', 'Jun']:
    m_df = all_df[all_df['month'] == m]
    filled = m_df[m_df['trade_filled'] & m_df['time_to_high_min'].notna()]
    if len(filled) > 0:
        avg = filled['time_to_high_min'].mean()
        med = filled['time_to_high_min'].median()
        print(f"  {m}: Mean {avg:.1f} min  |  Median {med:.1f} min")

# ── FINAL SYNTHESIS ──────────────────────────────────────────────────
print("\n" + "=" * 90)
print("SYNTHESIS: TOP CAUSES OF STRATEGY DEGRADATION")
print("=" * 90)

# Compute key deltas
filled_apr = all_df[(all_df['month'] == 'Apr') & all_df['trade_filled']]
filled_may = all_df[(all_df['month'] == 'May') & all_df['trade_filled']]
filled_jun = all_df[(all_df['month'] == 'Jun') & all_df['trade_filled']]
filled_am = pd.concat([filled_apr, filled_may])

if len(filled_am) > 0 and len(filled_jun) > 0:
    pnl_am = filled_am['pnl_at_close'].mean()
    pnl_jun = filled_jun['pnl_at_close'].mean()
    wr_am = (filled_am['pnl_at_close'] > 0).mean() * 100
    wr_jun = (filled_jun['pnl_at_close'] > 0).mean() * 100
    dd_am = filled_am['max_dd_from_entry'].mean()
    dd_jun = filled_jun['max_dd_from_entry'].mean()
    sl2_am = filled_am['sl_2pct_hit'].mean() * 100
    sl2_jun = filled_jun['sl_2pct_hit'].mean() * 100
    sl3_am = filled_am['sl_3pct_hit'].mean() * 100
    sl3_jun = filled_jun['sl_3pct_hit'].mean() * 100
    gain_am = filled_am['max_gain_from_entry'].mean()
    gain_jun = filled_jun['max_gain_from_entry'].mean()

    updn_am = all_df[all_df['month'].isin(['Apr', 'May'])]['updown_ratio'].median()
    updn_jun = all_df[all_df['month'] == 'Jun']['updown_ratio'].median()

    maxdn_am = all_df[all_df['month'].isin(['Apr', 'May'])]['max_dn'].mean()
    maxdn_jun = all_df[all_df['month'] == 'Jun']['max_dn'].mean()

    print(f"""
  1. SIMULATED PnL COLLAPSE:
     Apr-May avg PnL per trade: {pnl_am:+.3f}%  ->  Jun: {pnl_jun:+.3f}%
     Win rate: {wr_am:.1f}%  ->  {wr_jun:.1f}%

  2. DOWNSIDE EXPLOSION (Max Down from Open in Morning):
     Apr-May: {maxdn_am:.2f}%  ->  Jun: {maxdn_jun:.2f}%  ({(maxdn_jun-maxdn_am)/maxdn_am*100:+.1f}%)
     This means deeper drawdowns AFTER entry, hitting stop-losses more often.

  3. STOP-LOSS HIT RATE INCREASE:
     SL at 2%: {sl2_am:.1f}%  ->  {sl2_jun:.1f}%  ({sl2_jun-sl2_am:+.1f}pp)
     SL at 3%: {sl3_am:.1f}%  ->  {sl3_jun:.1f}%  ({sl3_jun-sl3_am:+.1f}pp)

  4. UP/DOWN ASYMMETRY SHIFT:
     Median ratio: {updn_am:.2f}  ->  {updn_jun:.2f}  ({(updn_jun-updn_am)/updn_am*100:+.1f}%)
     (>1 = up moves dominate = good for long; <1 = down moves dominate = bad)

  5. MAX DRAWDOWN FROM ENTRY vs MAX GAIN:
     Avg MaxDD: {dd_am:.2f}%  ->  {dd_jun:.2f}%  ({(dd_jun-dd_am)/dd_am*100:+.1f}%)
     Avg MaxGain: {gain_am:.2f}%  ->  {gain_jun:.2f}%  ({(gain_jun-gain_am)/gain_am*100:+.1f}%)
     Risk/Reward (Gain/DD): {gain_am/dd_am:.2f}  ->  {gain_jun/dd_jun:.2f}
""")

# Worst individual tickers
print("\n  WORST TICKERS BY PnL DETERIORATION (limit entry sim):")
print(f"  {'Ticker':<8s}  {'AM PnL':>8s}  {'Jun PnL':>8s}  {'AM WR':>7s}  {'Jun WR':>7s}  {'AM DD':>7s}  {'Jun DD':>7s}")
ticker_changes = []
for ticker in ALL_TICKERS:
    t_df = all_df[all_df['ticker'] == ticker]
    f_am = t_df[(t_df['month'].isin(['Apr', 'May'])) & t_df['trade_filled']]
    f_jun = t_df[(t_df['month'] == 'Jun') & t_df['trade_filled']]
    if len(f_am) > 0 and len(f_jun) > 0:
        pam = f_am['pnl_at_close'].mean()
        pjun = f_jun['pnl_at_close'].mean()
        wam = (f_am['pnl_at_close'] > 0).mean() * 100
        wjun = (f_jun['pnl_at_close'] > 0).mean() * 100
        dam = f_am['max_dd_from_entry'].mean()
        djun = f_jun['max_dd_from_entry'].mean()
        ticker_changes.append((ticker, pam, pjun, wam, wjun, dam, djun))

ticker_changes.sort(key=lambda x: x[2] - x[1])  # Sort by PnL deterioration
for ticker, pam, pjun, wam, wjun, dam, djun in ticker_changes:
    print(f"  {ticker:<8s}  {pam:+7.3f}%  {pjun:+7.3f}%  {wam:6.1f}%  {wjun:6.1f}%  {dam:6.2f}%  {djun:6.2f}%")

print("\n" + "=" * 90)
print("END OF DEEP DIVE")
print("=" * 90)
