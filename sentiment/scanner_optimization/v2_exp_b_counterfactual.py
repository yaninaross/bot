# %% EXP GROUP B: Counterfactual Universe (S3) + Cross-Sectional Ranking (S4)
# Loads from v2_features_full.csv. S3 makes its own Polygon API calls.

import os, sys, json, warnings, re
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib
try:
    get_ipython()
except NameError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

try:
    from polygon import RESTClient as PolygonClient
except ImportError:
    import subprocess; subprocess.check_call([sys.executable, "-m", "pip", "install", "polygon-api-client", "--quiet"])
    from polygon import RESTClient as PolygonClient
try:
    import requests as _requests
except ImportError:
    import subprocess; subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "--quiet"])
    import requests as _requests

try:
    OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    OUTPUT_DIR = os.getcwd()
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
ALL_DATES = ['2026-09-08', '2026-09-09', '2026-09-10', '2026-09-11']
FOLD_A = ['2026-09-08', '2026-09-09']
FOLD_B = ['2026-09-10', '2026-09-11']
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "2YBWWKHpvnM6OX5qVhd1MxhsRCorLkFs")
COUNTERFACTUAL_MAX_PER_DAY = 40
COUNTERFACTUAL_MIN_GAP_PCT = 3.0
MARKET_OPEN_ET = "09:30"; MARKET_CLOSE_ET = "16:00"
_W = 100

import pytz
ET = pytz.timezone('America/New_York')

df = pd.read_csv(f'{RESULTS_DIR}/v2_features_full.csv')
spy_daily = {}
if os.path.exists(f'{RESULTS_DIR}/v2_spy_daily.json'):
    with open(f'{RESULTS_DIR}/v2_spy_daily.json') as f: spy_daily = json.load(f)
print(f"Loaded {len(df)} trades")

def robust_stats(values):
    v = pd.Series(values).dropna(); n = len(v)
    if n == 0: return {'n': 0, 'avg': np.nan, 'med': np.nan, 'trim': np.nan, 'se': np.nan, 'wr': np.nan}
    avg = float(v.mean()); med = float(v.median()); se = float(v.std() / np.sqrt(n)) if n > 1 else 0.0
    wr = float((v > 0).sum() / n * 100)
    trim = float(v.sort_values().iloc[max(1, int(n*0.1)):-max(1, int(n*0.1))].mean()) if n >= 10 else avg
    return {'n': n, 'avg': avg, 'med': med, 'trim': trim, 'se': se, 'wr': wr}

def fold_robust(values, dates):
    v = pd.Series(values).dropna(); d = pd.Series(dates).loc[v.index]
    sa = robust_stats(v[d.isin(FOLD_A)]); sb = robust_stats(v[d.isin(FOLD_B)])
    cv = (sa['n'] > 0 and sb['n'] > 0 and sa['avg'] > 0 and sb['avg'] > 0)
    return {'all': robust_stats(v), 'A': sa, 'B': sb, 'cv': cv}

poly = PolygonClient(POLYGON_API_KEY)

def _fetch_minute_bars(ticker, eval_date):
    try:
        bars = list(poly.get_aggs(ticker=ticker, multiplier=1, timespan="minute",
                                   from_=eval_date, to=eval_date, limit=500))
        if not bars: return None
        bar_map = {}
        for b in bars:
            ts = pd.Timestamp(b.timestamp, unit='ms', tz='UTC').tz_convert(ET)
            bar_map[ts.strftime('%H:%M')] = {
                'open': float(b.open), 'high': float(b.high),
                'low': float(b.low), 'close': float(b.close), 'volume': int(b.volume)}
        return bar_map
    except: return None

# ==============================================================================
# EXP S3: COUNTERFACTUAL UNIVERSE
# ==============================================================================
print(f"\n{'='*_W}")
print(f"  EXP S3: COUNTERFACTUAL UNIVERSE")
print(f"  Do scanner-flagged gappers outperform ALL gappers?")
print(f"{'='*_W}")

prev_dates = {}
for dt in ALL_DATES:
    dt_obj = datetime.strptime(dt, '%Y-%m-%d')
    for offset in range(1, 7):
        prev = (dt_obj - timedelta(days=offset)).strftime('%Y-%m-%d')
        if prev in spy_daily: prev_dates[dt] = prev; break
    if dt not in prev_dates:
        prev_dates[dt] = (dt_obj - timedelta(days=1)).strftime('%Y-%m-%d')

all_gappers = []
scanner_tickers = set(zip(df['ticker'], df['eval_date']))

for dt in ALL_DATES:
    prev_dt = prev_dates.get(dt)
    print(f"\n  Fetching grouped daily for {dt} (prev={prev_dt})...")
    try:
        url_today = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{dt}"
        url_prev = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{prev_dt}"
        resp_today = _requests.get(url_today, params={"apiKey": POLYGON_API_KEY, "adjusted": "true"}, timeout=30)
        resp_prev = _requests.get(url_prev, params={"apiKey": POLYGON_API_KEY, "adjusted": "true"}, timeout=30)
        if resp_today.status_code != 200 or resp_prev.status_code != 200:
            print(f"    API error: {resp_today.status_code}, {resp_prev.status_code}"); continue
        today_data = {r['T']: r for r in resp_today.json().get('results', [])}
        prev_data = {r['T']: r for r in resp_prev.json().get('results', [])}
        n_gappers = 0
        for tk, td_r in today_data.items():
            if not re.match(r'^[A-Z]{1,5}$', tk): continue
            if tk not in prev_data: continue
            prev_c = prev_data[tk].get('c', 0); today_o = td_r.get('o', 0)
            if prev_c <= 0 or today_o <= 0: continue
            gap = (today_o - prev_c) / prev_c * 100
            if abs(gap) < COUNTERFACTUAL_MIN_GAP_PCT: continue
            flagged = (tk, dt) in scanner_tickers
            all_gappers.append({
                'ticker': tk, 'eval_date': dt, 'gap_pct': gap,
                'open': today_o, 'close': td_r.get('c', 0), 'prev_close': prev_c,
                'flagged': flagged,
                'day_return': (td_r.get('c', today_o) - today_o) / today_o * 100,
            })
            n_gappers += 1
        print(f"    {n_gappers} gappers (>{COUNTERFACTUAL_MIN_GAP_PCT}% gap)")
    except Exception as e:
        print(f"    Error: {e}")

df_gap = pd.DataFrame(all_gappers) if all_gappers else pd.DataFrame()

if len(df_gap) > 0:
    non_flagged = df_gap[~df_gap['flagged']].copy()
    non_flagged_sorted = non_flagged.reindex(non_flagged['gap_pct'].abs().sort_values(ascending=False).index)
    print(f"\n  Total gappers: {len(df_gap)} ({df_gap['flagged'].sum()} flagged, {(~df_gap['flagged']).sum()} not flagged)")

    nf_bars = {}
    nf_to_fetch = non_flagged_sorted.head(COUNTERFACTUAL_MAX_PER_DAY * len(ALL_DATES))
    print(f"  Fetching minute bars for {len(nf_to_fetch)} non-flagged gappers...")
    for i, (_, row) in enumerate(nf_to_fetch.iterrows()):
        bm = _fetch_minute_bars(row['ticker'], row['eval_date'])
        if bm: nf_bars[(row['ticker'], row['eval_date'])] = bm
        if (i + 1) % 20 == 0: print(f"    {i+1}/{len(nf_to_fetch)}")

    nf_pnl = []
    for _, row in nf_to_fetch.iterrows():
        bm = nf_bars.get((row['ticker'], row['eval_date']))
        if bm is None: continue
        entry_bar = bm.get('09:30')
        if not entry_bar:
            cands = sorted(k for k in bm if k >= '09:30')
            if not cands: continue
            entry_bar = bm[cands[0]]
        ep_nf = entry_bar['open']
        if ep_nf <= 0: continue
        bar_60 = None
        for m in range(55, 66):
            target = f"{(9*60+30+m)//60:02d}:{(9*60+30+m)%60:02d}"
            if target in bm: bar_60 = bm[target]; break
        if bar_60:
            raw_ret = (bar_60['close'] - ep_nf) / ep_nf * 100
            nf_pnl.append({
                'ticker': row['ticker'], 'eval_date': row['eval_date'],
                'gap_pct': row['gap_pct'], 'raw_60m': raw_ret, 'short_60m': -raw_ret,
                'gap_fade_60m': -raw_ret if row['gap_pct'] > 0 else raw_ret,
            })
    df_nf_pnl = pd.DataFrame(nf_pnl) if nf_pnl else pd.DataFrame()

    print(f"\n  Counterfactual Comparison:")
    print(f"  {'':─<{_W-4}}")
    print(f"  {'Group':<30} {'n':>5} {'Avg PnL':>9} {'Med':>9} {'WR':>6}")
    print(f"  {'':─<{_W-4}}")

    f_s = robust_stats(df['pnl_60m_nosl'])
    print(f"  {'Flagged: all signals':<30} {f_s['n']:>5} {f_s['avg']:>+8.3f}% {f_s['med']:>+8.3f}% {f_s['wr']:>5.0f}%")

    flagged_shorts = df[(df['direction'] == 'SHORT') & (df['gap_pct'] > COUNTERFACTUAL_MIN_GAP_PCT)]
    if len(flagged_shorts) > 0:
        fs_s = robust_stats(flagged_shorts['pnl_60m_nosl'])
        print(f"  {'Flagged: SHORT gap>3%':<30} {fs_s['n']:>5} {fs_s['avg']:>+8.3f}% {fs_s['med']:>+8.3f}% {fs_s['wr']:>5.0f}%")

    if len(df_nf_pnl) > 0:
        nf_s = robust_stats(df_nf_pnl['gap_fade_60m'])
        print(f"  {'Non-flagged: gap-fade':<30} {nf_s['n']:>5} {nf_s['avg']:>+8.3f}% {nf_s['med']:>+8.3f}% {nf_s['wr']:>5.0f}%")
        nf_sh = robust_stats(df_nf_pnl[df_nf_pnl['gap_pct'] > 0]['short_60m'])
        print(f"  {'Non-flagged: SHORT gap-ups':<30} {nf_sh['n']:>5} {nf_sh['avg']:>+8.3f}% {nf_sh['med']:>+8.3f}% {nf_sh['wr']:>5.0f}%")

    gap_up = df_gap[df_gap['gap_pct'] > 0]; gap_dn = df_gap[df_gap['gap_pct'] < 0]
    print(f"\n  All gappers day return (gap-fade = SHORT gap-ups, LONG gap-downs):")
    for lbl, sub in [('Gap-up', gap_up), ('Gap-down', gap_dn), ('All', df_gap)]:
        if len(sub) > 0:
            fade_ret = -sub['day_return'] if lbl == 'Gap-up' else sub['day_return']
            s = robust_stats(fade_ret if lbl != 'All' else sub['day_return'])
            fl = sub['flagged'].sum()
            print(f"    {lbl:<12} n={len(sub):>4} (flagged={fl:>3})  avg={s['avg']:+.3f}%  med={s['med']:+.3f}%")

    fig_s3, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig_s3.suptitle('EXP S3: Counterfactual Universe', fontsize=13, fontweight='bold')
    ax = axes[0]
    fl = df_gap[df_gap['flagged']]; nfl = df_gap[~df_gap['flagged']]
    ax.scatter(fl['gap_pct'], -fl['day_return'], c='#2ecc71', s=60, alpha=0.8,
               label=f'Flagged (n={len(fl)})', edgecolors='black', lw=0.5)
    ax.scatter(nfl['gap_pct'], -nfl['day_return'], c='#e74c3c', s=20, alpha=0.3,
               label=f'Not flagged (n={len(nfl)})', marker='x')
    ax.set_xlabel('Gap %'); ax.set_ylabel('Gap-Fade Return (intraday)')
    ax.set_title('Flagged vs Non-flagged Gappers'); ax.axhline(y=0, color='black', lw=0.8)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    if len(df_nf_pnl) > 0 and len(flagged_shorts) > 0:
        bins_cf = np.linspace(-10, 10, 30)
        ax.hist(flagged_shorts['pnl_60m_nosl'].dropna(), bins=bins_cf, alpha=0.7,
                color='#2ecc71', label=f'Flagged SHORT (n={len(flagged_shorts)})')
        ax.hist(df_nf_pnl['short_60m'], bins=bins_cf, alpha=0.4,
                color='#e74c3c', label=f'Non-flagged SHORT (n={len(df_nf_pnl)})')
        ax.axvline(x=0, color='black', lw=0.8); ax.set_xlabel('PnL @60m %')
        ax.set_title('60m PnL Distribution'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig_s3.savefig(f'{OUTPUT_DIR}/v2_s3_counterfactual.png', dpi=150, bbox_inches='tight'); plt.show()

# ==============================================================================
# EXP S4: CROSS-SECTIONAL RANKING
# ==============================================================================
print(f"\n\n{'='*_W}")
print(f"  EXP S4: CROSS-SECTIONAL RANKING — Daily tercile L/S")
print(f"{'='*_W}")

rank_features = ['remaining_move_pct', 'expected_move_pct', 'log_vol_ratio', 'gap_pct', 'abs_gap_pct']
print(f"\n  For each day, rank signals by feature, go LONG top tercile / SHORT bottom tercile.")
print(f"  Dollar-neutral — removes beta and day effect.")

for feat in rank_features:
    if feat not in df.columns: continue
    daily_rets = []
    for dt in ALL_DATES:
        day_df = df[(df['eval_date'] == dt) & df[feat].notna() & df['pnl_60m_nosl'].notna()].copy()
        if len(day_df) < 6: continue
        day_df['rank'] = day_df[feat].rank(pct=True)
        top = day_df[day_df['rank'] >= 0.67]['pnl_60m_nosl']
        bot = day_df[day_df['rank'] <= 0.33]['pnl_60m_nosl']
        if len(top) > 0 and len(bot) > 0:
            daily_rets.append({'date': dt, 'ls_return': top.mean() - bot.mean(),
                               'top_avg': top.mean(), 'bot_avg': bot.mean(),
                               'n_top': len(top), 'n_bot': len(bot)})
    if daily_rets:
        df_ls = pd.DataFrame(daily_rets)
        avg_ls = df_ls['ls_return'].mean()
        all_pos = (df_ls['ls_return'] > 0).all()
        print(f"\n  {feat}:")
        print(f"    Avg L/S return: {avg_ls:+.3f}%  ({'all days positive' if all_pos else 'mixed'})")
        for _, r in df_ls.iterrows():
            print(f"      {r['date']}: L/S={r['ls_return']:+.3f}%  "
                  f"(top={r['top_avg']:+.3f}% n={r['n_top']}, bot={r['bot_avg']:+.3f}% n={r['n_bot']})")

print(f"\n  Group B results saved. DONE.")
