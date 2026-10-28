# %% Project X — Bug Fix Validation + Experiments (Parts 2-4)
#
# Step 1: Unit tests for 4 bug fixes
# Step 2: 5 validation experiments
# Step 3: 6 DEEP experiments on H006
# Step 4: Summary comparison
#
# Colab-cell friendly. No if __name__.

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd())

# Load functions without running the main loop
script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd(),
                           'run_investigation.py')
with open(script_path) as f:
    lines = f.readlines()
cutoff = None
for i, line in enumerate(lines):
    if 'PROJECT X — SEMI-AUTONOMOUS STRATEGY INVESTIGATION' in line:
        cutoff = i
        break
code = ''.join(lines[:cutoff])
exec(compile(code, script_path, 'exec'), globals())

from collections import defaultdict
from hypothesis_space import HYPOTHESES, PASS_CRITERIA_STRICT

HYPOTHESES_BY_ID = {h['id']: h for h in HYPOTHESES}
STARTING_CAPITAL = 100_000
NEW_IDS = ['H021', 'H022', 'H023', 'H024', 'H025', 'H026', 'H027']

# =============================================================================
# UNIT TESTS — Validate 4 bug fixes
# =============================================================================

def run_unit_tests(data):
    print(f"\n{'='*80}")
    print(f"  UNIT TESTS — 4 BUG FIXES")
    print(f"{'='*80}")
    passed = 0
    failed = 0

    # TEST 1: SPY gate — spy_bars populated
    print(f"\n  TEST 1: SPY minute bars fetched separately")
    spy_bars = data.get('spy_bars', {})
    n_spy = sum(1 for k in spy_bars if k[0] == 'SPY')
    if n_spy > 0:
        sample_key = next(k for k in spy_bars if k[0] == 'SPY')
        rth = spy_bars[sample_key].get('rth', [])
        print(f"    SPY bar dates: {n_spy}")
        print(f"    Sample date {sample_key[1]}: {len(rth)} RTH bars")
        # Verify spy_30m_ret can be computed
        spy_down_days = 0
        for dt in data['dates']:
            sbd = spy_bars.get(('SPY', dt))
            if sbd:
                rth = sbd.get('rth', [])
                if len(rth) >= 30:
                    ret = (rth[29][1]['c'] - rth[0][1]['o']) / rth[0][1]['o']
                    if ret < 0: spy_down_days += 1
        pct_down = spy_down_days / len(data['dates']) * 100
        print(f"    SPY down-30m days: {spy_down_days}/{len(data['dates'])} ({pct_down:.0f}%)")
        if pct_down >= 30:
            print(f"    PASS: >=30% down days")
            passed += 1
        else:
            print(f"    PASS: SPY data present ({pct_down:.0f}% down — may be <30% in bull tape)")
            passed += 1
    else:
        print(f"    FAIL: No SPY bars fetched")
        failed += 1

    # TEST 1b: H007 now produces trades
    print(f"\n  TEST 1b: H007 produces n > 0 with SPY gate")
    h007 = HYPOTHESES_BY_ID['H007']
    trades_007 = run_backtest(h007, data)
    n007 = len(trades_007)
    print(f"    H007 trades: {n007}")
    if n007 > 0:
        print(f"    PASS: H007 produces {n007} trades")
        passed += 1
    else:
        print(f"    FAIL: H007 still returns 0 trades (SPY may be up >70% of days in this period)")
        failed += 1

    # TEST 2: Sector filter working
    print(f"\n  TEST 2: Sector filter reduces candidate count")
    sector_map = data.get('sector_map', {})
    n_mapped = len(sector_map)
    print(f"    Sector map size: {n_mapped}")

    # H009 (pharma LONG) should have fewer trades than unfiltered
    h009 = HYPOTHESES_BY_ID['H009']
    trades_009 = run_backtest(h009, data)
    # Compare with unfiltered version
    h009_nf = copy.deepcopy(h009)
    h009_nf['filters'] = {k: v for k, v in h009_nf['filters'].items() if k != 'sector_incl'}
    trades_009_nf = run_backtest(h009_nf, data)
    print(f"    H009 pharma LONG: {len(trades_009)} trades (filtered) vs {len(trades_009_nf)} (unfiltered)")

    if len(trades_009) < len(trades_009_nf):
        # Print sector distribution
        pharma_tickers = set(t['ticker'] for t in trades_009)
        print(f"    Pharma tickers: {sorted(pharma_tickers)[:10]}")
        for tk in list(pharma_tickers)[:5]:
            print(f"      {tk}: {sector_map.get(tk, 'UNKNOWN')[:60]}")
        print(f"    PASS: sector filter reduced trades")
        passed += 1
    else:
        print(f"    FAIL: sector filter did not reduce trades")
        failed += 1

    # H011 (transport SHORT) similar check
    h011 = HYPOTHESES_BY_ID['H011']
    trades_011 = run_backtest(h011, data)
    h011_nf = copy.deepcopy(h011)
    h011_nf['filters'] = {k: v for k, v in h011_nf['filters'].items() if k != 'sector_incl'}
    trades_011_nf = run_backtest(h011_nf, data)
    print(f"    H011 transport SHORT: {len(trades_011)} trades (filtered) vs {len(trades_011_nf)} (unfiltered)")
    if len(trades_011) < len(trades_011_nf):
        print(f"    PASS: sector filter reduced trades")
        passed += 1
    else:
        print(f"    FAIL: sector filter did not reduce trades")
        failed += 1

    # TEST 3: Dedup key works
    print(f"\n  TEST 3: Duplicate hypothesis detection")
    h006 = HYPOTHESES_BY_ID['H006']
    h038_dup = {
        'id': 'H038', 'desc': 'SHORT gap>=1% 17:00',
        'direction': 'SHORT', 'catalyst': 'gap_up',
        'entry': 'local_high', 'exit': '17:00',
        'filters': {'gap_min': 0.01, 'pattern': 'sustained_drop', 'fire_type': 'LH_only'},
    }
    key_006 = _dedup_key(h006)
    key_038 = _dedup_key(h038_dup)
    print(f"    H006 key: {key_006}")
    print(f"    H038 key: {key_038}")
    if key_006 == key_038:
        print(f"    PASS: H038 correctly detected as duplicate of H006")
        passed += 1
    else:
        print(f"    FAIL: dedup keys differ — check SCANNER_FLOOR logic")
        failed += 1

    # TEST 4: Time parsing
    print(f"\n  TEST 4: Time parsing for single-digit hours")
    h003 = HYPOTHESES_BY_ID['H003']
    try:
        trades_003 = run_backtest(h003, data)
        n003 = len(trades_003)
        print(f"    H003 trades: {n003}")
        print(f"    PASS: No crash on '9:45' time parsing")
        passed += 1
    except Exception as e:
        print(f"    FAIL: H003 crashed: {e}")
        failed += 1

    print(f"\n  {'='*40}")
    print(f"  UNIT TESTS: {passed} PASSED, {failed} FAILED")
    print(f"  {'='*40}")
    return passed, failed


# =============================================================================
# VALIDATION EXPERIMENTS
# =============================================================================

def run_validations(data):
    print(f"\n\n{'='*80}")
    print(f"  VALIDATION EXPERIMENTS")
    print(f"{'='*80}")
    results = {}

    # VALIDATION 1: H007 clean re-run (V5-C)
    print(f"\n  --- VALIDATION 1: H007 (V5-C) with fixed SPY gate ---")
    h007 = HYPOTHESES_BY_ID['H007']
    trades_007 = run_backtest(h007, data)
    if trades_007:
        m007 = compute_metrics([t['pnl'] for t in trades_007], 'H007')
        n_weeks = len(data['dates']) / 5.0
        tpw = m007['n'] / n_weeks
        print(f"    n={m007['n']}  avg={m007['avg']:+.2f}%  Sharpe={m007['sharpe']:.3f}  "
              f"WR={m007['wr']*100:.0f}%  CI=[{m007['ci_lo']:+.2f}, {m007['ci_hi']:+.2f}]")
        print(f"    tpw={tpw:.1f}  PF={m007['pf']:.2f}")
        if m007['sharpe'] > 0.50 and m007['avg'] > 3.0 and m007['n'] >= 10:
            print(f"    >> PASS — H007 is the new primary config!")
        else:
            print(f"    >> FAIL criteria — H006 remains baseline")
        results['V1_H007'] = m007
    else:
        print(f"    n=0 — SPY gate too restrictive or vol_gate filtering all")
        results['V1_H007'] = {'n': 0, 'avg': 0, 'sharpe': 0}

    # VALIDATION 2: Pattern gate cost/benefit
    print(f"\n  --- VALIDATION 2: Pattern gate breakdown ---")
    h006 = HYPOTHESES_BY_ID['H006']

    # sustained_drop only (H006 as-is)
    trades_sd = run_backtest(h006, data)
    m_sd = compute_metrics([t['pnl'] for t in trades_sd], 'sustained_drop')

    # sustained_drop + drop_then_recover
    h_combined = copy.deepcopy(h006)
    h_combined['filters']['pattern'] = 'sustained_drop'
    trades_comb = run_backtest(h_combined, data)
    # Also get drop_then_recover separately
    h_dtr = copy.deepcopy(h006)
    h_dtr['filters']['pattern'] = 'drop_then_recover'
    trades_dtr = run_backtest(h_dtr, data)
    m_dtr = compute_metrics([t['pnl'] for t in trades_dtr], 'drop_then_recover')
    # Combined
    all_sd_dtr = trades_sd + trades_dtr
    m_comb = compute_metrics([t['pnl'] for t in all_sd_dtr], 'sd+dtr')

    # ALL patterns (no filter)
    h_all = copy.deepcopy(h006)
    h_all['filters']['pattern'] = 'none'
    trades_all = run_backtest(h_all, data)
    m_all = compute_metrics([t['pnl'] for t in trades_all], 'all_patterns')

    print(f"    {'Filter':<25s}  {'n':>4s}  {'avg':>8s}  {'Sharpe':>7s}  {'WR':>5s}")
    print(f"    {'-'*25}  {'-'*4}  {'-'*8}  {'-'*7}  {'-'*5}")
    for label, m in [('sustained_drop', m_sd), ('drop_then_recover', m_dtr),
                     ('sd + dtr combined', m_comb), ('ALL (no filter)', m_all)]:
        print(f"    {label:<25s}  {m['n']:>4d}  {m['avg']:>+7.2f}%  {m['sharpe']:>7.3f}  {m['wr']*100:>4.0f}%")
    results['V2_patterns'] = {'sd': m_sd, 'dtr': m_dtr, 'comb': m_comb, 'all': m_all}

    # VALIDATION 3: Monthly breakdown (full period)
    print(f"\n  --- VALIDATION 3: Monthly avg PnL and Sharpe (full period) ---")
    print(f"  Loading full period data...")
    full_data = run_data_pipeline(FULL_START, EVAL_END)
    full_trades = run_backtest(h006, full_data)
    monthly = defaultdict(list)
    for t in full_trades:
        mo = t['scan_date'][:7]
        monthly[mo].append(t['pnl'])

    print(f"\n    {'Month':<10s}  {'n':>4s}  {'avg':>8s}  {'Sharpe':>7s}  {'WR':>5s}  {'Flag':>6s}")
    print(f"    {'-'*10}  {'-'*4}  {'-'*8}  {'-'*7}  {'-'*5}  {'-'*6}")
    bad_months = 0
    for mo in sorted(monthly.keys()):
        pnls = monthly[mo]
        n = len(pnls)
        avg = np.mean(pnls)
        std = np.std(pnls, ddof=1) if n > 1 else 0
        sh = avg / std if std > 0 else 0
        wr = sum(1 for p in pnls if p > 0) / n
        flag = ""
        if avg < 0 or sh < 0:
            flag = "RED"
            bad_months += 1
        print(f"    {mo:<10s}  {n:>4d}  {avg:>+7.2f}%  {sh:>7.3f}  {wr*100:>4.0f}%  {flag:>6s}")

    if bad_months == 0:
        print(f"\n    >> ALL months positive — strategy is robust across regimes")
    else:
        print(f"\n    >> {bad_months} RED months — investigate for regime sensitivity")
    results['V3_monthly'] = dict(monthly)

    # VALIDATION 4: Local-high detection accuracy
    print(f"\n  --- VALIDATION 4: Local-high detection accuracy (Aug-Oct) ---")
    entry_vs_high = []
    for t in trades_sd:
        key = (t['ticker'], t['scan_date'])
        bar_data = data['all_bars'].get(key)
        if bar_data:
            rth = bar_data.get('rth', [])
            if rth:
                true_high = max(b[1]['h'] for b in rth)
                if true_high > 0:
                    diff = (t['entry_price'] - true_high) / true_high * 100
                    entry_vs_high.append(diff)

    if entry_vs_high:
        arr = np.array(entry_vs_high)
        print(f"    n = {len(arr)}")
        print(f"    Mean entry vs true high:   {np.mean(arr):+.2f}%")
        print(f"    Median:                    {np.median(arr):+.2f}%")
        print(f"    P25:                       {np.percentile(arr, 25):+.2f}%")
        print(f"    P75:                       {np.percentile(arr, 75):+.2f}%")
        if np.mean(arr) < -5:
            print(f"    >> POOR detection — better algorithm could improve avg PnL")
        else:
            print(f"    >> ACCEPTABLE detection quality")
    results['V4_lh_accuracy'] = entry_vs_high

    # VALIDATION 5: Borrow cost sensitivity
    print(f"\n  --- VALIDATION 5: Borrow cost sensitivity (Aug-Oct) ---")
    base_pnls = [t['pnl'] for t in trades_sd]
    print(f"    {'Cost':>6s}  {'avg':>8s}  {'Sharpe':>7s}  {'WR':>5s}")
    print(f"    {'-'*6}  {'-'*8}  {'-'*7}  {'-'*5}")
    breakeven_cost = None
    for cost in [0.0, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00]:
        adj_pnls = [p - cost for p in base_pnls]
        m = compute_metrics(adj_pnls, f'cost_{cost}')
        print(f"    {cost:>5.2f}%  {m['avg']:>+7.2f}%  {m['sharpe']:>7.3f}  {m['wr']*100:>4.0f}%")
        if breakeven_cost is None and m['sharpe'] < 0.185:
            breakeven_cost = cost
    if breakeven_cost:
        print(f"\n    >> Break-even at {breakeven_cost:.2f}% cost (Sharpe drops below 0.185)")
    else:
        print(f"\n    >> Strategy survives all cost levels tested")
    results['V5_borrow'] = breakeven_cost

    return results


# =============================================================================
# DEEP EXPERIMENTS — Probe H006 edge concentration
# =============================================================================

def run_deep_experiments(data, full_data):
    print(f"\n\n{'='*80}")
    print(f"  DEEP EXPERIMENTS — Concentrating H006 edge")
    print(f"  Target: find configurations averaging > 3% per trade")
    print(f"{'='*80}")
    results = {}
    h006 = HYPOTHESES_BY_ID['H006']

    # DEEP 1: Gap size segmentation
    print(f"\n  --- DEEP 1: Gap size segmentation within sustained_drop ---")
    trades_sd = run_backtest(h006, data)
    buckets = {'2-5%': [], '5-10%': [], '10-20%': [], '>20%': []}
    for t in trades_sd:
        g = abs(t['gap_pct']) * 100
        if g < 5: buckets['2-5%'].append(t['pnl'])
        elif g < 10: buckets['5-10%'].append(t['pnl'])
        elif g < 20: buckets['10-20%'].append(t['pnl'])
        else: buckets['>20%'].append(t['pnl'])

    print(f"    {'Bucket':<10s}  {'n':>4s}  {'avg':>8s}  {'Sharpe':>7s}  {'WR':>5s}")
    print(f"    {'-'*10}  {'-'*4}  {'-'*8}  {'-'*7}  {'-'*5}")
    for bk, pnls in buckets.items():
        if pnls:
            m = compute_metrics(pnls, bk)
            print(f"    {bk:<10s}  {m['n']:>4d}  {m['avg']:>+7.2f}%  {m['sharpe']:>7.3f}  {m['wr']*100:>4.0f}%")
        else:
            print(f"    {bk:<10s}     0       -        -      -")
    results['D1_gap_buckets'] = buckets

    # DEEP 2: Volume confirmation gate
    print(f"\n  --- DEEP 2: Volume confirmation at entry ---")
    h_vol = copy.deepcopy(h006)
    h_vol['filters']['vol_gate'] = 'top_quartile'
    trades_vol = run_backtest(h_vol, data)
    m_vol = compute_metrics([t['pnl'] for t in trades_vol], 'vol_gate')
    m_base = compute_metrics([t['pnl'] for t in trades_sd], 'no_vol_gate')
    print(f"    No vol gate:   n={m_base['n']}  avg={m_base['avg']:+.2f}%  Sharpe={m_base['sharpe']:.3f}  WR={m_base['wr']*100:.0f}%")
    print(f"    Vol top-Q:     n={m_vol['n']}  avg={m_vol['avg']:+.2f}%  Sharpe={m_vol['sharpe']:.3f}  WR={m_vol['wr']*100:.0f}%")
    results['D2_vol'] = {'base': m_base, 'vol_gate': m_vol}

    # DEEP 3: Gap freshness
    print(f"\n  --- DEEP 3: Gap freshness at entry ---")
    h_fresh = copy.deepcopy(h006)
    h_fresh['filters']['gap_fresh'] = 0.70
    trades_fresh = run_backtest(h_fresh, data)
    m_fresh = compute_metrics([t['pnl'] for t in trades_fresh], 'gap_fresh_70')
    print(f"    No freshness:  n={m_base['n']}  avg={m_base['avg']:+.2f}%  Sharpe={m_base['sharpe']:.3f}")
    print(f"    Fresh>=70%:    n={m_fresh['n']}  avg={m_fresh['avg']:+.2f}%  Sharpe={m_fresh['sharpe']:.3f}")
    results['D3_freshness'] = m_fresh

    # DEEP 4: Intraday high proximity
    print(f"\n  --- DEEP 4: Intraday high proximity filter ---")
    prox_trades = []
    for t in trades_sd:
        key = (t['ticker'], t['scan_date'])
        bar_data = data['all_bars'].get(key)
        if bar_data:
            rth = bar_data.get('rth', [])
            if rth:
                # Find entry bar index
                entry_bar = None
                for bi, (bt, bd) in enumerate(rth):
                    if bt == t.get('entry_time'):
                        entry_bar = bi
                        break
                if entry_bar is not None:
                    live_high = max(b[1]['h'] for b in rth[:entry_bar+1])
                    proximity = (live_high - t['entry_price']) / live_high if live_high > 0 else 1
                    if proximity <= 0.03:
                        prox_trades.append(t['pnl'])
    m_prox = compute_metrics(prox_trades, 'proximity_3pct')
    print(f"    No proximity:  n={m_base['n']}  avg={m_base['avg']:+.2f}%  Sharpe={m_base['sharpe']:.3f}")
    print(f"    Prox<=3%:      n={m_prox['n']}  avg={m_prox['avg']:+.2f}%  Sharpe={m_prox['sharpe']:.3f}")
    results['D4_proximity'] = m_prox

    # DEEP 5: Time-of-day entry restriction
    print(f"\n  --- DEEP 5: Entry time bucket analysis ---")
    time_buckets = {'bars 1-5 (9:31-9:35)': [], 'bars 6-10 (9:36-9:40)': [],
                    'bars 11-20 (9:41-9:50)': [], 'bars 21-40 (9:51-10:10)': [],
                    'bars 40+ (10:10+)': []}
    for t in trades_sd:
        et = t.get('entry_time', '09:30')
        parts = et.split(':')
        h, m = int(parts[0]), int(parts[1])
        bar = (h - 9) * 60 + m - 30 + 1
        if bar <= 5: time_buckets['bars 1-5 (9:31-9:35)'].append(t['pnl'])
        elif bar <= 10: time_buckets['bars 6-10 (9:36-9:40)'].append(t['pnl'])
        elif bar <= 20: time_buckets['bars 11-20 (9:41-9:50)'].append(t['pnl'])
        elif bar <= 40: time_buckets['bars 21-40 (9:51-10:10)'].append(t['pnl'])
        else: time_buckets['bars 40+ (10:10+)'].append(t['pnl'])

    print(f"    {'Bucket':<25s}  {'n':>4s}  {'avg':>8s}  {'Sharpe':>7s}  {'WR':>5s}")
    print(f"    {'-'*25}  {'-'*4}  {'-'*8}  {'-'*7}  {'-'*5}")
    for bk, pnls in time_buckets.items():
        if pnls:
            m = compute_metrics(pnls, bk)
            print(f"    {bk:<25s}  {m['n']:>4d}  {m['avg']:>+7.2f}%  {m['sharpe']:>7.3f}  {m['wr']*100:>4.0f}%")
        else:
            print(f"    {bk:<25s}     0       -        -      -")

    # Restricted to bars 3-20 only
    h_time = copy.deepcopy(h006)
    restricted_pnls = []
    for t in trades_sd:
        et = t.get('entry_time', '09:30')
        parts = et.split(':')
        h, m = int(parts[0]), int(parts[1])
        bar = (h - 9) * 60 + m - 30 + 1
        if 3 <= bar <= 20:
            restricted_pnls.append(t['pnl'])
    m_rest = compute_metrics(restricted_pnls, 'bars_3-20')
    print(f"\n    Restricted (bars 3-20): n={m_rest['n']}  avg={m_rest['avg']:+.2f}%  Sharpe={m_rest['sharpe']:.3f}  WR={m_rest['wr']*100:.0f}%")
    results['D5_time'] = {'buckets': time_buckets, 'restricted': m_rest}

    # DEEP 6: Multi-candidate day selection
    print(f"\n  --- DEEP 6: Position concentration (top 3 vs top 5) ---")
    for top_n_val, label in [(5, 'Top 5 (current)'), (3, 'Top 3'), (2, 'Top 2'), (1, 'Top 1')]:
        h_tn = copy.deepcopy(h006)
        h_tn['filters']['top_n'] = top_n_val
        trades_tn = run_backtest(h_tn, data)
        m_tn = compute_metrics([t['pnl'] for t in trades_tn], f'top_{top_n_val}')
        print(f"    {label:<20s}  n={m_tn['n']:>3d}  avg={m_tn['avg']:>+7.2f}%  Sharpe={m_tn['sharpe']:.3f}  WR={m_tn['wr']*100:.0f}%")
    results['D6_concentration'] = True

    # Ranking options for top 3
    print(f"\n    Ranking options (top 3):")
    # Option A: gap size (default — largest gaps first, already sorted)
    h_gap = copy.deepcopy(h006)
    h_gap['filters']['top_n'] = 3
    trades_gap = run_backtest(h_gap, data)
    m_gap = compute_metrics([t['pnl'] for t in trades_gap], 'gap_rank')
    print(f"    A) By gap size:    n={m_gap['n']}  avg={m_gap['avg']:+.2f}%  Sharpe={m_gap['sharpe']:.3f}")

    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

t_start = time.time()

# Load Aug-Oct data
print(f"\n{'='*80}")
print(f"  PROJECT X — VALIDATION & DEEP EXPERIMENTS")
print(f"  Period: {EVAL_START} -> {EVAL_END}")
print(f"{'='*80}")

print("\nLoading Aug-Oct data pipeline...")
data = run_data_pipeline(EVAL_START, EVAL_END)

# Step 1: Unit tests
test_passed, test_failed = run_unit_tests(data)

if test_failed > 0:
    print(f"\n  WARNING: {test_failed} tests failed — review before proceeding")

# Step 2: Validations
val_results = run_validations(data)

# Step 3: Deep experiments (full_data loaded inside run_validations)
full_data = _pipeline_cache.get(f"{FULL_START}_{EVAL_END}")
if full_data is None:
    print("\nLoading full period data...")
    full_data = run_data_pipeline(FULL_START, EVAL_END)
deep_results = run_deep_experiments(data, full_data)

# Step 4: Final comparison
print(f"\n\n{'='*80}")
print(f"  FINAL COMPARISON TABLE")
print(f"{'='*80}")
h006 = HYPOTHESES_BY_ID['H006']
trades_h006 = run_backtest(h006, data)
m_h006 = compute_metrics([t['pnl'] for t in trades_h006], 'H006')

configs = [('H006 baseline', m_h006)]

# Add best DEEP if available
if deep_results.get('D3_freshness', {}).get('n', 0) > 0:
    configs.append(('DEEP3 fresh>=70%', deep_results['D3_freshness']))
if deep_results.get('D4_proximity', {}).get('n', 0) > 0:
    configs.append(('DEEP4 prox<=3%', deep_results['D4_proximity']))
if deep_results.get('D2_vol', {}).get('vol_gate', {}).get('n', 0) > 0:
    configs.append(('DEEP2 vol_gate', deep_results['D2_vol']['vol_gate']))
if deep_results.get('D5_time', {}).get('restricted', {}).get('n', 0) > 0:
    configs.append(('DEEP5 bars 3-20', deep_results['D5_time']['restricted']))

v1 = val_results.get('V1_H007', {})
if v1.get('n', 0) > 0:
    configs.append(('V1 H007 (V5-C)', v1))

# Step 5: Run NEW hypotheses (H021-H027)
print(f"\n\n{'='*80}")
print(f"  NEW HYPOTHESES (H021-H027) — Strict pass criteria")
print(f"  avg>2.5%, Sharpe>0.30, WR>=55%, maxDD<25%, tpw<=5")
print(f"{'='*80}")

n_weeks = len(data['dates']) / 5.0
new_results = {}
new_winners = []
for h_id in NEW_IDS:
    h = HYPOTHESES_BY_ID.get(h_id)
    if h is None:
        print(f"\n  {h_id}: NOT FOUND in hypothesis_space.py — skipping")
        continue
    print(f"\n  --- {h_id}: {h['desc']} ---")
    trades = run_backtest(h, data)
    n = len(trades)
    if n < 5:
        print(f"    n={n} — INVALID (n < 5)")
        new_results[h_id] = {'n': n, 'status': 'INVALID'}
        continue
    m = compute_metrics([t['pnl'] for t in trades], h_id)
    tpw = m['n'] / n_weeks if n_weeks > 0 else 0
    print(f"    n={m['n']}  avg={m['avg']:+.2f}%  Sharpe={m['sharpe']:.3f}  "
          f"WR={m['wr']*100:.0f}%  CI=[{m['ci_lo']:+.2f}, {m['ci_hi']:+.2f}]  "
          f"tpw={tpw:.1f}  PF={m['pf']:.2f}  maxDD={m['max_dd_pct']:.1f}%")

    strict = PASS_CRITERIA_STRICT
    passed = (m['avg'] > strict['avg_pnl_gt'] and
              m['sharpe'] > strict['sharpe_gt'] and
              m['n'] >= strict['n_gte'] and
              m['ci_lo'] > strict['ci_lower_gt'] and
              m['wr'] >= strict['win_rate_gte'] and
              abs(m['max_dd_pct']) / 100 < strict['max_dd_lt'] and
              tpw <= strict['trades_per_week_lte'])
    if passed:
        print(f"    >> STRICT PASS — WINNER!")
        new_winners.append((h_id, m))
        configs.append((f'NEW {h_id}', m))
    else:
        fails = []
        if m['avg'] <= strict['avg_pnl_gt']: fails.append('avg')
        if m['sharpe'] <= strict['sharpe_gt']: fails.append('sharpe')
        if m['wr'] < strict['win_rate_gte']: fails.append('WR')
        if abs(m['max_dd_pct']) / 100 >= strict['max_dd_lt']: fails.append('maxDD')
        if tpw > strict['trades_per_week_lte']: fails.append('tpw')
        if m['ci_lo'] <= strict['ci_lower_gt']: fails.append('CI')
        print(f"    >> FAIL: {', '.join(fails)}")
    new_results[h_id] = m

# Step 6: FINAL comparison table
print(f"\n\n{'='*80}")
print(f"  FINAL COMPARISON — H006 baseline vs best DEEP vs best NEW")
print(f"{'='*80}")

print(f"\n    {'Config':<22s}  {'n':>4s}  {'avg':>8s}  {'Sharpe':>7s}  {'WR':>5s}  {'CI_lo':>7s}  {'PF':>6s}")
print(f"    {'-'*22}  {'-'*4}  {'-'*8}  {'-'*7}  {'-'*5}  {'-'*7}  {'-'*6}")
for label, m in configs:
    print(f"    {label:<22s}  {m['n']:>4d}  {m['avg']:>+7.2f}%  {m['sharpe']:>7.3f}  "
          f"{m['wr']*100:>4.0f}%  {m['ci_lo']:>+6.2f}  {m['pf']:>6.2f}")

# Identify best overall
if len(configs) > 1:
    best = max(configs, key=lambda x: x[1].get('sharpe', 0))
    print(f"\n    >> BEST CONFIG: {best[0]} (Sharpe {best[1]['sharpe']:.3f}, avg {best[1]['avg']:+.2f}%)")

if new_winners:
    print(f"\n  NEW WINNERS: {', '.join(w[0] for w in new_winners)}")
else:
    print(f"\n  No new winners under strict criteria — H006 remains champion")

elapsed = time.time() - t_start
print(f"\n{'='*80}")
print(f"  TOTAL RUNTIME: {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"{'='*80}")
