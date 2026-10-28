# hypothesis_space.py
# Seed hypotheses — broad search across all strategy types
# Claude should extend this list autonomously as the loop runs

HYPOTHESES = [

    # ══════════════════════════════════════════════════════
    # CLUSTER 1: A6 LONG regime adaptations
    # Hypothesis: gap-down mean reversion works with
    # lower threshold + faster exit in compressed regime
    # ══════════════════════════════════════════════════════

    {
        "id": "H001",
        "desc": "A6 gap-down>=2% open entry T+2",
        "direction": "LONG",
        "catalyst": "gap_down",
        "entry": "open_9:30",
        "exit": "T+2_close",
        "filters": {"gap_min": 0.02, "pattern": "none"},
        "hold_approx": "2 days"
    },
    {
        "id": "H002",
        "desc": "A6 gap-down>=2% open entry T+1 open MOC",
        "direction": "LONG",
        "catalyst": "gap_down",
        "entry": "open_9:30",
        "exit": "T+1_open",
        "filters": {"gap_min": 0.02, "pattern": "none"},
        "hold_approx": "overnight"
    },
    {
        "id": "H003",
        "desc": "A6 gap-down>=2% entry 9:45 exit 10:30",
        "direction": "LONG",
        "catalyst": "gap_down",
        "entry": "9:45",
        "exit": "10:30",
        "filters": {"gap_min": 0.02, "pattern": "none"},
        "hold_approx": "45 minutes"
    },

    # ══════════════════════════════════════════════════════
    # CLUSTER 2: C1 LONG regime adaptations
    # Hypothesis: multi-day decline reversal works with
    # shorter lookback in a bull market
    # ══════════════════════════════════════════════════════

    {
        "id": "H004",
        "desc": "C1 3-day>=10% decline gap-up>=3% T+3",
        "direction": "LONG",
        "catalyst": "multi_day_decline_reversal",
        "entry": "open_9:30",
        "exit": "T+3_close",
        "filters": {
            "lookback_days": 3, "decline_thresh": 0.10,
            "gap_min": 0.03
        },
        "hold_approx": "3 days"
    },
    {
        "id": "H005",
        "desc": "C1 3-day>=10% decline gap-up>=3% T+1 open",
        "direction": "LONG",
        "catalyst": "multi_day_decline_reversal",
        "entry": "open_9:30",
        "exit": "T+1_open",
        "filters": {
            "lookback_days": 3, "decline_thresh": 0.10,
            "gap_min": 0.03
        },
        "hold_approx": "overnight"
    },

    # ══════════════════════════════════════════════════════
    # CLUSTER 3: SHORT gap-fade variations
    # Hypothesis: E2-A with quality filters beats baseline
    # ══════════════════════════════════════════════════════

    {
        "id": "H006",
        "desc": "E2-A SHORT sustained_drop only LH entry",
        "direction": "SHORT",
        "catalyst": "gap_up",
        "entry": "local_high",
        "exit": "17:00",
        "filters": {
            "pattern": "sustained_drop",
            "fire_type": "LH_only"
        },
        "hold_approx": "intraday"
    },
    {
        "id": "H007",
        "desc": "E2-A SHORT LH+SPY_down+vol_top_quartile",
        "direction": "SHORT",
        "catalyst": "gap_up",
        "entry": "local_high",
        "exit": "17:00",
        "filters": {
            "fire_type": "LH_only",
            "spy_gate": "down_30m",
            "vol_gate": "top_quartile",
            "gap_fresh": 0.70
        },
        "hold_approx": "intraday",
        "notes": "V5-C config from prior investigation"
    },
    {
        "id": "H008",
        "desc": "SHORT gap-up>=5% no-pharma sustained_drop 10:00 entry",
        "direction": "SHORT",
        "catalyst": "gap_up",
        "entry": "10:00",
        "exit": "17:00",
        "filters": {
            "gap_min": 0.05,
            "pattern": "sustained_drop",
            "sector_excl": "pharma_health_biotech",
            "entry_confirm": "drop_30m >= 3%"
        },
        "hold_approx": "intraday"
    },

    # ══════════════════════════════════════════════════════
    # CLUSTER 4: Sector-specific strategies
    # Hypothesis: certain sectors have regime-specific
    # behavior in the calm-bull Aug–Oct tape
    # ══════════════════════════════════════════════════════

    {
        "id": "H009",
        "desc": "LONG pharma/biotech gap-down>=3% open T+2",
        "direction": "LONG",
        "catalyst": "gap_down",
        "entry": "open_9:30",
        "exit": "T+2_close",
        "filters": {
            "gap_min": 0.03,
            "sector_incl": "pharma_health_biotech"
        },
        "hold_approx": "2 days",
        "notes": "Sector finding: pharma gap-downs have LONG edge"
    },
    {
        "id": "H010",
        "desc": "LONG utility/energy gap-down>=3% open T+2",
        "direction": "LONG",
        "catalyst": "gap_down",
        "entry": "open_9:30",
        "exit": "T+2_close",
        "filters": {
            "gap_min": 0.03,
            "sector_incl": "utilities_energy"
        },
        "hold_approx": "2 days",
        "notes": "Sector finding: utilities LONG edge in Aug-Oct"
    },
    {
        "id": "H011",
        "desc": "SHORT transport/retail gap-up>=5% open T+1",
        "direction": "SHORT",
        "catalyst": "gap_up",
        "entry": "open_9:30",
        "exit": "T+1_close",
        "filters": {
            "gap_min": 0.05,
            "sector_incl": "transport_retail"
        },
        "hold_approx": "1 day",
        "notes": "Sector finding: transport/retail gap-ups fade"
    },

    # ══════════════════════════════════════════════════════
    # CLUSTER 5: Catalyst-specific strategies
    # Hypothesis: specific catalyst types have predictable
    # multi-day direction regardless of gap
    # ══════════════════════════════════════════════════════

    {
        "id": "H012",
        "desc": "LONG post-offering stock T+1 to T+5",
        "direction": "LONG",
        "catalyst": "secondary_offering",
        "entry": "T+1_open",
        "exit": "T+5_close",
        "filters": {
            "gap_max": -0.03,
            "offering_discount": "any"
        },
        "hold_approx": "5 days",
        "notes": "Offering overhang clears — classic post-offering bounce"
    },
    {
        "id": "H013",
        "desc": "SHORT post-earnings gap-up>=5% open T+2",
        "direction": "SHORT",
        "catalyst": "earnings_beat",
        "entry": "open_9:30",
        "exit": "T+2_close",
        "filters": {
            "gap_min": 0.05,
            "catalyst_type": "earnings",
            "eps_surprise": "positive"
        },
        "hold_approx": "2 days",
        "notes": "Sell the news after earnings gap-up"
    },
    {
        "id": "H014",
        "desc": "LONG post-earnings gap-down>=5% open T+3",
        "direction": "LONG",
        "catalyst": "earnings_miss",
        "entry": "open_9:30",
        "exit": "T+3_close",
        "filters": {
            "gap_min": 0.05,
            "catalyst_type": "earnings",
            "eps_surprise": "negative"
        },
        "hold_approx": "3 days",
        "notes": "Post-earnings drift after overreaction"
    },
    {
        "id": "H015",
        "desc": "SHORT analyst downgrade gap-up>=3% same day",
        "direction": "SHORT",
        "catalyst": "analyst_upgrade",
        "entry": "open_9:30",
        "exit": "17:00",
        "filters": {
            "gap_min": 0.03,
            "catalyst_type": "analyst_upgrade",
            "named_firm": True
        },
        "hold_approx": "intraday",
        "notes": "Upgrade gap fades intraday — prior finding"
    },

    # ══════════════════════════════════════════════════════
    # CLUSTER 6: Vol regime strategies
    # Hypothesis: vol spike days have different dynamics
    # than calm days — exploit the difference
    # ══════════════════════════════════════════════════════

    {
        "id": "H016",
        "desc": "LONG gap-down>=3% on vol_bull days open T+2",
        "direction": "LONG",
        "catalyst": "gap_down",
        "entry": "open_9:30",
        "exit": "T+2_close",
        "filters": {
            "gap_min": 0.03,
            "regime": "vol_bull",
            "spy_10d_vol_min": 0.12
        },
        "hold_approx": "2 days",
        "notes": "Vol spikes create oversold bounces"
    },
    {
        "id": "H017",
        "desc": "SHORT gap-up>=5% on calm_bull days LH T+1",
        "direction": "SHORT",
        "catalyst": "gap_up",
        "entry": "local_high",
        "exit": "T+1_close",
        "filters": {
            "gap_min": 0.05,
            "regime": "calm_bull",
            "spy_10d_vol_max": 0.10
        },
        "hold_approx": "1 day",
        "notes": "Low vol gap-ups fade more reliably"
    },

    # ══════════════════════════════════════════════════════
    # CLUSTER 7: Multi-day swing strategies
    # Hypothesis: longer holds capture more of the move
    # in a trending tape — human-friendly, less monitoring
    # ══════════════════════════════════════════════════════

    {
        "id": "H018",
        "desc": "LONG 5-day loser gap-up>=3% hold T+5",
        "direction": "LONG",
        "catalyst": "multi_day_decline_reversal",
        "entry": "open_9:30",
        "exit": "T+5_close",
        "filters": {
            "lookback_days": 5, "decline_thresh": 0.12,
            "gap_min": 0.03
        },
        "hold_approx": "5 days"
    },
    {
        "id": "H019",
        "desc": "SHORT 5-day winner gap-down>=3% hold T+5",
        "direction": "SHORT",
        "catalyst": "multi_day_rise_reversal",
        "entry": "open_9:30",
        "exit": "T+5_close",
        "filters": {
            "lookback_days": 5, "rise_thresh": 0.15,
            "gap_min": 0.03
        },
        "hold_approx": "5 days",
        "notes": "Mirror of C1 on the SHORT side"
    },
    {
        "id": "H020",
        "desc": "LONG gap-down>=5% hold T+10 close",
        "direction": "LONG",
        "catalyst": "gap_down",
        "entry": "open_9:30",
        "exit": "T+10_close",
        "filters": {"gap_min": 0.05},
        "hold_approx": "2 weeks",
        "notes": "Longer hold — captures full mean reversion"
    },

    # ══════════════════════════════════════════════════════
    # CLUSTER 8: NEW hypotheses — Part 4 deep exploration
    # Added 2026-10-07 after H006 sustained_drop winner
    # ══════════════════════════════════════════════════════

    {
        "id": "H021",
        "desc": "SHORT post-earnings gap-up + sustained_drop pattern",
        "direction": "SHORT",
        "catalyst": "gap_up",
        "entry": "local_high",
        "exit": "17:00",
        "filters": {
            "gap_min": 0.03, "pattern": "sustained_drop",
            "fire_type": "LH_only",
            "catalyst_type": "earnings",
        },
        "hold_approx": "intraday",
        "notes": "Post-earnings overreaction + sustained_drop = high confidence SHORT"
    },
    {
        "id": "H022",
        "desc": "SHORT PM volume exhaustion gap-up LH 17:00",
        "direction": "SHORT",
        "catalyst": "gap_up",
        "entry": "local_high",
        "exit": "17:00",
        "filters": {
            "gap_min": 0.03, "pattern": "sustained_drop",
            "fire_type": "LH_only",
            "pm_vol_exhaustion": True,
        },
        "hold_approx": "intraday",
        "notes": "High PM volume = buyers exhausted by open; sustained_drop confirms"
    },
    {
        "id": "H023",
        "desc": "LONG sustained_drop recovery gap-up T+1 close",
        "direction": "LONG",
        "catalyst": "gap_up",
        "entry": "local_low",
        "exit": "T+1_close",
        "filters": {
            "gap_min": 0.03, "pattern": "sustained_drop",
            "fire_type": "LL_only",
        },
        "hold_approx": "1 day",
        "notes": "Buy the dip AFTER sustained_drop exhaustion — reversal play"
    },
    {
        "id": "H024",
        "desc": "SHORT sector-momentum gap-up pharma LH 17:00",
        "direction": "SHORT",
        "catalyst": "gap_up",
        "entry": "local_high",
        "exit": "17:00",
        "filters": {
            "gap_min": 0.03, "pattern": "sustained_drop",
            "fire_type": "LH_only",
            "sector_incl": "pharma_health_biotech",
        },
        "hold_approx": "intraday",
        "notes": "Pharma gap-ups fade harder — sector-specific sustained_drop"
    },
    {
        "id": "H025",
        "desc": "LONG gap-up continuation >10% open T+1 close",
        "direction": "LONG",
        "catalyst": "gap_up",
        "entry": "open_9:30",
        "exit": "T+1_close",
        "filters": {
            "gap_min": 0.10, "pattern": "flat_or_bounce",
        },
        "hold_approx": "1 day",
        "notes": "Massive gap-ups that hold flat continue next day (momentum)"
    },
    {
        "id": "H026",
        "desc": "LONG intraday reversal after flush gap-down LL T+0",
        "direction": "LONG",
        "catalyst": "gap_down",
        "entry": "local_low",
        "exit": "17:00",
        "filters": {
            "gap_min": 0.05, "pattern": "sustained_drop",
            "fire_type": "LL_only",
        },
        "hold_approx": "intraday",
        "notes": "Gap-down + sustained drop + local-low = capitulation reversal"
    },
    {
        "id": "H027",
        "desc": "SHORT pre-earnings run-up gap-up LH T+1",
        "direction": "SHORT",
        "catalyst": "gap_up",
        "entry": "local_high",
        "exit": "T+1_close",
        "filters": {
            "gap_min": 0.05, "pattern": "sustained_drop",
            "fire_type": "LH_only",
            "lookback_days": 5, "rise_thresh": 0.10,
        },
        "hold_approx": "1 day",
        "notes": "5-day winners that gap up more and drop = exhaustion SHORT"
    },
]

PASS_CRITERIA = {
    "avg_pnl_gt":       2.0,
    "sharpe_gt":        0.20,
    "n_gte":            15,
    "ci_lower_gt":      0.0,
    "win_rate_gte":     0.50,
    "max_dd_lt":        0.35,
    "trades_per_week_lte": 15,
    "avg_hold_hrs_gte": 1.0
}

PASS_CRITERIA_STRICT = {
    "avg_pnl_gt":       2.5,
    "sharpe_gt":        0.30,
    "n_gte":            15,
    "ci_lower_gt":      0.0,
    "win_rate_gte":     0.55,
    "max_dd_lt":        0.25,
    "trades_per_week_lte": 5,
    "avg_hold_hrs_gte": 1.0
}

BASELINE = {
    "strategy":  "E2-A SHORT",
    "period":    "Aug-Oct 2026",
    "avg_pnl":   2.28,
    "sharpe":    0.185,
    "n":         154,
}

OPEN_EXPLORATION_AREAS = [
    "momentum continuation on high-vol days",
    "post-halt resumption trades",
    "pre-market catalyst holds into next day",
    "sympathy plays on sector peers",
    "reversal after 2-standard-deviation intraday move",
    "options expiration week behavior",
    "end-of-month rebalancing flows",
    "Monday gap strategies",
    "short interest squeeze setups",
    "insider buying gap-up continuation",
    "M&A target peer trades",
    "FDA calendar plays on sector",
    "earnings drift by surprise magnitude bucket",
    "anything else you hypothesize based on the data"
]
