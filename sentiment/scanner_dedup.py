"""
Cross-run, cross-scanner ticker dedup state.

Ensures at most one live direction per ticker per day across all scanner runs.
Both premarket_scanner and discovery_scanner import this module.

State is stored in BQ table: scoring_logs.scanner_ticker_state
Each run reads state before selection, writes after.

Rules:
  - A ticker has at most one live direction per day.
  - A later run with the SAME direction: updates last_run, does NOT create a new trade.
  - A later run with OPPOSITE direction: supersedes only if EV > 2x existing EV,
    and logs the flip as an event. Otherwise the new signal is blocked.
"""

from google.cloud import bigquery
import json
from datetime import datetime

BQ_STATE_TABLE = "YOUR_GCP_PROJECT.scoring_logs.scanner_ticker_state"

_CREATE_TABLE_DDL = f"""
CREATE TABLE IF NOT EXISTS `{BQ_STATE_TABLE}` (
    scan_date          DATE,
    ticker             STRING,
    direction          STRING,
    ev                 FLOAT64,
    scanner            STRING,
    first_selected_run STRING,
    last_run           STRING,
    superseded_by      STRING,
    supersede_reason   STRING,
    update_timestamp   TIMESTAMP
)
"""

_state_cache = {}
_bq_client = None


def _get_client():
    global _bq_client
    if _bq_client is None:
        _bq_client = bigquery.Client()
    return _bq_client


def ensure_table():
    """Create state table if it doesn't exist."""
    try:
        _get_client().query(_CREATE_TABLE_DDL).result()
    except Exception as e:
        print(f"  [dedup] Table creation note: {e}")


def load_state(scan_date):
    """Load all ticker state for a given date into memory. Call once per run."""
    global _state_cache
    _state_cache = {}
    try:
        q = f"""
        SELECT ticker, direction, ev, scanner, first_selected_run, last_run
        FROM `{BQ_STATE_TABLE}`
        WHERE scan_date = '{scan_date}'
          AND superseded_by IS NULL
        """
        for row in _get_client().query(q).result():
            _state_cache[row['ticker']] = {
                'direction': row['direction'],
                'ev': float(row['ev'] or 0),
                'scanner': row['scanner'],
                'first_selected_run': row['first_selected_run'],
                'last_run': row['last_run'],
            }
        if _state_cache:
            print(f"  [dedup] Loaded {len(_state_cache)} active ticker states for {scan_date}")
    except Exception as e:
        print(f"  [dedup] State load failed ({e}), starting fresh")
    return _state_cache


def check_ticker(ticker, direction, ev, scanner, run_time):
    """
    Check if a ticker+direction is allowed given current state.

    Returns:
        (allowed, reason)
        - allowed=True:  this signal can proceed to selection
        - allowed=False: this signal is blocked by dedup
        reason: human-readable explanation
    """
    EV_SUPERSEDE_RATIO = 2.0

    existing = _state_cache.get(ticker)

    if existing is None:
        return True, "new_ticker"

    if existing['direction'] == direction:
        return False, f"DEDUP_SAME_DIR (already {direction} from {existing['scanner']}@{existing['first_selected_run']}, ev={existing['ev']:.1f})"

    # Opposite direction — supersede only if EV is 2x the existing
    if ev > existing['ev'] * EV_SUPERSEDE_RATIO:
        return True, f"SUPERSEDE ({direction} ev={ev:.1f} > 2x prior {existing['direction']} ev={existing['ev']:.1f})"

    return False, f"DEDUP_DIR_CONFLICT ({existing['direction']} from {existing['scanner']}@{existing['first_selected_run']} ev={existing['ev']:.1f}, new {direction} ev={ev:.1f} < 2x threshold)"


def record_selection(scan_date, ticker, direction, ev, scanner, run_time):
    """
    Record a selected ticker in the state table.
    Call AFTER a ticker is confirmed as selected.

    If this supersedes an existing opposite-direction entry,
    the old row is marked as superseded.
    """
    existing = _state_cache.get(ticker)
    client = _get_client()
    now_utc = datetime.utcnow().isoformat()

    if existing and existing['direction'] != direction:
        # Supersede the old entry
        try:
            supersede_q = f"""
            UPDATE `{BQ_STATE_TABLE}`
            SET superseded_by = '{scanner}_{run_time}',
                supersede_reason = 'EV_FLIP ({direction} ev={ev:.1f} > {existing["direction"]} ev={existing["ev"]:.1f})',
                update_timestamp = CURRENT_TIMESTAMP()
            WHERE scan_date = '{scan_date}'
              AND ticker = '{ticker}'
              AND superseded_by IS NULL
            """
            client.query(supersede_q).result()
        except Exception as e:
            print(f"  [dedup] Supersede update failed for {ticker}: {e}")

    # Insert new state row
    row = {
        'scan_date': scan_date,
        'ticker': ticker,
        'direction': direction,
        'ev': ev,
        'scanner': scanner,
        'first_selected_run': run_time,
        'last_run': run_time,
        'superseded_by': None,
        'supersede_reason': None,
        'update_timestamp': now_utc,
    }

    try:
        errors = client.insert_rows_json(BQ_STATE_TABLE, [row])
        if errors:
            print(f"  [dedup] BQ insert error for {ticker}: {errors}")
    except Exception as e:
        print(f"  [dedup] BQ insert failed for {ticker}: {e}")

    # Update local cache
    _state_cache[ticker] = {
        'direction': direction,
        'ev': ev,
        'scanner': scanner,
        'first_selected_run': run_time,
        'last_run': run_time,
    }


def update_last_run(scan_date, ticker, run_time):
    """Update last_run for a same-direction re-scan (no new trade)."""
    existing = _state_cache.get(ticker)
    if not existing:
        return
    try:
        q = f"""
        UPDATE `{BQ_STATE_TABLE}`
        SET last_run = '{run_time}',
            update_timestamp = CURRENT_TIMESTAMP()
        WHERE scan_date = '{scan_date}'
          AND ticker = '{ticker}'
          AND superseded_by IS NULL
        """
        _get_client().query(q).result()
        existing['last_run'] = run_time
    except Exception as e:
        print(f"  [dedup] last_run update failed for {ticker}: {e}")


def get_state_summary():
    """Return current state for logging."""
    if not _state_cache:
        return "No active state"
    lines = []
    for ticker, s in sorted(_state_cache.items()):
        lines.append(f"  {ticker:<6} {s['direction']:<6} ev={s['ev']:.1f} "
                      f"({s['scanner']}@{s['first_selected_run']})")
    return '\n'.join(lines)
