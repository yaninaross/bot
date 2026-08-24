#!/bin/bash
# Launch bulk_micro_all.py with service account auth.
# Logs to /tmp/bulk_micro_refresh.log with timestamps.
# Usage:
#   ./run_bulk_refresh.sh              # auto-detect stale tickers
#   ./run_bulk_refresh.sh --all        # force all tickers
#   ./run_bulk_refresh.sh MARA KEEL    # specific tickers

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="/tmp/bulk_micro_refresh_$(date +%Y%m%d_%H%M%S).log"

echo "Starting bulk refresh at $(date -u)" | tee "$LOG"
echo "Log: $LOG" | tee -a "$LOG"
echo "Args: $@" | tee -a "$LOG"

cd "$SCRIPT_DIR"

python3 bulk_micro_all.py --sa "$@" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "Finished at $(date -u)" | tee -a "$LOG"
