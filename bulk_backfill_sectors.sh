#!/bin/bash
# bulk_backfill_sectors.sh — Full data pipeline for 62 sector tickers
# Power & Grid, AI Hardware, Aerospace & Defense, E-Commerce & Retail
#
# Phase 1: OHLCV backfill from Polygon (2022-01-01 → today)
# Phase 2: T&Q download + micro rebuild + master_inference stitch
#
# Run AFTER bulk_backfill_new50.sh completes (same Polygon API key).
#
# Usage:
#   nohup bash bulk_backfill_sectors.sh > /tmp/sectors_backfill.log 2>&1 &

export PYTHONUNBUFFERED=1
cd /home/admin_yaninaso_altostrat_com/bot-RAG

# 62 missing tickers across 4 sectors
# Note: MOG.A uses Polygon ticker MOG-A (will try both)
TICKERS="PWR NVT HUBB TLN ORA AEP SO D MYRG AYI MTRX AGR SRE FE EXC PNW AES FN LRCX ASML AMAT KLAC ONTO CAMT NVMI GLW CSCO DIOD IPGP HWM CRS ATI TDG HEI LDOS BWXT HII BA TXT BHE WWD GOGO AXON CAE ERJ TGI AIR VSEC WING SG DECK MELI SE CELH CPNG PDD ELF BOOT RVLV W SFM MOG-A"

echo "============================================================"
echo "  SECTOR TICKERS — FULL DATA BACKFILL (62 tickers)"
echo "  Started: $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "  Tickers: $(echo $TICKERS | wc -w)"
echo "============================================================"

echo ""
echo "============================================================"
echo "  PHASE 1: OHLCV BACKFILL (Polygon aggs)"
echo "  Started: $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "============================================================"
python3 bulk_ohlcv_download.py $TICKERS

echo ""
echo "============================================================"
echo "  PHASE 2: T&Q + MICRO + MASTER_INFERENCE"
echo "  Started: $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "============================================================"
python3 bulk_micro_all.py $TICKERS

echo ""
echo "============================================================"
echo "  ALL PHASES COMPLETE"
echo "  Finished: $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "============================================================"
