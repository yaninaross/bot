#!/bin/bash
# bulk_backfill_new50.sh — Full data pipeline for 50 new tickers
# Phase 1: OHLCV backfill from Polygon (2022-01-01 → today)
# Phase 2: T&Q download + micro rebuild + master_inference stitch
#
# Usage:
#   nohup bash bulk_backfill_new50.sh > /tmp/new50_backfill.log 2>&1 &

cd /home/admin_yaninaso_altostrat_com/bot-RAG

TICKERS="BNY MGRT RXT SIF CRNG ROMA SNDK AXTI BUUU BLZE DELL ZTG INTC WDC SCKT IRDM CJMB NCEW FLEX CIEN DFNS MRVL BW APP ATEX FIX STX DD DOV SWK CENX VLO SBLK SHOP META AAOI NUA LITE ORCL ANET SPOT NFLX NRG KGS CVNA CAVA ANF STRL EME COHR"

echo "============================================================"
echo "  NEW 50 TICKERS — FULL DATA BACKFILL"
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
