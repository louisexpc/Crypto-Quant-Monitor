#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

START="${START:-2025-05-01}"
END="${END:-2025-10-20}"

LONG_TRAIN_INFER="${LONG_TRAIN_INFER:-runs/BTC_3candles_long_rb40/trial_001_mcc=+0.097/trial_001_long.csv}"
SHORT_TRAIN_INFER="${SHORT_TRAIN_INFER:-runs/BTC_3candles_short_rb40/trial_000_mcc=+0.015/trial_000_short.csv}"
RUNTIME_PRED="${RUNTIME_PRED:-trader/testcase/test_case_pred.csv}"
COMPARE_OUT="${COMPARE_OUT:-trader/testcase/compare_diff.csv}"

cd "$ROOT_DIR"

echo "[1/2] Running trader/testcase/test_case.py ..."
"$PYTHON_BIN" trader/testcase/test_case.py \
  --start "$START" \
  --end "$END"

echo "[2/2] Comparing runtime prediction with train inference CSVs ..."
"$PYTHON_BIN" trader/testcase/compare_result.py \
  --train-long "$LONG_TRAIN_INFER" \
  --train-short "$SHORT_TRAIN_INFER" \
  --runtime "$RUNTIME_PRED" \
  --output "$COMPARE_OUT"

echo "[OK] Pipeline done."
echo "  runtime: $RUNTIME_PRED"
echo "  compare: $COMPARE_OUT"

# trader/testcase/test_pipeline.sh
