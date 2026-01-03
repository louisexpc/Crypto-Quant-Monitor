#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-./.venv/bin/python}"
CFG="${CFG:-feature_selection/statistics/ts_kmeans_msm/config.yaml}"

$PYTHON feature_selection/statistics/ts_kmeans_msm/run_ts_kmeans_msm.py -c "$CFG"
$PYTHON feature_selection/statistics/ts_kmeans_msm/reporter.py -c "$CFG"
