#!/usr/bin/env bash
set -euo pipefail

# 批次執行 3 個 lookback × 2 個 keep_sides 共 6 組實驗

CONFIG_PATH="${CONFIG_PATH:-train/config.yaml}"

declare -A TBM_CSV_MAP=(
  [36]="data/TBM_label/win_rate/BTC-USDT_1h_atr_up4_dn4_lookback36_label.csv"
  [72]="data/TBM_label/win_rate/BTC-USDT_1h_atr_up6_dn6_lookback72_label.csv"
  [108]="data/TBM_label/win_rate/BTC-USDT_1h_ewma_up8_dn8_lookback108_label.csv"
)

LOOKBACKS=(36 72 108)
SIDES=("short" "long")

for side in "${SIDES[@]}"; do
  for lookback in "${LOOKBACKS[@]}"; do
    tbm_csv="${TBM_CSV_MAP[$lookback]}"
    if [[ -z "${tbm_csv}" ]]; then
      echo "[WARN] No TBM CSV mapped for lookback=${lookback}, skip."
      continue
    fi

    project_name="BTC_2-stream_rank_biserial_${lookback}_${side}"
    data_path="feature_selection/results/rank_biserial/${side}_${lookback}/rank_biserial_60_feat.csv"

    echo "============================================================"
    echo "Running project=${project_name}, lookback=${lookback}, side=${side}"
    echo "data_path=${data_path}"
    echo "tbm_csv=${tbm_csv}"
    echo "============================================================"

    python -m train.main_train \
      --config "${CONFIG_PATH}" \
      --project-name "${project_name}" \
      --feat-path "${data_path}" \
      --tbm-csv-path "${tbm_csv}" \
      --tbm-keep-sides "${side}"
  done
done
