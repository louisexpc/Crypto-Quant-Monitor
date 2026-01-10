# Predictor (Inference) 指南

## 簡介
`train/inference/predictor.py` 提供輕量推論器，直接吃預算好的 15m 特徵（可選 1m micro 展平），載入訓練時存下的 checkpoint，產出帶預測欄位的 TBM DataFrame：

- `predict(df, model_path)`: 單一 checkpoint，輸出欄位 `pred`、`pred_p1`。
- `predict_vote(df, model_paths_or_dir)`: 多 checkpoint 投票，輸出 `pred_i`（各 fold）、`pred_vote_votes_total`、`pred_vote_margin`、`pred_vote`。
- 不做 scaler/標準化，假設特徵已在 precompute 階段完成。

TrialRunner 的 post-infer（若 `post_infer.tbm_concat.enabled=True`）會自動調用這支 Predictor。

## 輸入需求
- **特徵檔**：15m 預算特徵 DataFrame；若有 1m micro，需先展平為 `m0_...~m(window_len-1)_...`。在 `trial_runner` / `predictor_testing.py` 會用 `flatten_micro_features` 處理，手動呼叫時請先套用相同邏輯並裁剪到 `[cv_start, ts_end]`。
- **TBM CSV**：由 `cfg.label.tbm_csv_path`（或 `post_infer.csv_path_override`）讀取，至少含 `t0`、`side`、`label`；`__rid` 若缺會自動補。
- **Checkpoint**：訓練時保存的 `model_state.pt`，內含 `state_dict`、`feature_columns`（可選）、`model_cfg`、`best_val_thresh`、`temperature`。

## 必要/選用的設定鍵位
```yaml
device: "cuda:0"
data:
  path: ...
  micro:
    enabled: true
    path: ...
    window_len: 15

cv:
  start_date: "YYYY-MM-DD"
  end_date: "YYYY-MM-DD"

sequence:
  seq_len: 144 # 往前看幾根 15 min 的 k bar

label:
  tbm_csv_path: ...
  keep_sides: "<side>"   # long | short | both
  align_method: pad

post_infer:
  enabled: true
  date_start: "YYYY-MM-DD"
  date_end: "YYYY-MM-DD"
  collapse_mask_enable: true

# model type
model:
  name: "TwoStreamHybrid"
  num_classes: 2

# (optional) 使用與 train 相同的 dtype/bs
train:
  batch_size: 256
  amp: true
  amp_dtype: "bf16"

```

## 使用方式
### 1) 直接在程式內呼叫
```python
from pathlib import Path
import pandas as pd

from train.core.config_loader import load_cfg
from train.data.dataloaders.base import flatten_micro_features, load_precomputed_features
from train.inference.predictor import Predictor

cfg = load_cfg("train/inference/test_predictor.yaml")

feat_df = load_precomputed_features(path=cfg["data"]["path"])
micro_cfg = cfg["data"].get("micro", {})
if micro_cfg.get("enabled"):
    micro_df = load_precomputed_features(path=micro_cfg["path"])
    feat_df = flatten_micro_features(
        feat_df=feat_df,
        micro_df=micro_df,
        cv_start=pd.Timestamp(cfg["cv"]["start_date"]),
        ts_end=pd.Timestamp(cfg["post_infer"]["date_end"]),
        window_len=int(micro_cfg.get("window_len", 15)),
    )

model_paths = [
    Path("runs/.../fold_0/model_state.pt"),
    Path("runs/.../fold_1/model_state.pt"),
]

pred = Predictor(cfg).predict_vote(feat_df, model_paths)
# pred 內含 pred_0/pred_1/.../pred_vote_* 欄位，可自行輸出 CSV
```

### 2) 使用預設 test case 腳本
```bash
# 預設會讀 train/inference/test_predictor.yaml 與三個 BTC fold 模型
python3 train/inference/test_case.py --output pred.csv

# 若要改用其他設定或模型
python3 train/inference/test_case.py \
  --config path/to/your_cfg.yaml \
  --models runs/xxx/fold_0/model_state.pt runs/xxx/fold_1/model_state.pt \
  --output your_pred.csv
```

## 回傳資料型態與範例
- `Predictor.predict` / `predict_vote` 都回傳 `pd.DataFrame`，內容為 TBM 事件 + 預測欄位（索引被重設為欄位）。
- 單模型欄位：`pred`、`pred_p1`；多模型另含 `pred_i`、`pred_vote_votes_total`、`pred_vote_margin`、`pred_vote`。

範例（`predict_vote` head）：
```text
   __rid        t0        side  label  pred_0  pred_0_p1  pred_1  pred_1_p1  pred_vote_votes_total  pred_vote_margin  pred_vote
0      5 2025-05-01  short/-1?      1       1   0.612345       0   0.488765                      2          0.000000          0
1     12 2025-05-02  short/-1?      0       1   0.734210       1   0.702100                      2          2.000000          1
2     28 2025-05-03  short/-1?      0       0   0.423100       0   0.401230                      2         -2.000000          0
3     44 2025-05-04  short/-1?      1       1   0.812340       1   0.805400                      2          2.000000          1
4     63 2025-05-05  short/-1?      1     NaN        NaN       1   0.551230                      1          1.000000          1
```

## 輸出欄位
- 單模型：`pred`（0/1）、`pred_p1`（機率）。
- 多模型投票：`pred_i`（每 checkpoint 0/1）、`pred_vote_votes_total`（有效票數）、`pred_vote_margin`（#1 - #0）、`pred_vote`（投票結果）。
- 非 keep_sides 或無對齊事件的列保持 NaN/NA。

## 常見注意事項
- 門檻/溫度：自動取 checkpoint 的 `best_val_thresh`、`temperature`，缺值時門檻為 0.5。
- 對齊：沿用 `cfg.label.align_method`（默認 pad），對齊事件到特徵格點；需要 `cfg.label.keep_sides` 設定 long/short/both。
- NaN/Inf：推論前會檢查特徵數值欄不可含 NaN/Inf，若展平後仍有缺值，需回到 precompute 或展平流程修正。

## 相關檔案
- `train/inference/predictor.py`：核心推論邏輯。
- `train/data/dataloaders/base.py.flatten_micro_features`：1m→15m 展平共用工具。
- `train/pipeline/trial_runner.py`：post-infer 入口，呼叫 Predictor + TBMExporter。
