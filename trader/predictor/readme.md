# Trader Predictor

## 概要
- 輕量推論模組，支援多 checkpoint 載入，專注單窗推論（序列長度由 config 提供）。
- 不依賴 `train/`，僅保留 TwoStreamHybrid 模型的 inference 版本。
- 若多個 checkpoint 的 AMP 設定不同，會沿用第一個並給警告。

## 主要介面
- `Predictor(cfg: dict, side)`  
  - `cfg` 需包含：`model_path_list`（list[str|Path]）、`seq_len`（int）、`device`（預設 "cuda:0"）。
  - 其中`model_path_list`底下需要分別設定 `long/short` 的 `model list`，並由 `side` 決定用於建立 `Predictor` 的 `model`。
  - 會一次載入所有 checkpoints，驗證 feature_columns/amp/threshold/temperature 一致。
- `predict(feat_df, model_idx=0) -> (inference_time_sec, pred_bool)`  
  - 使用指定模型推論單一時間窗，返回耗時（秒）與布林預測。
- `predict_vote(feat_df) -> (inference_time_sec, pred_bool)`  
  - 逐模型推論後投票（各自門檻），返回耗時與票決結果。

## 輸入要求
- `feat_df` 為單窗特徵 DataFrame：
  - 行數必須等於 `rolling_win + seq_len`
    (若特徵計算需要 rolling z-score，請在上游先準備 ，送入FeatureComputer 並只截取尾端 `seq_len` 送入。)
  - 欄位需 == checkpoint 的 `feature_columns`
  - index 必須為單調遞增的 `DatetimeIndex`，且相鄰差為 15 分鐘（等距）
  - 數值欄不得包含 NaN/Inf


## Checkpoint 需求
- 應包含：`state_dict`、`feature_columns`、`model_cfg`、`best_val_thresh`、`temperature`、`amp`、`amp_dtype`。
- 僅支援 TwoStreamHybrid。

## 常見用法
```python
from trader.predictor.predictor import Predictor
import pandas as pd
# 或使用 yaml
cfg = {
    "model_path_list": ["runs/.../fold_0/model_state.pt", "runs/.../fold_1/model_state.pt"],
    "seq_len": 144,
    "device": "cuda:0",
}
side = "long" # or short
pred = Predictor(cfg, side) # 可以 long/short 建立兩個 predictor 物件
feat_df = pd.DataFrame(...)  # 應滿足上述輸入要求
inference_time, yhat = pred.predict_vote(feat_df)
```
