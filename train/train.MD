# Train

## 流程
```
Precomputed features (CSV/Parquet) + TBM labels
        │  (config.data.path, config.label)
        ▼
train/main_train.py.load_cfg        # 載入 YAML、設定亂數種子、建立輸出位置
        ▼
Optuna objective( trial, cfg, df )  # objective.objective
        │
        ├─ make_folds()             # objective_utils.make_folds → FoldGenerator
        ├─ make_*_loaders_for_fold  # build_feature_loader.dataloader
        │      │
        │      ├─ select_plan_columns()  # 根據 features.plan 過濾欄位
        │      ├─ create_label()         # build_features.create_label / TBM 對齊
        │      └─ SeqDataset/EventDataset
        │
        ├─ build_model()            # models.model_factory
        ├─ train_one_fold()         # trainer_cls / trainer_reg
        └─ compute_trial_score()    # objective_utils
        ▼
Optuna Trial Summary               # save_cv_summary, dump_best_yaml
        ▼
Post infer / TBM concat (可選)     # objective.export_tbm_predictions_for_trial
```

## 1. 啟動入口 (train/main_train.py)
- `load_cfg` 讀取 YAML (`train/config.yaml`) 並以 UTF-8 載入整份訓練設定。
- `set_seed` 與 `setup_cuda_acceleration` 固定亂數、開啟 TF32/AMP 等 GPU 加速選項。
- `prepare_dataframe` 只負責讀取預算特徵檔，建立時間索引 (UTC)；實際特徵/標籤組裝在每個 fold 內處理。
- `build_study` 以 SQLite 儲存 Optuna Study，對應 `cfg.search` 的 sampler、pruner、超參限制。
- `study.optimize` 呼叫 `objective`：
  - 設定輸出資料夾 `runs/<project_name>/trial_xxx/` (多 GPU 會分 `gpu*/trial_xxx`)。
  - 每個 trial 會覆寫 `cfg.seed = base_seed + trial.number`，確保可重現。
  - 完成後呼叫 `dump_best_yaml` 匯出最佳設定、超參與特徵組合。

## 2. 資料與 DataLoader 管線
### 2.1 預算特徵檔需求 (`config.data`)
- `path` 指向含有 **完整 15 分鐘資料網格** 的 CSV 或 Parquet。檔案需包含：
  - 時間欄位 `datetime` 或 `timestamp` (秒或毫秒)。
  - 基礎 OHLCV 欄位 (`open`, `high`, `low`, `close`, `volume`)。
  - 其他自訂特徵欄位（與 `features.plan` 預期相符）。
- `freq` 決定重建網格的頻率 (例 `"15min"`)；若資料有缺洞會透過 reindex 產生 NaN，缺值會在後續清理時被過濾。
- 如果配置 `data.micro.enabled`，請確保帶有 `m_-k_feature` 類欄位；會在 `min_trade_feat` 清單中選取。

### 2.2 特徵篩選 (`select_plan_columns`)
- 依 `features.plan.features` 只保留啟用的指標：
  - 各指標名稱對應 `train/build_feature_loader/indicators.py` 中的 builder 定義。
  - 特徵表中若不存在對應欄位會被忽略；若所有欄位都不存在則報錯。
- OHLCV 是否保留由 `OPEN/HIGH/...` 這些條目控制。
- `features.min_trade_feat` 對 1 分鐘展開欄位（以 `m_` 前綴）做額外挑選。

### 2.3 標籤生成 (`build_features.create_label`)
- 時間驅動 (classification/regression)：
  - `label.mode` 可選 `vol`、`bps`、`quantile`、`triple` 等，每種模式會依 `label.ret_shift`、`ret_type`（`logret` 或 `fractionally`）生成 `y_reg` 或 `y_cls`。
  - `label.fracdiff` 控制分數差分目標 (`ret_type=fractionally`)。
  - 門檻相關參數 (`k_vol`, `flat_band_bps`, `roundtrip_cost_bps`...) 影響三分類切割。
- 事件驅動 (`mode=event_tbm`)：
  - `assign_tbm_labels_to_df` 讀 `label.tbm_csv_path`，依 `keep_sides` 選擇 long/short，並以 `align_method` (`exact` 或 `pad`) 對齊到 15m 網格，產生二元標籤 `y_cls`。
  - 若 `keep_sides=both`，要求同一時間戳不得同時存在雙邊訊號。

### 2.4 時間驅動 DataLoader (`make_time_loaders_for_fold`)
- 從預算特徵檔重建完整網格後：
  1. 依 `features.plan` 過濾欄位並轉成 `float32`。
  2. 呼叫 `create_label` 產生 `y_reg` 或 `y_cls`，移除包含 NaN/Inf 的列。
  3. 使用 `cv.start_date` `cv.end_date` 篩選訓練期間。
  4. 利用 `FoldGenerator` 提供的布林遮罩或時間戳切分 train/val/test。
  5. 以 `sequence.scaler` 與 `pick_cols_to_scale` 選取需標準化的欄位：
     - 若 scaler 支援 timesafe (`rolling`, `ewm` 等)，在完整資料上做 `transform_full`。
     - 其他 sklearn 類 scaler 則於 train split 擬合後套用。
  6. 建立 `SeqDataset`：
     - 使用滑動視窗長度 `sequence.seq_len`、`stride`/`stride_anchor` 控制窗口移動。
     - `preload_to_gpu=true` 時，資料會預先搬到 GPU；否則停在 CPU 並在 DataLoader 中透過 `pin_memory` 加速。
  7. 回傳三個 `DataLoader` 與 XGBoost 用的 numpy bundle (`info['XGB']`)。

### 2.5 事件驅動 DataLoader (`make_event_loaders_for_fold`)
- 適用於 TBM / 事件序列任務：
  - 先以 `select_plan_columns` 選特徵欄，再依 `sequence.scaler` 對整張特徵表做時間安全縮放。
  - 將每個 fold 的 train/val/test 時段對齊到底層網格 (`align_times`)；只會保留可產生完整視窗 (`seq_len`) 的事件。
  - `EventDataset` 讀 `tbm_csv_path`，計算 `(N,L,F)` 張量與標籤；支援 `keep_sides`、`drop_incomplete`、裝置預載。
  - DataLoader 不啟用 `pin_memory`（資料已在目標裝置）。
  - `stats` 會列出事件總數、被濾除原因，以便檢查資料覆蓋率。

### 2.6 交叉驗證折疊 (`objective_utils.make_folds`)
- `FoldGenerator` 會將索引正規化為 UTC，依 `cv.start_date`、`end_date` 限定可用時間：
  - `type=Purged_kfold`：均分月份，對測試區塊套 `embargo_hours`，確保訓練集不含測試鄰近樣本。
  - `type=Rolling`：以 rolling window 方式構建 train/test，支援測試頻率 (`test_freq`) 與禁運期。
- 每個 fold 存有：
  - `train_val_mask`、`test_mask`（布林向量或時間集合）。
  - `train_val_times`、`test_times`（事件驅動回落時使用）。
  - `test_month` 等 metadata 會在 log 中顯示。

## 3. 模型訓練與評估
### 3.1 模型建構 (`models/model_factory.build_model`)
- 根據 `model.name` 建立對應神經網路或 XGBoost 模型：
  - `LSTM_SE`、`LSTM_Temporal`：序列式 LSTM，支援 SE 模組、雙向等選項。
  - `TemporalTransformer`：含 sin-cos 位置編碼、Pre-Norm block、可選 `pooling` (`attn`, `cls`, `mean`, `last`)。
  - `TwoStreamHybrid`：同時處理 15m 特徵與 1m 展開特徵，最後以 Bilinear 融合。
  - `xgb` 相關名稱會回傳 `XGBRegressorModel` 包裝器。
- `suggest_model_hparams` 依 Optuna trial 調整 `d_model`、`n_layers`、`mlp_ratio` 等超參。

### 3.2 訓練流程 (`trainer_base` + `trainer_cls`/`trainer_reg`)
- `get_trainer` 依任務型別選擇分類或回歸訓練器。
- 共同特性：
  - Optimizer 使用 AdamW (fused 版可用時自動啟用)。
  - `build_warmup_scheduler` 僅於前 `train.warmup_epochs` 內進行暖身。
  - AMP 依 `train.amp`、`amp_dtype` 決定混合精度 dtype；loss 調整使用 `GradScaler`。
- 分類 (`trainer_cls`):
  - 損失函數：交叉熵 (含 label smoothing) 或 `BCEWithLogitsFocalLoss` (含 focal、confidence penalty)。
  - 類別權重：`infer_class_weights` 使用訓練集頻率計算並正規化。
  - Collapse guard：可監控正類占比、熵，必要時觸發 LR 衰減或調整懲罰。
  - 溫度校準：`fit_temperature_ce` 對驗證 logits 做後校準。
  - 門檻搜尋：根據 `train.threshold_mode` (`auto_auc` 或 `auto_fbeta`) 掃描閾值；亦可固定門檻。
  - 評估：`compute_cls_metrics` 輸出 ACC/F1/F0.5/MCC、PR/ROC 曲線與混淆矩陣。
- 回歸 (`trainer_reg`):
  - 損失支援 `emamse_pearson`（結合 EMA、MSE、皮爾森）、Huber、MSE。
  - 會輸出 RMSE/MAE/皮爾森/Spearman 等統計，並可執行門檻掃描將回歸轉分類訊號。

### 3.3 評估與匯出 (`train_utils.compute_export_metrices`)
- 於每個 fold：
  - `save_fold_metrics` 將訓練、驗證、測試指標與圖表存入 `trial_xxx/fold_k/`。
  - `plot_test_eval` 繪製 PR/ROC/混淆矩陣（分類）或殘差圖（回歸）。
- 於整體 trial：
  - `save_cv_summary` 聚合所有 fold 指標（平均值、標準差），輸出 JSON/Markdown 摘要。
  - `compute_trial_score` 依 `objective.primary_metric`（例 `test_mcc`, `pearson`）計算 Optuna 分數。

## 4. Trial 產出與後處理
- `objective` 在 trial 結束時：
  - 將 `fold_scores` 取平均，並把實際 seed 寫回 `cfg['seed']`。
  - 以主指標在資料夾名稱加上分數標籤（例 `_mcc+0.312`）。
  - 將完整設定輸出為 `trial_config_<metric>=<score>.yaml`，方便重現。
- `dump_best_yaml`（在主流程完成後）寫出：
  - `runs/<project>/best/best_config.yaml`
  - `runs/<project>/best/best_params.yaml`
  - `runs/<project>/best/selected_features.txt`
- `post_infer.tbm_concat.enabled=true` 時：
  - `export_tbm_predictions_for_trial` 會蒐集各 fold 最佳模型，對指定時間窗的 TBM 事件生成預測：
    - `*_prob` 欄位提供平均機率與機率閾值。
    - `*_vote` 欄位提供多數決結果與票數統計。
  - 產出 CSV (`tbm_with_<column>_<date>.csv`) 與 debug 日誌。

## 5. 操作指南
1. **準備資料**
   - 建立預算特徵檔：確保欄位包含 `datetime`/`timestamp` 與基本 OHLCV，以及配置中啟用的特徵名稱。
   - 事件任務需額外準備 TBM 標籤 CSV (`t0`,`side`,`label` 以及可選 `t1`,`entry_price`)。
2. **調整 `train/config.yaml`**
   - `project_name`：決定輸出路徑與 study 名稱。
   - `data.path`、`freq`、`columns`：對應檔案欄位。
   - `cv`：設定起訖日期、拆分策略、禁運期、fold 數量或 rolling 視窗。
   - `sequence`：視窗長度、stride、scaler（若使用 rolling/ewm scaler，資料需連續）。
   - `train` / `loss`：學習率、batch size、梯度裁剪、loss 選項、門檻搜尋模式。
   - `model`：選擇模型與超參空間；可透過 list 指定 Optuna 搜尋範圍。
   - `label`：定義標籤模式、horizon、ret_type 或 TBM 對齊方式。
   - `features.plan`：啟用或停用預算特徵；必要時同步更新預算檔。
3. **啟動訓練**
   ```bash
   python train/main_train.py
   ```
   - 單機多卡：設定 `search.multi_gpu.enabled` 與 `gpu_ids`，再執行相同指令。
4. **監控進度**
   - 主進程會列印當前 trial 編號、fold 分數、CV 平均等資訊。
   - `runs/<project_name>/trial_xxx/fold_k/` 可隨時查看中間結果、評估圖表。
5. **查看成果**
   - `runs/<project_name>/best/`：最佳 trial 配置、特徵清單。
   - `study.db`：Optuna Study 檔，可用 `optuna dashboard` 或 `optuna studies` 分析。
   - 若啟用 post infer：檢查 `trial_xxx/tbm_with_*.csv` 與 `post_infer_debug.txt`。

## 6. 常見調整與建議
- **增減特徵**：更新 `features.plan` 後，確保預算檔已有對應欄位；無需改動 DataLoader。
- **改變標籤策略**：
  - 時間驅動：調整 `label.mode`、`ret_shift`、`ret_type` 等即可。
  - 事件驅動：TBM CSV 需維持 `t0` 與 `side`；若同一時間有雙邊訊號，請先於資料前處理去重或使用 `keep_sides`。
- **調整序列長度/stride**：在 `sequence.seq_len/stride` 使用 list 讓 Optuna 幫忙搜尋；`stride_anchor` 可鎖定窗口起點。
- **混合模型**：`also_XGB=true` 會為每個 fold 生成 numpy bundle，允許在 objective 中平行訓練 XGBoost。
- **閾值與後處理**：`train.threshold_mode`、`threshold_fbeta` 決定二分類輸出的評估標準；若實務上需固定門檻可設 `mode=fixed`。
- **資源管理**：
  - 大量事件任務建議將 `sequence.preload_to_gpu` 設為 `false`，避免顯存不足。
  - 透過 `train.batch_size`、`grad_accum_steps` 調整計算負載。

## 7. 輸出資料夾結構範例
```
runs/<project_name>/
├─ study.db
├─ best/
│  ├─ best_config.yaml
│  ├─ best_params.yaml
│  └─ selected_features.txt
├─ trial_000_mcc+0.312/
│  ├─ trial_config_test_mcc=0.312.yaml
│  ├─ fold_0/
│  │  ├─ train_metrics.json
│  │  ├─ val_metrics.json
│  │  ├─ pr_curve.png
│  │  ├─ roc_curve.png
│  │  └─ confusion_matrix.png
│  ├─ fold_1/
│  ├─ ...
│  ├─ cv_summary.json
│  ├─ post_infer_debug.txt
│  └─ tbm_with_pred_20230101_20250822.csv
└─ trial_001_...
```

## 8. 模組對照表
| 模組 | 角色 | 重點內容 |
| ----- | ----- | ----- |
| `train/main_train.py` | 入口；建立 Optuna study | `load_cfg`、`prepare_dataframe`、`run_single`/`run_multi` |
| `objective/objective.py` | Trial 主流程 | 超參建議、fold 迴圈、訓練、分數與後處理 |
| `objective/objective_utils.py` | Optuna 輔助 | `suggest_*`、`make_folds`、`compute_trial_score` |
| `build_feature_loader/dataloader.py` | DataLoader 工具 | 特徵/標籤生成、scaler、SeqDataset/EventDataset 組裝 |
| `build_feature_loader/build_features.py` | 標籤邏輯 | 時間驅動與事件驅動標籤計算、TBM 對齊 |
| `dataset/SeqDataset` `dataset/EventDataset` | 資料集物件 | 時序視窗化、事件向量化 |
| `models/model_factory.py` | 模型工廠 | LSTM/Transformer/TwoStream/XGB 組態 |
| `trainer/trainer_base.py` | 訓練共用 | Optimizer、AMP、類別權重、溫度校準 |
| `trainer/trainer_cls.py` | 分類訓練 | Focal loss、collapse guard、門檻搜尋、指標輸出 |
| `trainer/trainer_reg.py` | 回歸訓練 | 混合損失、殘差圖、Pearson/Spearman |
| `train_utils/compute_export_metrices.py` | 評估匯出 | PR/ROC/Confusion matrix、CV 摘要、TBM 匯出支援 |

## 9. 故障排除提示
- 缺少欄位：確認預算檔包含 YAML 指定的時間欄與 OHLCV；若特徵欄名不同，調整 `features.plan` 或重新命名。
- 標籤為空：時間驅動請檢查 `ret_shift` 是否超出資料範圍；事件任務請確認 `tbm_csv_path`、`keep_sides`、`align_method` 設定正確。
- Fold 沒有樣本：縮小 `cv.start_date/end_date` 範圍或調整 `embargo_hours`、`min_train_days`。
- 類別完全傾斜：可啟用 `train.dist_align_weight`、`loss.cls.focal_alpha` 或調整 TBM 標籤策略。
- Post infer 失敗：檢查 `post_infer_debug.txt`，常見原因為 TBM CSV 路徑錯誤或日期範圍超出特徵資料。

