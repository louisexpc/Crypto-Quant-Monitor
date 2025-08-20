# Train Core
## 專案目錄與檔案說明
```
train_core/
├─ build_feature_loader/
│  ├─ build_features.py       # 特徵與標籤對齊、報酬計算
│  ├─ dataloader.py           # SeqDataset、FoldGenerator、DataLoader
│  └─ indicators.py           # 技術指標計算、快取與除錯
├─ models/
│  ├─ LSTM.py                 # LSTM_SE 模型
│  ├─ Transformer.py          # TemporalTransformer 模型
│  ├─ fastCNN.py              # AdaptiveKernelCNN
│  ├─ hybird.py               # CNN+Transformer 雙模態融合
│  ├─ model_factory.py        # 模型生成工廠
│  └─ xgb_model.py            # XGBoost wrapper
├─ objective/
│  ├─ objective.py            # Optuna 主目標函式與 trial 執行
│  └─ objective_utils.py      # 超參搜尋、fold 生成、特徵選取
├─ trainer/
│  ├─ trainer_base.py         # 公用工具、AMP、optimizer 建立
│  ├─ trainer_cls.py          # 分類訓練流程
│  ├─ trainer_reg.py          # 回歸訓練流程
│  └─ xgb_trainer.py          # XGBoost 訓練流程
├─ utils/
│  ├─ compute_export_metrices.py # 度量計算、圖表輸出、best YAML 匯出
│  ├─ init_train.py              # 隨機種子、CUDA/TF32 初始化
│  └─ regression_utils.py        # 回歸損失與指標
├─ main_train.py              # 整體入口：讀取 YAML、準備資料、執行 Optuna
└─ config.yaml                # 預設設定檔（資料路徑、模型、CV、特徵等）
```

### build_feature_loader

| 檔案 | 功能概要 |
|-----|---------|
| `build_features.py` | 把 `Indicators` 產生的技術指標與原始 OHLCV 對齊並回傳 `X, y`。支援 `regression` 與 `classification`：<br> - 將時間轉成 UTC 並強制網格化<br> - `horizon` 未來報酬計算（`logret` 或 `simple`）<br> - 回歸回傳連續 `target`；分類依 `cls_threshold` 產生 0/1 `label`<br> - 清除 NaN / inf、限制時間範圍 |
| `dataloader.py` | 提供訓練資料載入：<br> - `SeqDataset`：把 `DataFrame` → `[N,T,F]` 時序張量、對應標籤<br> - `FoldGenerator`：支持 `Anchored`、`Rolling`、`OddEven` 等時間窗切分，含 `embargo` 保護<br> - `make_loaders_for_fold`：依 fold 建立 `train/val/test` 的 `DataLoader`，可同時輸出 XGB 所需的 numpy |
| `indicators.py` | 技術指標計算核心：<br> - 正規化 OHLCV（UTC、排序、去重、缺口警示）<br> - `PAPER_TOP8_PLAN` / `BEST20_PLAN` 預設特徵方案<br> - 自訂多項 `pandas_ta` 指標與原始欄位 builder<br> - 以「資料指紋 + 計畫指紋」進行 Parquet 快取<br> - 防洩漏：全部特徵 `shift(1)`<br> - 除錯：欄位重複、未知指標、缺欄位、頻率缺口等皆會拋錯或提示 |

### models

| 檔案 | 模型摘要 |
|-----|---------|
| `LSTM.py` | `LSTM_SE`：LSTM + Squeeze-and-Excitation + mean/max pooling + MLP 頭；輸出 `[B]` 回歸或 `[B,C]` 分類 |
| `Transformer.py` | `TemporalTransformer`：<br> - Sin-Cos 位置編碼<br> - Pre-Norm Transformer block + DropPath<br> - 可選 pooling (`cls/mean/last/attn`)<br> - 支援 causal mask、防止看未來 |
| `fastCNN.py` | `AdaptiveKernelCNN`：多 kernel 1D convolution + 注意力加權 + 殘差，對 OHLCV 等快特徵進行特徵萃取 |
| `hybird.py` | `CNNTransformerBilinearModel`：<br> - `AdaptiveKernelCNN` 處理快窗特徵<br> - `TemporalTransformer` 處理慢窗特徵<br> - `FactorizedBilinear` 低秩雙線性交叉融合 + 兩個輔助頭 (deep supervision) |
| `model_factory.py` | 依 `config.yaml` 的 `model.name` 建立對應模型 (`LSTM_SE` / `TemporalTransformer` / `XGBRegressorModel`) |
| `xgb_model.py` | 封裝 XGBoost：`from_cfg` 讀 YAML -> `build()` 產生 `xgboost.XGBRegressor`，含 `early_stopping_rounds` 等參數 |

### objective

| 檔案 | 用途 |
|----|----|
| `objective.py` | Optuna trial 主入口：<br> - 複製 `base_cfg` 並隨機建議超參（learning rate、grad_clip 等）<br> - 呼叫 `suggest_model_hparams`、`suggest_rolling_and_cv`、特徵抽樣等<br> - 依 `make_folds` 建立所有 fold，透過 `trainer` 逐 fold 訓練<br> - `compute_trial_score` 根據 `primary_metric`（回歸/分類不同）計算分數<br> - 保存每個 trial 的 config 與結果 |
| `objective_utils.py` | 支援函式：<br> - `suggest_*` 系列為 Optuna 提供搜尋空間 (float/int/categorical)<br> - `make_folds` 建立 `OddEven`/`Anchored`/`Rolling` 折疊<br> - `map_feature_to_columns` & `get_enabled_feature_names`：從 YAML 特徵計畫映射成實際欄位，驗證欄位存在<br> - 模型超參數設定 (`suggest_model_hparams`) |

### trainer

| 檔案 | 描述 |
|-----|-----|
| `trainer_base.py` | 公用訓練工具：<br> - 取得任務型別、AMP dtype、optimizer、scheduler<br> - class weight/最佳 threshold/溫度校準等分類工具<br> - `get_trainer` 對應 `classification` 或 `regression` |
| `trainer_cls.py` | 分類訓練迴圈：<br> - 支援 CE / Focal Loss、class weight、自動 threshold/溫度校準<br> - early stopping、最佳模型 state 儲存、驗證/測試度量輸出 |
| `trainer_reg.py` | 回歸訓練迴圈：<br> - 支援 `EMAMSE_PearsonLoss`、MSE、Huber 等<br> - (可選) 若模型為 `XGBRegressorModel` 則改走 `_train_one_fold_xgb`<br> - 計算 `mixed` objective、Pearson、RMSE 等指標<br> - early stopping、結果與圖表輸出 |
| `xgb_trainer.py` | XGBoost 專用訓練流程：<br> - 支援 GPU handoff (CuPy)<br> - 早停 (`es_step` / `es_patience` / `es_min_delta`)<br> - 儲存每輪 RMSE & Pearson，最後依驗證 RMSE 選最佳輪 |

### utils

| 檔案 | 功能 |
|-----|------|
| `init_train.py` | 設定隨機種子、啟用 CUDA TF32、設定 flash attention/matmul 精度等 |
| `compute_export_metrices.py` | 計算分類/回歸指標、畫 ROC/PR/混淆矩陣、回歸分析圖、`dump_best_yaml` 輸出最佳 trial 設定 |
| `regression_utils.py` | 回歸損失 (`EMAMSE_PearsonLoss`) 與常用度量、回歸→二分類門檻搜尋 (`find_best_threshold_for_regression`) |

### 其他

| 檔案 | 描述 |
|-----|-----|
| `main_train.py` | 主要入口：<br> 1. 讀取 YAML (`load_cfg`)<br> 2. `prepare_dataframe`：讀 OHLCV → `Indicators` 計算技術指標 → `build_features_and_label` 產生 `X, y`<br> 3. `build_study`：建立 Optuna Study（sqlite DB）<br> 4. `study.optimize`：呼叫 `objective` 逐 trial 搜尋<br> 5. 最佳結果輸出、`dump_best_yaml` |
| `config.yaml` | 預設設定檔，包含：<br> - `data`：原始 OHLCV 檔路徑、頻率、欄位<br> - `cv`：跨月/rolling 等 cross-validation 參數<br> - `model`：模型種類與超參<br> - `features`：指標計畫與開關<br> - `train`、`loss`、`objective`、`sequence` 等細節 |

---

## 資料處理 Pipeline

1. **讀取原始 OHLCV**  
   `prepare_dataframe` 以 `config.data.path` 指向的 CSV/Parquet 讀資料。  
   - 允許 `datetime` 或 `timestamp` 欄位；全部轉 UTC `DatetimeIndex`。  
   - 檢查重複、排序並可提醒缺 K 線。  

2. **技術指標計算 (`Indicators`)**  
   - 規格化 OHLCV 後，依 `config.features.plan` 逐項指標計算。  
   - 以資料指紋 + 計畫指紋決定快取檔名；已有快取直接讀取。  
   - 所有特徵 `shift(1)` 防洩漏；欄位重複會拋錯。  

3. **特徵與標籤建立 (`build_features_and_label`)**  
   - 轉換時區、強制完整網格，確保特徵與 OHLCV 對齊。  
   - 計算未來 `horizon` 步的 `logret` 或 `simple` 報酬。  
   - `regression`：連續 `target`; `classification`：`target` > `cls_threshold` → `label` 0/1。  
   - 清除無效時間點、NaN/inf、依 `start_date` ~ `end_date` 範圍過濾。  

4. **Fold 生成 (`FoldGenerator`)**  
   - 支援 `OddEven`：奇數月訓練、偶數月測試。  
   - `Anchored`：從起始時間開始累積資料，每個測試月前加 `embargo` 小時禁區。  
   - `Rolling`：固定訓練窗大小往前滑動。  

5. **DataLoader (`make_loaders_for_fold`)**  
   - 依 fold 時間 index 切出 `train/val/test`。  
   - `scaler` (Robust/Standard/MinMax) 僅 fit 在訓練集。  
   - `SeqDataset` 產生 `[batch, seq_len, features]`；標籤型別依任務決定。  
   - 可同時產生 XGB 需要的 numpy 陣列。  

---

## 指標產生與排錯

* `Indicators._normalize_ohlcv`：  
  - 不存在時間欄時拋錯。  
  - 缺 OHLCV 欄位或時區錯誤會提出。  
  - 若 `freq_check` 設定則印出缺口警告。

* `Indicators.compute`：  
  - `plan` 會先經 `_prune_plan` 移除未啟用指標。  
  - `Unknown indicator` 會拋 `ValueError`。  
  - 指標欄名重複時拋錯，避免 features 覆蓋。  
  - 指標計算後統一 `float32`，NaN/inf 會被清除。  

* `objective_utils.map_feature_to_columns`：  
  - 輸入 YAML 中 `name/kwargs`，自動推測實際欄位名稱。  
  - 欄位不存在時拋錯，避免 Optuna 選到不存在的特徵。  

* `train` / `loss`：  
  - 若模型輸出維度與 `num_classes` 不符或 XGB 設定缺值，會即時報錯。  
  - `trainer_cls`、`trainer_reg` 均在第一個 batch 印出 debug shape，以利排查。  

---

## Pipeline 執行流程

1. **入口 `main_train.py`**  
   - `load_cfg` → 讀取 `config.yaml`  
   - `set_seed`、`setup_cuda_acceleration` 初始化  
   - `prepare_dataframe` → 產生 `df` (特徵+label) 與 `pt_bundle` (指標快取路徑等)  
   - `build_study` → Optuna Study (SQLite)  
   - `study.optimize` → 迴圈執行 `objective`  

2. **在 `objective` 中的每次 trial**  
   - 隨機建議：序列長度、embargo、learning rate、模型隱層大小等  
   - `get_enabled_feature_names` 解析目前啟用的特徵欄  
   - `make_folds` → 依 CV 參數產生折疊  
   - 逐 fold：  
     - `make_loaders_for_fold` → DataLoader / XGB numpy  
     - `build_model` → 建立 DL 或 XGB 模型  
     - `train_one_fold` → 執行訓練、驗證、測試；輸出指標與圖表  
     - `compute_trial_score` → 回傳該 fold 分數給 Optuna  
   - 平均所有 fold 分數返回給 Optuna  
   - 記錄 trial 詳細設定、選到的特徵等  

3. **完成搜尋後**  
   - 印出 `best_trial.params`、最佳分數  
   - `dump_best_yaml`：輸出 `best_params.yaml`、`best_config.yaml`、`selected_features.txt`  

---

## 排錯 & 調試建議

| 情況 | 解法 |
|-----|-----|
| **缺少欄位/時間欄錯誤** | 確認原始 OHLCV 資料含 `timestamp` 或 `datetime`，且欄位名與 `config.data.columns` 相符。 |
| **freq 缺口警告** | `Indicators` 只提醒不補值；若缺口過大請自行填補或調整資料。 |
| **Unknown indicator** | 檢查 `config.features.plan` 中的 `name` 是否有對應 builder；必要時在 `Indicators` 補入建構函式。 |
| **特徵欄名重複** | 代表計畫中同名指標重複或 builder 產生相同欄位；需修改 plan 或自訂命名。 |
| **模型輸出維度不符** | `trainer_cls`、`trainer_reg` 初始 batch 會檢查 `out_dim` 與 `num_classes`，必要時調整 `model.head` 或 `num_classes`。 |
| **Optuna trial 被 pruned** | 受 `MedianPruner` 控制；可關閉 `enable_prune` 或增加 `pruner_warmup_folds`。 |
| **XGBoost GPU 不可用** | `xgb_trainer` 會自動判斷 `device`；若無 CuPy 會退回 CPU。請確保 CUDA 驅動與依賴安裝完整。 |

---

## 快速開始（範例流程）

1. **準備資料**：將 1H 交易所 OHLCV CSV 放置於 `config.yaml` 中 `data.path` 指定位置，欄位需包含 `timestamp` 或 `datetime` 與 OHLCV。  
2. **調整 `config.yaml`**：  
   - 調整 `cv` 參數、想測試的 `model`、啟用的 `features.plan` 指標等。  
   - 若只想跑單 trial，可將 `search.n_trials` 設為 1。  
3. **執行**：  
   ```bash
   python train_core/main_train.py  # 預設會讀 train_core/config.yaml
   ```  
4. **查看結果**：  
   - `runs/<project_name>/` 下有 `study.db` 與各 `trial_*` 目錄。  
   - 最佳結果於 `runs/<project_name>/best/`：`best_params.yaml`, `best_config.yaml`, `selected_features.txt` 等。  

---

## 依賴
見 train_requirements.txt

---

## 補充說明

* `Indicators` 產生的特徵會快取於 `config.features.cache_dir`（預設 `cache_features`），避免重複計算。  
* 若要增添新指標，只需在 `Indicators._builders` 中註冊對應函式，並於 `config.yaml` 的 `features.plan` 啟用即可。  
* `regression_to_class` 區塊可將回歸輸出轉成訊號（例如多空判斷），支援以 F-beta 最優門檻或固定門檻。  
* `dump_best_yaml` 會將搜尋空間替換為實際值，方便後續直接載入最佳設定重現結果。  

---
