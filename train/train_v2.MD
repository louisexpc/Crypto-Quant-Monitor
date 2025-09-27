# Crypto-Quant Train Pipeline

本說明文件整理 `train/` 目錄下的訓練平台，涵蓋從預算特徵載入、交叉驗證切分、模型建構、訓練與後處理到事件推論輸出的完整流程。除了描述各模組與主要函式，也提供典型使用方式與管線資料流，方便快速導覽與擴充。

---

## 專案架構
```
train/
├── main_train.py                  # Optuna 入口：讀取 YAML、準備資料、啟動 trial 搜尋
├── config.yaml                    # 預設專案設定（資料路徑、CV、模型、特徵、後處理）
├── NOTE.md                        # 內部開發備忘錄
├── research_problem.MD            # 研究議題與 TODO
├── train.MD                       # 原始簡版說明
├── train_v2.MD                    # 本文件
├── train_requirements.txt         # 訓練環境依賴
├── __init__.py                    # 暴露主要 namespace（例如 `train.main_train`）
│
├── build_feature_loader/
│   ├── build_features.py          # 標籤計算、TBM 對齊與分數差分工具
│   ├── dataloader.py              # Fold 切分、特徵選取、Seq/Event DataLoader 建構
│   ├── indicators.py              # 特徵計算元件與指標維護
│   └── scalar.py                  # 時序安全 scaler 與 ColumnSubsetScaler
│
├── dataset/
│   ├── time_dataset.py            # `SeqDataset`：時間視窗張量化
│   └── event_dataset.py           # `EventDataset`：TBM 事件序列資料集
│
├── models/
│   ├── LSTM.py                    # LSTM 系列模型（含 SE、Temporal head）
│   ├── Transformer.py             # `TemporalTransformer` backbone
│   ├── two_stream_model.py        # Minute×15m 雙模態融合模型
│   ├── xgb_model.py               # XGBoost wrapper
│   └── model_factory.py           # 依 config 建立對應模型
│
├── objective/
│   ├── objective.py               # Optuna trial 主函式
│   ├── objective_utils.py         # 搜尋空間、fold 生成、小工具
│   └── export_tbm_pred.py         # Trial 完成後的 TBM 併回匯出
│
├── trainer/
│   ├── trainer_base.py            # Optimizer/AMP/threshold/溫度校準共用
│   ├── trainer_cls.py             # 分類任務訓練與指標輸出
│   ├── trainer_reg.py             # 回歸訓練流程
│   └── xgb_trainer.py             # XGBoost 訓練流程（also_XGB）
│
├── train_utils/
│   ├── init_train.py              # 隨機種子、CUDA/TF32 初始化
│   ├── compute_export_metrices.py # PR/ROC/匯整 YAML 與 TBM 匯出支援
│   ├── metrics_cls.py             # 分類指標計算
│   ├── metrics_reg.py             # 回歸指標與混合目標
│   ├── regression_utils.py        # 回歸損失組件
│   ├── analysis.py                # SNR 水平策略分析工具
│   └── label.py                   # 事件訊號生成輔助（TBM 前處理）
│
├── trainer_utils/ (無)            # 保留
├── export_model/                  # 產線/部署輸出暫存（目前空）
├── post_infer/                    # 後處理擴充掛鉤（目前空）
└── random_alpha_generator/
    ├── random_alpha_generator.py  # 遺傳演算法 Alpha 因子搜尋器
    ├── node.py                    # 表達式樹節點與安全運算
    └── __init__.py
```

---

## 實作流程總覽
1. **設定載入**：`main_train.load_cfg` 讀取 YAML，`set_seed` 與 `setup_cuda_acceleration` 初始化環境。
2. **資料準備**：`prepare_dataframe` 建立 UTC 索引並驗證預算特徵檔；實際特徵/標籤在 fold 內再生成。
3. **Optuna Trial**：`objective.objective` 複製設定、建議超參、生成 fold，並為每個 fold 準備 DataLoader 與模型。
4. **訓練**：`trainer.get_trainer` 選擇分類或回歸訓練器，執行評估、threshold 搜尋、溫度校準並儲存結果。
5. **評估與匯出**：`compute_export_metrices` 匯整指標、圖表、best YAML；`export_tbm_predictions_for_trial` 可選地將各 fold 模型推論併回 TBM。
6. **Trial 管理**：每個 trial 會在 `runs/<project>/trial_xxx[_score-tag]/` 留存設定、指標與後處理檔案。

---

## Pipeline Data Flow
```mermaid
flowchart TD
    CFG[train/config.yaml] --> MainTrain[main_train.run_single()] -->|df, cfg| Objective[objective.objective]
    Objective -->|make_folds| FoldGen[build_feature_loader.dataloader.FoldGenerator]
    Objective -->|feature plan| Plan[build_feature_loader.dataloader.select_plan_columns]
    Objective -->|time-driven| TimeLoad[build_feature_loader.dataloader.make_time_loaders_for_fold]
    Objective -->|event-driven| EventLoad[build_feature_loader.dataloader.make_event_loaders_for_fold]
    TimeLoad --> Trainer[get_trainer(cfg)]
    EventLoad --> Trainer
    Trainer --> Metrics[train_utils.compute_export_metrices.save_fold_metrics]
    Trainer --> Models[models.model_factory.build_model]
    Metrics --> CVSummary[save_cv_summary]
    Objective --> TBMExport[objective.export_tbm_pred.export_tbm_predictions_for_trial]
    TBMExport --> Outputs[[runs/<project>/trial_xxx/tbm_with_*.csv]]
    CVSummary --> BestDump[dump_best_yaml]
    BestDump --> Final[[runs/<project>/best/]]
```

---

## 核心模組與 API
以下依目錄列出主要 Python 檔案、核心函式與使用建議。若省略 `from ... import ...`，皆以 `train/` 為相對根目錄。

### Root 層級
- **main_train.py**
  - `load_cfg(path) -> dict`：讀取 YAML。常與 `cfg = load_cfg("train/config.yaml")` 搭配。
  - `prepare_dataframe(cfg) -> (pd.DataFrame, dict|None)`：驗證 `cfg.data.path`，回傳索引用 df；若缺時間欄位會直接 raise。
  - `build_study(cfg, run_dir, parallel=False)`：建立 Optuna Study。可在 notebook 中呼叫以重用現有 `run_dir`。
  - `run_single(cfg_path, worker_tag=None)` / `run_multi(cfg_path)`：訓練主入口。CLI 常用 `python train/main_train.py`。

- **config.yaml**
  - 定義專案名稱、資料路徑、CV 範圍、模型結構、訓練參數、標籤策略、特徵 plan、post-infer 等。
  - 建議先複製為自訂檔案再修改，以保留參考設定。

- **train_requirements.txt**：列出訓練所需套件版本，可用 `pip install -r train/train_requirements.txt` 設定環境。

- **__init__.py**：將常用工具（如 `set_seed`, `setup_cuda_acceleration`）暴露給外部匯入。

### build_feature_loader/
- **build_features.py**
  - `assign_tbm_labels_to_df(df_15m, tbm_csv_path, ...) -> pd.DataFrame`：將 TBM 事件標籤對齊進 15m 網格。
  - `create_labels_adaptive(df, mode, ...) -> pd.DataFrame`：時間驅動標籤；支援 `mode`（`vol`, `bps`, `quantile`, `triple`, `event_tbm`）。
  - `create_label(df_base, cfg, return_what="auto") -> pd.Series`：依任務輸出 `y_reg` 或 `y_cls`；常由 DataLoader 呼叫。
  - **使用**：
    ```python
    from train.build_feature_loader.build_features import create_label
    y = create_label(price_df, cfg, return_what="cls")
    ```

- **dataloader.py**
  - `select_plan_columns(feat_df, cfg) -> list[str]`：讀取 `features.plan` 過濾特徵欄位。
  - `FoldGenerator(dt_index, mode, start_month, end_month, ...)`：生成 Rolling 或 Purged k-fold 的時間遮罩，並附 embargo。
  - `make_time_loaders_for_fold(df, feat_cols, target_col, fold, cfg, also_XGB, pre_feat_df)`：建立時間驅動的 `SeqDataset` DataLoader 與 XGB numpy 包。
  - `make_event_loaders_for_fold(df, feat_cols, target_col, fold, cfg, also_XGB, pre_feat_df)`：建立事件驅動的 `EventDataset` DataLoader。
  - `split_fold_to_indices(df, fold, cfg)`：回傳 train/val/test DatetimeIndex。
  - **使用**：Optuna trial 內會自動呼叫；若需手動驗證資料，可在 notebook 直接載入。

- **indicators.py**
  - `FeatureComputer`：集中管理所有特徵 builder，並避免洩漏（透過 `shift(1)`）。
  - `DEFAULT_MINUTE_PREFIXES`、`PAPER_TOP8_PLAN`、`BEST20_PLAN` 等：預設特徵方案。
  - 發現未知指標或欄位缺失時會即時拋錯，提示在 YAML 中修正。

- **scalar.py**
  - `pick_cols_to_scale(df, feat_cols)`：自動略過 sign-like 欄位。
   - `ColumnSubsetScaler`：將 sklearn scaler 限制在指定欄位；`fit_df` / `transform`。
  - `TimeSafeScaler(mode, window, min_frac)`：滾動/ewm/robust 滙前標準化，全部統計量皆 `shift(1)`，避免看未來。
  - `_get_scaler(kind, window, min_frac)`：統一取得 scaler 或 `None`。

### dataset/
- **time_dataset.py**
  - `SeqDataset(X_df, y_s, seq_len, scaler=None, device='cuda', label_dtype='auto', stride=1, anchor=0)`：將 DataFrame 轉為 `[N,T,F]` 序列張量，支援 stride 與 anchor 控制；`__getitem__` 回傳 `(X_i, y_i)`。

- **event_dataset.py**
  - `_to_utc_index`, `_drop_label_like_cols`：內部工具，確保索引一致。
  - `EventDataset(feat_df, tbm_csv_path, seq_len, feature_cols, keep_sides, align_method, device, allowed_align_index)`：一次性將 TBM 事件轉成 `[N,L,F]` 張量，並紀錄 drop statistics。
  - `load_features_csv(path, time_col, assume_tz)`：輔助載入外部特徵 CSV。
  - `EventDataset.stats` 提供 `total/kept/drop_short/drop_nan_or_inf/drop_outside`，可協助檢查資料品質。

### models/
- **model_factory.py**
  - `build_model(cfg, n_features, columns)`：依 `cfg.model.name` 建立模型，支援 `LSTM_SE`, `LSTM_Temporal`, `TemporalTransformer`, `TwoStreamHybrid`, `xgb` 系列。

- **LSTM.py**
  - `LSTM_SE(input_dim, hidden_size, n_layers, dropout, bidirectional, num_classes)`：含 Squeeze-and-Excitation 與 pooling 的 LSTM。
  - `LSTM_Temporal(...)`：LSTM + 注意力/Pooling Head，可選是否回傳注意力 α。

- **Transformer.py**
  - `TemporalTransformer(input_dim, num_classes, d_model, n_heads, num_layers, mlp_ratio, dropout, attn_dropout, pooling)`：Pre-Norm Transformer，支援多種 pooling 模式與 DropPath。

- **two_stream_model.py**
  - `TwoStreamHybrid(...)`：15m 特徵走 Transformer、1m 特徵走 LSTM，最後以 Bilinear 融合並提供輔助頭。

- **xgb_model.py**
  - `XGBRegressorModel.from_cfg(cfg)`：載入設定並建構 `xgboost.XGBRegressor`。可與 `trainer/xgb_trainer.py` 配合。

### objective/
- **objective.py**
  - `objective(trial, base_cfg, df, run_dir, pt_bundle)`：Optuna trial 主函式，負責：
    1. 建議 learning rate、grad clip、threshold 等超參。
    2. 依 `cfg.label.mode` 選擇時間或事件 DataLoader。
    3. 建構模型並調用 `train_one_fold`，將結果記錄於 trial 目錄。
    4. 平均 fold 分數、儲存可重現 config、呼叫 TBM 匯出。

- **objective_utils.py**
  - `suggest_float/int/cat`：自動判斷 list/區間/單值。
  - `suggest_rolling_and_cv`：針對序列長度、stride、Rolling 參數提供 Optuna 搜尋。
  - `make_folds(df, cfg)`：依 `cv.type` 呼叫 `FoldGenerator`。
  - `suggest_model_hparams(trial, cfg)`：依模型類型調整 d_model、n_heads 等。
  - `compute_trial_score(result, cfg)`：讀取 trainer 回傳指標決定 Optuna 目標值。

- **export_tbm_pred.py**
  - `_load_precomputed_features`, `_align_times`：輔助對齊特徵與事件。
  - `export_tbm_predictions_for_trial(cfg, df_index, folds, fold_models, date_start, date_end, ...)`：將各 fold 模型推論合併，支援 mean-prob / fold-vote / both 決策模式，並輸出 CSV。
  - **使用**：`objective` trial 完成後自動觸發；亦可在離線腳本自行呼叫以重建推論。

### trainer/
- **trainer_base.py**
  - `get_trainer(cfg)`：依任務型別返回 `train_one_fold` 函式。
  - `build_optimizer`, `build_warmup_scheduler`, `build_grad_scaler`：統一訓練器使用的 Optimizer/AMP 工具。
  - `infer_class_weights(train_loader, num_classes, device)`：推估類別權重。
  - `find_best_threshold_by_auc`, `fit_temperature_ce`：二分類門檻與溫度校準工具。

- **trainer_cls.py**
  - `BCEWithLogitsFocalLoss`：支援 focal γ, α、confidence penalty、各種 logits 形狀自動對齊。
  - `build_classification_loss(cfg, class_weights)`：根據設定建立 CE 或 focal loss。
  - `train_one_fold(model, train_loader, val_loader, test_loader, cfg, device, fold_id, export_dir)`：主要訓練流程；輸出訓練紀錄、最佳 checkpoint、指標 JSON、圖表。
  - 支援 collapse guard、分佈對齊、threshold grid search、溫度校準。

- **trainer_reg.py**
  - `train_one_fold`：回歸任務訓練流程，支援 `emamse_pearson`、Huber、MSE；可將回歸輸出轉成訊號評估。

- **xgb_trainer.py**
  - `train_xgb_per_fold(xgb_pack, cfg, device, fold_id, export_dir)`：訓練 XGBoost，輸出與 DL 相同的指標 JSON；搭配 `also_XGB=true` 使用。

### train_utils/
- **init_train.py**
  - `set_seed(seed, deterministic=True)`：一次設定 Python/numpy/torch 的亂數。
  - `setup_cuda_acceleration()`：啟用 TF32、CUDNN benchmark、靜音部分警告。

- **compute_export_metrices.py**
  - `ClassificationReporter.plot_eval(...)`：繪製 ROC/PR/Confusion Matrix 並輸出 JSON 摘要。
  - `RegressionReporter`：輸出散點、殘差、Pearson/Spearman、可選 threshold sweep。
  - `save_fold_metrics(result, export_dir, task_type)`：將 fold 指標寫成 JSON/圖檔。
  - `save_cv_summary(fold_results, export_dir, task_type)`：聚合全部 fold 指標。
  - `dump_best_yaml(study, cfg, run_dir)`：匯出最佳試驗的 YAML 與特徵清單。

- **metrics_cls.py / metrics_reg.py**
  - 提供計算分類（ACC/F1/Fβ/MCC/PR-AUC/ROC-AUC）與回歸（Pearson/Spearman/RMSE/MAE）指標的純函式。

- **regression_utils.py**
  - `EMAMSELoss`, `PearsonLoss`, `MixedObjective` 等：回歸損失與混合指標的 PyTorch 實作。

- **analysis.py**
  - `StrategyAnalyzer`、`Level`：模擬支撐/阻力、產生策略訊號，供 `train_utils/label.py` 或外部生成事件用。

- **label.py**
  - `generate_signals(df, lookback, symbol, timeframe, volume_ema_window)`：從策略分析器輸出事件 `DataFrame`，可用於製作 TBM CSV。

### random_alpha_generator/
- **random_alpha_generator.py**
  - `random_tree(depth, terminals, operators)`：隨機生成 Alpha 表達式樹。
  - `Individual(tree)` 類別：封裝 alpha 的 fitness、IC、Sharpe；支援 multiprocessing 評估（`_mp_init`, `_mp_eval_worker`）。
  - `calc_ic`, `calc_sharpe`, `safe_corr`, `safe_sharpe`：因子評估指標。
  - 適合獨立啟動因子搜尋；可將結果轉換成 `node_to_dict`/`node_from_dict` 作序列化。

- **node.py**
  - 定義 `Node`, `Leaf`, `OpNode` 類別與 `OP_META`，提供安全的對齊、窗口處理、常數檢查等工具。
  - 可搭配 `random_alpha_generator.py` 進行 crossover/mutation。

### 其他目錄
- **export_model/**：預留部署時導出模型/指標的空目錄。
- **post_infer/**：後處理模組掛鉤位置，預設為空，若需要自訂 post-infer 流程可放於此。

---

## 操作指南
1. **資料準備**
   - 檢查 `config.yaml` 中 `data.path` 是否指向包含 `datetime/timestamp`、OHLCV 與特徵欄位的預算檔。
   - 若使用 TBM 事件，`label.tbm_csv_path` 必須包含 `t0`, `label`, `side`（可選 `t1`, `entry_price`）。

2. **啟動訓練**
   ```bash
   python train/main_train.py                # 單卡訓練（執行 run_single）
   python train/main_train.py --help         # 閱讀 CLI （在自訂 wrapper 時）
   ```
   - 多卡：設定 `search.multi_gpu.enabled=true` 並指定 `gpu_ids`，`main_train` 會以 multiprocessing 啟動 `run_multi`。

3. **觀察輸出**
   - `runs/<project_name>/trial_xxx/`：trial 設定、fold 指標、圖表、post-infer 結果。
   - `runs/<project_name>/best/`：最佳試驗的 `best_config.yaml`, `best_params.yaml`, `selected_features.txt`。

4. **重複推論 / TBM 併回**
   ```python
   from train.objective.export_tbm_pred import export_tbm_predictions_for_trial
   export_tbm_predictions_for_trial(
       cfg=cfg,
       df_index=df.index,
       folds=folds,
       fold_models=fold_models,
       date_start="2023-01-01",
       date_end="2025-08-22",
       save_to_path="runs/demo/tbm.csv"
   )
   ```

5. **調整特徵或標籤**
   - 修改 `features.plan` 或 `features.min_trade_feat` 後，確保預算檔已有對應欄位。
   - 調整 `label.mode`, `ret_shift`, `ret_type`, `fracdiff`，或替換為 `event_tbm` 模式。

6. **XGBoost 同步訓練**
   - 設 `also_XGB: true`，`make_*_loaders_for_fold` 會另外回傳 numpy 包給 `trainer/xgb_trainer.py`。
   - 可在 trial 目錄檢視 `xgb_metrics.json` 與 `xgb_feature_importance.csv`（若啟用）。

---

## 常見擴充建議
- **新增自訂指標**：在 `build_feature_loader/indicators.py` 登錄 builder，並於 YAML `features.plan` 啟用。
- **自訂模型**：擴充 `models/` 下的檔案，並在 `model_factory.build_model` 加入新分支，Optuna 即可納入搜尋。
- **調整後處理**：在 `train_utils/compute_export_metrices.py` 新增報表函式，或在 `post_infer/` 建立自訂腳本並於 `objective` 完成後呼叫。
- **離線資料檢查**：可單獨引入 `FoldGenerator` 與 `make_time_loaders_for_fold` 在 notebook 中檢視 fold/特徵內容，避免 trial 期間才發現問題。

---

若要進一步客製化（例如整合新的縮放器、替換損失函式、添加 callback），建議依據上述檔案位置擴充對應模組，並在 `config.yaml` 加入可調旗標，以便透過 Optuna 搜尋或環境變數控制。


train/
├── core/                         # “只能被依賴、不可向下依賴”的超薄核心
│   ├── config_loader.py          # 讀/驗 YAML + schema 鎖定
│   ├── context.py                # RunContext (cfg、paths、seed 等不可變物件)
│   └── orchestrator.py           # CLI/頂層流程（只做導線，不做業務）
│
├── data/                         # 單一“真相源頭”：特徵、標籤、CV 分割與縮放
│   ├── feature_store.py          # 預算特徵與標籤的集中快取（一次建好，處處使用）
│   ├── folds.py                  # Rolling/Purged/Embargo/CPCV 的 fold 規格
│   ├── scalers.py                # 時序安全 scaler 工廠
│   └── dataloaders/
│       ├── base.py               # 共用切片/對齊/XGB pack
│       ├── time_loader.py        # 時間驅動 DataLoader
│       └── event_loader.py       # 事件驅動 DataLoader（TBM）
│
├── models/                       # 只負責“構建網路”；無 I/O、無 CV、無訓練程式碼
│   ├── registry.py               # name -> builder 的登錄表
│   ├── LSTM.py
│   ├── Transformer.py
│   ├── two_stream_model.py
│   └── xgb_model.py
│
├── training/                     # “一個 fold 的訓練怎麼跑”
│   ├── trainers/
│   │   ├── classification.py     # 單 fold 訓練；回傳 FoldResult
│   │   ├── regression.py
│   │   └── xgb.py
│   ├── metrics/                  # 純函式，無狀態
│   └── hooks.py                  # threshold grid / 溫度校準 / collapse guard
│
├── evaluation/                   # “把結果說人話”（圖表/JSON/CSV）
│   ├── reporters/
│   │   ├── classification_reporter.py
│   │   └── regression_reporter.py
│   └── exporters/
│       ├── best_yaml.py
│       └── tbm_exporter.py
│
├── pipeline/                     # 管試驗，不管“怎麼訓練”
│   ├── trial_runner.py           # Loop folds、串 trainer + reporter
│   └── search/
│       ├── objective.py          # Optuna 的薄包裝
│       └── space.py              # suggest_* 與搜尋空間定義
│
└── post_infer/                   # 可選後處理掛鉤（TBM 併回等）
