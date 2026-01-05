# Feature Selection Module

本資料夾提供一系列指標計算與降維/統計方法，協助從龐大的特徵庫中挑選高訊號欄位。流程大致可分為：

```
Raw data → features_computer (indicator & normalization)
        → statistics/* modules (Rank Biserial / PCA / UMAP / Hierarchical Corr / SoftDTW KMeans ...)
        → results/* (報表、圖像、篩選後清單)
```

## 目錄結構

```
feature_selection/
├─ features_computer/          # 產生或轉換原始特徵
│  ├─ indicators.py            # 指標/特徵計算
│  ├─ feat_normlization.py     # 常用標準化工具
│  ├─ feature_computer.py      # 計算 feat API
│  └─ features_config.yaml     # 指標配置範例
├─ statistics/                 # 各種特徵評估/降維方法
│  ├─ rank_biserial/           # Rank-Biserial correlation 特徵篩選
│  ├─ pca/                     # Principal Component Analysis
│  ├─ umap/                    # UMAP 可視化/降維
│  ├─ hierarchical_corr/       # 階層式相關矩陣 + 叢集濾除
│  ├─ softdtw_kmeans/          # Soft-DTW KMeans 分群
│  └─ common/                  # 共用工具、Reporter
├─ results/                    # 各方法產生的輸出
└─ readme.md                   # 本說明
```

<!-- ## 各模組說明

| 路徑 | 功能摘要 |
| --- | --- |
| `features_computer/indicators.py` | 讀取原始 OHLCV/成交量等欄位，轉為多種技術指標、統計特徵。可搭配 `features_config.yaml` 決定 pipeline。 |
| `features_computer/feat_normlization.py` | 針對高波動特徵提供 standard/min-max/rolling/ewm 等常見 normalization。 |
| `statistics/rank_biserial/` | `run_rank_biserial.py` 依 label 計算 Rank-Biserial correlation，輸出排序表與 `rank_biserial_*.csv`。`config.yaml` 控制資料路徑、視窗、最終輸出欄數。 |
| `statistics/pca/` | `run_pca.py` 進行 PCA，產生 explained variance、loading、投影資料及圖表。`pca_config.yaml` 為範例設定，`reporter.py` 負責寫入 meta 與圖。 |
| `statistics/umap/` | `run_umap.py` 將高維特徵降至 2D/3D 以供視覺檢查，並輸出 scatter 圖與 CSV。`umap_config.yaml` 可調整 n_neighbors / min_dist / metric。 |
| `statistics/hierarchical_corr/` | `run_hcorr.py` 依相關矩陣做階層式叢集，找出冗餘特徵並輸出報表。`config.yaml` 定義距離門檻、保留策略。 |
| `statistics/softdtw_kmeans/` | 透過 Soft-DTW 度量和 KMeans 對序列特徵做聚類，適合分析多窗口時間序列。`config.yaml` 控制 cluster 數、標準化方式。 |
| `statistics/common/` | 放共用的 `Reporter`、檔案工具與 CLI 介面輔助。 |
| `results/` | 各模組輸出會依模組名稱分資料夾保存，例如 `results/rank_biserial/rank_biserial_60_feat.csv` 或 `results/pca/pca_summary.csv`。 | -->

<!-- ## 執行範例 -->

<!-- 以下命令皆於專案根目錄執行，必要時可透過 `--config` 指定自訂 YAML。

### 1. Rank Biserial 特徵篩選

```bash
python feature_selection/statistics/rank_biserial/run_rank_biserial.py \
  --config feature_selection/statistics/rank_biserial/config.yaml \
  --topk 60 \
  --output_dir feature_selection/results/rank_biserial
```

- `--label_csv`：覆寫配置中的標籤檔路徑。
- `--min_samples`：忽略樣本不足的特徵。

### 2. PCA 降維

```bash
python feature_selection/statistics/pca/run_pca.py \
  --config feature_selection/statistics/pca/pca_config.yaml \
  --standardize true \
  --components 20
```

輸出：`results/pca/` 內含 `cumulative_variance.png`、投影後資料 `pca_output.csv` 等。

### 3. UMAP 可視化

```bash
python feature_selection/statistics/umap/run_umap.py \
  --config feature_selection/statistics/umap/umap_config.yaml \
  --dimensions 2 \
  --metric cosine
```

會生成 `umap_output_oos.csv` 與 `umap_scatter.png`，用於觀察資料分佈。

### 4. SoftDTW KMeans

```bash
python feature_selection/statistics/softdtw_kmeans/run_softdtw_kmeans.py \
  --config feature_selection/statistics/softdtw_kmeans/config.yaml \
  --clusters 6
```

將序列型特徵分群，輸出每群成員列表與 meta 資訊。 -->

## 指標/特徵前處理

1. ###  `FeatureComputer.compute()`
    使用範例:  
    先在 `feature_selection/features_computer/features_config.yaml`中設定
    ```yaml
    # raw data path
    data:
      # 1. Raw ohlcv + FNG
      ohlcv_fng_path: data/derived/ohlcv_fng_15m.csv 
      index_col: timestamp            # 可為 "timestamp" 或 "datetime"；需與原檔一致
      freq: "15min"                   # 用於缺口提示（不自動補齊；補齊請在前處理階段做）
      columns:
        time: ["datetime", "timestamp"]
        ohlcv: ["open", "high", "low", "close", "volume"]

      # 2. Trades feat
      trades:                           # 1 分鐘訊號（15m bar 聚合）    
        trades_min_path: data/derived/btcusdt_trades_1m_stats.csv
        enabled: true
        window_len: 15                 # 每根 15m bar 對應過去 15 根的 1m 視窗（嚴格使用 t-1 以前）
    ```
    
    - `ohlcv_fng_15m.csv`:  cols = `[datetime,timestamp,open,high,low,close,volume,fng]`  
      產出方式見`crawler/utils/merge_fng_into_15m.py`  

    - `btcusdt_trades_1m_stats.csv`: cols = `[datetime,timestamp,open_1m,high_1m,low_1m,close_1m,volume_1m,vwap_1m,trade_count_1m,buy_volume_1m,sell_volume_1m,buy_trades_1m,sell_trades_1m,rv_var_1m,hl_range_1m,mean_intertrade_time_1m,max_intertrade_time_1m]`  
    產出方式見`feature_selection/statistics/trades/trades_analysis.py` (有點複雜，因為他是每天下載一個zip檔案解壓縮之後再計算的之後再說)
    
    
    
    ```python
    # 設定要選用的 feat、export path ...，格式見"feature_selection/features_computer/features_config.yaml"
    cfg_path = Path("feature_selection/features_computer/features_config.yaml")
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # FeatureComputer 給 cfg
    fc = FeatureComputer(cfg)

    # 分別回傳 15min 的 df_ohlcv_feat 跟 1min的 df_trades_feat
    df_ohlcv_feat, df_trades_feat = fc.compute()  # 亦可以傳入自己的 df, e.g. fc.compute(your_df_raw_ohlcv, your_df_raw_trades)
    ```

2. 產生 precomputed_feat:
    於 config 設定輸出 path，以及 nan 處理方法
    ```yaml
    export:
      out_dir: data/precomputed/btcusdt_15m_features/   # folder
      # 1. slow 15min
      ohlcv_feat: btcusdt_15m_features.csv 
      # 2. fast_feat 1min
      trades_feat: btcusdt_1m_trades.csv

      # 3. rolling z_normalization
      feat_normlization: 
        rolling_window: 144     # rolling 往前算 z-score 的 window_size

      # 4. 缺值策略：drop | linear_interp | none
      nan_policy: "linear_interp"                          
    ```
    ```bash
    # 執行
    python feature_selection/features_computer/feature_computer.py

    # output: df_ohlcv_feat, df_trades_feat的.csv
    ```

